//! `OpenAI` embedding client implementation.
//!
//! Provides embedding generation using `OpenAI`'s embedding models:
//! - `text-embedding-3-small`: 1536 dimensions (default), supports dimension reduction
//! - `text-embedding-3-large`: 3072 dimensions (default), supports dimension reduction
//! - `text-embedding-ada-002`: 1536 dimensions (legacy, no dimension reduction)
//!
//! # Example
//!
//! ```no_run
//! use neuromance_client::embedding::{EmbeddingClient, EmbeddingConfig};
//! use neuromance_client::openai::OpenAIEmbedding;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = EmbeddingConfig::openai_small("sk-...");
//! let client = OpenAIEmbedding::new(config)?;
//!
//! // Single embedding
//! let vector = client.embed("Hello, world!").await?;
//! println!("Embedding dimensions: {}", vector.len());
//!
//! // Batch embedding (more efficient)
//! let vectors = client.embed_batch(&["Hello", "World"]).await?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use base64::prelude::*;
use log::{debug, error, warn};
use reqwest_middleware::ClientWithMiddleware;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use reqwest_retry_after::RetryAfterMiddleware;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};

use crate::embedding::{
    Embedding, EmbeddingClient, EmbeddingConfig, EmbeddingInput, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, EncodingFormat, models,
};
use crate::error::{ClientError, ErrorResponse};

/// Maximum number of inputs allowed in a single batch request.
///
/// `OpenAI` limits embedding requests to 2048 inputs per request.
const MAX_BATCH_SIZE: usize = 2048;

/// Dimension constraints for `OpenAI` embedding models.
struct DimensionConstraints {
    min: u32,
    max: u32,
}

impl DimensionConstraints {
    const fn new(min: u32, max: u32) -> Self {
        Self { min, max }
    }

    fn validate(&self, dimensions: u32, model: &str) -> Result<(), ClientError> {
        if dimensions < self.min || dimensions > self.max {
            return Err(ClientError::ConfigurationError(format!(
                "Invalid dimensions {dimensions} for model '{model}': must be between {} and {}",
                self.min, self.max
            )));
        }
        Ok(())
    }
}

/// Get dimension constraints for a model, if it supports dimension reduction.
fn get_dimension_constraints(model: &str) -> Option<DimensionConstraints> {
    match model {
        // text-embedding-3-small: 1536 default, supports 512-1536
        models::TEXT_EMBEDDING_3_SMALL => Some(DimensionConstraints::new(512, 1536)),
        // text-embedding-3-large: 3072 default, supports 256-3072
        models::TEXT_EMBEDDING_3_LARGE => Some(DimensionConstraints::new(256, 3072)),
        // text-embedding-ada-002 does not support dimension reduction
        _ => None,
    }
}

/// `OpenAI` embedding API request format.
#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a EmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
    #[serde(skip_serializing_if = "EncodingFormat::is_default")]
    encoding_format: EncodingFormat,
}

/// `OpenAI` embedding API response format.
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIEmbeddingUsage,
}

/// Raw embedding data that can be either a float array or base64-encoded bytes.
///
/// `OpenAI` returns embeddings as `Vec<f32>` by default, but when `encoding_format: "base64"`
/// is requested, it returns a base64-encoded string of little-endian f32 bytes.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawEmbedding {
    /// Standard float array format (default).
    Float(Vec<f32>),
    /// Base64-encoded little-endian f32 bytes.
    Base64(String),
}

impl RawEmbedding {
    /// Convert the raw embedding to a vector of floats.
    ///
    /// For base64-encoded data, decodes the string and interprets bytes as little-endian f32.
    fn into_floats(self) -> Result<Vec<f32>, ClientError> {
        match self {
            Self::Float(v) => Ok(v),
            Self::Base64(s) => {
                let bytes = BASE64_STANDARD.decode(&s).map_err(|e| {
                    ClientError::EmbeddingError(format!("Failed to decode base64 embedding: {e}"))
                })?;

                if bytes.len() % 4 != 0 {
                    return Err(ClientError::EmbeddingError(format!(
                        "Invalid base64 embedding: byte length {} is not a multiple of 4",
                        bytes.len()
                    )));
                }

                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        // Each f32 is 4 bytes in little-endian format
                        let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        f32::from_le_bytes(arr)
                    })
                    .collect())
            }
        }
    }
}

/// Individual embedding data from `OpenAI` response.
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    index: u32,
    embedding: RawEmbedding,
}

/// Usage statistics from `OpenAI` response.
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

/// `OpenAI` embedding client.
///
/// Implements the `EmbeddingClient` trait for `OpenAI`'s embedding API.
/// Supports all current `OpenAI` embedding models including dimension reduction
/// for the v3 models.
#[derive(Clone)]
pub struct OpenAIEmbedding {
    client: ClientWithMiddleware,
    config: Arc<EmbeddingConfig>,
    /// Pre-validated embeddings endpoint URL.
    embeddings_url: String,
}

// Custom Debug to avoid exposing API key
impl std::fmt::Debug for OpenAIEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbedding")
            .field("config", &self.config)
            .field("embeddings_url", &self.embeddings_url)
            .finish_non_exhaustive()
    }
}

impl OpenAIEmbedding {
    /// Create a new `OpenAI` embedding client.
    ///
    /// # Arguments
    ///
    /// * `config` - The embedding configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP client cannot be created
    /// - The base URL is invalid
    /// - Dimensions are specified for a model that doesn't support them
    /// - Dimensions are outside the valid range for the model
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        // Validate dimensions if specified
        if let Some(dimensions) = config.dimensions {
            Self::validate_dimensions(&config.model, dimensions)?;
        }

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        // Construct and validate the embeddings URL once at initialization
        let embeddings_url = format!("{base_url}/embeddings");
        reqwest::Url::parse(&embeddings_url)
            .map_err(|e| anyhow::anyhow!("Invalid base URL '{base_url}': {e}"))?;

        // Build retry policy from config
        let retry_policy = ExponentialBackoff::builder()
            .retry_bounds(
                config.retry_config.initial_delay,
                config.retry_config.max_delay,
            )
            .build_with_max_retries(config.retry_config.max_retries);

        // Create reqwest client with optional timeout
        let reqwest_client = match config.timeout_seconds {
            Some(timeout) => reqwest::Client::builder()
                .timeout(Duration::from_secs(timeout))
                .build()?,
            None => reqwest::Client::builder().build()?,
        };

        // Create client with retry middleware
        let client = reqwest_middleware::ClientBuilder::new(reqwest_client)
            .with(RetryAfterMiddleware::new())
            .with(RetryTransientMiddleware::new_with_policy(retry_policy))
            .build();

        Ok(Self {
            client,
            config: Arc::new(config),
            embeddings_url,
        })
    }

    /// Validate dimensions against the model's constraints.
    fn validate_dimensions(model: &str, dimensions: u32) -> Result<()> {
        match get_dimension_constraints(model) {
            Some(constraints) => {
                constraints.validate(dimensions, model)?;
                Ok(())
            }
            None => Err(ClientError::ConfigurationError(format!(
                "Model '{model}' does not support dimension reduction"
            ))
            .into()),
        }
    }

    /// Validate the embedding input.
    fn validate_input(input: &EmbeddingInput) -> Result<(), ClientError> {
        match input {
            EmbeddingInput::Single(text) => {
                if text.is_empty() {
                    return Err(ClientError::EmbeddingError(
                        "Input text cannot be empty".to_string(),
                    ));
                }
            }
            EmbeddingInput::Batch(texts) => {
                if texts.is_empty() {
                    return Err(ClientError::EmbeddingError(
                        "Batch input cannot be empty".to_string(),
                    ));
                }
                if texts.len() > MAX_BATCH_SIZE {
                    return Err(ClientError::EmbeddingError(format!(
                        "Batch size {} exceeds maximum of {MAX_BATCH_SIZE}",
                        texts.len()
                    )));
                }
                // Check for empty strings in batch
                if texts.iter().any(String::is_empty) {
                    return Err(ClientError::EmbeddingError(
                        "Batch contains empty text inputs".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Make an embedding API request.
    async fn make_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<OpenAIEmbeddingResponse, ClientError> {
        // Validate input
        Self::validate_input(&request.input)?;

        // Determine dimensions to use and validate if from request
        let dimensions = request.dimensions.or(self.config.dimensions);
        if let Some(dims) = request.dimensions {
            // Only validate request-level dimensions (config dimensions already validated in new())
            if self.config.dimensions.is_none() {
                Self::validate_dimensions(&self.config.model, dims)
                    .map_err(|e| ClientError::ConfigurationError(e.to_string()))?;
            }
        }

        // Build the request body
        let openai_request = OpenAIEmbeddingRequest {
            model: &self.config.model,
            input: &request.input,
            dimensions,
            user: request.user.as_deref(),
            encoding_format: request.encoding_format,
        };

        let response = self
            .client
            .post(&self.embeddings_url)
            .header(
                "Authorization",
                format!("Bearer {}", self.config.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(&openai_request).map_err(ClientError::SerializationError)?)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();

            // Extract Retry-After header before consuming the response body
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs);

            let error_text = response.text().await.map_err(|e| {
                warn!("Failed to read error response body: {e}");
                ClientError::NetworkError(e)
            })?;

            // Extract error message from structured response or use raw text
            let error_message = match serde_json::from_str::<ErrorResponse>(&error_text) {
                Ok(parsed) => {
                    debug!("Parsed structured error response");
                    parsed.error.message
                }
                Err(parse_err) => {
                    debug!("Failed to parse error response as JSON: {parse_err}. Using raw text.");
                    error_text
                }
            };

            error!(
                "Embedding API request failed with status {}: {}",
                status.as_u16(),
                error_message
            );

            return Err(match status.as_u16() {
                401 => ClientError::AuthenticationError(error_message),
                429 => ClientError::RateLimitError { retry_after },
                400 => ClientError::EmbeddingError(error_message),
                _ if status.is_server_error() => ClientError::ServiceUnavailable(error_message),
                _ => ClientError::EmbeddingError(error_message),
            });
        }

        let response_text = response.text().await?;
        debug!("Embedding API response: {}", &response_text);

        serde_json::from_str(&response_text).map_err(ClientError::SerializationError)
    }

    /// Get the default dimensions for a model.
    fn model_default_dimensions(model: &str) -> Option<u32> {
        match model {
            models::TEXT_EMBEDDING_3_LARGE => Some(3072),
            models::TEXT_EMBEDDING_3_SMALL | models::TEXT_EMBEDDING_ADA_002 => Some(1536),
            _ => None,
        }
    }

    /// Check if a model supports dimension reduction.
    fn model_supports_dimensions(model: &str) -> bool {
        matches!(
            model,
            models::TEXT_EMBEDDING_3_SMALL | models::TEXT_EMBEDDING_3_LARGE
        )
    }
}

#[async_trait]
impl EmbeddingClient for OpenAIEmbedding {
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    async fn embed_request(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let openai_response = self.make_request(request).await?;

        let embeddings: Vec<Embedding> = openai_response
            .data
            .into_iter()
            .map(|d| {
                Ok(Embedding {
                    index: d.index,
                    embedding: d.embedding.into_floats()?,
                })
            })
            .collect::<Result<Vec<_>, ClientError>>()?;

        Ok(EmbeddingResponse {
            embeddings,
            model: openai_response.model,
            usage: Some(EmbeddingUsage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                total_tokens: openai_response.usage.total_tokens,
            }),
        })
    }

    fn default_dimensions(&self) -> Option<u32> {
        Self::model_default_dimensions(&self.config.model)
    }

    fn supports_dimensions(&self) -> bool {
        Self::model_supports_dimensions(&self.config.model)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn create_test_config(base_url: &str) -> EmbeddingConfig {
        EmbeddingConfig::openai_small("test-key").with_base_url(base_url)
    }

    #[tokio::test]
    async fn test_successful_single_embedding() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("authorization", "Bearer test-key"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
                }],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let embedding = client.embed("Hello, world!").await.unwrap();

        assert_eq!(embedding.len(), 5);
        assert!((embedding[0] - 0.1).abs() < f32::EPSILON);
        assert!((embedding[4] - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_successful_batch_embedding() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3]
                    },
                    {
                        "object": "embedding",
                        "index": 1,
                        "embedding": [0.4, 0.5, 0.6]
                    }
                ],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 4,
                    "total_tokens": 4
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let embeddings = client.embed_batch(&["Hello", "World"]).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 3);
        assert_eq!(embeddings[1].len(), 3);
    }

    #[tokio::test]
    async fn test_batch_embedding_sorts_by_index() {
        let mock_server = MockServer::start().await;

        // Return embeddings out of order (index 1 before index 0)
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 1,
                        "embedding": [0.4, 0.5, 0.6]
                    },
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3]
                    }
                ],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 4,
                    "total_tokens": 4
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let embeddings = client.embed_batch(&["Hello", "World"]).await.unwrap();

        // Verify embeddings are sorted by index, not response order
        assert_eq!(embeddings.len(), 2);
        assert!((embeddings[0][0] - 0.1).abs() < f32::EPSILON); // index 0 embedding
        assert!((embeddings[1][0] - 0.4).abs() < f32::EPSILON); // index 1 embedding
    }

    #[tokio::test]
    async fn test_embed_request_with_dimensions() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2]
                }],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 2,
                    "total_tokens": 2
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let request = EmbeddingRequest::builder()
            .input(EmbeddingInput::Single("test".to_string()))
            .dimensions(Some(512)) // Valid for text-embedding-3-small (512-1536)
            .build();

        let response = client.embed_request(&request).await.unwrap();

        assert_eq!(response.model, "text-embedding-3-small");
        assert!(response.usage.is_some());
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 2);
    }

    #[tokio::test]
    async fn test_authentication_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let result = client.embed("test").await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid API key"));
    }

    #[tokio::test]
    async fn test_rate_limit_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let result = client.embed("test").await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Rate limit"));
    }

    #[tokio::test]
    async fn test_rate_limit_error_with_retry_after() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "30")
                    .set_body_json(serde_json::json!({
                        "error": {
                            "message": "Rate limit exceeded",
                            "type": "rate_limit_error"
                        }
                    })),
            )
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let result = client.embed_request(&EmbeddingRequest::new("test")).await;
        assert!(result.is_err());

        let err = result
            .unwrap_err()
            .downcast::<ClientError>()
            .expect("Expected ClientError");
        assert!(err.is_rate_limit_error());
        assert_eq!(err.retry_after(), Some(Duration::from_secs(30)));
    }

    #[tokio::test]
    async fn test_embedding_error_bad_request() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {
                    "message": "Invalid model specified",
                    "type": "invalid_request_error"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        // Use valid input - the server returns 400 for other reasons (e.g., invalid model)
        let result = client.embed("test").await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid model specified"));
    }

    #[tokio::test]
    async fn test_service_unavailable_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                "error": {
                    "message": "Service temporarily unavailable",
                    "type": "server_error"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let result = client.embed("test").await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("unavailable"));
    }

    #[tokio::test]
    async fn test_empty_response_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let result = client.embed("test").await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("empty response"));
        assert!(error_msg.contains("no embeddings in data array"));
    }

    #[tokio::test]
    async fn test_empty_embedding_vector_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": []
                }],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 1,
                    "total_tokens": 1
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let result = client.embed("test").await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("malformed response"));
        assert!(error_msg.contains("embedding vector is empty"));
    }

    #[test]
    fn test_model_default_dimensions() {
        assert_eq!(
            OpenAIEmbedding::model_default_dimensions("text-embedding-3-small"),
            Some(1536)
        );
        assert_eq!(
            OpenAIEmbedding::model_default_dimensions("text-embedding-3-large"),
            Some(3072)
        );
        assert_eq!(
            OpenAIEmbedding::model_default_dimensions("text-embedding-ada-002"),
            Some(1536)
        );
        assert_eq!(
            OpenAIEmbedding::model_default_dimensions("unknown-model"),
            None
        );
    }

    #[test]
    fn test_model_supports_dimensions() {
        assert!(OpenAIEmbedding::model_supports_dimensions(
            "text-embedding-3-small"
        ));
        assert!(OpenAIEmbedding::model_supports_dimensions(
            "text-embedding-3-large"
        ));
        assert!(!OpenAIEmbedding::model_supports_dimensions(
            "text-embedding-ada-002"
        ));
    }

    #[test]
    fn test_client_trait_methods() {
        let config = EmbeddingConfig::openai_small("test-key");
        let client = OpenAIEmbedding::new(config).unwrap();

        assert_eq!(client.default_dimensions(), Some(1536));
        assert!(client.supports_dimensions());
        assert_eq!(client.config().model, "text-embedding-3-small");
    }

    #[test]
    fn test_openai_embedding_debug_redacts_api_key() {
        let config = EmbeddingConfig::openai_small("super-secret-key");
        let client = OpenAIEmbedding::new(config).unwrap();
        let debug_output = format!("{client:?}");
        assert!(!debug_output.contains("super-secret"));
        assert!(debug_output.contains("[REDACTED]"));
    }

    #[test]
    fn test_raw_embedding_float_format() {
        let raw = RawEmbedding::Float(vec![0.1, 0.2, 0.3]);
        let floats = raw.into_floats().unwrap();
        assert_eq!(floats.len(), 3);
        assert!((floats[0] - 0.1).abs() < f32::EPSILON);
        assert!((floats[1] - 0.2).abs() < f32::EPSILON);
        assert!((floats[2] - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_raw_embedding_base64_format() {
        // Encode [0.1_f32, 0.2_f32, 0.3_f32] as little-endian bytes, then base64
        let floats: Vec<f32> = vec![0.1, 0.2, 0.3];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base64_str = BASE64_STANDARD.encode(&bytes);

        let raw = RawEmbedding::Base64(base64_str);
        let decoded = raw.into_floats().unwrap();

        assert_eq!(decoded.len(), 3);
        assert!((decoded[0] - 0.1).abs() < f32::EPSILON);
        assert!((decoded[1] - 0.2).abs() < f32::EPSILON);
        assert!((decoded[2] - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_raw_embedding_base64_invalid_length() {
        // 5 bytes is not a multiple of 4
        let bytes = vec![0u8, 1, 2, 3, 4];
        let base64_str = BASE64_STANDARD.encode(&bytes);

        let raw = RawEmbedding::Base64(base64_str);
        let result = raw.into_floats();

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("not a multiple of 4"));
    }

    #[test]
    fn test_raw_embedding_base64_invalid_encoding() {
        let raw = RawEmbedding::Base64("not-valid-base64!!!".to_string());
        let result = raw.into_floats();

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Failed to decode base64"));
    }

    #[tokio::test]
    async fn test_base64_encoded_embedding_response() {
        let mock_server = MockServer::start().await;

        // Create base64-encoded embedding data
        let floats: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base64_str = BASE64_STANDARD.encode(&bytes);

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": base64_str
                }],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let request =
            EmbeddingRequest::new("Hello, world!").with_encoding_format(EncodingFormat::Base64);
        let response = client.embed_request(&request).await.unwrap();

        let embedding = response.first().unwrap();
        assert_eq!(embedding.len(), 5);
        assert!((embedding[0] - 0.1).abs() < f32::EPSILON);
        assert!((embedding[4] - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_base64_encoded_batch_embedding_response() {
        let mock_server = MockServer::start().await;

        // Create base64-encoded embeddings
        let floats1: Vec<f32> = vec![0.1, 0.2, 0.3];
        let bytes1: Vec<u8> = floats1.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base64_str1 = BASE64_STANDARD.encode(&bytes1);

        let floats2: Vec<f32> = vec![0.4, 0.5, 0.6];
        let bytes2: Vec<u8> = floats2.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base64_str2 = BASE64_STANDARD.encode(&bytes2);

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": base64_str1
                    },
                    {
                        "object": "embedding",
                        "index": 1,
                        "embedding": base64_str2
                    }
                ],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 4,
                    "total_tokens": 4
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = OpenAIEmbedding::new(config).unwrap();

        let input: EmbeddingInput = (&["Hello", "World"][..]).into();
        let request = EmbeddingRequest::new(input).with_encoding_format(EncodingFormat::Base64);
        let response = client.embed_request(&request).await.unwrap();

        assert_eq!(response.embeddings.len(), 2);
        assert!((response.embeddings[0].embedding[0] - 0.1).abs() < f32::EPSILON);
        assert!((response.embeddings[1].embedding[0] - 0.4).abs() < f32::EPSILON);
    }

    // ===================
    // Input validation tests
    // ===================

    #[test]
    fn test_dimension_validation_small_model_valid() {
        // text-embedding-3-small supports 512-1536
        let config = EmbeddingConfig::openai_small("test-key").with_dimensions(512);
        assert!(OpenAIEmbedding::new(config).is_ok());

        let config = EmbeddingConfig::openai_small("test-key").with_dimensions(1536);
        assert!(OpenAIEmbedding::new(config).is_ok());

        let config = EmbeddingConfig::openai_small("test-key").with_dimensions(1024);
        assert!(OpenAIEmbedding::new(config).is_ok());
    }

    #[test]
    fn test_dimension_validation_small_model_invalid() {
        // Below minimum
        let config = EmbeddingConfig::openai_small("test-key").with_dimensions(256);
        let result = OpenAIEmbedding::new(config);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("must be between 512 and 1536"));

        // Above maximum
        let config = EmbeddingConfig::openai_small("test-key").with_dimensions(2048);
        let result = OpenAIEmbedding::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_validation_large_model_valid() {
        // text-embedding-3-large supports 256-3072
        let config = EmbeddingConfig::openai_large("test-key").with_dimensions(256);
        assert!(OpenAIEmbedding::new(config).is_ok());

        let config = EmbeddingConfig::openai_large("test-key").with_dimensions(3072);
        assert!(OpenAIEmbedding::new(config).is_ok());

        let config = EmbeddingConfig::openai_large("test-key").with_dimensions(1024);
        assert!(OpenAIEmbedding::new(config).is_ok());
    }

    #[test]
    fn test_dimension_validation_large_model_invalid() {
        // Below minimum
        let config = EmbeddingConfig::openai_large("test-key").with_dimensions(128);
        let result = OpenAIEmbedding::new(config);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("must be between 256 and 3072"));

        // Above maximum
        let config = EmbeddingConfig::openai_large("test-key").with_dimensions(4096);
        let result = OpenAIEmbedding::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_validation_ada_model_not_supported() {
        // text-embedding-ada-002 does not support dimension reduction
        let config = EmbeddingConfig::openai_ada("test-key").with_dimensions(1536);
        let result = OpenAIEmbedding::new(config);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("does not support dimension reduction"));
    }

    #[test]
    fn test_input_validation_empty_single() {
        let input = EmbeddingInput::Single(String::new());
        let result = OpenAIEmbedding::validate_input(&input);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Input text cannot be empty"));
    }

    #[test]
    fn test_input_validation_empty_batch() {
        let input = EmbeddingInput::Batch(vec![]);
        let result = OpenAIEmbedding::validate_input(&input);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Batch input cannot be empty"));
    }

    #[test]
    fn test_input_validation_batch_with_empty_string() {
        let input = EmbeddingInput::Batch(vec![
            "hello".to_string(),
            String::new(),
            "world".to_string(),
        ]);
        let result = OpenAIEmbedding::validate_input(&input);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Batch contains empty text inputs"));
    }

    #[test]
    fn test_input_validation_batch_exceeds_max_size() {
        let input = EmbeddingInput::Batch(vec!["test".to_string(); MAX_BATCH_SIZE + 1]);
        let result = OpenAIEmbedding::validate_input(&input);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("exceeds maximum"));
        assert!(error_msg.contains("2048"));
    }

    #[test]
    fn test_input_validation_valid_single() {
        let input = EmbeddingInput::Single("hello world".to_string());
        assert!(OpenAIEmbedding::validate_input(&input).is_ok());
    }

    #[test]
    fn test_input_validation_valid_batch() {
        let input = EmbeddingInput::Batch(vec!["hello".to_string(), "world".to_string()]);
        assert!(OpenAIEmbedding::validate_input(&input).is_ok());
    }

    #[test]
    fn test_input_validation_max_batch_size() {
        // Exactly at the limit should be ok
        let input = EmbeddingInput::Batch(vec!["test".to_string(); MAX_BATCH_SIZE]);
        assert!(OpenAIEmbedding::validate_input(&input).is_ok());
    }

    #[tokio::test]
    async fn test_empty_input_rejected_before_request() {
        // Empty input should be rejected client-side before making a request
        let config = EmbeddingConfig::openai_small("test-key");
        let client = OpenAIEmbedding::new(config).unwrap();

        let result = client.embed("").await;
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Input text cannot be empty"));
    }

    #[tokio::test]
    async fn test_empty_batch_rejected_before_request() {
        let config = EmbeddingConfig::openai_small("test-key");
        let client = OpenAIEmbedding::new(config).unwrap();

        let result = client.embed_batch(&[]).await;
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Batch input cannot be empty"));
    }
}
