//! Embedding client trait and types.
//!
//! This module provides a unified interface for generating embeddings from text
//! using various providers. Currently supports `OpenAI` embedding models.
//!
//! # Example
//!
//! ```no_run
//! use neuromance_client::embedding::{EmbeddingClient, EmbeddingConfig};
//! use neuromance_client::openai::OpenAIEmbedding;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a client with a preset configuration
//! let config = EmbeddingConfig::openai_small("sk-...");
//! let client = OpenAIEmbedding::new(config)?;
//!
//! // Generate a single embedding
//! let vector = client.embed("Hello, world!").await?;
//!
//! // Generate batch embeddings
//! let vectors = client.embed_batch(&["Hello", "World"]).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use async_trait::async_trait;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use neuromance_common::client::{Config, RetryConfig};

/// `OpenAI` embedding model identifiers.
///
/// These constants provide type-safe references to embedding models,
pub mod models {
    /// `OpenAI`'s `text-embedding-3-small` model.
    ///
    /// - Default dimensions: 1536
    /// - Supports dimension reduction: 512-1536
    /// - Best for: Cost-effective embeddings with good quality
    pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";

    /// `OpenAI`'s `text-embedding-3-large` model.
    ///
    /// - Default dimensions: 3072
    /// - Supports dimension reduction: 256-3072
    /// - Best for: Highest quality embeddings
    pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";

    /// `OpenAI`'s legacy `text-embedding-ada-002` model.
    ///
    /// - Dimensions: 1536 (fixed)
    /// - Does not support dimension reduction
    /// - Note: Consider using `text-embedding-3-small` for new projects
    pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";
}

/// Configuration for embedding clients.
///
/// Contains API credentials, model settings, and optional parameters
/// for embedding generation.
///
/// # Creating from existing Config
///
/// If you already have a [`Config`] from the common crate, you can create
/// an `EmbeddingConfig` using the `From` implementation:
///
/// ```
/// use neuromance_common::Config;
/// use neuromance_client::embedding::EmbeddingConfig;
///
/// let config = Config::new("openai", "text-embedding-3-small")
///     .with_api_key("sk-...")
///     .with_base_url("https://api.openai.com/v1");
///
/// let embedding_config = EmbeddingConfig::from(&config);
/// ```
#[derive(Clone)]
pub struct EmbeddingConfig {
    /// The API key for authentication.
    pub api_key: SecretString,
    /// The embedding model to use.
    pub model: String,
    /// Optional base URL override for the API endpoint.
    pub base_url: Option<String>,
    /// Optional dimension override (for models that support it).
    pub dimensions: Option<u32>,
    /// Retry configuration for transient failures.
    pub retry_config: RetryConfig,
    /// Optional request timeout in seconds.
    pub timeout_seconds: Option<u64>,
}

// Custom Debug to avoid exposing API key
impl std::fmt::Debug for EmbeddingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("dimensions", &self.dimensions)
            .field("retry_config", &self.retry_config)
            .field("timeout_seconds", &self.timeout_seconds)
            .finish()
    }
}

impl EmbeddingConfig {
    /// Create a new embedding configuration with the specified model and API key.
    ///
    /// # Arguments
    ///
    /// * `model` - The embedding model identifier
    /// * `api_key` - The API key for authentication
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            model: model.into(),
            base_url: None,
            dimensions: None,
            retry_config: RetryConfig::default(),
            timeout_seconds: None,
        }
    }

    /// Create a configuration for `OpenAI`'s `text-embedding-3-small` model.
    ///
    /// This model produces 1536-dimensional embeddings by default, but supports
    /// dimension reduction via the `dimensions` parameter.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The `OpenAI` API key
    #[must_use]
    pub fn openai_small(api_key: impl Into<String>) -> Self {
        Self::new(models::TEXT_EMBEDDING_3_SMALL, api_key)
    }

    /// Create a configuration for `OpenAI`'s `text-embedding-3-large` model.
    ///
    /// This model produces 3072-dimensional embeddings by default, but supports
    /// dimension reduction via the `dimensions` parameter.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The `OpenAI` API key
    #[must_use]
    pub fn openai_large(api_key: impl Into<String>) -> Self {
        Self::new(models::TEXT_EMBEDDING_3_LARGE, api_key)
    }

    /// Create a configuration for `OpenAI`'s legacy `text-embedding-ada-002` model.
    ///
    /// This model produces 1536-dimensional embeddings and does not support
    /// dimension reduction.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The `OpenAI` API key
    #[must_use]
    pub fn openai_ada(api_key: impl Into<String>) -> Self {
        Self::new(models::TEXT_EMBEDDING_ADA_002, api_key)
    }

    /// Set the base URL for the API endpoint.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL (e.g., `https://api.openai.com/v1`)
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the output dimensions for the embedding.
    ///
    /// Only supported by `text-embedding-3-small` and `text-embedding-3-large`.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The desired output dimensions
    #[must_use]
    pub const fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set the retry configuration.
    ///
    /// # Arguments
    ///
    /// * `retry_config` - The retry configuration
    #[must_use]
    pub const fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    /// Set the request timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout_seconds` - The timeout in seconds
    #[must_use]
    pub const fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = Some(timeout_seconds);
        self
    }
}

impl From<&Config> for EmbeddingConfig {
    /// Create an `EmbeddingConfig` from an existing [`Config`].
    ///
    /// This allows reusing configuration from the common crate's `Config` type.
    /// The API key must be set in the source `Config`, otherwise this will
    /// use an empty string (which will fail authentication).
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::Config;
    /// use neuromance_client::embedding::EmbeddingConfig;
    ///
    /// let config = Config::new("openai", "text-embedding-3-small")
    ///     .with_api_key("sk-...");
    ///
    /// let embedding_config = EmbeddingConfig::from(&config);
    /// assert_eq!(embedding_config.model, "text-embedding-3-small");
    /// ```
    fn from(config: &Config) -> Self {
        use secrecy::ExposeSecret;

        // Clone the API key if present, otherwise use an empty string
        // (authentication will fail, but this matches the pattern of
        // allowing optional API keys in Config)
        let api_key = config.api_key.as_ref().map_or_else(
            || SecretString::from(String::new()),
            |s| SecretString::from(s.expose_secret().to_string()),
        );

        Self {
            api_key,
            model: config.model.clone(),
            base_url: config.base_url.clone(),
            dimensions: None,
            retry_config: config.retry_config.clone(),
            timeout_seconds: config.timeout_seconds,
        }
    }
}

/// Encoding format for embedding vectors.
///
/// Controls how embedding vectors are returned from the API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    /// Return embeddings as arrays of floats (default).
    #[default]
    Float,
    /// Return embeddings as base64-encoded strings.
    ///
    /// This can reduce network transfer size for large batches,
    /// but requires decoding on the client side.
    Base64,
}

impl EncodingFormat {
    /// Returns `true` if this is the default encoding format ([`Float`](Self::Float)).
    ///
    /// Used by serde's `skip_serializing_if` to omit the field when it's the default.
    #[must_use]
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Float)
    }
}

/// Input for embedding generation.
///
/// Supports both single text inputs and batch inputs.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single text string to embed.
    Single(String),
    /// Multiple text strings to embed in one request.
    Batch(Vec<String>),
}

impl From<&str> for EmbeddingInput {
    fn from(s: &str) -> Self {
        Self::Single(s.to_string())
    }
}

impl From<String> for EmbeddingInput {
    fn from(s: String) -> Self {
        Self::Single(s)
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(v: Vec<String>) -> Self {
        Self::Batch(v)
    }
}

impl From<&[&str]> for EmbeddingInput {
    fn from(v: &[&str]) -> Self {
        Self::Batch(v.iter().map(|s| (*s).to_string()).collect())
    }
}

impl From<&[String]> for EmbeddingInput {
    fn from(v: &[String]) -> Self {
        Self::Batch(v.to_vec())
    }
}

/// A request for embedding generation.
///
/// Contains the input text(s) and optional parameters.
#[derive(Debug, Clone, Serialize, TypedBuilder)]
pub struct EmbeddingRequest {
    /// The input text(s) to embed.
    pub input: EmbeddingInput,
    /// Optional dimension override for supported models.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Optional user identifier for tracking.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Encoding format for the returned embeddings.
    ///
    /// Defaults to `Float`. Use `Base64` to reduce network transfer size
    /// for large batches (requires client-side decoding).
    #[builder(default)]
    #[serde(skip_serializing_if = "EncodingFormat::is_default")]
    pub encoding_format: EncodingFormat,
}

impl EmbeddingRequest {
    /// Create a new embedding request with the specified input.
    #[must_use]
    pub fn new(input: impl Into<EmbeddingInput>) -> Self {
        Self {
            input: input.into(),
            dimensions: None,
            user: None,
            encoding_format: EncodingFormat::default(),
        }
    }

    /// Set the user identifier for tracking.
    #[must_use]
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set the encoding format for the response.
    #[must_use]
    pub const fn with_encoding_format(mut self, format: EncodingFormat) -> Self {
        self.encoding_format = format;
        self
    }
}

/// A single embedding vector.
#[derive(Debug, Clone, Deserialize)]
pub struct Embedding {
    /// The index of this embedding in the request.
    pub index: u32,
    /// The embedding vector.
    pub embedding: Vec<f32>,
}

/// Token usage statistics for an embedding request.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingUsage {
    /// Number of tokens in the input.
    pub prompt_tokens: u32,
    /// Total tokens used.
    ///
    /// For `OpenAI`, this equals `prompt_tokens` since embeddings have no output tokens.
    /// Other providers may report this differently.
    pub total_tokens: u32,
}

/// Response from an embedding generation request.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// The generated embeddings.
    pub embeddings: Vec<Embedding>,
    /// The model used for generation.
    pub model: String,
    /// Token usage statistics.
    pub usage: Option<EmbeddingUsage>,
}

impl EmbeddingResponse {
    /// Get the first embedding vector, if any.
    #[must_use]
    pub fn first(&self) -> Option<&[f32]> {
        self.embeddings.first().map(|e| e.embedding.as_slice())
    }

    /// Get all embedding vectors.
    #[must_use]
    pub fn vectors(&self) -> Vec<&[f32]> {
        self.embeddings
            .iter()
            .map(|e| e.embedding.as_slice())
            .collect()
    }
}

/// Extract a single embedding from a response.
///
/// Used by trait default implementations.
fn extract_single_embedding(response: &EmbeddingResponse) -> Result<Vec<f32>> {
    let embedding = response.first().ok_or_else(|| {
        anyhow::anyhow!("API returned empty response: no embeddings in data array")
    })?;

    if embedding.is_empty() {
        return Err(anyhow::anyhow!(
            "API returned malformed response: embedding vector is empty"
        ));
    }

    Ok(embedding.to_vec())
}

/// Extract batch embeddings from a response, sorted by index.
///
/// Used by trait default implementations.
fn extract_batch_embeddings(response: EmbeddingResponse) -> Vec<Vec<f32>> {
    let mut embeddings = response.embeddings;
    embeddings.sort_by_key(|e| e.index);
    embeddings.into_iter().map(|e| e.embedding).collect()
}

/// Trait for embedding client implementations.
///
/// Provides a unified interface for generating embeddings from text
/// across different providers.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Get the client's configuration.
    fn config(&self) -> &EmbeddingConfig;

    /// Generate an embedding for a single text input.
    ///
    /// This is a convenience method that calls `embed_request` internally.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// The embedding vector for the input text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest::new(text);
        let response = self.embed_request(&request).await?;
        extract_single_embedding(&response)
    }

    /// Generate embeddings for multiple text inputs in a single request.
    ///
    /// This is more efficient than calling `embed` multiple times.
    ///
    /// # Arguments
    ///
    /// * `texts` - The texts to embed
    ///
    /// # Returns
    ///
    /// A vector of embedding vectors, in the same order as the inputs.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let input: EmbeddingInput = texts.into();
        let request = EmbeddingRequest::new(input);
        let response = self.embed_request(&request).await?;
        Ok(extract_batch_embeddings(response))
    }

    /// Generate embeddings with full control over request parameters.
    ///
    /// # Arguments
    ///
    /// * `request` - The embedding request with all parameters
    ///
    /// # Returns
    ///
    /// The full embedding response including metadata.
    async fn embed_request(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Generate an embedding for a single text input with user tracking.
    ///
    /// Like [`embed`](Self::embed), but includes a user identifier for tracking.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    /// * `user` - User identifier for tracking
    async fn embed_with_user(&self, text: &str, user: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest::new(text).with_user(user);
        let response = self.embed_request(&request).await?;
        extract_single_embedding(&response)
    }

    /// Generate embeddings for multiple text inputs with user tracking.
    ///
    /// Like [`embed_batch`](Self::embed_batch), but includes a user identifier for tracking.
    ///
    /// # Arguments
    ///
    /// * `texts` - The texts to embed
    /// * `user` - User identifier for tracking
    async fn embed_batch_with_user(&self, texts: &[&str], user: &str) -> Result<Vec<Vec<f32>>> {
        let input: EmbeddingInput = texts.into();
        let request = EmbeddingRequest::new(input).with_user(user);
        let response = self.embed_request(&request).await?;
        Ok(extract_batch_embeddings(response))
    }

    /// Get the default dimensions for the configured model.
    ///
    /// Returns `None` if the model's dimensions are not known.
    fn default_dimensions(&self) -> Option<u32> {
        None
    }

    /// Check if the configured model supports dimension reduction.
    fn supports_dimensions(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_embedding_config_presets() {
        let small = EmbeddingConfig::openai_small("test-key");
        assert_eq!(small.model, "text-embedding-3-small");

        let large = EmbeddingConfig::openai_large("test-key");
        assert_eq!(large.model, "text-embedding-3-large");

        let ada = EmbeddingConfig::openai_ada("test-key");
        assert_eq!(ada.model, "text-embedding-ada-002");
    }

    #[test]
    fn test_embedding_config_builder_methods() {
        let config = EmbeddingConfig::openai_small("test-key")
            .with_base_url("https://custom.api.com/v1")
            .with_dimensions(512)
            .with_timeout(30);

        assert_eq!(
            config.base_url,
            Some("https://custom.api.com/v1".to_string())
        );
        assert_eq!(config.dimensions, Some(512));
        assert_eq!(config.timeout_seconds, Some(30));
    }

    #[test]
    fn test_embedding_input_conversions() {
        let single: EmbeddingInput = "hello".into();
        assert!(matches!(single, EmbeddingInput::Single(_)));

        let single_string: EmbeddingInput = String::from("hello").into();
        assert!(matches!(single_string, EmbeddingInput::Single(_)));

        let batch: EmbeddingInput = vec!["hello".to_string(), "world".to_string()].into();
        assert!(matches!(batch, EmbeddingInput::Batch(_)));

        let slice: &[&str] = &["hello", "world"];
        let batch_from_slice: EmbeddingInput = slice.into();
        assert!(matches!(batch_from_slice, EmbeddingInput::Batch(_)));

        let string_vec = vec!["hello".to_string(), "world".to_string()];
        let string_slice: &[String] = &string_vec;
        let batch_from_string_slice: EmbeddingInput = string_slice.into();
        assert!(matches!(batch_from_string_slice, EmbeddingInput::Batch(_)));
    }

    #[test]
    fn test_embedding_request_builder() {
        let request = EmbeddingRequest::builder()
            .input(EmbeddingInput::Single("test".to_string()))
            .dimensions(Some(256))
            .user(Some("test-user".to_string()))
            .build();

        assert!(matches!(request.input, EmbeddingInput::Single(_)));
        assert_eq!(request.dimensions, Some(256));
        assert_eq!(request.user, Some("test-user".to_string()));
    }

    #[test]
    fn test_embedding_response_accessors() {
        let response = EmbeddingResponse {
            embeddings: vec![
                Embedding {
                    index: 0,
                    embedding: vec![0.1, 0.2, 0.3],
                },
                Embedding {
                    index: 1,
                    embedding: vec![0.4, 0.5, 0.6],
                },
            ],
            model: "test-model".to_string(),
            usage: Some(EmbeddingUsage {
                prompt_tokens: 10,
                total_tokens: 10,
            }),
        };

        assert_eq!(response.first(), Some([0.1, 0.2, 0.3].as_slice()));
        let vectors = response.vectors();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], &[0.1, 0.2, 0.3]);
        assert_eq!(vectors[1], &[0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_embedding_config_debug_redacts_api_key() {
        let config = EmbeddingConfig::openai_small("super-secret-key");
        let debug_output = format!("{config:?}");
        assert!(!debug_output.contains("super-secret"));
        assert!(debug_output.contains("[REDACTED]"));
    }
}
