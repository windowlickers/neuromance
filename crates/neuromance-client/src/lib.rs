//! # neuromance-client
//!
//! Client library for interacting with LLM inference providers.
//!
//! Provides a unified `LLMClient` trait for various LLM providers. Currently supports
//! OpenAI-compatible APIs with tool/function calling and streaming.

// ClientError contains EventSourceError (~176 bytes), but this is acceptable
// for network-bound code where HTTP latency dwarfs any stack size concerns.
#![allow(clippy::result_large_err)]

use std::pin::Pin;

use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;

use neuromance_common::client::ChatChunk;
use neuromance_common::{ChatRequest, ChatResponse, Config};

pub mod anthropic;
pub mod embedding;
mod error;
pub mod openai;
pub mod responses;

pub use anthropic::AnthropicClient;
pub use embedding::{
    EmbeddingClient, EmbeddingConfig, EmbeddingInput, EmbeddingRequest, EmbeddingResponse,
};
pub use error::ClientError;
pub use openai::{OpenAIClient, OpenAIEmbedding};
pub use responses::ResponsesClient;

/// A retry policy for SSE streams that never retries.
///
/// Useful for handling retries at a higher level rather than automatic reconnection.
pub struct NoRetryPolicy;

impl reqwest_eventsource::retry::RetryPolicy for NoRetryPolicy {
    fn retry(
        &self,
        _error: &reqwest_eventsource::Error,
        _last_retry: Option<(usize, std::time::Duration)>,
    ) -> Option<std::time::Duration> {
        // Never retry - return None to indicate no retry should happen
        None
    }

    fn set_reconnection_time(&mut self, _duration: std::time::Duration) {
        // Ignored - we never retry anyway
    }
}

/// Trait for LLM client implementations.
///
/// Provides a unified interface for interacting with different LLM providers.
/// Implementations must be async and thread-safe (Send + Sync).
#[must_use = "LLMClient must be used to make requests"]
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Get the client's configuration.
    fn config(&self) -> &Config;

    /// Send a chat completion request to the LLM.
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse>;

    /// Send a streaming chat completion request to the LLM.
    ///
    /// Returns a stream of incremental response chunks for real-time display.
    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send>>>;

    /// Check if the client supports tool/function calling.
    fn supports_tools(&self) -> bool;

    /// Check if the client supports streaming responses.
    fn supports_streaming(&self) -> bool;

    /// Validate a configuration object.
    ///
    /// Checks parameter ranges: `temperature` (0.0-2.0), `top_p` (0.0-1.0), `frequency_penalty` (-2.0-2.0).
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of range.
    fn validate_config(&self, config: Config) -> Result<()> {
        if config
            .temperature
            .is_some_and(|t| !(0.0..=2.0).contains(&t))
        {
            return Err(ClientError::InvalidTemperature.into());
        }

        if config.top_p.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
            return Err(ClientError::InvalidTopP.into());
        }

        if config
            .frequency_penalty
            .is_some_and(|f| !(-2.0..=2.0).contains(&f))
        {
            return Err(ClientError::InvalidFrequencyPenalty.into());
        }

        Ok(())
    }

    /// Validate a chat request before sending.
    ///
    /// Checks messages exist and that tools/streaming are supported if requested.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    fn validate_request(&self, request: &ChatRequest) -> Result<()> {
        request
            .validate_has_messages()
            .map_err(|e| ClientError::InvalidRequest(e.to_string()))?;

        if !self.supports_tools() && request.has_tools() {
            return Err(ClientError::ToolsNotSupported.into());
        }

        if !self.supports_streaming() && request.is_streaming() {
            return Err(ClientError::StreamingNotSupported.into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_common::chat::Message;
    use neuromance_common::client::{ChatResponse, ToolChoice};
    use neuromance_common::features::{ReasoningLevel, ThinkingMode};
    use neuromance_common::tools::Tool;
    use std::collections::HashMap;
    use uuid::Uuid;

    // Mock implementation for testing
    struct MockLLMClient {
        config: Config,
        supports_tools: bool,
        supports_streaming: bool,
    }

    impl MockLLMClient {
        fn new() -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                supports_tools: true,
                supports_streaming: true,
            }
        }

        fn without_tools() -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                supports_tools: false,
                supports_streaming: true,
            }
        }

        fn without_streaming() -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                supports_tools: true,
                supports_streaming: false,
            }
        }
    }

    #[async_trait]
    impl LLMClient for MockLLMClient {
        fn config(&self) -> &Config {
            &self.config
        }

        async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: create_test_message(),
                model: "mock-model".to_string(),
                usage: None,
                finish_reason: None,
                created_at: chrono::Utc::now(),
                response_id: Some("test-response".to_string()),
                metadata: HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send>>> {
            use futures::stream;

            // Create a simple mock stream with one chunk
            let chunk = ChatChunk {
                model: "mock-model".to_string(),
                delta_content: Some("Hello".to_string()),
                delta_reasoning_content: None,
                delta_role: Some(neuromance_common::chat::MessageRole::Assistant),
                delta_tool_calls: None,
                finish_reason: None,
                usage: None,
                response_id: Some("test-response".to_string()),
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };

            Ok(Box::pin(stream::iter(vec![Ok(chunk)])))
        }

        fn supports_tools(&self) -> bool {
            self.supports_tools
        }

        fn supports_streaming(&self) -> bool {
            self.supports_streaming
        }
    }

    fn create_test_message() -> Message {
        Message::user(Uuid::new_v4(), "Test message")
    }

    fn create_test_tool() -> Tool {
        use neuromance_common::tools::{Function, Parameters, Property};
        use std::collections::HashMap;

        let mut properties = HashMap::new();
        properties.insert("arg".to_string(), Property::string("A test argument"));

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "test_function".to_string(),
                description: "A test function".to_string(),
                parameters: Parameters::new(properties, vec!["arg".into()]).into(),
            },
        }
    }

    #[test]
    fn test_validate_request_empty_messages() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_err());
        let error = result.unwrap_err();
        let client_error = error.downcast_ref::<ClientError>().unwrap();
        assert!(matches!(client_error, ClientError::InvalidRequest(_)));
    }

    #[test]
    fn test_validate_request_valid() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_request_tools_not_supported() {
        let client = MockLLMClient::without_tools();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: Some(vec![create_test_tool()]),
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_err());
        let error = result.unwrap_err();
        let client_error = error.downcast_ref::<ClientError>().unwrap();
        assert!(matches!(client_error, ClientError::ToolsNotSupported));
    }

    #[test]
    fn test_validate_request_tools_supported() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: Some(vec![create_test_tool()]),
            tool_choice: Some(ToolChoice::Auto),
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_request_streaming_not_supported() {
        let client = MockLLMClient::without_streaming();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: true,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_err());
        let error = result.unwrap_err();
        let client_error = error.downcast_ref::<ClientError>().unwrap();
        assert!(matches!(client_error, ClientError::StreamingNotSupported));
    }

    #[test]
    fn test_validate_request_streaming_supported() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: true,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_client_config_access() {
        let client = MockLLMClient::new();
        let config = client.config();
        assert_eq!(config.provider, "mock");
        assert_eq!(config.model, "mock-model");
    }

    #[test]
    fn test_supports_tools() {
        let client_with_tools = MockLLMClient::new();
        let client_without_tools = MockLLMClient::without_tools();

        assert!(client_with_tools.supports_tools());
        assert!(!client_without_tools.supports_tools());
    }

    #[test]
    fn test_supports_streaming() {
        let client_with_streaming = MockLLMClient::new();
        let client_without_streaming = MockLLMClient::without_streaming();

        assert!(client_with_streaming.supports_streaming());
        assert!(!client_without_streaming.supports_streaming());
    }

    #[tokio::test]
    async fn test_chat_method() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.chat(&request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.response_id, Some("test-response".to_string()));
        assert_eq!(response.model, "mock-model");
    }
}
