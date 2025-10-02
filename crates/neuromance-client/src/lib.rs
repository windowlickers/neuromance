//! # neuromance-client
//!
//! Client library for interacting with LLM inference providers.
//!
//! This crate provides a unified interface for communicating with various LLM providers
//! through the `LLMClient` trait. Currently supports:
//! - OpenAI-compatible APIs
//! - Tool/function calling
//! - Non-streaming chat completions
//!
//! ## Example
//!
//! ```no_run
//! use neuromance_client::{LLMClient, OpenAIClient};
//! use neuromance_common::{Config, Message};
//! use uuid::Uuid;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a client configuration
//! let config = Config::new("openai", "gpt-4")
//!     .with_api_key("your-api-key")
//!     .with_base_url("https://api.openai.com/v1");
//!
//! // Initialize the client
//! let client = OpenAIClient::new(config)?;
//!
//! // Create a chat request
//! let conversation_id = Uuid::new_v4();
//! let message = Message::user(conversation_id, "Hello, world!");
//! let request = (client.config().clone(), vec![message]).into();
//!
//! // Send the request
//! let response = client.chat(request).await?;
//! println!("Response: {}", response.message.content);
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use async_trait::async_trait;

use neuromance_common::{ChatRequest, ChatResponse, Config};

pub mod error;
pub mod openai;

pub use error::ClientError;
pub use openai::OpenAIClient;

/// Trait for LLM client implementations.
///
/// Provides a unified interface for interacting with different LLM providers.
/// Implementations must support async operations and be thread-safe (Send + Sync).
#[must_use = "LLMClient must be used to make requests"]
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Get the client's configuration.
    ///
    /// Returns a reference to the configuration used to initialize this client.
    fn config(&self) -> &Config;

    /// Send a chat completion request to the LLM.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat completion request containing messages and parameters
    ///
    /// # Returns
    ///
    /// A `ChatResponse` containing the model's reply, usage information, and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The request fails validation
    /// - Network communication fails
    /// - The API returns an error (authentication, rate limit, etc.)
    /// - The response cannot be parsed
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;

    /// Check if the client supports tool/function calling.
    ///
    /// Returns `true` if this client can handle requests with tools.
    fn supports_tools(&self) -> bool;

    /// Check if the client supports streaming responses.
    ///
    /// Returns `true` if this client can handle streaming chat completions.
    /// Note: Streaming is not yet implemented.
    fn supports_streaming(&self) -> bool;

    /// Validate a configuration object.
    ///
    /// Checks that configuration parameters are within valid ranges:
    /// - `temperature`: 0.0 to 2.0
    /// - `top_p`: 0.0 to 1.0
    /// - `frequency_penalty`: -2.0 to 2.0
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
    /// Checks that:
    /// - At least one message is provided
    /// - Tools are not used if the client doesn't support them
    /// - Streaming is not requested if the client doesn't support it
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
    use super::*;
    use neuromance_common::chat::Message;
    use neuromance_common::client::{ChatResponse, ToolChoice};
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

        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse> {
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
        use neuromance_common::tools::{Function, Property};
        use std::collections::HashMap;

        let mut properties = HashMap::new();
        properties.insert(
            "arg".to_string(),
            Property {
                prop_type: "string".to_string(),
                description: "A test argument".to_string(),
            },
        );

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "test_function".to_string(),
                description: "A test function".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": ["arg"]
                }),
            },
        }
    }

    #[test]
    fn test_validate_request_empty_messages() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
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
            messages: vec![create_test_message()],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
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
            messages: vec![create_test_message()],
            tools: Some(vec![create_test_tool()]),
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
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
            messages: vec![create_test_message()],
            tools: Some(vec![create_test_tool()]),
            tool_choice: Some(ToolChoice::Auto),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
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
            messages: vec![create_test_message()],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: true,
            user: None,
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
            messages: vec![create_test_message()],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: true,
            user: None,
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
            messages: vec![create_test_message()],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            metadata: HashMap::new(),
        };

        let result = client.chat(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.response_id, Some("test-response".to_string()));
        assert_eq!(response.model, "mock-model");
    }
}
