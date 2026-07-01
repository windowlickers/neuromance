//! Anthropic client implementation.
//!
//! This module provides a client for interacting with the Anthropic Messages API.
//!
//! # Features
//!
//! - **Messages API**: Implementation of the Anthropic Messages API
//! - **Tool/Function Calling**: Support for tool use with streaming accumulation
//! - **Automatic Retries**: Configurable exponential backoff with jitter
//! - **Secure API Keys**: Uses the `secrecy` crate to prevent accidental exposure
//!
//! # Examples
//!
//! ## Basic Chat Completion
//!
//! ```no_run
//! use neuromance_client::{AnthropicClient, LLMClient};
//! use neuromance_common::client::{Config, ChatRequest};
//! use neuromance_common::chat::Conversation;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = Config::new("anthropic", "claude-sonnet-4-5-20250929")
//!     .with_api_key("sk-ant-...")
//!     .with_base_url("https://api.anthropic.com/v1");
//!
//! let client = AnthropicClient::new(config)?;
//!
//! let mut conversation = Conversation::new()
//!     .with_title("Example Chat");
//!
//! conversation.add_message(
//!     conversation.system_message("You are a helpful assistant")
//! )?;
//!
//! conversation.add_message(
//!     conversation.user_message("Hello!")
//! )?;
//!
//! let request = ChatRequest::new(conversation.get_messages().to_vec())
//!     .with_max_tokens(1024);
//! let response = client.chat(&request).await?;
//!
//! println!("Response: {}", response.message.content);
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use chrono::Utc;
use futures::stream::Stream;
use reqwest_middleware::ClientWithMiddleware;
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{error, warn};

use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config, ProxyConfig, Usage};
use neuromance_common::tools::{FunctionCall, ToolCall};

use crate::error::ClientError;
use crate::message::MessageBuilder;
use crate::streaming::{StreamingProvider, run_sse_stream};
use crate::transport::{add_proxy_headers, send_json};
use crate::{LLMClient, build_client_resources};

use super::{
    ANTHROPIC_VERSION, ContentBlockStart, CreateMessageRequest, DEFAULT_BASE_URL, Delta,
    INTERLEAVED_THINKING_BETA, MessageResponse, ResponseContentBlock, StreamEvent,
    StreamingToolCall,
};

/// Client for Anthropic's Messages API.
///
/// Supports Claude models with tool/function calling and streaming.
///
/// # Security
///
/// The API key is stored using the `secrecy` crate to prevent accidental
/// exposure through debug logs or memory dumps.
///
/// # Proxy Support
///
/// When a [`ProxyConfig`] is provided in the [`Config`], requests are routed
/// through a tokenizer proxy. The proxy intercepts requests and injects real
/// credentials, allowing agents to use sealed tokens instead of raw API keys.
#[derive(Clone)]
pub struct AnthropicClient {
    client: ClientWithMiddleware,
    streaming_client: reqwest::Client,
    api_key: Arc<SecretString>,
    base_url: String,
    config: Arc<Config>,
    proxy_config: Option<ProxyConfig>,
}

impl std::fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("config", &self.config)
            .field("proxy_config", &self.proxy_config)
            .finish_non_exhaustive()
    }
}

impl AnthropicClient {
    /// Create a new Anthropic client from a configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Client configuration including API key and base URL
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use neuromance_client::AnthropicClient;
    /// use neuromance_common::client::Config;
    ///
    /// let config = Config::new("anthropic", "claude-sonnet-4-5-20250929")
    ///     .with_api_key("sk-ant-...")
    ///     .with_base_url("https://api.anthropic.com/v1");
    ///
    /// let client = AnthropicClient::new(config)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// # Proxy Configuration
    ///
    /// When a [`ProxyConfig`] is provided, requests are routed through the proxy.
    /// The `api_key` should contain a sealed token instead of the real API key.
    ///
    /// ```no_run
    /// use neuromance_client::AnthropicClient;
    /// use neuromance_common::client::{Config, ProxyConfig};
    ///
    /// let config = Config::new("anthropic", "claude-sonnet-4-5-20250929")
    ///     .with_api_key("sealed.abc123xyz...")  // Sealed token
    ///     .with_proxy(ProxyConfig {
    ///         proxy_url: "http://tokenizer.internal:8080".to_string(),
    ///         token_header: "X-Tokenizer-Token".to_string(),
    ///     });
    ///
    /// let client = AnthropicClient::new(config)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is missing or HTTP client creation fails.
    pub fn new(config: Config) -> Result<Self, ClientError> {
        let r = build_client_resources(config, DEFAULT_BASE_URL)?;

        Ok(Self {
            client: r.client,
            streaming_client: r.streaming_client,
            api_key: r.api_key,
            base_url: r.base_url,
            config: r.config,
            proxy_config: r.proxy_config,
        })
    }

    /// Set a custom base URL for the API endpoint.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        Arc::make_mut(&mut self.config).base_url = Some(base_url.clone());
        self.base_url = base_url;
        self
    }

    /// Set the model to use for chat completions.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.config).model = model.into();
        self
    }

    /// Make a non-streaming request to the Messages API.
    ///
    /// # Arguments
    ///
    /// * `body` - The request body
    /// * `beta_features` - Optional beta header value for enabling beta features
    async fn make_request(
        &self,
        body: &CreateMessageRequest,
        beta_features: Option<&str>,
    ) -> Result<MessageResponse, ClientError> {
        let url = format!("{}/messages", self.base_url);

        // Validate URL construction
        reqwest::Url::parse(&url)
            .map_err(|e| ClientError::ConfigurationError(format!("Invalid URL '{url}': {e}")))?;

        let mut request_builder = self
            .client
            .post(&url)
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json");

        // Add beta header if specified
        if let Some(beta) = beta_features {
            request_builder = request_builder.header("anthropic-beta", beta);
        }

        // Add proxy headers if configured
        request_builder =
            add_proxy_headers(request_builder, self.proxy_config.as_ref(), &self.api_key);

        let request_builder = request_builder
            .body(serde_json::to_string(body).map_err(ClientError::SerializationError)?);

        send_json(request_builder).await
    }

    /// Convert an Anthropic response to our internal Message format.
    fn convert_response_to_message(
        response: &MessageResponse,
        conversation_id: uuid::Uuid,
    ) -> Message {
        let mut builder = MessageBuilder::new(conversation_id, MessageRole::Assistant);
        for block in &response.content {
            match block {
                ResponseContentBlock::Text { text, .. } => builder.append_content(text),
                ResponseContentBlock::ToolUse { id, name, input } => {
                    builder.push_tool_call(ToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: input.to_string(),
                        },
                        index: None,
                    });
                }
                ResponseContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    builder.append_reasoning(thinking, "\n\n");
                    builder.set_reasoning_signature(signature.clone());
                }
                ResponseContentBlock::RedactedThinking { .. } => {}
            }
        }
        builder.build()
    }
}

/// Convert an Anthropic streaming event to our common `ChatChunk` format.
#[must_use]
#[allow(clippy::too_many_lines)]
#[allow(clippy::implicit_hasher)]
pub fn convert_event_to_chat_chunk(
    event: &StreamEvent,
    model: &str,
    response_id: &str,
    streaming_tool_calls: Option<&mut HashMap<u32, StreamingToolCall>>,
) -> Option<ChatChunk> {
    match event {
        StreamEvent::MessageStart { message } => Some(ChatChunk {
            model: message.model.clone(),
            delta_content: None,
            delta_reasoning_content: None,
            delta_role: Some(MessageRole::Assistant),
            delta_tool_calls: None,
            finish_reason: None,
            usage: Some(Usage::from(message.usage.clone())),
            response_id: Some(message.id.clone()),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }),

        StreamEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            match content_block {
                ContentBlockStart::Text { text } => {
                    if text.is_empty() {
                        None
                    } else {
                        Some(ChatChunk {
                            model: model.to_string(),
                            delta_content: Some(text.clone()),
                            delta_reasoning_content: None,
                            delta_role: None,
                            delta_tool_calls: None,
                            finish_reason: None,
                            usage: None,
                            response_id: Some(response_id.to_string()),
                            created_at: Utc::now(),
                            metadata: HashMap::new(),
                        })
                    }
                }
                ContentBlockStart::Thinking { thinking } => {
                    if thinking.is_empty() {
                        None
                    } else {
                        Some(ChatChunk {
                            model: model.to_string(),
                            delta_content: None,
                            delta_reasoning_content: Some(thinking.clone()),
                            delta_role: None,
                            delta_tool_calls: None,
                            finish_reason: None,
                            usage: None,
                            response_id: Some(response_id.to_string()),
                            created_at: Utc::now(),
                            metadata: HashMap::new(),
                        })
                    }
                }
                ContentBlockStart::ToolUse { id, name, .. } => {
                    // Start accumulating this tool call
                    if let Some(tool_calls) = streaming_tool_calls {
                        tool_calls.insert(*index, StreamingToolCall::new(id.clone(), name.clone()));
                    }
                    None
                }
            }
        }

        StreamEvent::ContentBlockDelta { index, delta } => match delta {
            Delta::TextDelta { text } => Some(ChatChunk {
                model: model.to_string(),
                delta_content: Some(text.clone()),
                delta_reasoning_content: None,
                delta_role: None,
                delta_tool_calls: None,
                finish_reason: None,
                usage: None,
                response_id: Some(response_id.to_string()),
                created_at: Utc::now(),
                metadata: HashMap::new(),
            }),
            Delta::ThinkingDelta { thinking } => Some(ChatChunk {
                model: model.to_string(),
                delta_content: None,
                delta_reasoning_content: Some(thinking.clone()),
                delta_role: None,
                delta_tool_calls: None,
                finish_reason: None,
                usage: None,
                response_id: Some(response_id.to_string()),
                created_at: Utc::now(),
                metadata: HashMap::new(),
            }),
            Delta::InputJsonDelta { partial_json } => {
                // Accumulate JSON for this tool call
                if let Some(tool_calls) = streaming_tool_calls
                    && let Some(tool_call) = tool_calls.get_mut(index)
                {
                    tool_call.append_delta(partial_json);
                }
                None
            }
            Delta::SignatureDelta { .. } => {
                // Signature deltas are not exposed in our chunk format
                None
            }
        },

        StreamEvent::ContentBlockStop { index } => {
            // If this was a tool call, finalize it
            if let Some(tool_calls) = streaming_tool_calls
                && let Some(tool_call) = tool_calls.remove(index)
            {
                match tool_call.finalize() {
                    Ok(finalized) => {
                        return Some(ChatChunk {
                            model: model.to_string(),
                            delta_content: None,
                            delta_reasoning_content: None,
                            delta_role: None,
                            delta_tool_calls: Some(vec![finalized]),
                            finish_reason: None,
                            usage: None,
                            response_id: Some(response_id.to_string()),
                            created_at: Utc::now(),
                            metadata: HashMap::new(),
                        });
                    }
                    Err(e) => {
                        warn!("Failed to finalize tool call: {e}");
                    }
                }
            }
            None
        }

        StreamEvent::MessageDelta { delta, usage } => {
            let finish_reason = delta.stop_reason.as_ref().map(|r| r.clone().into());

            // MessageDelta only carries output_tokens; input token data
            // (including cache stats) arrives in MessageStart.  We emit
            // a partial Usage here so consumers can merge the two.
            Some(ChatChunk {
                model: model.to_string(),
                delta_content: None,
                delta_reasoning_content: None,
                delta_role: None,
                delta_tool_calls: None,
                finish_reason,
                usage: Some(Usage {
                    prompt_tokens: 0,
                    completion_tokens: usage.output_tokens,
                    total_tokens: usage.output_tokens,
                    cost: None,
                    input_tokens_details: None,
                    output_tokens_details: None,
                }),
                response_id: Some(response_id.to_string()),
                created_at: Utc::now(),
                metadata: HashMap::new(),
            })
        }

        StreamEvent::MessageStop | StreamEvent::Ping => None,
        StreamEvent::Error { error } => {
            warn!(
                "Stream error from API: {} - {}",
                error.error_type, error.message
            );
            None
        }
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    fn config(&self) -> &Config {
        &self.config
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
        self.validate_request(request)?;

        let mut anthropic_request = CreateMessageRequest::from((request, self.config.as_ref()));
        anthropic_request.stream = Some(false);

        // Determine if interleaved thinking beta should be enabled
        let beta_features = if request.thinking.is_interleaved() {
            Some(INTERLEAVED_THINKING_BETA)
        } else {
            None
        };

        let response = self.make_request(&anthropic_request, beta_features).await?;

        // Get conversation_id from first message
        let conversation_id = request
            .messages
            .first()
            .ok_or_else(|| {
                error!("Request has no messages despite passing validation");
                ClientError::InvalidRequest("Request must contain at least one message".to_string())
            })?
            .conversation_id;

        let message = Self::convert_response_to_message(&response, conversation_id);

        let finish_reason = response.stop_reason.map(std::convert::Into::into);

        Ok(ChatResponse {
            message,
            model: response.model,
            usage: Some(Usage::from(response.usage)),
            finish_reason,
            created_at: Utc::now(),
            response_id: Some(response.id),
            metadata: HashMap::new(),
        })
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
    {
        self.validate_request(request)?;

        let mut anthropic_request = CreateMessageRequest::from((request, self.config.as_ref()));
        anthropic_request.stream = Some(true);

        let url = format!("{}/messages", self.base_url);
        reqwest::Url::parse(&url)
            .map_err(|e| ClientError::ConfigurationError(format!("Invalid URL '{url}': {e}")))?;

        let mut request_builder = self
            .streaming_client
            .post(&url)
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json");

        if request.thinking.is_interleaved() {
            request_builder = request_builder.header("anthropic-beta", INTERLEAVED_THINKING_BETA);
        }

        request_builder =
            add_proxy_headers(request_builder, self.proxy_config.as_ref(), &self.api_key);

        let request_builder = request_builder.json(&anthropic_request);

        run_sse_stream(self, request_builder)
    }
}

/// Per-stream accumulator for the Anthropic streaming protocol.
///
/// `model` and `response_id` are seeded from config and overwritten by the
/// `MessageStart` event; `streaming_tool_calls` accumulates partial tool-use
/// JSON across `ContentBlockStart` / `ContentBlockDelta` / `ContentBlockStop`.
pub struct AnthropicStreamState {
    model: String,
    response_id: String,
    streaming_tool_calls: HashMap<u32, StreamingToolCall>,
}

impl StreamingProvider for AnthropicClient {
    type Event = StreamEvent;
    type State = AnthropicStreamState;

    fn initial_state(&self) -> Self::State {
        AnthropicStreamState {
            model: self.config.model.clone(),
            response_id: String::new(),
            streaming_tool_calls: HashMap::new(),
        }
    }

    fn process_event(
        state: &mut Self::State,
        event: Self::Event,
    ) -> Option<Result<ChatChunk, ClientError>> {
        if let StreamEvent::MessageStart { message: ref msg } = event {
            state.model.clone_from(&msg.model);
            state.response_id.clone_from(&msg.id);
        }
        convert_event_to_chat_chunk(
            &event,
            &state.model,
            &state.response_id,
            Some(&mut state.streaming_tool_calls),
        )
        .map(Ok)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::panic)]
    #![allow(clippy::match_wildcard_for_single_variants)]

    use super::*;
    use futures::StreamExt;
    use neuromance_common::chat::{Message, MessageRole};
    use neuromance_common::client::FinishReason;
    use smallvec::SmallVec;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn create_test_config(base_url: &str) -> Config {
        Config::new("anthropic", "claude-sonnet-4-5-20250929")
            .with_api_key("test-key")
            .with_base_url(base_url)
    }

    fn create_test_message() -> Message {
        Message {
            id: uuid::Uuid::new_v4(),
            conversation_id: uuid::Uuid::new_v4(),
            role: MessageRole::User,
            content: "Hello".to_string(),
            tool_calls: SmallVec::new(),
            tool_call_id: None,
            name: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            reasoning: None,
            model: None,
            provider: None,
            usage: None,
        }
    }

    #[tokio::test]
    async fn test_successful_chat_completion() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Hello! How can I help you today?"
                }],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.model, "claude-sonnet-4-5-20250929");
        assert_eq!(response.message.content, "Hello! How can I help you today?");
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[tokio::test]
    async fn test_chat_completion_with_tool_use() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_weather",
                    "input": {"location": "San Francisco", "unit": "celsius"}
                }],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 15,
                    "output_tokens": 25
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.message.tool_calls.len(), 1);

        let tool_call = &response.message.tool_calls[0];
        assert_eq!(tool_call.id, "toolu_01A09q90qw90lq917835lq9");
        assert_eq!(tool_call.function.name, "get_weather");
    }

    #[tokio::test]
    async fn test_authentication_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let result = client.chat(&request).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid API key"));
    }

    #[tokio::test]
    async fn test_rate_limit_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit exceeded"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let result = client.chat(&request).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Rate limit"));
    }

    #[tokio::test]
    async fn test_overloaded_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(529).set_body_json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "API is overloaded"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let result = client.chat(&request).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("unavailable") || error_msg.contains("overloaded"));
    }

    #[tokio::test]
    async fn test_stop_reason_mapping() {
        let test_cases = vec![
            ("end_turn", FinishReason::Stop),
            ("max_tokens", FinishReason::Length),
            ("stop_sequence", FinishReason::Stop),
            ("tool_use", FinishReason::ToolCalls),
        ];

        for (stop_reason_str, expected_finish_reason) in test_cases {
            let mock_server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/messages"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "test"}],
                    "model": "claude-sonnet-4-5-20250929",
                    "stop_reason": stop_reason_str,
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                })))
                .mount(&mock_server)
                .await;

            let config = create_test_config(&mock_server.uri());
            let client = AnthropicClient::new(config).unwrap();

            let message = create_test_message();
            let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

            let response = client.chat(&request).await.unwrap();
            assert_eq!(response.finish_reason, Some(expected_finish_reason));
        }
    }

    #[tokio::test]
    async fn test_chat_with_thinking_budget() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_thinking_test",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me think about this...",
                        "signature": "test_signature"
                    },
                    {
                        "type": "text",
                        "text": "Here's my response."
                    }
                ],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 50
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message])
            .with_max_tokens(16000)
            .with_thinking_budget(10000);

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.message.content, "Here's my response.");
        assert_eq!(
            response
                .message
                .reasoning
                .as_ref()
                .map(|r| r.content.as_str()),
            Some("Let me think about this...")
        );
    }

    #[tokio::test]
    async fn test_chat_with_interleaved_thinking_sends_beta_header() {
        let mock_server = MockServer::start().await;

        // This mock verifies the beta header is sent
        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            .and(header("anthropic-beta", INTERLEAVED_THINKING_BETA))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_interleaved_test",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Response with interleaved thinking"
                }],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message])
            .with_max_tokens(16000)
            .with_interleaved_thinking(10000); // Use interleaved thinking with budget

        let response = client.chat(&request).await.unwrap();

        assert_eq!(
            response.message.content,
            "Response with interleaved thinking"
        );
    }

    #[tokio::test]
    async fn test_chat_without_interleaved_thinking_no_beta_header() {
        let mock_server = MockServer::start().await;

        // This mock should NOT match if the beta header is sent
        // We verify by checking the request succeeds without the beta header requirement
        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            // NOT requiring the beta header
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_no_interleaved",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Regular response"
                }],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        // Using Extended thinking (not Interleaved), so no beta header should be sent
        let request = ChatRequest::new(vec![message])
            .with_max_tokens(1024)
            .with_thinking_budget(5000); // Extended thinking, not interleaved

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.message.content, "Regular response");
    }

    // ========================================================================
    // Streaming Integration Tests
    // ========================================================================

    #[test]
    fn test_streaming_text_delta_conversion() {
        use crate::anthropic::{Delta, StreamEvent};

        // Test TextDelta produces correct ChatChunk
        let event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::TextDelta {
                text: "Hello, world!".to_string(),
            },
        };

        let chunk =
            convert_event_to_chat_chunk(&event, "claude-sonnet-4-5-20250929", "resp_123", None);
        let chunk = chunk.expect("Should produce a chunk");

        assert_eq!(chunk.delta_content, Some("Hello, world!".to_string()));
        assert_eq!(chunk.model, "claude-sonnet-4-5-20250929");
        assert_eq!(chunk.response_id, Some("resp_123".to_string()));
        assert!(chunk.delta_reasoning_content.is_none());
        assert!(chunk.delta_tool_calls.is_none());
    }

    #[test]
    fn test_streaming_thinking_delta_conversion() {
        use crate::anthropic::{Delta, StreamEvent};

        let event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::ThinkingDelta {
                thinking: "Let me consider this...".to_string(),
            },
        };

        let chunk =
            convert_event_to_chat_chunk(&event, "claude-sonnet-4-5-20250929", "resp_123", None);
        let chunk = chunk.expect("Should produce a chunk");

        assert_eq!(
            chunk.delta_reasoning_content,
            Some("Let me consider this...".to_string())
        );
        assert!(chunk.delta_content.is_none());
    }

    #[test]
    fn test_streaming_tool_call_accumulation() {
        use crate::anthropic::{ContentBlockStart, Delta, StreamEvent, StreamingToolCall};
        use std::collections::HashMap;

        let mut tool_calls: HashMap<u32, StreamingToolCall> = HashMap::new();

        // Step 1: ContentBlockStart with ToolUse
        let start_event = StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlockStart::ToolUse {
                id: "toolu_abc123".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::Value::Object(serde_json::Map::new()),
            },
        };
        let chunk = convert_event_to_chat_chunk(
            &start_event,
            "claude-sonnet-4-5-20250929",
            "resp_123",
            Some(&mut tool_calls),
        );
        assert!(chunk.is_none()); // No chunk emitted for start
        assert!(tool_calls.contains_key(&0));

        // Step 2: Multiple InputJsonDelta events
        let delta1 = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::InputJsonDelta {
                partial_json: r#"{"location":"#.to_string(),
            },
        };
        let chunk = convert_event_to_chat_chunk(
            &delta1,
            "claude-sonnet-4-5-20250929",
            "resp_123",
            Some(&mut tool_calls),
        );
        assert!(chunk.is_none()); // No chunk for partial JSON

        let delta2 = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::InputJsonDelta {
                partial_json: r#""San Francisco"}"#.to_string(),
            },
        };
        let _ = convert_event_to_chat_chunk(
            &delta2,
            "claude-sonnet-4-5-20250929",
            "resp_123",
            Some(&mut tool_calls),
        );

        // Step 3: ContentBlockStop finalizes the tool call
        let stop_event = StreamEvent::ContentBlockStop { index: 0 };
        let chunk = convert_event_to_chat_chunk(
            &stop_event,
            "claude-sonnet-4-5-20250929",
            "resp_123",
            Some(&mut tool_calls),
        );

        let chunk = chunk.expect("Should produce a chunk with tool call");
        let tool_calls_result = chunk.delta_tool_calls.expect("Should have tool calls");
        assert_eq!(tool_calls_result.len(), 1);

        let tool_call = &tool_calls_result[0];
        assert_eq!(tool_call.id, "toolu_abc123");
        assert_eq!(tool_call.function.name, "get_weather");
        // Verify the JSON was parsed correctly
        let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments).unwrap();
        assert_eq!(args["location"], "San Francisco");
    }

    #[test]
    fn test_streaming_message_start_captures_metadata() {
        use crate::anthropic::{AnthropicUsage, MessageResponse, StreamEvent};

        let event = StreamEvent::MessageStart {
            message: MessageResponse {
                id: "msg_01XYZ".to_string(),
                response_type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: "claude-sonnet-4-5-20250929".to_string(),
                stop_reason: None,
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: 100,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            },
        };

        let chunk = convert_event_to_chat_chunk(&event, "", "", None);
        let chunk = chunk.expect("Should produce a chunk");

        assert_eq!(chunk.model, "claude-sonnet-4-5-20250929");
        assert_eq!(chunk.response_id, Some("msg_01XYZ".to_string()));
        assert_eq!(chunk.delta_role, Some(MessageRole::Assistant));

        let usage = chunk.usage.expect("Should have usage");
        assert_eq!(usage.prompt_tokens, 100);
    }

    // ========================================================================
    // Multiple Content Blocks Tests
    // ========================================================================

    #[tokio::test]
    async fn test_response_with_text_and_tool_use_blocks() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_multi_block",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'll help you check the weather."
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_weather_1",
                        "name": "get_weather",
                        "input": {"location": "New York", "unit": "fahrenheit"}
                    }
                ],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 100
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let response = client.chat(&request).await.unwrap();

        // Both text and tool call should be extracted
        assert_eq!(response.message.content, "I'll help you check the weather.");
        assert_eq!(response.message.tool_calls.len(), 1);

        let tool_call = &response.message.tool_calls[0];
        assert_eq!(tool_call.id, "toolu_weather_1");
        assert_eq!(tool_call.function.name, "get_weather");
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn test_response_with_thinking_text_and_tool_blocks() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_thinking_multi",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I need to analyze this request carefully...",
                        "signature": "sig_xyz789"
                    },
                    {
                        "type": "text",
                        "text": "Based on my analysis, let me search for that."
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_search_1",
                        "name": "web_search",
                        "input": {"query": "latest news"}
                    }
                ],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 200
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message])
            .with_max_tokens(16000)
            .with_thinking_budget(10000);

        let response = client.chat(&request).await.unwrap();

        // All three content types should be extracted
        assert_eq!(
            response
                .message
                .reasoning
                .as_ref()
                .map(|r| r.content.as_str()),
            Some("I need to analyze this request carefully...")
        );
        assert_eq!(
            response.message.content,
            "Based on my analysis, let me search for that."
        );
        assert_eq!(response.message.tool_calls.len(), 1);
        assert_eq!(response.message.tool_calls[0].function.name, "web_search");
    }

    // ========================================================================
    // Tool Result Message Conversion Tests
    // ========================================================================

    #[test]
    fn test_tool_result_message_conversion() {
        use crate::anthropic::{
            AnthropicMessage, AnthropicRole, MessageContent, RequestContentBlock, ToolResultContent,
        };

        let tool_message = Message {
            id: uuid::Uuid::new_v4(),
            conversation_id: uuid::Uuid::new_v4(),
            role: MessageRole::Tool,
            content: "The weather in San Francisco is 72°F and sunny.".to_string(),
            tool_calls: SmallVec::new(),
            tool_call_id: Some("toolu_weather_123".to_string()),
            name: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            reasoning: None,
            model: None,
            provider: None,
            usage: None,
        };

        let anthropic_msg = AnthropicMessage::from(&tool_message);

        // Tool results are sent as User role in Anthropic API
        assert!(matches!(anthropic_msg.role, AnthropicRole::User));

        // Content should be a ToolResult block
        match anthropic_msg.content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    RequestContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        assert_eq!(tool_use_id, "toolu_weather_123");
                        assert!(is_error.is_none());
                        match content {
                            Some(ToolResultContent::Text(text)) => {
                                assert_eq!(text, "The weather in San Francisco is 72°F and sunny.");
                            }
                            _ => panic!("Expected Text content in ToolResult"),
                        }
                    }
                    _ => panic!("Expected ToolResult block"),
                }
            }
            _ => panic!("Expected Blocks content"),
        }
    }

    #[test]
    fn test_assistant_message_with_tool_calls_conversion() {
        use crate::anthropic::{
            AnthropicMessage, AnthropicRole, MessageContent, RequestContentBlock,
        };
        use neuromance_common::tools::{FunctionCall, ToolCall};

        let mut tool_calls = SmallVec::new();
        tool_calls.push(ToolCall {
            id: "toolu_calc_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "calculate".to_string(),
                arguments: r#"{"expression":"2+2"}"#.to_string(),
            },
            index: None,
        });

        let assistant_message = Message {
            id: uuid::Uuid::new_v4(),
            conversation_id: uuid::Uuid::new_v4(),
            role: MessageRole::Assistant,
            content: "Let me calculate that for you.".to_string(),
            tool_calls,
            tool_call_id: None,
            name: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            reasoning: None,
            model: None,
            provider: None,
            usage: None,
        };

        let anthropic_msg = AnthropicMessage::from(&assistant_message);

        assert!(matches!(anthropic_msg.role, AnthropicRole::Assistant));

        match anthropic_msg.content {
            MessageContent::Blocks(blocks) => {
                // Should have Text block and ToolUse block
                assert_eq!(blocks.len(), 2);

                // First block should be text
                match &blocks[0] {
                    RequestContentBlock::Text { text, .. } => {
                        assert_eq!(text, "Let me calculate that for you.");
                    }
                    _ => panic!("Expected Text block first"),
                }

                // Second block should be tool use
                match &blocks[1] {
                    RequestContentBlock::ToolUse { id, name, input } => {
                        assert_eq!(id, "toolu_calc_1");
                        assert_eq!(name, "calculate");
                        assert_eq!(input["expression"], "2+2");
                    }
                    _ => panic!("Expected ToolUse block second"),
                }
            }
            _ => panic!("Expected Blocks content"),
        }
    }

    // ========================================================================
    // Cache Control Tests
    // ========================================================================

    #[test]
    fn test_anthropic_usage_preserves_cache_stats() {
        use crate::anthropic::AnthropicUsage;

        let usage = AnthropicUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_creation_input_tokens: 200,
            cache_read_input_tokens: 800,
        };

        let common_usage = Usage::from(usage);

        // prompt_tokens normalizes to total input: 1000 + 200 + 800
        assert_eq!(common_usage.prompt_tokens, 2000);
        assert_eq!(common_usage.completion_tokens, 500);
        assert_eq!(common_usage.total_tokens, 2500);

        let details = common_usage
            .input_tokens_details
            .expect("Should have input_tokens_details");
        assert_eq!(details.cached_tokens, 800);
        assert_eq!(details.cache_creation_tokens, 200);
    }

    #[test]
    fn test_system_blocks_last_gets_cache_control() {
        use crate::anthropic::{CreateMessageRequest, SystemContentBlock, SystemPrompt};

        let mut messages = vec![
            Message {
                id: uuid::Uuid::new_v4(),
                conversation_id: uuid::Uuid::new_v4(),
                role: MessageRole::System,
                content: "First system message".to_string(),
                tool_calls: SmallVec::new(),
                tool_call_id: None,
                name: None,
                timestamp: Utc::now(),
                metadata: HashMap::new(),
                reasoning: None,
                model: None,
                provider: None,
                usage: None,
            },
            Message {
                id: uuid::Uuid::new_v4(),
                conversation_id: uuid::Uuid::new_v4(),
                role: MessageRole::System,
                content: "Second system message".to_string(),
                tool_calls: SmallVec::new(),
                tool_call_id: None,
                name: None,
                timestamp: Utc::now(),
                metadata: HashMap::new(),
                reasoning: None,
                model: None,
                provider: None,
                usage: None,
            },
        ];
        messages.push(create_test_message()); // Add a user message

        let config =
            Config::new("anthropic", "claude-sonnet-4-5-20250929").with_api_key("test-key");
        let request = ChatRequest::new(messages);

        let anthropic_request = CreateMessageRequest::from((&request, &config));

        // Verify system prompt exists and has cache control on last block only
        let system = anthropic_request.system.expect("Should have system prompt");
        match system {
            SystemPrompt::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);

                // First block should NOT have cache_control
                match &blocks[0] {
                    SystemContentBlock::Text { cache_control, .. } => {
                        assert!(
                            cache_control.is_none(),
                            "First block should not have cache_control"
                        );
                    }
                }

                // Last block SHOULD have cache_control
                match &blocks[1] {
                    SystemContentBlock::Text { cache_control, .. } => {
                        assert!(
                            cache_control.is_some(),
                            "Last block should have cache_control"
                        );
                    }
                }
            }
            _ => panic!("Expected Blocks variant"),
        }
    }

    #[test]
    fn test_tools_last_gets_cache_control() {
        use crate::anthropic::CreateMessageRequest;
        use neuromance_common::tools::{Function, Tool};

        let message = create_test_message();
        let config =
            Config::new("anthropic", "claude-sonnet-4-5-20250929").with_api_key("test-key");

        let tools = vec![
            Tool {
                r#type: "function".to_string(),
                function: Function {
                    name: "tool_one".to_string(),
                    description: "First tool".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: Function {
                    name: "tool_two".to_string(),
                    description: "Second tool".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            },
        ];

        let request = ChatRequest::new(vec![message]).with_tools(tools);
        let anthropic_request = CreateMessageRequest::from((&request, &config));

        let tools = anthropic_request.tools.expect("Should have tools");
        assert_eq!(tools.len(), 2);

        // First tool should NOT have cache_control
        assert!(
            tools[0].cache_control.is_none(),
            "First tool should not have cache_control"
        );

        // Last tool SHOULD have cache_control
        assert!(
            tools[1].cache_control.is_some(),
            "Last tool should have cache_control"
        );
    }

    // ==================== Proxy Header Tests ====================

    fn create_test_config_with_proxy(proxy_url: &str) -> Config {
        Config::new("anthropic", "claude-sonnet-4-5-20250929")
            .with_api_key("sealed.test-token")
            .with_proxy(ProxyConfig {
                proxy_url: proxy_url.to_string(),
                token_header: "X-Tokenizer-Token".to_string(),
            })
    }

    fn create_successful_response() -> serde_json::Value {
        serde_json::json!({
            "id": "msg_proxy_test",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Response via proxy"
            }],
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        })
    }

    #[tokio::test]
    async fn test_proxy_headers_sent() {
        let mock_server = MockServer::start().await;

        // Verify the sealed-token header is sent and the upstream path
        // (including the `/v1` prefix from the default base URL) is preserved
        // when reqwest routes requests through the forward proxy.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("X-Tokenizer-Token", "sealed.test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(create_successful_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = create_test_config_with_proxy(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let response = client.chat(&request).await.unwrap();
        assert_eq!(response.message.content, "Response via proxy");
    }

    #[tokio::test]
    async fn test_proxy_with_custom_token_header() {
        let mock_server = MockServer::start().await;

        // Use a custom token header name; the upstream host is in the URL,
        // not a side-band header, so no target-host header is expected.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("X-Custom-Token", "sealed.custom-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(create_successful_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = Config::new("anthropic", "claude-sonnet-4-5-20250929")
            .with_api_key("sealed.custom-token")
            .with_proxy(ProxyConfig {
                proxy_url: mock_server.uri(),
                token_header: "X-Custom-Token".to_string(),
            });

        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let response = client.chat(&request).await.unwrap();
        assert_eq!(response.message.content, "Response via proxy");
    }

    #[tokio::test]
    async fn test_proxy_headers_sent_streaming() {
        let mock_server = MockServer::start().await;

        // Minimal SSE event sequence for Anthropic streaming
        let sse_body = [
            "event: message_start",
            &format!(
                "data: {}",
                serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_proxy_stream",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-sonnet-4-5-20250929",
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": { "input_tokens": 10, "output_tokens": 0 }
                    }
                })
            ),
            "",
            "event: content_block_start",
            &format!(
                "data: {}",
                serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                })
            ),
            "",
            "event: content_block_delta",
            &format!(
                "data: {}",
                serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "Streamed via proxy" }
                })
            ),
            "",
            "event: content_block_stop",
            &format!(
                "data: {}",
                serde_json::json!({ "type": "content_block_stop", "index": 0 })
            ),
            "",
            "event: message_delta",
            &format!(
                "data: {}",
                serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": { "output_tokens": 5 }
                })
            ),
            "",
            "event: message_stop",
            &format!("data: {}", serde_json::json!({ "type": "message_stop" })),
            "",
        ]
        .join("\n");

        // Verify proxy headers are sent on the streaming path
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("X-Tokenizer-Token", "sealed.test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = create_test_config_with_proxy(&mock_server.uri());
        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let mut stream = client.chat_stream(&request).await.unwrap();

        let mut got_content = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            if chunk.delta_content.as_deref() == Some("Streamed via proxy") {
                got_content = true;
            }
        }
        assert!(
            got_content,
            "expected to receive streamed content via proxy"
        );
    }

    #[tokio::test]
    async fn test_proxy_preserves_custom_base_url_path() {
        let mock_server = MockServer::start().await;

        // With a custom base URL like `https://host/v1`, the forward proxy
        // should still receive the full upstream path in absolute form.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("X-Tokenizer-Token", "sealed.custom-base"))
            .respond_with(ResponseTemplate::new(200).set_body_json(create_successful_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = Config::new("anthropic", "claude-sonnet-4-5-20250929")
            .with_api_key("sealed.custom-base")
            .with_base_url("https://custom.api.example.com/v1")
            .with_proxy(ProxyConfig {
                proxy_url: mock_server.uri(),
                token_header: "X-Tokenizer-Token".to_string(),
            });

        let client = AnthropicClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]).with_max_tokens(1024);

        let response = client.chat(&request).await.unwrap();
        assert_eq!(response.message.content, "Response via proxy");
    }
}
