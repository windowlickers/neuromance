//! Chat Completions client implementation.
//!
//! This module provides a client for any API that implements the Chat Completions specification.
//!
//! # Features
//!
//! - **Chat Completions**: Implementation of the Chat Completions API
//! - **Tool/Function Calling**: Support for function calling and tool use
//! - **Automatic Retries**: Configurable exponential backoff with jitter for transient failures
//! - **Secure API Keys**: Uses the `secrecy` crate to prevent accidental exposure
//!
//! # Examples
//!
//! ## Basic Chat Completion
//!
//! ```no_run
//! use neuromance_client::{ChatCompletionsClient, LLMClient};
//! use neuromance_common::client::{Config, ChatRequest};
//! use neuromance_common::chat::Conversation;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Configure the client
//! let config = Config::new("openai", "gpt-4")
//!     .with_api_key("sk-...")
//!     .with_base_url("https://api.openai.com/v1");
//!
//! let client = ChatCompletionsClient::new(config)?;
//!
//! // Create a conversation and add messages
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
//! // Send the chat request
//! let request = ChatRequest::new(conversation.get_messages().to_vec());
//! let response = client.chat(&request).await?;
//!
//! println!("Response: {}", response.message.content);
//! # Ok(())
//! # }
//! ```
//!
//! ## Using Custom Retry Configuration
//!
//! ```no_run
//! use neuromance_client::ChatCompletionsClient;
//! use neuromance_common::client::{Config, RetryConfig};
//! use std::time::Duration;
//!
//! # fn example() -> anyhow::Result<()> {
//! let retry_config = RetryConfig {
//!     max_retries: 5,
//!     initial_delay: Duration::from_millis(500),
//!     max_delay: Duration::from_secs(60),
//!     backoff_multiplier: 2.0,
//!     jitter: true,
//! };
//!
//! let config = Config::new("openai", "gpt-4")
//!     .with_api_key("sk-...")
//!     .with_retry_config(retry_config);
//!
//! let client = ChatCompletionsClient::new(config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Message Builder Pattern
//!
//! The module provides a type-safe builder for constructing Chat Completions messages:
//!
//! ```
//! use neuromance_client::chat_completions::ChatCompletionsMessage;
//! use neuromance_common::chat::MessageRole;
//!
//! let message = ChatCompletionsMessage::builder()
//!     .role(MessageRole::User)
//!     .content(Some("Hello!".to_string()))
//!     .build();
//! ```
//!
//! # Error Handling
//!
//! The client handles various error scenarios:
//!
//! - **Authentication errors (401)**: Invalid or missing API keys
//! - **Rate limiting (429)**: Retryable; carries the `Retry-After` delay when sent
//! - **Server errors (5xx)**: Transient failures
//! - **Invalid responses**: Missing or malformed response data
//!
//! Retries happen in two places, depending on the call. Non-streaming requests
//! are retried inside this client by the `reqwest-retry` middleware, using
//! [`RetryConfig`](neuromance_common::client::RetryConfig). Streaming requests
//! are not — `reqwest-eventsource` needs a raw `reqwest::RequestBuilder` and
//! cannot carry the middleware stack — so a stream that fails to open is
//! retried by the caller (`Core::stream_with_retry`) on any
//! [`ClientError::is_retryable`] error. Once chunks start arriving, neither
//! layer retries: a partially consumed stream cannot be replayed.
//!
//! # Security
//!
//! API keys are stored using the `secrecy` crate, which:
//! - Prevents accidental logging or display of sensitive data
//! - Zeros memory on drop to minimize exposure window
//! - Requires explicit `expose_secret()` calls for access

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::Stream;
use reqwest_middleware::ClientWithMiddleware;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{Instrument as _, error, warn};

use neuromance_common::chat::Message;
use neuromance_common::client::{
    ChatChunk, ChatRequest, ChatResponse, Config, FinishReason, ProxyConfig, Usage,
};
use neuromance_common::tools::{FunctionCall, ToolCall};

use crate::chat_completions::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionsMessage,
};
use crate::error::ClientError;
use crate::message::MessageBuilder;
use crate::streaming::{StreamingProvider, run_sse_stream};
use crate::telemetry::GenAiOp;
use crate::transport::{add_proxy_headers, inject_trace_context, send_json};
use crate::{LLMClient, build_client_resources};

/// Type-state marker types for compile-time validation.
///
/// These types are used to enforce correct message construction at compile time
/// using the type-state pattern.
mod builder_states {
    /// Initial builder state before a role is set.
    pub struct NoRole;
    /// Builder state after a role has been set.
    pub struct HasRole;
}

/// Builder for constructing Chat Completions messages with compile-time validation.
///
/// Uses the type-state pattern to ensure messages are built correctly:
/// - Messages must have a role set before being built
/// - Invalid state transitions are prevented at compile time
///
/// # Examples
///
/// ```
/// use neuromance_client::chat_completions::ChatCompletionsMessage;
/// use neuromance_common::chat::MessageRole;
///
/// // Valid: role is set
/// let message = ChatCompletionsMessage::builder()
///     .role(MessageRole::User)
///     .content(Some("Hello".to_string()))
///     .build();
/// ```
pub struct ChatCompletionsMessageBuilder<State> {
    _state: PhantomData<State>,
    role: Option<neuromance_common::chat::MessageRole>,
    content: Option<String>,
    name: Option<String>,
    tool_calls: Option<SmallVec<[crate::chat_completions::ChatCompletionsToolCall; 2]>>,
    tool_call_id: Option<String>,
}

impl ChatCompletionsMessageBuilder<builder_states::NoRole> {
    /// Creates a new message builder in the initial state.
    ///
    /// The builder starts in `NoRole` state and requires calling `role()`
    /// before `build()` can be called.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _state: PhantomData,
            role: None,
            content: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Sets the message role and transitions to `HasRole` state.
    ///
    /// This is the only transition from `NoRole` to `HasRole` state,
    /// enforcing that every message must have a role.
    ///
    /// # Arguments
    ///
    /// * `role` - The message role (User, Assistant, System, or Tool)
    #[must_use]
    pub fn role(
        self,
        role: neuromance_common::chat::MessageRole,
    ) -> ChatCompletionsMessageBuilder<builder_states::HasRole> {
        ChatCompletionsMessageBuilder {
            _state: PhantomData,
            role: Some(role),
            content: self.content,
            name: self.name,
            tool_calls: self.tool_calls,
            tool_call_id: self.tool_call_id,
        }
    }
}

impl ChatCompletionsMessageBuilder<builder_states::HasRole> {
    /// Sets the message content.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the message
    #[must_use]
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Sets the message name (optional author identifier).
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the message author
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the tool calls for this message.
    ///
    /// Used when the assistant wants to call functions/tools.
    ///
    /// # Arguments
    ///
    /// * `tool_calls` - Vector of tool calls to execute
    #[must_use]
    pub fn tool_calls(
        mut self,
        tool_calls: SmallVec<[crate::chat_completions::ChatCompletionsToolCall; 2]>,
    ) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Sets the tool call ID for tool response messages.
    ///
    /// Used when this message is a response to a tool call.
    ///
    /// # Arguments
    ///
    /// * `tool_call_id` - The ID of the tool call this message responds to
    #[must_use]
    pub fn tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    /// Builds the Chat Completions message.
    ///
    /// Only available in `HasRole` state, ensuring the role is always set.
    ///
    /// # Panics
    ///
    /// Panics if the role is not set (should not happen in `HasRole` state).
    #[must_use]
    pub fn build(self) -> ChatCompletionsMessage {
        ChatCompletionsMessage {
            role: self
                .role
                .unwrap_or_else(|| unreachable!("Role must be set in HasRole state")),
            content: self.content,
            name: self.name,
            tool_calls: self.tool_calls,
            tool_call_id: self.tool_call_id,
            reasoning_content: None,
            refusal: None,
        }
    }
}

impl Default for ChatCompletionsMessageBuilder<builder_states::NoRole> {
    fn default() -> Self {
        Self::new()
    }
}

/// Client for Chat Completions APIs.
///
/// Supports chat completions with tool/function calling for any API
/// that implements the Chat Completions specification.
///
/// # Security
///
/// The API key is stored using the `secrecy` crate to prevent accidental
/// exposure through debug logs or memory dumps. `SecretString` automatically
/// zeroes memory when dropped via zeroize.
///
/// # Proxy Support
///
/// When a [`ProxyConfig`] is provided in the [`Config`], requests are routed
/// through a tokenizer proxy. The proxy intercepts requests and injects real
/// credentials, allowing agents to use sealed tokens instead of raw API keys.
#[derive(Clone)]
pub struct ChatCompletionsClient {
    client: ClientWithMiddleware,
    streaming_client: reqwest::Client,
    api_key: Arc<SecretString>,
    base_url: String,
    config: Arc<Config>,
    proxy_config: Option<ProxyConfig>,
}

// Custom Debug implementation to avoid exposing API key
impl std::fmt::Debug for ChatCompletionsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatCompletionsClient")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("config", &self.config)
            .field("proxy_config", &self.proxy_config)
            .finish_non_exhaustive()
    }
}

/// Convert a Chat Completions streaming chunk to our common `ChatChunk` format.
///
/// Handles delta updates for content, role, and tool calls.
pub fn convert_chunk_to_chat_chunk(chunk: &ChatCompletionChunk) -> ChatChunk {
    let choice = chunk.choices.first();

    // A refusal streams in place of content. Surfacing it as content keeps the reason visible;
    // `ChatCompletionsClient::process_event` separately marks the stream as filtered.
    let delta_content =
        choice.and_then(|c| c.delta.content.clone().or_else(|| c.delta.refusal.clone()));
    let delta_reasoning_content = choice.and_then(|c| c.delta.reasoning_content.clone());
    let delta_role = choice.and_then(|c| c.delta.role);
    let finish_reason = choice
        .and_then(|c| c.finish_reason.as_ref())
        .and_then(|reason| reason.parse().ok());

    // Convert tool call deltas if present.
    //
    // OpenAI streaming sends `id`, `type`, and `function.name` only in the
    // first chunk for a given tool call; subsequent chunks carry only `index`
    // and an `arguments` fragment. We must propagate every chunk (including
    // those without an `id`) and tag each with `delta.index` so that
    // `ToolCall::merge_deltas` can stitch fragments by slot.
    let delta_tool_calls =
        choice
            .and_then(|c| c.delta.tool_calls.as_ref())
            .map(|tool_call_deltas| {
                let mut result = Vec::with_capacity(tool_call_deltas.len());

                for delta in tool_call_deltas {
                    let id = delta.id.clone().unwrap_or_default();
                    let call_type = delta
                        .r#type
                        .clone()
                        .unwrap_or_else(|| "function".to_string());
                    let (name, arguments) = delta.function.as_ref().map_or_else(
                        || (String::new(), String::new()),
                        |f| {
                            (
                                f.name.clone().unwrap_or_default(),
                                f.arguments.clone().unwrap_or_default(),
                            )
                        },
                    );

                    result.push(ToolCall {
                        id,
                        call_type,
                        function: FunctionCall { name, arguments },
                        index: Some(delta.index),
                    });
                }

                result
            });

    ChatChunk {
        model: chunk.model.clone(),
        delta_content,
        delta_reasoning_content,
        delta_role,
        delta_tool_calls,
        finish_reason,
        usage: chunk.usage.clone().map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cost: None,
            input_tokens_details: u.input_tokens_details,
            output_tokens_details: u.output_tokens_details,
        }),
        response_id: Some(chunk.id.clone()),
        created_at: DateTime::from_timestamp(i64::try_from(chunk.created).unwrap_or(0), 0)
            .unwrap_or_else(Utc::now),
        metadata: HashMap::new(),
    }
}

impl ChatCompletionsClient {
    /// Create a new Chat Completions client from a configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Client configuration including API key and base URL
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use neuromance_client::ChatCompletionsClient;
    /// use neuromance_client::ClientError;
    /// use neuromance_common::client::Config;
    ///
    /// let config = Config::new("openai", "gpt-4")
    ///     .with_api_key("sk-...")
    ///     .with_base_url("https://api.openai.com/v1");
    ///
    /// let client = ChatCompletionsClient::new(config)?;
    /// # Ok::<(), ClientError>(())
    /// ```
    ///
    /// # Proxy Configuration
    ///
    /// When a [`ProxyConfig`] is provided, requests are routed through the proxy.
    /// The `api_key` should contain a sealed token instead of the real API key.
    ///
    /// ```no_run
    /// use neuromance_client::ChatCompletionsClient;
    /// use neuromance_client::ClientError;
    /// use neuromance_common::client::{Config, ProxyConfig};
    ///
    /// let config = Config::new("openai", "gpt-4")
    ///     .with_api_key("sealed.abc123xyz...")  // Sealed token
    ///     .with_proxy(ProxyConfig {
    ///         proxy_url: "http://tokenizer.internal:8080".to_string(),
    ///         token_header: "X-Tokenizer-Token".to_string(),
    ///     });
    ///
    /// let client = ChatCompletionsClient::new(config)?;
    /// # Ok::<(), ClientError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is missing or HTTP client creation fails.
    pub fn new(config: Config) -> Result<Self, ClientError> {
        let r = build_client_resources(config, "https://api.openai.com/v1")?;

        Ok(Self {
            client: r.client,
            streaming_client: r.streaming_client,
            api_key: r.api_key,
            base_url: r.base_url,
            config: r.config,
            proxy_config: r.proxy_config,
        })
    }

    /// Set the model to use for chat completions.
    ///
    /// # Arguments
    ///
    /// * `model` - The model name (e.g., "gpt-4", "gpt-3.5-turbo")
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.config).model = model.into();
        self
    }

    async fn make_request<T: for<'de> Deserialize<'de>, B: Serialize + Sync>(
        &self,
        endpoint: &str,
        body: &B,
    ) -> Result<T, ClientError> {
        let url = format!("{}/{}", self.base_url, endpoint);

        // Validate URL construction
        reqwest::Url::parse(&url)
            .map_err(|e| ClientError::ConfigurationError(format!("Invalid URL '{url}': {e}")))?;

        let mut request_builder = self
            .client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json");

        // Add proxy headers if configured
        request_builder = inject_trace_context(add_proxy_headers(
            request_builder,
            self.proxy_config.as_ref(),
            &self.api_key,
        ));

        let request_builder = request_builder
            .body(serde_json::to_string(body).map_err(ClientError::SerializationError)?);

        send_json(request_builder).await
    }

    /// Convert a Chat Completions message to our internal message format.
    ///
    /// # Note on Tool Arguments
    ///
    /// This method does not validate the JSON structure of tool call arguments.
    /// The arguments come directly from the API response and are passed through as-is.
    /// Users should validate and parse these arguments when executing tools:
    ///
    /// ```rust,ignore
    /// use anyhow::{Context, Result};
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct ToolArgs {
    ///     // Your tool-specific fields
    /// }
    ///
    /// fn parse_tool_args(arguments: &str) -> Result<ToolArgs> {
    ///     serde_json::from_str(arguments)
    ///         .context("Failed to parse tool arguments")
    /// }
    /// ```
    fn convert_message(msg: &ChatCompletionsMessage, conversation_id: uuid::Uuid) -> Message {
        let mut builder = MessageBuilder::new(conversation_id, msg.role);
        if let Some(content) = msg.content.as_deref() {
            builder.set_content(content.to_string());
        }
        // A refusal arrives instead of content, never alongside it. Carrying it as content keeps
        // the reason visible to callers, which an otherwise empty message would lose.
        if let Some(refusal) = msg.refusal.as_deref() {
            builder.append_content(refusal);
        }
        if let Some(tcs) = msg.tool_calls.as_ref() {
            for tc in tcs {
                builder.push_tool_call(ToolCall {
                    id: tc.id.to_string(),
                    call_type: tc.r#type.to_string(),
                    function: FunctionCall {
                        name: tc.function.name.to_string(),
                        arguments: tc.function.arguments.to_string(),
                    },
                    index: None,
                });
            }
        }
        if let Some(id) = msg.tool_call_id.as_ref() {
            builder.set_tool_call_id(id.clone());
        }
        if let Some(name) = msg.name.as_ref() {
            builder.set_name(name.clone());
        }
        if let Some(reasoning) = msg.reasoning_content.as_ref() {
            builder.append_reasoning(reasoning, "\n");
        }
        builder.build()
    }
}

#[async_trait]
impl LLMClient for ChatCompletionsClient {
    fn config(&self) -> &Config {
        &self.config
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_structured_output(&self) -> bool {
        true
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
        // Validation failures never reach the provider, so they are not a
        // GenAI operation and get no span.
        self.validate_request(request)?;

        let op = GenAiOp::chat(&self.config, request);
        match self.send_chat(request).instrument(op.span().clone()).await {
            Ok(response) => {
                op.finish_response(&response);
                Ok(response)
            }
            Err(error) => {
                op.finish_error(&error);
                Err(error)
            }
        }
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
    {
        self.validate_request(request)?;

        // The span opens before the request is built so `inject_trace_context`
        // sees it, and closes only when the returned stream ends.
        let op = GenAiOp::chat(&self.config, request);
        let stream = match op.in_scope(|| {
            let mut chat_request = ChatCompletionRequest::from((request, self.config.as_ref()));
            chat_request.stream = Some(true);
            chat_request.stream_options = Some(serde_json::json!({
                "include_usage": true
            }));

            let url = format!("{}/{}", self.base_url, "chat/completions");
            reqwest::Url::parse(&url).map_err(|e| {
                ClientError::ConfigurationError(format!("Invalid URL '{url}': {e}"))
            })?;

            let mut request_builder = self
                .streaming_client
                .post(&url)
                .header(
                    "Authorization",
                    format!("Bearer {}", self.api_key.expose_secret()),
                )
                .header("Content-Type", "application/json");

            request_builder = inject_trace_context(add_proxy_headers(
                request_builder,
                self.proxy_config.as_ref(),
                &self.api_key,
            ));

            run_sse_stream(self, request_builder.json(&chat_request))
        }) {
            Ok(stream) => stream,
            Err(error) => {
                op.finish_error(&error);
                return Err(error);
            }
        };

        Ok(Box::pin(op.into_instrumented(stream)))
    }
}

impl ChatCompletionsClient {
    /// Issue the request and map the provider's response.
    ///
    /// Split out of [`LLMClient::chat`] so the whole exchange runs inside the
    /// GenAI span: the request builder injects `traceparent` from whatever
    /// span is current, and it must name this operation, not its caller.
    async fn send_chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
        let mut chat_request = ChatCompletionRequest::from((request, self.config.as_ref()));
        chat_request.stream = Some(false);

        let response: ChatCompletionResponse =
            self.make_request("chat/completions", &chat_request).await?;

        // Validate response has at least one choice
        let choice = response.choices.first().ok_or_else(|| {
            warn!(
                "Received empty choices array from API. Response ID: {}, Model: {}",
                response.id, response.model
            );
            ClientError::InvalidResponse("API returned no choices in response".to_string())
        })?;

        // Get conversation_id from first message (validated earlier, but handle defensively)
        let conversation_id = request
            .messages
            .first()
            .ok_or_else(|| {
                error!("Request has no messages despite passing validation");
                ClientError::InvalidRequest("Request must contain at least one message".to_string())
            })?
            .conversation_id;

        let message = Self::convert_message(&choice.message, conversation_id);

        // A refusal outranks the reported reason. The API sends `finish_reason: "stop"` when the
        // model declines, so without this the refusal prose reaches callers as a real answer.
        let finish_reason = if choice.message.refusal.is_some() {
            Some(FinishReason::ContentFilter)
        } else {
            choice
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.parse().ok())
        };

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cost: None,
            input_tokens_details: u.input_tokens_details,
            output_tokens_details: u.output_tokens_details,
        });

        Ok(ChatResponse {
            message,
            model: response.model,
            usage,
            finish_reason,
            created_at: DateTime::from_timestamp(i64::try_from(response.created).unwrap_or(0), 0)
                .unwrap_or_else(Utc::now),
            response_id: Some(response.id),
            metadata: HashMap::new(),
        })
    }
}

/// Streaming state for [`ChatCompletionsClient`].
///
/// Refusal deltas and the terminating `finish_reason` arrive in different chunks, so detecting a
/// refusal needs memory across the stream.
#[derive(Debug, Default)]
#[doc(hidden)]
pub struct ChatCompletionsStreamState {
    saw_refusal: bool,
}

impl StreamingProvider for ChatCompletionsClient {
    type Event = ChatCompletionChunk;
    type State = ChatCompletionsStreamState;

    fn initial_state(&self) -> Self::State {
        ChatCompletionsStreamState::default()
    }

    fn is_stream_end(data: &str) -> bool {
        data == "[DONE]"
    }

    fn process_event(
        state: &mut Self::State,
        event: Self::Event,
    ) -> Option<Result<ChatChunk, ClientError>> {
        if event
            .choices
            .first()
            .is_some_and(|choice| choice.delta.refusal.is_some())
        {
            state.saw_refusal = true;
        }

        let mut chunk = convert_chunk_to_chat_chunk(&event);

        // The API reports `finish_reason: "stop"` even when the model refused. Rewriting the
        // terminal chunk is what lets callers tell a refusal from a real answer.
        if state.saw_refusal && chunk.finish_reason.is_some() {
            chunk.finish_reason = Some(FinishReason::ContentFilter);
        }

        Some(Ok(chunk))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::time::Duration;

    use super::*;
    use futures::StreamExt;
    use neuromance_common::chat::{Message, MessageRole};
    use neuromance_common::client::FinishReason;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn create_test_config(base_url: &str) -> Config {
        Config::new("openai", "gpt-4")
            .with_api_key("test-key")
            .with_base_url(base_url)
            .with_retry_config(crate::fast_retry_config())
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
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1_677_652_288,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.message.content, "Hello! How can I help you today?");
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[tokio::test]
    async fn test_chat_completion_with_different_finish_reasons() {
        let test_cases = vec![
            ("stop", FinishReason::Stop),
            ("length", FinishReason::Length),
            ("tool_calls", FinishReason::ToolCalls),
            ("content_filter", FinishReason::ContentFilter),
        ];

        for (reason_str, expected_reason) in test_cases {
            let mock_server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1_677_652_288,
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Test response"
                        },
                        "finish_reason": reason_str
                    }]
                })))
                .mount(&mock_server)
                .await;

            let config = create_test_config(&mock_server.uri());
            let client = ChatCompletionsClient::new(config).unwrap();

            let message = create_test_message();
            let request = ChatRequest::new(vec![message]);

            let response = client.chat(&request).await.unwrap();
            assert_eq!(response.finish_reason, Some(expected_reason));
        }
    }

    #[tokio::test]
    async fn test_unknown_finish_reason() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1_677_652_288,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response"
                    },
                    "finish_reason": "unknown_reason"
                }]
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();
        // Unknown finish reasons should parse as None (using and_then with parse().ok())
        assert_eq!(response.finish_reason, None);
    }

    #[tokio::test]
    async fn test_authentication_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
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
        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let result = client.chat(&request).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid API key"));
    }

    #[tokio::test]
    async fn test_rate_limit_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let result = client.chat(&request).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Rate limit"));
    }

    #[tokio::test]
    async fn test_model_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "error": {
                    "message": "Internal server error",
                    "type": "server_error"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let result = client.chat(&request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_empty_choices_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1_677_652_288,
                "model": "gpt-4",
                "choices": []
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let result = client.chat(&request).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("no choices"));
    }

    #[tokio::test]
    async fn test_chat_completion_with_tool_calls() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1_677_652_288,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\":\"San Francisco\",\"unit\":\"celsius\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.message.tool_calls.len(), 1);

        let tool_call = &response.message.tool_calls[0];
        assert_eq!(tool_call.id, "call_abc123");
        assert_eq!(tool_call.call_type, "function");
        assert_eq!(tool_call.function.name, "get_weather");
        assert_eq!(
            tool_call.function.arguments,
            "{\"location\":\"San Francisco\",\"unit\":\"celsius\"}"
        );
    }

    #[tokio::test]
    async fn test_chat_completion_with_usage_details() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1_677_652_288,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "input_tokens_details": {
                        "cached_tokens": 5
                    },
                    "output_tokens_details": {
                        "reasoning_tokens": 3
                    }
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);

        let input_details = usage.input_tokens_details.unwrap();
        assert_eq!(input_details.cached_tokens, 5);

        let output_details = usage.output_tokens_details.unwrap();
        assert_eq!(output_details.reasoning_tokens, 3);
    }

    // ==================== Settings Plumbing Tests ====================

    #[tokio::test]
    async fn test_transient_failure_is_retried_until_it_succeeds() {
        let mock_server = MockServer::start().await;

        // Two 503s then a 200. `up_to_n_times` makes the failing mock stop
        // matching after two hits, so the success mock takes over.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .expect(2)
            .mount(&mock_server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(create_successful_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();
        let request = ChatRequest::new(vec![create_test_message()]);

        let response = client.chat(&request).await.unwrap();
        assert_eq!(response.message.content, "Response via proxy");

        // Assert the attempt count directly: without it, a client that never
        // retried but happened to hit the success mock would also pass.
        let attempts = mock_server.received_requests().await.unwrap().len();
        assert_eq!(attempts, 3, "expected 2 failures plus 1 success");
    }

    #[tokio::test]
    async fn test_retries_stop_at_max_retries() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let max_retries = config.retry_config.max_retries as usize;
        let client = ChatCompletionsClient::new(config).unwrap();
        let request = ChatRequest::new(vec![create_test_message()]);

        let error = client.chat(&request).await.unwrap_err();
        assert!(error.is_retryable(), "a 503 should stay retryable: {error}");

        let attempts = mock_server.received_requests().await.unwrap().len();
        assert_eq!(attempts, max_retries + 1, "one try plus max_retries");
    }

    #[tokio::test]
    async fn test_timeout_seconds_is_enforced() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(create_successful_response())
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&mock_server)
            .await;

        // `timeout_seconds` has one-second granularity and a timeout is
        // transient, so retries would multiply the test's wall clock by the
        // attempt count. Retry behaviour is covered separately.
        let mut config = create_test_config(&mock_server.uri()).with_timeout(1);
        config.retry_config.max_retries = 0;
        let client = ChatCompletionsClient::new(config).unwrap();
        let request = ChatRequest::new(vec![create_test_message()]);

        let error = client.chat(&request).await.unwrap_err();
        assert!(
            matches!(error, ClientError::TimeoutError),
            "expected TimeoutError, got {error:?}"
        );
        assert!(error.is_retryable());
    }

    #[test]
    fn test_missing_api_key_is_a_configuration_error() {
        let config = Config::new("openai", "gpt-4").with_base_url("http://localhost:1");

        let error = ChatCompletionsClient::new(config).unwrap_err();
        assert!(
            matches!(&error, ClientError::ConfigurationError(msg) if msg.contains("API key")),
            "expected a configuration error naming the API key, got {error:?}"
        );
    }

    // ==================== Proxy Header Tests ====================

    fn create_successful_response() -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-proxy",
            "object": "chat.completion",
            "created": 1_677_652_288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response via proxy"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
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
            .and(path("/v1/chat/completions"))
            .and(header("X-Tokenizer-Token", "sealed.test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(create_successful_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = Config::new("openai", "gpt-4")
            .with_api_key("sealed.test-token")
            .with_proxy(ProxyConfig {
                proxy_url: mock_server.uri(),
                token_header: "X-Tokenizer-Token".to_string(),
            });

        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();
        assert_eq!(response.message.content, "Response via proxy");
    }

    #[tokio::test]
    async fn test_proxy_with_custom_token_header() {
        let mock_server = MockServer::start().await;

        // Use a custom token header name; the upstream host is in the URL,
        // not a side-band header, so no target-host header is expected.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("X-Custom-Token", "sealed.custom-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(create_successful_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = Config::new("openai", "gpt-4")
            .with_api_key("sealed.custom-token")
            .with_proxy(ProxyConfig {
                proxy_url: mock_server.uri(),
                token_header: "X-Custom-Token".to_string(),
            });

        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();
        assert_eq!(response.message.content, "Response via proxy");
    }

    #[tokio::test]
    async fn test_proxy_headers_sent_streaming() {
        let mock_server = MockServer::start().await;

        // Minimal SSE event sequence for Chat Completions streaming
        let sse_body = [
            &format!(
                "data: {}",
                serde_json::json!({
                    "id": "chatcmpl-proxy-stream",
                    "object": "chat.completion.chunk",
                    "created": 1_677_652_288,
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant", "content": "Streamed via proxy" },
                        "finish_reason": null
                    }]
                })
            ),
            "",
            &format!(
                "data: {}",
                serde_json::json!({
                    "id": "chatcmpl-proxy-stream",
                    "object": "chat.completion.chunk",
                    "created": 1_677_652_288,
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                })
            ),
            "",
            "data: [DONE]",
            "",
        ]
        .join("\n");

        // Verify proxy headers are sent on the streaming path
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("X-Tokenizer-Token", "sealed.test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = Config::new("openai", "gpt-4")
            .with_api_key("sealed.test-token")
            .with_proxy(ProxyConfig {
                proxy_url: mock_server.uri(),
                token_header: "X-Tokenizer-Token".to_string(),
            });

        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

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
            .and(path("/v1/chat/completions"))
            .and(header("X-Tokenizer-Token", "sealed.custom-base"))
            .respond_with(ResponseTemplate::new(200).set_body_json(create_successful_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = Config::new("openai", "gpt-4")
            .with_api_key("sealed.custom-base")
            .with_base_url("https://custom.api.example.com/v1")
            .with_proxy(ProxyConfig {
                proxy_url: mock_server.uri(),
                token_header: "X-Tokenizer-Token".to_string(),
            });

        let client = ChatCompletionsClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();
        assert_eq!(response.message.content, "Response via proxy");
    }

    /// Reproduces the canonical `OpenAI` streaming shape where `id`, `type`, and
    /// `function.name` are sent only in the first chunk for a given tool call,
    /// and subsequent chunks carry just `index` plus a fragment of `arguments`.
    /// Before this regression test, the converter dropped every non-first chunk
    /// because it gated emission on `delta.id.is_some()`, leaving the
    /// downstream tool dispatcher with empty arguments.
    #[test]
    fn test_convert_chunk_stitches_streamed_tool_call_arguments() {
        // Frame 1: id + name + opening of args.
        // Frames 2-4: arguments-only deltas (no id, no name) — the case that
        // used to be silently dropped.
        let chunks_json = [
            r#"{
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "qwen3",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{\""}
                        }]
                    },
                    "finish_reason": null
                }]
            }"#,
            r#"{
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "qwen3",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "command\": "}
                        }]
                    },
                    "finish_reason": null
                }]
            }"#,
            r#"{
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "qwen3",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "\"ls /\""}
                        }]
                    },
                    "finish_reason": null
                }]
            }"#,
            r#"{
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "qwen3",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }"#,
        ];

        let mut tool_calls: Vec<neuromance_common::tools::ToolCall> = Vec::new();
        for raw in chunks_json {
            let chunk: ChatCompletionChunk =
                serde_json::from_str(raw).expect("test chunk should deserialize");
            let chat_chunk = convert_chunk_to_chat_chunk(&chunk);
            if let Some(deltas) = chat_chunk.delta_tool_calls.as_deref() {
                tool_calls = neuromance_common::tools::ToolCall::merge_deltas(tool_calls, deltas);
            }
        }

        assert_eq!(tool_calls.len(), 1, "should stitch into a single tool call");
        let merged = &tool_calls[0];
        assert_eq!(merged.id, "call_abc");
        assert_eq!(merged.function.name, "bash");
        assert_eq!(merged.function.arguments, r#"{"command": "ls /"}"#);
        // Round-trip the merged arguments to ensure they form valid JSON object
        // ready for the tool dispatcher (which calls `as_object()`).
        let parsed: serde_json::Value =
            serde_json::from_str(&merged.function.arguments).expect("merged args must be JSON");
        assert!(parsed.is_object());
        assert_eq!(parsed["command"], "ls /");
    }

    /// Parallel tool calls arrive interleaved across chunks with distinct
    /// `index` values; only the first chunk per index carries `id`/`name`.
    #[test]
    fn test_convert_chunk_stitches_parallel_streamed_tool_calls() {
        let chunks_json = [
            // First chunk opens both slots in one frame.
            r#"{
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "qwen3",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_a", "type": "function",
                             "function": {"name": "read", "arguments": "{\"p"}},
                            {"index": 1, "id": "call_b", "type": "function",
                             "function": {"name": "bash", "arguments": "{\"c"}}
                        ]
                    },
                    "finish_reason": null
                }]
            }"#,
            // Subsequent frames extend each slot independently.
            r#"{
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "qwen3",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": "ath\": \"/etc\"}"}}
                        ]
                    },
                    "finish_reason": null
                }]
            }"#,
            r#"{
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "qwen3",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 1, "function": {"arguments": "ommand\": \"ls\"}"}}
                        ]
                    },
                    "finish_reason": "tool_calls"
                }]
            }"#,
        ];

        let mut tool_calls: Vec<neuromance_common::tools::ToolCall> = Vec::new();
        for raw in chunks_json {
            let chunk: ChatCompletionChunk =
                serde_json::from_str(raw).expect("test chunk should deserialize");
            let chat_chunk = convert_chunk_to_chat_chunk(&chunk);
            if let Some(deltas) = chat_chunk.delta_tool_calls.as_deref() {
                tool_calls = neuromance_common::tools::ToolCall::merge_deltas(tool_calls, deltas);
            }
        }

        assert_eq!(tool_calls.len(), 2);
        let by_id: std::collections::HashMap<_, _> =
            tool_calls.iter().map(|tc| (tc.id.as_str(), tc)).collect();
        assert_eq!(
            by_id["call_a"].function.arguments, r#"{"path": "/etc"}"#,
            "slot 0 should accumulate independently of slot 1",
        );
        assert_eq!(
            by_id["call_b"].function.arguments, r#"{"command": "ls"}"#,
            "slot 1 should accumulate independently of slot 0",
        );
        assert_eq!(by_id["call_a"].function.name, "read");
        assert_eq!(by_id["call_b"].function.name, "bash");
    }

    /// The API reports `finish_reason: "stop"` on a refusal, so the reported reason alone would
    /// pass a refusal off as a real answer.
    #[tokio::test]
    async fn test_refusal_becomes_content_filter() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-refusal",
                "object": "chat.completion",
                "created": 1_677_652_288,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "refusal": "I can't help with that"
                    },
                    "finish_reason": "stop"
                }]
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ChatCompletionsClient::new(config).unwrap();
        let request = ChatRequest::new(vec![create_test_message()]);

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.finish_reason, Some(FinishReason::ContentFilter));
        assert_eq!(
            response.message.content, "I can't help with that",
            "the refusal text must survive: an empty message would trip Core's empty-turn retry"
        );
    }

    fn refusal_chunk(refusal: Option<&str>, finish_reason: Option<&str>) -> ChatCompletionChunk {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1_677_652_288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"refusal": refusal},
                "finish_reason": finish_reason
            }]
        }))
        .unwrap()
    }

    /// The refusal delta and the terminating `finish_reason` arrive in separate chunks, so the
    /// stream state has to remember the refusal to rewrite the last one.
    #[test]
    fn test_streamed_refusal_rewrites_the_terminal_finish_reason() {
        let mut state = ChatCompletionsStreamState::default();

        let first = ChatCompletionsClient::process_event(
            &mut state,
            refusal_chunk(Some("I can't help with that"), None),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            first.delta_content.as_deref(),
            Some("I can't help with that")
        );
        assert_eq!(first.finish_reason, None);

        let last =
            ChatCompletionsClient::process_event(&mut state, refusal_chunk(None, Some("stop")))
                .unwrap()
                .unwrap();
        assert_eq!(last.finish_reason, Some(FinishReason::ContentFilter));
    }

    /// A stream with no refusal keeps the reported reason.
    #[test]
    fn test_streamed_answer_keeps_its_finish_reason() {
        let mut state = ChatCompletionsStreamState::default();

        let last =
            ChatCompletionsClient::process_event(&mut state, refusal_chunk(None, Some("stop")))
                .unwrap()
                .unwrap();

        assert_eq!(last.finish_reason, Some(FinishReason::Stop));
    }

    fn structured_output_schema() -> neuromance_common::client::OutputSchema {
        neuromance_common::client::OutputSchema::new(
            "answer",
            serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": false
            }),
        )
        .unwrap()
    }

    fn structured_output_tool() -> neuromance_common::tools::Tool {
        neuromance_common::tools::Tool::builder()
            .function(neuromance_common::tools::Function {
                name: "lookup".to_string(),
                description: "look something up".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            })
            .build()
    }

    #[test]
    fn test_output_schema_serializes_as_response_format() {
        let request = ChatRequest::new(vec![create_test_message()])
            .with_output_schema(structured_output_schema());
        let wire = crate::chat_completions::ChatCompletionRequest::from((
            &request,
            &create_test_config("http://localhost"),
        ));

        let value = serde_json::to_value(&wire).unwrap();
        assert_eq!(value["response_format"]["type"], "json_schema");
        assert_eq!(value["response_format"]["json_schema"]["name"], "answer");
        assert_eq!(value["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            value["response_format"]["json_schema"]["schema"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn test_output_schema_and_tools_serialize_together() {
        let request = ChatRequest::new(vec![create_test_message()])
            .with_tools(vec![structured_output_tool()])
            .with_output_schema(structured_output_schema());
        let wire = crate::chat_completions::ChatCompletionRequest::from((
            &request,
            &create_test_config("http://localhost"),
        ));

        let value = serde_json::to_value(&wire).unwrap();
        assert_eq!(value["response_format"]["type"], "json_schema");
        assert_eq!(value["tools"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn test_request_without_output_schema_omits_response_format() {
        let request = ChatRequest::new(vec![create_test_message()]);
        let wire = crate::chat_completions::ChatCompletionRequest::from((
            &request,
            &create_test_config("http://localhost"),
        ));

        let value = serde_json::to_value(&wire).unwrap();
        assert!(value.get("response_format").is_none());
    }
}

#[cfg(test)]
mod fuzz_tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use crate::chat_completions::{ChatCompletionResponse, ChatCompletionsMessage};
    use neuromance_common::chat::MessageRole;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn fuzz_chat_completions_response_parsing(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            // Should not panic on malformed responses
            let _ = serde_json::from_slice::<ChatCompletionResponse>(&data);
        }

        #[test]
        fn fuzz_chat_completions_message_parsing(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            // Should not panic on malformed message data
            let _ = serde_json::from_slice::<ChatCompletionsMessage>(&data);
        }

        #[test]
        fn fuzz_chat_completions_response_with_invalid_fields(
            id_str in ".*",
            model_str in ".*",
            created_val in any::<u64>(),
        ) {
            // Create various malformed response JSON
            let json_variants = vec![
                format!(r#"{{"id":"{}","object":"chat.completion","created":{},"model":"{}","choices":[]}}"#,
                    id_str, created_val, model_str),
                r#"{"choices":[]}"#.to_string(),
                r#"{"id":"test","choices":null}"#.to_string(),
                format!(r#"{{"id":"{}","created":{},"model":"{}"}}"#, id_str, created_val, model_str),
            ];

            for json in json_variants {
                let _ = serde_json::from_str::<ChatCompletionResponse>(&json);
            }
        }

        #[test]
        fn fuzz_chat_completions_message_with_missing_fields(
            role_idx in 0usize..4,
            content in prop::option::of(".*"),
        ) {
            let role_str = match role_idx {
                0 => "user",
                1 => "assistant",
                2 => "system",
                _ => "tool",
            };

            let json = content.map_or_else(|| format!(r#"{{"role":"{role_str}"}}"#), |c| {
                let escaped = c.replace('\\', "\\\\").replace('"', "\\\"");
                format!(r#"{{"role":"{role_str}","content":"{escaped}"}}"#)
            });

            let _ = serde_json::from_str::<ChatCompletionsMessage>(&json);
        }

        #[test]
        fn fuzz_chat_completions_message_with_tool_calls(
            num_tool_calls in 0usize..5,
        ) {
            let mut tool_calls_json = Vec::new();
            for i in 0..num_tool_calls {
                tool_calls_json.push(format!(
                    r#"{{"id":"call_{i}","type":"function","function":{{"name":"func_{i}","arguments":"{{}}"}}}}"#
                ));
            }

            let json = if num_tool_calls > 0 {
                format!(
                    r#"{{"role":"assistant","content":null,"tool_calls":[{}]}}"#,
                    tool_calls_json.join(",")
                )
            } else {
                r#"{"role":"assistant","content":"test"}"#.to_string()
            };

            let result = serde_json::from_str::<ChatCompletionsMessage>(&json);
            if result.is_ok() {
                let msg = result.unwrap();
                assert_eq!(msg.role, MessageRole::Assistant);
            }
        }

        #[test]
        fn fuzz_chat_completion_with_extreme_values(
            created_timestamp in any::<u64>(),
            num_choices in 0usize..10,
        ) {
            let choices: Vec<String> = (0..num_choices)
                .map(|i| format!(
                    r#"{{"index":{i},"message":{{"role":"assistant","content":"Response {i}"}},"finish_reason":"stop"}}"#
                ))
                .collect();

            let json = format!(
                r#"{{"id":"test","object":"chat.completion","created":{},"model":"gpt-4","choices":[{}]}}"#,
                created_timestamp,
                choices.join(",")
            );

            let result = serde_json::from_str::<ChatCompletionResponse>(&json);
            if result.is_ok() {
                let response = result.unwrap();
                assert_eq!(response.choices.len(), num_choices);
                assert_eq!(response.created, created_timestamp);
            }
        }

        #[test]
        fn fuzz_chat_completions_function_arguments(
            func_name in ".*",
            args_json in ".*",
        ) {
            let json = format!(
                r#"{{"name":"{}","arguments":"{}"}}"#,
                func_name.replace('\\', "\\\\").replace('"', "\\\""),
                args_json.replace('\\', "\\\\").replace('"', "\\\"")
            );

            let _ = serde_json::from_str::<crate::chat_completions::ChatCompletionsFunction>(&json);
        }

        #[test]
        fn fuzz_response_with_usage_details(
            prompt_tokens in 0u32..100_000,
            completion_tokens in 0u32..100_000,
        ) {
            let total = prompt_tokens + completion_tokens;
            let json = format!(
                r#"{{
                    "id":"test",
                    "object":"chat.completion",
                    "created":1234567890,
                    "model":"gpt-4",
                    "choices":[{{"index":0,"message":{{"role":"assistant","content":"test"}},"finish_reason":"stop"}}],
                    "usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{completion_tokens},"total_tokens":{total}}}
                }}"#
            );

            let result = serde_json::from_str::<ChatCompletionResponse>(&json);
            if result.is_ok() {
                let response = result.unwrap();
                let usage = response.usage.unwrap();
                assert_eq!(usage.prompt_tokens, prompt_tokens);
                assert_eq!(usage.completion_tokens, completion_tokens);
                assert_eq!(usage.total_tokens, total);
            }
        }
    }
}
