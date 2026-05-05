//! Responses API client implementation.
//!
//! This module provides a client for interacting with the `OpenAI` Responses API.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::Stream;
use log::{debug, error, warn};
use reqwest_middleware::ClientWithMiddleware;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use neuromance_common::chat::MessageRole;
use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config, Usage};
use neuromance_common::tools::{FunctionCall, ToolCall};

use crate::error::{ClientError, ErrorResponse};
use crate::streaming::{StreamingProvider, run_sse_stream};
use crate::{LLMClient, build_client_resources};

use super::{
    OutputItem, ResponsesRequest, ResponsesResponse, StreamEvent, StreamingFunctionCall,
    convert_response_to_message,
};

/// Default base URL for the Responses API.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Client for the `OpenAI` Responses API.
///
/// Supports stateless mode, streaming, and tool calling.
#[derive(Clone)]
pub struct ResponsesClient {
    /// HTTP client with retry middleware for non-streaming requests.
    client: ClientWithMiddleware,
    /// HTTP client without middleware for streaming requests.
    streaming_client: reqwest::Client,
    /// API key for authentication.
    api_key: Arc<SecretString>,
    /// Base URL for API requests.
    base_url: String,
    /// Client configuration.
    config: Arc<Config>,
}

impl std::fmt::Debug for ResponsesClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesClient")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ResponsesClient {
    /// Create a new Responses API client from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Client configuration including API key
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
        })
    }

    /// Set a custom base URL for the API endpoint.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL (e.g., `https://api.openai.com/v1`)
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        Arc::make_mut(&mut self.config).base_url = Some(base_url.clone());
        self.base_url = base_url;
        self
    }

    /// Set the model to use for requests.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.config).model = model.into();
        self
    }

    /// Make a non-streaming request to the Responses API.
    async fn make_request<T: for<'de> Deserialize<'de>>(
        &self,
        body: &ResponsesRequest,
    ) -> Result<T, ClientError> {
        let url = format!("{}/responses", self.base_url);

        // Validate URL construction
        reqwest::Url::parse(&url)
            .map_err(|e| ClientError::ConfigurationError(format!("Invalid URL '{url}': {e}")))?;

        let response = self
            .client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(body).map_err(ClientError::SerializationError)?)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.map_err(|e| {
                warn!("Failed to read error response body: {e}");
                ClientError::NetworkError(e)
            })?;

            // Extract the error message from structured response or use raw text
            let error_message = match serde_json::from_str::<ErrorResponse>(&error_text) {
                Ok(parsed) => {
                    debug!("Parsed structured error response");
                    parsed.error.message
                }
                Err(parse_err) => {
                    debug!(
                        "Failed to parse error response as JSON: {parse_err}. Using raw text instead."
                    );
                    error_text
                }
            };

            error!(
                "API request failed with status {}: {}",
                status.as_u16(),
                error_message
            );

            return Err(match status.as_u16() {
                401 => ClientError::AuthenticationError(error_message),
                429 => ClientError::RateLimitError { retry_after: None },
                _ => ClientError::RequestError(error_message),
            });
        }

        let response_text = response.text().await?;
        debug!("Raw API response: {response_text}");
        let parsed_response: T =
            serde_json::from_str(&response_text).map_err(ClientError::SerializationError)?;

        Ok(parsed_response)
    }
}

#[async_trait]
impl LLMClient for ResponsesClient {
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

        let mut responses_request = ResponsesRequest::from((request, self.config.as_ref()));
        responses_request.stream = Some(false);

        let response: ResponsesResponse = self.make_request(&responses_request).await?;

        // Get conversation_id from first message
        let conversation_id = request
            .messages
            .first()
            .ok_or_else(|| {
                error!("Request has no messages despite passing validation");
                ClientError::InvalidRequest("Request must contain at least one message".to_string())
            })?
            .conversation_id;

        let message = convert_response_to_message(&response, conversation_id);

        let has_tool_calls = !message.tool_calls.is_empty();
        let finish_reason = super::finish_reason_from_status(
            &response.status,
            response.incomplete_details.as_ref(),
            has_tool_calls,
        );

        let usage = response.usage.map(Usage::from);

        Ok(ChatResponse {
            message,
            model: response.model,
            usage,
            finish_reason,
            created_at: DateTime::from_timestamp(response.created_at, 0).unwrap_or_else(Utc::now),
            response_id: Some(response.id),
            metadata: response.metadata,
        })
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
    {
        self.validate_request(request)?;

        let mut responses_request = ResponsesRequest::from((request, self.config.as_ref()));
        responses_request.stream = Some(true);

        let url = format!("{}/responses", self.base_url);
        reqwest::Url::parse(&url)
            .map_err(|e| ClientError::ConfigurationError(format!("Invalid URL '{url}': {e}")))?;

        let request_builder = self
            .streaming_client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .json(&responses_request);

        run_sse_stream(self, request_builder)
    }
}

/// Per-stream accumulator for the Responses streaming protocol.
///
/// `model` and `response_id` are populated by the first `ResponseCreated`
/// event; `streaming_function_calls` accumulates partial tool-use arguments
/// across `OutputItemAdded` / `FunctionCallArgumentsDelta` /
/// `FunctionCallArgumentsDone` / `OutputItemDone`.
#[derive(Default)]
pub struct ResponsesStreamState {
    model: String,
    response_id: String,
    streaming_function_calls: HashMap<u32, StreamingFunctionCall>,
}

impl StreamingProvider for ResponsesClient {
    type Event = StreamEvent;
    type State = ResponsesStreamState;

    fn initial_state(&self) -> Self::State {
        ResponsesStreamState::default()
    }

    fn is_stream_end(data: &str) -> bool {
        data == "[DONE]"
    }

    fn process_event(
        state: &mut Self::State,
        event: Self::Event,
    ) -> Option<Result<ChatChunk, ClientError>> {
        convert_event_to_chunk(event, state)
    }
}

/// Translate one Responses stream event into a `ChatChunk` (or error).
#[allow(clippy::too_many_lines)]
fn convert_event_to_chunk(
    event: StreamEvent,
    state: &mut ResponsesStreamState,
) -> Option<Result<ChatChunk, ClientError>> {
    match event {
        StreamEvent::ResponseCreated { response }
        | StreamEvent::ResponseInProgress { response } => {
            state.model.clone_from(&response.model);
            state.response_id.clone_from(&response.id);

            Some(Ok(ChatChunk {
                model: response.model,
                delta_content: None,
                delta_reasoning_content: None,
                delta_role: Some(MessageRole::Assistant),
                delta_tool_calls: None,
                finish_reason: None,
                usage: None,
                response_id: Some(response.id),
                created_at: DateTime::from_timestamp(response.created_at, 0)
                    .unwrap_or_else(Utc::now),
                metadata: HashMap::new(),
            }))
        }

        StreamEvent::ResponseCompleted { response }
        | StreamEvent::ResponseIncomplete { response } => {
            let has_tool_calls = response
                .output
                .iter()
                .any(|item| matches!(item, OutputItem::FunctionCall { .. }));
            let finish_reason = super::finish_reason_from_status(
                &response.status,
                response.incomplete_details.as_ref(),
                has_tool_calls,
            );
            let usage = response.usage.map(Usage::from);

            Some(Ok(ChatChunk {
                model: response.model,
                delta_content: None,
                delta_reasoning_content: None,
                delta_role: None,
                delta_tool_calls: None,
                finish_reason,
                usage,
                response_id: Some(response.id),
                created_at: DateTime::from_timestamp(response.created_at, 0)
                    .unwrap_or_else(Utc::now),
                metadata: response.metadata,
            }))
        }

        StreamEvent::ResponseFailed { response } => {
            let error_msg = response
                .error
                .map_or_else(|| "Unknown error".to_string(), |e| e.message);
            warn!("Response failed: {error_msg}");
            Some(Err(ClientError::RequestError(error_msg)))
        }

        StreamEvent::OutputTextDelta { delta, .. } => Some(Ok(ChatChunk {
            model: state.model.clone(),
            delta_content: Some(delta),
            delta_reasoning_content: None,
            delta_role: None,
            delta_tool_calls: None,
            finish_reason: None,
            usage: None,
            response_id: Some(state.response_id.clone()),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        })),

        StreamEvent::ReasoningSummaryTextDelta { delta, .. } => Some(Ok(ChatChunk {
            model: state.model.clone(),
            delta_content: None,
            delta_reasoning_content: Some(delta),
            delta_role: None,
            delta_tool_calls: None,
            finish_reason: None,
            usage: None,
            response_id: Some(state.response_id.clone()),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        })),

        StreamEvent::OutputItemAdded { output_index, item } => {
            if let OutputItem::FunctionCall { call_id, name, .. } = item {
                state
                    .streaming_function_calls
                    .insert(output_index, StreamingFunctionCall::new(call_id, name));
            }
            None
        }

        StreamEvent::FunctionCallArgumentsDelta {
            output_index,
            delta,
            ..
        } => {
            // Fallback path for OutputItemDone — primary finalization uses
            // FunctionCallArgumentsDone with full arguments.
            if let Some(fc) = state.streaming_function_calls.get_mut(&output_index) {
                fc.append_delta(&delta);
            }
            None
        }

        StreamEvent::FunctionCallArgumentsDone {
            output_index,
            arguments,
            ..
        } => {
            // call_id (not item_id) is the correct identifier for correlating
            // tool results back in multi-turn conversations.
            let (call_id, function_name) = state
                .streaming_function_calls
                .remove(&output_index)
                .map(|fc| (fc.call_id, fc.name))
                .unwrap_or_default();

            let tool_call = ToolCall {
                id: call_id,
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: function_name,
                    arguments,
                },
            };

            Some(Ok(ChatChunk {
                model: state.model.clone(),
                delta_content: None,
                delta_reasoning_content: None,
                delta_role: None,
                delta_tool_calls: Some(vec![tool_call]),
                finish_reason: None,
                usage: None,
                response_id: Some(state.response_id.clone()),
                created_at: Utc::now(),
                metadata: HashMap::new(),
            }))
        }

        StreamEvent::OutputItemDone { output_index, .. } => {
            // Finalize a function call that wasn't already closed via
            // FunctionCallArgumentsDone.
            let maybe_fc = state.streaming_function_calls.remove(&output_index);

            if let Some(fc) = maybe_fc {
                let tool_call = fc.finalize();
                return Some(Ok(ChatChunk {
                    model: state.model.clone(),
                    delta_content: None,
                    delta_reasoning_content: None,
                    delta_role: None,
                    delta_tool_calls: Some(vec![tool_call]),
                    finish_reason: None,
                    usage: None,
                    response_id: Some(state.response_id.clone()),
                    created_at: Utc::now(),
                    metadata: HashMap::new(),
                }));
            }
            None
        }

        StreamEvent::Error { error } => {
            warn!(
                "Stream error from API: {} - {}",
                error.error_type, error.message
            );
            Some(Err(ClientError::RequestError(error.message)))
        }

        StreamEvent::ContentPartAdded { .. }
        | StreamEvent::ContentPartDone { .. }
        | StreamEvent::OutputTextDone { .. }
        | StreamEvent::ReasoningSummaryTextDone { .. }
        | StreamEvent::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::panic)]

    use super::*;
    use futures::StreamExt;
    use neuromance_common::chat::Message;
    use neuromance_common::client::FinishReason;
    use smallvec::SmallVec;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn create_test_config(base_url: &str) -> Config {
        Config::new("responses", "gpt-4o")
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
        }
    }

    #[tokio::test]
    async fn test_successful_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(header("authorization", "Bearer test-key"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_123",
                "object": "response",
                "created_at": 1_677_652_288,
                "model": "gpt-4o",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "Hello! How can I help you today?"
                    }]
                }],
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ResponsesClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.message.content, "Hello! How can I help you today?");
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[tokio::test]
    async fn test_response_with_function_call() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_123",
                "object": "response",
                "created_at": 1_677_652_288,
                "model": "gpt-4o",
                "output": [{
                    "type": "function_call",
                    "call_id": "call_abc123",
                    "name": "get_weather",
                    "arguments": "{\"location\":\"San Francisco\",\"unit\":\"celsius\"}"
                }],
                "status": "completed",
                "usage": {
                    "input_tokens": 15,
                    "output_tokens": 25,
                    "total_tokens": 40
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ResponsesClient::new(config).unwrap();

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
    async fn test_authentication_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
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
        let client = ResponsesClient::new(config).unwrap();

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
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error"
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ResponsesClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let result = client.chat(&request).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Rate limit"));
    }

    // ========================================================================
    // Streaming unit tests for convert_event_to_chunk
    // ========================================================================

    fn make_state() -> ResponsesStreamState {
        ResponsesStreamState::default()
    }

    fn make_partial_response(id: &str, model: &str) -> super::super::PartialResponse {
        super::super::PartialResponse {
            id: id.to_string(),
            object: "response".to_string(),
            created_at: 1_700_000_000,
            model: model.to_string(),
            status: super::super::ResponseStatus::InProgress,
            output: vec![],
            error: None,
            incomplete_details: None,
            usage: None,
        }
    }

    #[tokio::test]
    async fn test_stream_response_created_sets_model_and_role() {
        let mut state = make_state();

        let event = StreamEvent::ResponseCreated {
            response: make_partial_response("resp_abc", "gpt-4o"),
        };

        let result = convert_event_to_chunk(event, &mut state);
        let chunk = result.unwrap().unwrap();

        assert_eq!(chunk.model, "gpt-4o");
        assert_eq!(chunk.response_id.as_deref(), Some("resp_abc"));
        assert_eq!(chunk.delta_role, Some(MessageRole::Assistant));
        assert!(chunk.delta_content.is_none());

        // Shared state should be updated
        assert_eq!(state.model, "gpt-4o");
        assert_eq!(state.response_id, "resp_abc");
    }

    #[tokio::test]
    async fn test_stream_text_delta() {
        let mut state = make_state();
        state.model = "gpt-4o".to_string();
        state.response_id = "resp_abc".to_string();

        let event = StreamEvent::OutputTextDelta {
            output_index: 0,
            content_index: 0,
            delta: "Hello ".to_string(),
        };

        let result = convert_event_to_chunk(event, &mut state);
        let chunk = result.unwrap().unwrap();

        assert_eq!(chunk.delta_content.as_deref(), Some("Hello "));
        assert_eq!(chunk.model, "gpt-4o");
        assert_eq!(chunk.response_id.as_deref(), Some("resp_abc"));
        assert!(chunk.delta_role.is_none());
    }

    #[tokio::test]
    async fn test_stream_reasoning_delta() {
        let mut state = make_state();
        state.model = "o3".to_string();
        state.response_id = "resp_r".to_string();

        let event = StreamEvent::ReasoningSummaryTextDelta {
            output_index: 0,
            summary_index: 0,
            delta: "Let me think...".to_string(),
        };

        let result = convert_event_to_chunk(event, &mut state);
        let chunk = result.unwrap().unwrap();

        assert_eq!(
            chunk.delta_reasoning_content.as_deref(),
            Some("Let me think...")
        );
        assert!(chunk.delta_content.is_none());
        assert_eq!(chunk.model, "o3");
    }

    #[tokio::test]
    async fn test_stream_response_completed_with_usage() {
        let mut state = make_state();

        let response = super::super::ResponsesResponse {
            id: "resp_done".to_string(),
            object: "response".to_string(),
            created_at: 1_700_000_000,
            model: "gpt-4o".to_string(),
            status: super::super::ResponseStatus::Completed,
            output: vec![super::super::OutputItem::Message {
                role: "assistant".to_string(),
                content: vec![],
            }],
            error: None,
            incomplete_details: None,
            usage: Some(super::super::ResponsesUsage {
                input_tokens: 10,
                output_tokens: 20,
                total_tokens: 30,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            metadata: HashMap::new(),
        };

        let event = StreamEvent::ResponseCompleted { response };

        let result = convert_event_to_chunk(event, &mut state);
        let chunk = result.unwrap().unwrap();

        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[tokio::test]
    async fn test_stream_response_incomplete_with_usage() {
        let mut state = make_state();

        let response = super::super::ResponsesResponse {
            id: "resp_inc".to_string(),
            object: "response".to_string(),
            created_at: 1_700_000_000,
            model: "gpt-4o".to_string(),
            status: super::super::ResponseStatus::Incomplete,
            output: vec![super::super::OutputItem::Message {
                role: "assistant".to_string(),
                content: vec![],
            }],
            error: None,
            incomplete_details: Some(super::super::IncompleteDetails {
                reason: super::super::IncompleteReason::MaxOutputTokens,
            }),
            usage: Some(super::super::ResponsesUsage {
                input_tokens: 100,
                output_tokens: 512,
                total_tokens: 612,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            metadata: HashMap::new(),
        };

        let event = StreamEvent::ResponseIncomplete { response };

        let result = convert_event_to_chunk(event, &mut state);
        let chunk = result.unwrap().unwrap();

        assert_eq!(chunk.finish_reason, Some(FinishReason::Length));
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 512);
        assert_eq!(usage.total_tokens, 612);
    }

    #[tokio::test]
    async fn test_stream_response_failed() {
        let mut state = make_state();

        let mut partial = make_partial_response("resp_fail", "gpt-4o");
        partial.status = super::super::ResponseStatus::Failed;
        partial.error = Some(super::super::ResponseError {
            code: "server_error".to_string(),
            message: "Internal server error".to_string(),
        });

        let event = StreamEvent::ResponseFailed { response: partial };

        let result = convert_event_to_chunk(event, &mut state);
        let err = result.unwrap().unwrap_err();
        assert!(err.to_string().contains("Internal server error"));
    }

    #[tokio::test]
    async fn test_stream_function_call_flow() {
        let mut state = make_state();
        state.model = "gpt-4o".to_string();
        state.response_id = "resp_fc".to_string();

        // 1. OutputItemAdded with function call
        let added_event = StreamEvent::OutputItemAdded {
            output_index: 0,
            item: super::super::OutputItem::FunctionCall {
                call_id: "call_123".to_string(),
                name: "get_weather".to_string(),
                arguments: String::new(),
            },
        };
        let result = convert_event_to_chunk(added_event, &mut state);
        assert!(result.is_none());

        // 2. First arguments delta
        let delta1 = StreamEvent::FunctionCallArgumentsDelta {
            output_index: 0,
            call_id: String::new(),
            delta: r#"{"loc"#.to_string(),
        };
        let result = convert_event_to_chunk(delta1, &mut state);
        assert!(result.is_none());

        // 3. Second arguments delta
        let delta2 = StreamEvent::FunctionCallArgumentsDelta {
            output_index: 0,
            call_id: String::new(),
            delta: r#"ation":"SF"}"#.to_string(),
        };
        let result = convert_event_to_chunk(delta2, &mut state);
        assert!(result.is_none());

        // 4. FunctionCallArgumentsDone
        let done_event = StreamEvent::FunctionCallArgumentsDone {
            output_index: 0,
            item_id: "item_456".to_string(),
            arguments: r#"{"location":"SF"}"#.to_string(),
            sequence_number: 0,
        };
        let result = convert_event_to_chunk(done_event, &mut state);
        let chunk = result.unwrap().unwrap();

        let tool_calls = chunk.delta_tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, r#"{"location":"SF"}"#);
    }

    #[tokio::test]
    async fn test_stream_function_call_finalized_on_output_item_done() {
        let mut state = make_state();
        state.model = "gpt-4o".to_string();
        state.response_id = "resp_fc2".to_string();

        // 1. OutputItemAdded
        let added = StreamEvent::OutputItemAdded {
            output_index: 1,
            item: super::super::OutputItem::FunctionCall {
                call_id: "call_xyz".to_string(),
                name: "do_thing".to_string(),
                arguments: String::new(),
            },
        };
        convert_event_to_chunk(added, &mut state);

        // No deltas — empty args

        // 2. OutputItemDone (without FunctionCallArgumentsDone)
        let done = StreamEvent::OutputItemDone {
            output_index: 1,
            item: super::super::OutputItem::FunctionCall {
                call_id: "call_xyz".to_string(),
                name: "do_thing".to_string(),
                arguments: String::new(),
            },
        };
        let result = convert_event_to_chunk(done, &mut state);
        let chunk = result.unwrap().unwrap();

        let tool_calls = chunk.delta_tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_xyz");
        assert_eq!(tool_calls[0].function.name, "do_thing");
        // Empty args should default to "{}"
        assert_eq!(tool_calls[0].function.arguments, "{}");
    }

    #[tokio::test]
    async fn test_stream_output_item_done_no_pending_function_call() {
        let mut state = make_state();

        // OutputItemDone for a message item (no pending function call)
        let done = StreamEvent::OutputItemDone {
            output_index: 0,
            item: super::super::OutputItem::Message {
                role: "assistant".to_string(),
                content: vec![],
            },
        };
        let result = convert_event_to_chunk(done, &mut state);
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_stream_error_event() {
        let mut state = make_state();

        let event = StreamEvent::Error {
            error: super::super::ApiError {
                error_type: "invalid_request_error".to_string(),
                code: Some("bad_request".to_string()),
                message: "Invalid model specified".to_string(),
            },
        };

        let result = convert_event_to_chunk(event, &mut state);
        let err = result.unwrap().unwrap_err();
        assert!(err.to_string().contains("Invalid model specified"));
    }

    #[tokio::test]
    async fn test_stream_unknown_events_return_none() {
        let mut state = make_state();

        // ContentPartAdded
        let event = StreamEvent::ContentPartAdded {
            output_index: 0,
            content_index: 0,
            part: super::super::OutputContentBlock::OutputText {
                text: String::new(),
            },
        };
        assert!(convert_event_to_chunk(event, &mut state).is_none());

        // ContentPartDone
        let event = StreamEvent::ContentPartDone {
            output_index: 0,
            content_index: 0,
            part: super::super::OutputContentBlock::OutputText {
                text: "done".to_string(),
            },
        };
        assert!(convert_event_to_chunk(event, &mut state).is_none());

        // OutputTextDone
        let event = StreamEvent::OutputTextDone {
            output_index: 0,
            content_index: 0,
            text: "done".to_string(),
        };
        assert!(convert_event_to_chunk(event, &mut state).is_none());

        // ReasoningSummaryTextDone
        let event = StreamEvent::ReasoningSummaryTextDone {
            output_index: 0,
            summary_index: 0,
            text: "done".to_string(),
        };
        assert!(convert_event_to_chunk(event, &mut state).is_none());

        // Unknown
        assert!(convert_event_to_chunk(StreamEvent::Unknown, &mut state).is_none());
    }

    // ========================================================================
    // Integration test: chat_stream via raw TCP SSE server
    // ========================================================================

    #[tokio::test]
    async fn test_chat_stream_text_response() {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn a minimal SSE server
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();

            // Read the HTTP request (consume it so the connection doesn't stall)
            let mut buf = vec![0u8; 4096];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;

            // SSE events to send
            let events = [
                r#"data: {"type":"response.created","response":{"id":"resp_s1","object":"response","created_at":1700000000,"model":"gpt-4o","status":"in_progress","output":[]}}"#,
                r#"data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello"}"#,
                r#"data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":" world"}"#,
                r#"data: {"type":"response.completed","response":{"id":"resp_s1","object":"response","created_at":1700000000,"model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":5,"output_tokens":10,"total_tokens":15}}}"#,
                "data: [DONE]",
            ];

            let mut body = String::new();
            for e in &events {
                body.push_str(e);
                body.push_str("\n\n");
            }

            let response = format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: text/event-stream\r\n\
                 Cache-Control: no-cache\r\n\
                 Connection: keep-alive\r\n\
                 \r\n\
                 {body}"
            );

            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
            // Keep connection alive briefly so the client can read all events
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        });

        let base_url = format!("http://{addr}");
        let config = create_test_config(&base_url);
        let client = ResponsesClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let stream = client.chat_stream(&request).await.unwrap();
        let chunks: Vec<ChatChunk> = stream.filter_map(|r| async { r.ok() }).collect().await;

        // Should have: ResponseCreated chunk, two text deltas, ResponseCompleted chunk
        assert!(
            chunks.len() >= 3,
            "Expected at least 3 chunks, got {}",
            chunks.len()
        );

        // First chunk should set the role
        assert_eq!(chunks[0].delta_role, Some(MessageRole::Assistant));
        assert_eq!(chunks[0].model, "gpt-4o");
        assert_eq!(chunks[0].response_id.as_deref(), Some("resp_s1"));

        // Text delta chunks
        assert_eq!(chunks[1].delta_content.as_deref(), Some("Hello"));
        assert_eq!(chunks[2].delta_content.as_deref(), Some(" world"));

        // Final chunk should have finish_reason and usage
        let last = chunks.last().unwrap();
        assert_eq!(last.finish_reason, Some(FinishReason::Stop));
        let usage = last.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 10);
        assert_eq!(usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn test_response_with_reasoning() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_123",
                "object": "response",
                "created_at": 1_677_652_288,
                "model": "o1-preview",
                "output": [
                    {
                        "type": "reasoning",
                        "content": [{
                            "type": "summary_text",
                            "text": "Let me think about this..."
                        }]
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "Here is my answer."
                        }]
                    }
                ],
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 50,
                    "total_tokens": 60,
                    "output_tokens_details": {
                        "reasoning_tokens": 30
                    }
                }
            })))
            .mount(&mock_server)
            .await;

        let config = create_test_config(&mock_server.uri());
        let client = ResponsesClient::new(config).unwrap();

        let message = create_test_message();
        let request = ChatRequest::new(vec![message]);

        let response = client.chat(&request).await.unwrap();

        assert_eq!(response.message.content, "Here is my answer.");
        assert!(response.message.reasoning.is_some());
        assert_eq!(
            response.message.reasoning.as_ref().unwrap().content,
            "Let me think about this..."
        );

        let usage = response.usage.unwrap();
        assert_eq!(usage.output_tokens_details.unwrap().reasoning_tokens, 30);
    }

    // ========================================================================
    // ChatRequest -> ResponsesRequest conversion tests
    // ========================================================================

    fn make_message(role: MessageRole, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4(),
            conversation_id: uuid::Uuid::new_v4(),
            role,
            content: content.to_string(),
            tool_calls: SmallVec::new(),
            tool_call_id: None,
            name: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            reasoning: None,
        }
    }

    fn default_config() -> Config {
        Config::new("responses", "gpt-4o").with_api_key("test-key")
    }

    #[test]
    fn test_conversion_tool_message_becomes_function_call_output() {
        let user_msg = make_message(MessageRole::User, "Hello");
        let mut tool_msg = make_message(MessageRole::Tool, r#"{"temp": 72}"#);
        tool_msg.tool_call_id = Some("call_abc".to_string());

        let request = ChatRequest::new(vec![user_msg, tool_msg]);
        let config = default_config();
        let responses_req = super::super::ResponsesRequest::from((&request, &config));

        // Should have 2 input items: user message + function call output
        assert_eq!(responses_req.input.len(), 2);

        match &responses_req.input[1] {
            super::super::InputItem::FunctionCallOutput { call_id, output } => {
                assert_eq!(call_id, "call_abc");
                assert_eq!(output, r#"{"temp": 72}"#);
            }
            other => panic!("Expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn test_conversion_tool_message_without_call_id_is_skipped() {
        let user_msg = make_message(MessageRole::User, "Hello");
        let tool_msg = make_message(MessageRole::Tool, "some output");
        // tool_call_id is None

        let request = ChatRequest::new(vec![user_msg, tool_msg]);
        let config = default_config();
        let responses_req = super::super::ResponsesRequest::from((&request, &config));

        // Only the user message should be present; tool message without call_id is skipped
        assert_eq!(responses_req.input.len(), 1);
        assert!(matches!(
            &responses_req.input[0],
            super::super::InputItem::Message {
                role: super::super::ResponsesRole::User,
                ..
            }
        ));
    }

    #[test]
    fn test_conversion_multiple_system_messages_concatenated() {
        let sys1 = make_message(MessageRole::System, "You are helpful.");
        let sys2 = make_message(MessageRole::System, "Be concise.");
        let user_msg = make_message(MessageRole::User, "Hi");

        let request = ChatRequest::new(vec![sys1, sys2, user_msg]);
        let config = default_config();
        let responses_req = super::super::ResponsesRequest::from((&request, &config));

        assert_eq!(
            responses_req.instructions.as_deref(),
            Some("You are helpful.\n\nBe concise.")
        );

        // System messages should not appear as input items
        assert_eq!(responses_req.input.len(), 1);
    }

    #[test]
    fn test_conversion_previous_response_id_from_metadata() {
        let user_msg = make_message(MessageRole::User, "Continue");

        let mut metadata = HashMap::new();
        metadata.insert(
            "previous_response_id".to_string(),
            serde_json::json!("resp_prev_123"),
        );

        let request = ChatRequest::new(vec![user_msg]).with_metadata(metadata);
        let config = default_config();
        let responses_req = super::super::ResponsesRequest::from((&request, &config));

        assert_eq!(
            responses_req.previous_response_id.as_deref(),
            Some("resp_prev_123")
        );
    }

    #[test]
    fn test_conversion_reasoning_level_to_config() {
        use neuromance_common::features::ReasoningLevel;

        let user_msg = make_message(MessageRole::User, "Think hard");

        // High reasoning level
        let request =
            ChatRequest::new(vec![user_msg.clone()]).with_reasoning_level(ReasoningLevel::High);
        let config = default_config();
        let responses_req = super::super::ResponsesRequest::from((&request, &config));

        let reasoning = responses_req.reasoning.unwrap();
        assert_eq!(reasoning.effort, super::super::ReasoningEffort::High);
        assert_eq!(
            reasoning.summary,
            Some(super::super::ReasoningSummary::Concise)
        );

        // Low reasoning level
        let request =
            ChatRequest::new(vec![user_msg.clone()]).with_reasoning_level(ReasoningLevel::Low);
        let responses_req = super::super::ResponsesRequest::from((&request, &config));
        let reasoning = responses_req.reasoning.unwrap();
        assert_eq!(reasoning.effort, super::super::ReasoningEffort::Low);

        // Medium reasoning level
        let request =
            ChatRequest::new(vec![user_msg.clone()]).with_reasoning_level(ReasoningLevel::Medium);
        let responses_req = super::super::ResponsesRequest::from((&request, &config));
        let reasoning = responses_req.reasoning.unwrap();
        assert_eq!(reasoning.effort, super::super::ReasoningEffort::Medium);

        // Default reasoning level -> no reasoning config
        let request =
            ChatRequest::new(vec![user_msg]).with_reasoning_level(ReasoningLevel::Default);
        let responses_req = super::super::ResponsesRequest::from((&request, &config));
        assert!(responses_req.reasoning.is_none());
    }
}
