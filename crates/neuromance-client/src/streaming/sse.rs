//! Generic SSE event-source driver.
//!
//! [`run_sse_stream`] consumes a [`reqwest::RequestBuilder`] and a
//! [`StreamingProvider`], driving the resulting event source to a stream of
//! [`ChatChunk`]s. Provider-specific behaviour (event type, accumulator
//! state, sentinel detection, event-to-chunk translation) is supplied by
//! the trait implementation.

use std::pin::Pin;

use futures::{Stream, StreamExt};
use reqwest_eventsource::{Event, EventSource};
use serde::de::DeserializeOwned;
use tracing::{debug, error, warn};

use neuromance_common::client::ChatChunk;

use crate::NoRetryPolicy;
use crate::error::{ClientError, ErrorResponse};

/// Boxed, pinned, send-able stream of [`ChatChunk`] results — the public
/// shape returned by streaming chat APIs across all providers.
pub type ChatChunkStream = Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>;

/// Provider-specific contract for streaming chat completions over SSE.
///
/// Implementors plug in:
/// - the wire event type ([`Self::Event`])
/// - per-stream accumulator state ([`Self::State`]; use `()` if none)
/// - the seed for that state ([`Self::initial_state`]; can read provider config)
/// - any stream-end sentinel ([`Self::is_stream_end`]; e.g. `OpenAI`'s `[DONE]`)
/// - the event-to-chunk translation ([`Self::process_event`])
///
/// The driver ([`run_sse_stream`]) handles connection setup, retry-policy
/// disablement, parsing, stream termination, and HTTP-status error
/// extraction — all the boilerplate that's identical across providers.
pub trait StreamingProvider {
    /// The provider's wire event type. Each SSE `data:` line is parsed
    /// into a value of this type.
    type Event: DeserializeOwned + Send + 'static;

    /// Per-stream accumulator state. Threaded mutably through every
    /// [`Self::process_event`] call so providers can track values like
    /// the active model id, response id, or in-flight tool calls without
    /// reaching for `Arc<Mutex<…>>`.
    type State: Send + 'static;

    /// Build the initial accumulator state for a stream. Reads provider
    /// config (e.g. configured model id) so the state has something to
    /// emit before the first server-supplied identifier event arrives.
    fn initial_state(&self) -> Self::State;

    /// True if `data` is a sentinel that ends the stream cleanly.
    ///
    /// Default returns `false` (no sentinel). `OpenAI`-shaped APIs override
    /// to recognise `[DONE]`.
    fn is_stream_end(_data: &str) -> bool {
        false
    }

    /// Translate one provider event into a stream item.
    ///
    /// - `None` skips emission — for events that update accumulator state
    ///   without producing user-visible output (e.g. Anthropic `Ping`,
    ///   `MessageStart`).
    /// - `Some(Ok(chunk))` yields a [`ChatChunk`] to the consumer.
    /// - `Some(Err(err))` surfaces a typed error from the wire data without
    ///   terminating the stream — for application-level failures embedded
    ///   in events (e.g. Responses `ResponseFailed`). Transport errors are
    ///   handled by the driver, not here.
    fn process_event(
        state: &mut Self::State,
        event: Self::Event,
    ) -> Option<Result<ChatChunk, ClientError>>;
}

/// Drive an SSE event source through a [`StreamingProvider`], yielding a
/// stream of [`ChatChunk`]s.
///
/// The provider is borrowed only to seed initial state via
/// [`StreamingProvider::initial_state`] before the unfold loop starts; it
/// does not need to outlive the returned stream.
///
/// Behaviour:
/// - `EventSource::Open` events are logged and skipped.
/// - Provider stream-end sentinels (via [`StreamingProvider::is_stream_end`])
///   terminate the stream cleanly.
/// - JSON parse failures yield [`ClientError::SerializationError`] but do
///   not terminate the stream — subsequent valid events still flow.
/// - [`reqwest_eventsource::Error::StreamEnded`] terminates cleanly.
/// - [`reqwest_eventsource::Error::InvalidStatusCode`] is unwrapped into a
///   typed [`ClientError`] via the response body, then terminates.
/// - All other event-source errors are mapped via [`ClientError::from`] and
///   terminate the stream.
///
/// # Errors
///
/// Returns [`ClientError::ConfigurationError`] if the [`EventSource`] cannot
/// be constructed from the supplied request builder.
pub fn run_sse_stream<P: StreamingProvider>(
    provider: &P,
    request: reqwest::RequestBuilder,
) -> Result<ChatChunkStream, ClientError> {
    let mut event_source = EventSource::new(request).map_err(|e| {
        ClientError::ConfigurationError(format!("Failed to create event source: {e}"))
    })?;
    event_source.set_retry_policy(Box::new(NoRetryPolicy));

    let stream = futures::stream::unfold(
        StreamState::<P::State> {
            event_source,
            provider_state: provider.initial_state(),
            terminated: false,
        },
        |mut s| async move {
            if s.terminated {
                return None;
            }
            loop {
                match s.event_source.next().await {
                    None => return None,
                    Some(Ok(Event::Open)) => {
                        debug!("Stream connection opened");
                    }
                    Some(Ok(Event::Message(message))) => {
                        if P::is_stream_end(&message.data) {
                            debug!("Stream completed via provider sentinel");
                            return None;
                        }
                        match serde_json::from_str::<P::Event>(&message.data) {
                            Ok(event) => match P::process_event(&mut s.provider_state, event) {
                                Some(Ok(chunk)) => return Some((Ok(chunk), s)),
                                Some(Err(err)) => return Some((Err(err), s)),
                                None => {}
                            },
                            Err(e) => {
                                warn!("Failed to parse streaming event: {e}");
                                debug!("Problematic event data: {}", message.data);
                                return Some((Err(ClientError::SerializationError(e)), s));
                            }
                        }
                    }
                    Some(Err(reqwest_eventsource::Error::StreamEnded)) => {
                        debug!("Stream ended normally");
                        return None;
                    }
                    Some(Err(reqwest_eventsource::Error::InvalidStatusCode(status, response))) => {
                        let error = extract_error_from_response(status, response).await;
                        error!("API error: {error}");
                        s.terminated = true;
                        return Some((Err(error), s));
                    }
                    Some(Err(other)) => {
                        let error = ClientError::from(other);
                        error!("Stream error: {error}");
                        s.terminated = true;
                        return Some((Err(error), s));
                    }
                }
            }
        },
    );

    Ok(Box::pin(stream))
}

/// Internal carrier for state threaded through the unfold closure.
struct StreamState<S> {
    event_source: EventSource,
    provider_state: S,
    terminated: bool,
}

/// Extract a typed [`ClientError`] from an HTTP error response body.
///
/// Tries to parse the body as a structured [`ErrorResponse`], falling back
/// to raw text when it isn't JSON. Maps the HTTP status to a specific
/// [`ClientError`] variant.
async fn extract_error_from_response(
    status: reqwest::StatusCode,
    response: reqwest::Response,
) -> ClientError {
    let error_text = response.text().await.unwrap_or_default();

    let error_message = match serde_json::from_str::<ErrorResponse>(&error_text) {
        Ok(parsed) => parsed.error.message,
        Err(_) => {
            if error_text.is_empty() {
                format!("HTTP {status}")
            } else {
                error_text
            }
        }
    };

    match status.as_u16() {
        401 => ClientError::AuthenticationError(error_message),
        429 => ClientError::RateLimitError { retry_after: None },
        500..=599 => ClientError::ServiceUnavailable(error_message),
        _ => ClientError::RequestError(error_message),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::panic)]

    use std::collections::HashMap;

    use chrono::Utc;
    use futures::StreamExt;
    use serde::Deserialize;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use neuromance_common::client::FinishReason;

    use super::*;

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum TestEvent {
        Hello { model: String },
        Delta { text: String },
        Done,
        Ping,
        Boom { message: String },
    }

    struct TestProvider;

    impl StreamingProvider for TestProvider {
        type Event = TestEvent;
        type State = String;

        fn initial_state(&self) -> Self::State {
            String::new()
        }

        fn is_stream_end(data: &str) -> bool {
            data == "[DONE]"
        }

        fn process_event(
            state: &mut Self::State,
            event: Self::Event,
        ) -> Option<Result<ChatChunk, ClientError>> {
            match event {
                TestEvent::Hello { model } => {
                    *state = model;
                    None
                }
                TestEvent::Ping => None,
                TestEvent::Delta { text } => Some(Ok(chunk(state, Some(text), None))),
                TestEvent::Done => Some(Ok(chunk(state, None, Some(FinishReason::Stop)))),
                TestEvent::Boom { message } => Some(Err(ClientError::RequestError(message))),
            }
        }
    }

    fn chunk(model: &str, content: Option<String>, finish: Option<FinishReason>) -> ChatChunk {
        ChatChunk {
            model: model.to_string(),
            delta_content: content,
            delta_reasoning_content: None,
            delta_role: None,
            delta_tool_calls: None,
            finish_reason: finish,
            usage: None,
            response_id: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    fn sse_body(events: &[&str]) -> String {
        let mut body = String::new();
        for event in events {
            body.push_str("data: ");
            body.push_str(event);
            body.push_str("\n\n");
        }
        body
    }

    fn post_request(server: &MockServer) -> reqwest::RequestBuilder {
        reqwest::Client::new()
            .post(format!("{}/stream", server.uri()))
            .header("Content-Type", "application/json")
    }

    #[tokio::test]
    async fn yields_chunks_in_order_and_terminates_on_sentinel() {
        let server = MockServer::start().await;
        let body = sse_body(&[
            r#"{"type":"hello","model":"test-model"}"#,
            r#"{"type":"delta","text":"hello "}"#,
            r#"{"type":"delta","text":"world"}"#,
            r#"{"type":"done"}"#,
            "[DONE]",
        ]);

        Mock::given(method("POST"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;

        let stream = run_sse_stream(&TestProvider, post_request(&server)).unwrap();
        let chunks: Vec<_> = stream.collect().await;

        assert_eq!(chunks.len(), 3);

        let first = chunks[0].as_ref().unwrap();
        assert_eq!(first.model, "test-model");
        assert_eq!(first.delta_content.as_deref(), Some("hello "));

        let second = chunks[1].as_ref().unwrap();
        assert_eq!(second.delta_content.as_deref(), Some("world"));

        let third = chunks[2].as_ref().unwrap();
        assert_eq!(third.finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn skips_events_that_produce_no_chunk() {
        let server = MockServer::start().await;
        let body = sse_body(&[
            r#"{"type":"hello","model":"m"}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"delta","text":"after pings"}"#,
            "[DONE]",
        ]);

        Mock::given(method("POST"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;

        let stream = run_sse_stream(&TestProvider, post_request(&server)).unwrap();
        let chunks: Vec<_> = stream.collect().await;

        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].as_ref().unwrap().delta_content.as_deref(),
            Some("after pings")
        );
    }

    #[tokio::test]
    async fn provider_data_error_emits_error_but_stream_continues() {
        let server = MockServer::start().await;
        let body = sse_body(&[
            r#"{"type":"hello","model":"m"}"#,
            r#"{"type":"delta","text":"before"}"#,
            r#"{"type":"boom","message":"upstream said no"}"#,
            r#"{"type":"delta","text":"after"}"#,
            "[DONE]",
        ]);

        Mock::given(method("POST"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;

        let stream = run_sse_stream(&TestProvider, post_request(&server)).unwrap();
        let results: Vec<_> = stream.collect().await;

        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].as_ref().unwrap().delta_content.as_deref(),
            Some("before")
        );
        match &results[1] {
            Err(ClientError::RequestError(msg)) => assert!(msg.contains("upstream said no")),
            other => panic!("expected RequestError, got {other:?}"),
        }
        assert_eq!(
            results[2].as_ref().unwrap().delta_content.as_deref(),
            Some("after")
        );
    }

    #[tokio::test]
    async fn parse_failure_emits_error_but_stream_continues() {
        let server = MockServer::start().await;
        let body = sse_body(&[
            r#"{"type":"hello","model":"m"}"#,
            "not json",
            r#"{"type":"delta","text":"recovered"}"#,
            "[DONE]",
        ]);

        Mock::given(method("POST"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;

        let stream = run_sse_stream(&TestProvider, post_request(&server)).unwrap();
        let results: Vec<_> = stream.collect().await;

        assert_eq!(results.len(), 2);
        assert!(matches!(
            &results[0],
            Err(ClientError::SerializationError(_))
        ));
        assert_eq!(
            results[1].as_ref().unwrap().delta_content.as_deref(),
            Some("recovered")
        );
    }

    #[tokio::test]
    async fn http_401_unwraps_to_authentication_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": { "message": "bad key" }
            })))
            .mount(&server)
            .await;

        let stream = run_sse_stream(&TestProvider, post_request(&server)).unwrap();
        let results: Vec<_> = stream.collect().await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Err(ClientError::AuthenticationError(msg)) => assert!(msg.contains("bad key")),
            other => panic!("expected AuthenticationError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_429_unwraps_to_rate_limit_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(429).set_body_string(""))
            .mount(&server)
            .await;

        let stream = run_sse_stream(&TestProvider, post_request(&server)).unwrap();
        let results: Vec<_> = stream.collect().await;

        assert_eq!(results.len(), 1);
        assert!(matches!(
            &results[0],
            Err(ClientError::RateLimitError { .. })
        ));
    }

    #[tokio::test]
    async fn http_500_unwraps_to_service_unavailable() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                "error": { "message": "upstream down" }
            })))
            .mount(&server)
            .await;

        let stream = run_sse_stream(&TestProvider, post_request(&server)).unwrap();
        let results: Vec<_> = stream.collect().await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Err(ClientError::ServiceUnavailable(msg)) => assert!(msg.contains("upstream down")),
            other => panic!("expected ServiceUnavailable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_stream_yields_no_chunks() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("", "text/event-stream"))
            .mount(&server)
            .await;

        let stream = run_sse_stream(&TestProvider, post_request(&server)).unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert!(chunks.is_empty());
    }
}
