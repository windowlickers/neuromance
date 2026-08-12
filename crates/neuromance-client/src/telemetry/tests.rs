//! Span-shape assertions for [`GenAiOp`](super::GenAiOp).
//!
//! These check what an exporter would actually receive: the span name, the
//! kind, the attribute types. Attribute *names* only matter once a backend
//! indexes them, so getting one wrong is invisible in a unit test that only
//! reads back `tracing` fields — hence the in-memory OTLP exporter.

#![allow(clippy::expect_used)]
#![allow(clippy::panic)]

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{Value, trace::SpanKind};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use tracing_subscriber::layer::SubscriberExt as _;

use neuromance_common::chat::Message;
use neuromance_common::client::{ChatRequest, ChatResponse, Config, FinishReason, Usage};
use neuromance_common::telemetry::genai;

use super::GenAiOp;
use crate::error::ClientError;

/// Run `body` under a subscriber that exports finished spans in memory.
fn exported_spans(body: impl FnOnce()) -> Vec<SpanData> {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));

    tracing::subscriber::with_default(subscriber, body);
    provider.force_flush().ok();

    exporter.get_finished_spans().expect("in-memory spans")
}

fn attribute<'a>(span: &'a SpanData, key: &str) -> Option<&'a Value> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| &kv.value)
}

fn config() -> Config {
    Config::new("openai", "gpt-4o")
}

fn response() -> ChatResponse {
    ChatResponse {
        message: Message::assistant(uuid::Uuid::new_v4(), "hi"),
        model: "gpt-4o-2024-08-06".to_string(),
        usage: Some(Usage {
            prompt_tokens: 900,
            completion_tokens: 40,
            total_tokens: 940,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        }),
        finish_reason: Some(FinishReason::ToolCalls),
        created_at: chrono::Utc::now(),
        response_id: Some("resp_123".to_string()),
        metadata: std::collections::HashMap::new(),
    }
}

/// The conventions name the span `{operation} {model}`, but a `tracing` span
/// name must be a literal. `otel.name` is what makes the dynamic name reach
/// the exporter; without it every chat span would be called `gen_ai`.
#[test]
fn test_chat_span_is_named_operation_and_model() {
    let spans = exported_spans(|| {
        let request = ChatRequest::new(Vec::new()).with_model("gpt-4o");
        GenAiOp::chat(&config(), &request).finish_response(&response());
    });

    let span = spans.first().expect("one chat span");
    assert_eq!(span.name, "chat gpt-4o");
    assert_eq!(span.span_kind, SpanKind::Client);
}

/// Finish reasons are an array in the conventions, and `ClickHouse` types them
/// as `Array(String)`. Joining them into one string would compile and export
/// happily while breaking every array query against the backend.
#[test]
fn test_finish_reasons_export_as_a_string_array() {
    let spans = exported_spans(|| {
        let request = ChatRequest::new(Vec::new()).with_model("gpt-4o");
        GenAiOp::chat(&config(), &request).finish_response(&response());
    });

    let span = spans.first().expect("one chat span");
    let value = attribute(span, genai::RESPONSE_FINISH_REASONS).expect("finish reasons");
    let Value::Array(opentelemetry::Array::String(reasons)) = value else {
        panic!("expected a string array, got {value:?}");
    };
    assert_eq!(
        reasons.iter().map(ToString::to_string).collect::<Vec<_>>(),
        vec!["tool_calls".to_string()]
    );
}

#[test]
fn test_chat_span_carries_request_and_response_attributes() {
    let spans = exported_spans(|| {
        let request = ChatRequest::new(Vec::new())
            .with_model("gpt-4o")
            .with_temperature(0.25)
            .with_max_tokens(512);
        GenAiOp::chat(&config(), &request).finish_response(&response());
    });

    let span = spans.first().expect("one chat span");
    for (key, expected) in [
        (genai::OPERATION_NAME, Value::from("chat")),
        (genai::PROVIDER_NAME, Value::from("openai")),
        (genai::REQUEST_MODEL, Value::from("gpt-4o")),
        (genai::REQUEST_MAX_TOKENS, Value::I64(512)),
        (genai::RESPONSE_MODEL, Value::from("gpt-4o-2024-08-06")),
        (genai::RESPONSE_ID, Value::from("resp_123")),
        (genai::USAGE_INPUT_TOKENS, Value::I64(900)),
        (genai::USAGE_OUTPUT_TOKENS, Value::I64(40)),
        (genai::SERVER_ADDRESS, Value::from("api.openai.com")),
        (genai::SERVER_PORT, Value::I64(443)),
    ] {
        assert_eq!(attribute(span, key), Some(&expected), "wrong {key}");
    }
}

/// A failed call must be findable as a failure. `error.type` is the metric
/// attribute too, so the trace and the duration histogram agree.
#[test]
fn test_failed_chat_span_records_the_error_type_and_status() {
    let spans = exported_spans(|| {
        let request = ChatRequest::new(Vec::new()).with_model("gpt-4o");
        GenAiOp::chat(&config(), &request).finish_error(&ClientError::TimeoutError);
    });

    let span = spans.first().expect("one chat span");
    assert_eq!(
        attribute(span, genai::ERROR_TYPE),
        Some(&Value::from("timeout"))
    );
    assert!(
        matches!(span.status, opentelemetry::trace::Status::Error { .. }),
        "expected an error status, got {:?}",
        span.status
    );
}

/// `Core` cancels by dropping the future. Without the `Drop` marker the span
/// would still export, just with no outcome, which reads as a successful call
/// that happened to record nothing.
#[test]
fn test_abandoned_operation_is_marked_cancelled() {
    let spans = exported_spans(|| {
        let request = ChatRequest::new(Vec::new()).with_model("gpt-4o");
        drop(GenAiOp::chat(&config(), &request));
    });

    let span = spans.first().expect("one chat span");
    assert_eq!(
        attribute(span, genai::ERROR_TYPE),
        Some(&Value::from("cancelled"))
    );
}

/// A schema-carrying request asks the provider for JSON, and the attribute is
/// what lets a backend separate structured calls from free-form ones.
#[test]
fn test_output_type_reflects_a_requested_schema() {
    let schema = neuromance_common::client::OutputSchema::new(
        "answer",
        serde_json::json!({"type": "object", "properties": {}, "additionalProperties": false}),
    )
    .expect("valid schema");

    let spans = exported_spans(|| {
        let request = ChatRequest::new(Vec::new())
            .with_model("gpt-4o")
            .with_output_schema(schema);
        GenAiOp::chat(&config(), &request).finish_response(&response());
    });

    let span = spans.first().expect("one chat span");
    assert_eq!(
        attribute(span, genai::OUTPUT_TYPE),
        Some(&Value::from("json"))
    );
}
