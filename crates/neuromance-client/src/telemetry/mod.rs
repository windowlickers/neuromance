//! OpenTelemetry GenAI instrumentation for provider calls.
//!
//! [`GenAiOp`] owns one operation's span from the moment the request is built
//! until the operation ends, and records the two `gen_ai.client.*` metrics as
//! it closes. It is the only place in this crate that names a `gen_ai`
//! attribute, so adding one means editing [`attrs`] rather than six call
//! sites.
//!
//! Attribute keys come from `neuromance_common::telemetry::genai`.

mod attrs;
mod metrics;
#[cfg(test)]
mod tests;

use std::time::Instant;

use tracing::{Span, field, info_span};

use neuromance_common::client::{ChatRequest, ChatResponse, Config, Usage};
use neuromance_common::telemetry::genai;

use crate::error::ClientError;

pub use attrs::GenAiAttrs;

/// An in-flight GenAI operation and the span describing it.
///
/// The span stays open until one of the `finish_*` methods runs, or until the
/// value is dropped. A streaming call hands this to the stream adapter so the
/// span outlives the function that opened it.
pub struct GenAiOp {
    span: Span,
    attrs: GenAiAttrs,
    start: Instant,
    /// Cleared by whichever `finish_*` runs, so [`Drop`] can tell an
    /// abandoned operation from a completed one.
    pending: bool,
}

impl GenAiOp {
    /// Open a `chat` span carrying every request-side attribute.
    pub fn chat(config: &Config, request: &ChatRequest) -> Self {
        let attrs = GenAiAttrs::chat(config, request);
        let span = new_span(&attrs);
        record_chat_request(&span, request);
        Self::start(span, attrs)
    }

    fn start(span: Span, attrs: GenAiAttrs) -> Self {
        attrs.apply_to_span(&span);
        Self {
            span,
            attrs,
            start: Instant::now(),
            pending: true,
        }
    }

    /// The span, for entering or for instrumenting a future.
    pub const fn span(&self) -> &Span {
        &self.span
    }

    /// Close a successful non-streaming operation.
    pub fn finish_response(mut self, response: &ChatResponse) {
        self.pending = false;
        let finish_reasons = response
            .finish_reason
            .iter()
            .map(ToString::to_string)
            .collect();
        self.finish_ok(
            Some(response.model.as_str()),
            response.response_id.as_deref(),
            finish_reasons,
            response.usage.as_ref(),
        );
    }

    /// Close an operation that produced no usable result.
    pub fn finish_error(mut self, error: &ClientError) {
        self.pending = false;
        self.record_failure(error.reason(), &error.to_string());
    }

    fn finish_ok(
        &self,
        response_model: Option<&str>,
        response_id: Option<&str>,
        finish_reasons: Vec<String>,
        usage: Option<&Usage>,
    ) {
        if let Some(model) = response_model {
            self.span.record(genai::RESPONSE_MODEL, model);
        }
        if let Some(id) = response_id {
            self.span.record(genai::RESPONSE_ID, id);
        }
        attrs::set_string_array(&self.span, genai::RESPONSE_FINISH_REASONS, finish_reasons);

        if let Some(usage) = usage {
            self.span
                .record(genai::USAGE_INPUT_TOKENS, i64::from(usage.prompt_tokens));
            self.span.record(
                genai::USAGE_OUTPUT_TOKENS,
                i64::from(usage.completion_tokens),
            );
            metrics::instruments().record_token_usage(&self.attrs, response_model, usage);
        }
        self.record_duration(response_model, None);
    }

    fn record_failure(&self, error_type: &'static str, description: &str) {
        self.span.record(genai::ERROR_TYPE, error_type);
        self.span.record("otel.status_code", "ERROR");
        self.span.record("otel.status_description", description);
        self.record_duration(None, Some(error_type));
    }

    fn record_duration(&self, response_model: Option<&str>, error_type: Option<&str>) {
        metrics::instruments().record_duration(
            &self.attrs,
            response_model,
            error_type,
            self.start.elapsed().as_secs_f64(),
        );
    }
}

impl Drop for GenAiOp {
    /// An operation dropped without a `finish_*` was abandoned mid-flight —
    /// the future or stream was dropped, which is how `Core` implements
    /// cancellation. Mark it rather than exporting a span that just stops.
    fn drop(&mut self) {
        if self.pending {
            self.record_failure("cancelled", "operation dropped before completion");
        }
    }
}

/// Create the span with every attribute this operation may set declared.
///
/// A field must exist at creation for `Span::record` to reach it; recording an
/// undeclared field is silently discarded. `otel.name` carries the
/// `{operation} {model}` name the conventions require, because a `tracing`
/// span name has to be a literal.
fn new_span(attrs: &GenAiAttrs) -> Span {
    info_span!(
        "gen_ai",
        otel.name = %attrs.span_name(),
        otel.kind = "client",
        otel.status_code = field::Empty,
        otel.status_description = field::Empty,
        { genai::OPERATION_NAME } = field::Empty,
        { genai::PROVIDER_NAME } = field::Empty,
        { genai::REQUEST_MODEL } = field::Empty,
        { genai::REQUEST_TEMPERATURE } = field::Empty,
        { genai::REQUEST_TOP_P } = field::Empty,
        { genai::REQUEST_MAX_TOKENS } = field::Empty,
        { genai::REQUEST_FREQUENCY_PENALTY } = field::Empty,
        { genai::REQUEST_PRESENCE_PENALTY } = field::Empty,
        { genai::REQUEST_CHOICE_COUNT } = field::Empty,
        { genai::OUTPUT_TYPE } = field::Empty,
        { genai::RESPONSE_MODEL } = field::Empty,
        { genai::RESPONSE_ID } = field::Empty,
        { genai::USAGE_INPUT_TOKENS } = field::Empty,
        { genai::USAGE_OUTPUT_TOKENS } = field::Empty,
        { genai::SERVER_ADDRESS } = field::Empty,
        { genai::SERVER_PORT } = field::Empty,
        { genai::ERROR_TYPE } = field::Empty,
    )
}

/// Record the sampling parameters the request actually carries.
///
/// Integers are widened to `i64` on the way in. `tracing-opentelemetry`'s span
/// visitor implements `record_i64` but not `record_u64`, so an unsigned field
/// falls through to the `Debug` path and exports as a string — which type-checks,
/// exports cleanly, and quietly breaks every numeric query in the backend.
///
/// `gen_ai.request.stop_sequences` is array-valued, so it goes through
/// `set_attribute` and must not be declared as a field.
fn record_chat_request(span: &Span, request: &ChatRequest) {
    if let Some(temperature) = request.temperature {
        span.record(genai::REQUEST_TEMPERATURE, f64::from(temperature));
    }
    if let Some(top_p) = request.top_p {
        span.record(genai::REQUEST_TOP_P, f64::from(top_p));
    }
    if let Some(max_tokens) = request.max_tokens.or(request.max_completion_tokens) {
        span.record(genai::REQUEST_MAX_TOKENS, i64::from(max_tokens));
    }
    if let Some(penalty) = request.frequency_penalty {
        span.record(genai::REQUEST_FREQUENCY_PENALTY, f64::from(penalty));
    }
    if let Some(penalty) = request.presence_penalty {
        span.record(genai::REQUEST_PRESENCE_PENALTY, f64::from(penalty));
    }
    // Every provider here returns exactly one choice; the attribute is set so
    // a backend does not have to infer it.
    span.record(genai::REQUEST_CHOICE_COUNT, 1_i64);
    span.record(
        genai::OUTPUT_TYPE,
        if request.output_schema.is_some() {
            genai::output_type::JSON
        } else {
            genai::output_type::TEXT
        },
    );
    if let Some(ref stop) = request.stop {
        attrs::set_string_array(span, genai::REQUEST_STOP_SEQUENCES, stop.clone());
    }
}
