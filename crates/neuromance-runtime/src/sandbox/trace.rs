//! W3C trace-context propagation across the sandbox gRPC hop.
//!
//! The orchestrator and the sandbox are separate processes, so a `tool_call`
//! span in the orchestrator and the tool's actual execution in the sandbox
//! land in different traces unless the context travels with the request.
//! `opentelemetry-http` only covers `http::HeaderMap`; tonic carries its own
//! [`MetadataMap`], so the carriers live here.
//!
//! Both directions are no-ops when no propagator is installed, so a runtime
//! built without telemetry pays only an empty map walk.

use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::{Context, global};
use tonic::metadata::{KeyRef, MetadataKey, MetadataMap, MetadataValue};
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Writes propagator output into a request's gRPC metadata.
struct MetadataInjector<'a>(&'a mut MetadataMap);

impl Injector for MetadataInjector<'_> {
    /// Malformed pairs are dropped rather than propagated. A propagator only
    /// emits ASCII header names and values, so this rejects nothing in
    /// practice — but a bad key must not abort the RPC it decorates.
    fn set(&mut self, key: &str, value: String) {
        if let Ok(name) = MetadataKey::from_bytes(key.as_bytes())
            && let Ok(value) = MetadataValue::try_from(&value)
        {
            self.0.insert(name, value);
        }
    }
}

/// Reads propagator input from an incoming request's gRPC metadata.
struct MetadataExtractor<'a>(&'a MetadataMap);

impl Extractor for MetadataExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    /// Binary (`-bin`) metadata keys are skipped: trace context is always
    /// ASCII, and their values are not `&str`.
    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .filter_map(|key| match key {
                KeyRef::Ascii(key) => Some(key.as_str()),
                KeyRef::Binary(_) => None,
            })
            .collect()
    }
}

/// Wrap `message` in a request carrying the current span's trace context.
pub fn request<T>(message: T) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    let context = tracing::Span::current().context();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut MetadataInjector(request.metadata_mut()));
    });
    request
}

/// Attach the caller's trace context to `span`, so sandbox spans continue the
/// orchestrator's trace instead of rooting one per RPC.
///
/// Extracts against an empty context rather than calling `extract`, which
/// defaults to `Context::current()` and returns it unchanged when no
/// `traceparent` is present — that would silently adopt whatever span is active
/// on the serving thread instead of starting a root.
pub fn set_parent(span: &tracing::Span, metadata: &MetadataMap) {
    let parent = global::get_text_map_propagator(|propagator| {
        propagator.extract_with_context(&Context::new(), &MetadataExtractor(metadata))
    });
    if let Err(e) = span.set_parent(parent) {
        tracing::debug!(error = %e, "sandbox span keeps its own trace");
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use opentelemetry::propagation::TextMapPropagator;
    use opentelemetry::trace::{TraceContextExt as _, TracerProvider as _};
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_subscriber::layer::SubscriberExt;

    use super::*;

    /// Run `f` with a real `OTel` tracer layer installed, so `Span::current()`
    /// resolves to a sampled context the propagator will emit.
    fn with_otel_layer<T>(f: impl FnOnce() -> T) -> T {
        let provider = SdkTracerProvider::builder().build();
        let layer = tracing_opentelemetry::layer().with_tracer(provider.tracer("test"));
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, f)
    }

    fn traceparent<T>(request: &tonic::Request<T>) -> Option<String> {
        request
            .metadata()
            .get("traceparent")
            .map(|v| v.to_str().unwrap().to_string())
    }

    /// Three properties, in one test because the propagator registry is
    /// process-global and cannot be uninstalled — asserting the empty state
    /// requires running before any install, which separate tests cannot order.
    ///
    /// 1. Injection is a no-op with no propagator, so a telemetry-free
    ///    deployment is unaffected.
    /// 2. With one installed, an instrumented call carries a `traceparent`.
    /// 3. Metadata *without* a `traceparent` starts a new trace rather than
    ///    adopting the span active on this thread. `TextMapPropagator::extract`
    ///    defaults to `Context::current()` and returns it unchanged when the
    ///    header is absent; using it here made every RPC that arrived without
    ///    trace context join whatever the worker was already running.
    ///
    /// Serial because assertion 1 reads the propagator registry before this
    /// test installs one, so it must not run beside a test that installs.
    #[test]
    #[serial_test::serial]
    fn test_trace_context_crosses_only_when_a_traceparent_is_present() {
        assert_eq!(
            with_otel_layer(|| tracing::info_span!("caller").in_scope(|| traceparent(&request(())))),
            None,
            "no propagator installed should add no metadata"
        );

        global::set_text_map_propagator(TraceContextPropagator::new());

        let injected = with_otel_layer(|| {
            tracing::info_span!("caller").in_scope(|| traceparent(&request(())))
        });
        assert!(
            injected.is_some_and(|v| v.starts_with("00-")),
            "an instrumented call should carry a W3C traceparent"
        );

        let empty = MetadataMap::new();
        let (active_trace, incoming_trace) = with_otel_layer(|| {
            let active = tracing::info_span!("unrelated_work");
            active.in_scope(|| {
                let incoming = tracing::info_span!("incoming");
                set_parent(&incoming, &empty);
                (
                    active.context().span().span_context().trace_id(),
                    incoming.context().span().span_context().trace_id(),
                )
            })
        });

        assert_ne!(
            incoming_trace, active_trace,
            "an untraced request must start its own trace, not join the active span"
        );
    }

    /// The sandbox reads back the trace the orchestrator wrote — the join that
    /// makes a tool execution show up under its `tool_call` span. Uses an
    /// explicit propagator, so it does not depend on the global registry.
    #[test]
    fn test_extractor_reads_the_incoming_trace_id() {
        let mut metadata = MetadataMap::new();
        metadata.insert(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
                .parse()
                .unwrap(),
        );

        let context = TraceContextPropagator::new().extract(&MetadataExtractor(&metadata));

        let span_context = context.span().span_context().clone();
        assert!(span_context.is_valid(), "extracted context is not sampled");
        assert_eq!(
            span_context.trace_id().to_string(),
            "4bf92f3577b34da6a3ce929d0e0e4736"
        );
    }
}
