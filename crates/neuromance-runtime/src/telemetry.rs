//! Optional OTLP telemetry export for traces and logs.
//!
//! Activated when `OTEL_EXPORTER_OTLP_ENDPOINT` is set in the environment.
//! When absent, [`try_init`] returns `Ok(None)` and the runtime emits only
//! Prometheus metrics and stderr logs, as before.
//!
//! Metrics are intentionally **not** dual-exported in-process. The
//! recommended pattern is to point an OpenTelemetry Collector's
//! `prometheus` receiver at this runtime's `/metrics` endpoint
//! (default `:8081`) and have the collector forward them via OTLP.
//!
//! Standard OTEL environment variables are honored:
//!
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — gate, plus default endpoint
//! - `OTEL_EXPORTER_OTLP_PROTOCOL` — `grpc` (default) or `http/protobuf`
//! - `OTEL_EXPORTER_OTLP_HEADERS` — handled by `opentelemetry-otlp`
//! - `OTEL_SERVICE_NAME` — falls back to the runtime's `agent.id`
//! - `OTEL_RESOURCE_ATTRIBUTES` — merged on top of defaults
//! - Per-signal endpoint overrides (`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`,
//!   `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`) — supported transparently

use opentelemetry::{KeyValue, global, trace::TracerProvider as _};
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_otlp::{LogExporter, Protocol, SpanExporter, WithExportConfig};
use opentelemetry_sdk::{
    Resource, logs::SdkLoggerProvider, propagation::TraceContextPropagator,
    trace::SdkTracerProvider,
};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing::Subscriber;
use tracing_subscriber::{Layer, registry::LookupSpan};

use crate::error::RuntimeError;

const ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
const PROTOCOL_ENV: &str = "OTEL_EXPORTER_OTLP_PROTOCOL";
const SERVICE_NAME_ENV: &str = "OTEL_SERVICE_NAME";

/// Owns the OTLP trace and log providers for the process lifetime.
///
/// [`shutdown`](Self::shutdown) flushes buffered exports and tears
/// down the providers; the `Drop` impl invokes it if the caller forgets.
pub struct TelemetryGuard {
    tracer_provider: Option<SdkTracerProvider>,
    logger_provider: Option<SdkLoggerProvider>,
}

impl std::fmt::Debug for TelemetryGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryGuard")
            .field("tracer_provider", &self.tracer_provider.is_some())
            .field("logger_provider", &self.logger_provider.is_some())
            .finish()
    }
}

impl TelemetryGuard {
    /// Flush buffered exports and shut down both providers. Idempotent.
    pub fn shutdown(mut self) {
        self.shutdown_in_place();
    }

    fn shutdown_in_place(&mut self) {
        if let Some(provider) = self.tracer_provider.take()
            && let Err(e) = provider.shutdown()
        {
            tracing::warn!(error = %e, "otel tracer provider shutdown failed");
        }
        if let Some(provider) = self.logger_provider.take()
            && let Err(e) = provider.shutdown()
        {
            tracing::warn!(error = %e, "otel logger provider shutdown failed");
        }
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        self.shutdown_in_place();
    }
}

/// Boxed `tracing-subscriber` layer produced by [`try_init`].
///
/// Returned as `Option<BoxedLayer<S>>` rather than `Vec<BoxedLayer<S>>`:
/// an empty `Vec<L>` reports `Interest::never()` from its `Layer` impl,
/// which silences every event in the subscriber when attached. `Option::None`
/// reports `Interest::always()` and is a safe no-op.
pub type BoxedLayer<S> = Box<dyn Layer<S> + Send + Sync + 'static>;

/// Output of a successful [`try_init`] when OTLP export is enabled.
pub type TelemetryInit<S> = (TelemetryGuard, BoxedLayer<S>);

/// Initialize OTLP export when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
///
/// Returns `Ok(None)` if the env var is absent (telemetry disabled).
/// Returns `Ok(Some((guard, layer)))` otherwise; the caller must install
/// `layer` into its `tracing-subscriber` registry and keep `guard` alive
/// for the process lifetime.
///
/// `default_service_name` is used as `service.name` when `OTEL_SERVICE_NAME`
/// is unset.
///
/// # Errors
/// Returns [`RuntimeError::Telemetry`] if `OTEL_EXPORTER_OTLP_PROTOCOL`
/// is set to an unsupported value, or if exporter construction fails.
pub fn try_init<S>(default_service_name: &str) -> Result<Option<TelemetryInit<S>>, RuntimeError>
where
    S: Subscriber + for<'a> LookupSpan<'a> + Send + Sync + 'static,
{
    if std::env::var_os(ENDPOINT_ENV).is_none() {
        return Ok(None);
    }

    let protocol = resolve_protocol(std::env::var(PROTOCOL_ENV).ok().as_deref())?;
    let service_name = std::env::var(SERVICE_NAME_ENV)
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_service_name.to_string());
    let resource = build_resource(&service_name);

    global::set_text_map_propagator(TraceContextPropagator::new());

    let tracer_provider = build_tracer_provider(protocol, resource.clone())?;
    let logger_provider = build_logger_provider(protocol, resource)?;

    let tracer = tracer_provider.tracer("neuromance-runtime");
    let trace_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let log_layer = OpenTelemetryTracingBridge::new(&logger_provider);
    let combined: BoxedLayer<S> = trace_layer.and_then(log_layer).boxed();

    Ok(Some((
        TelemetryGuard {
            tracer_provider: Some(tracer_provider),
            logger_provider: Some(logger_provider),
        },
        combined,
    )))
}

fn resolve_protocol(raw: Option<&str>) -> Result<Protocol, RuntimeError> {
    match raw.map(str::trim) {
        None | Some("" | "grpc") => Ok(Protocol::Grpc),
        Some("http/protobuf") => Ok(Protocol::HttpBinary),
        Some(other) => Err(RuntimeError::Telemetry(format!(
            "unsupported {PROTOCOL_ENV}={other:?}; expected 'grpc' or 'http/protobuf'"
        ))),
    }
}

fn build_resource(service_name: &str) -> Resource {
    Resource::builder()
        .with_attributes([
            KeyValue::new(SERVICE_NAME, service_name.to_string()),
            KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
        ])
        .build()
}

fn build_tracer_provider(
    protocol: Protocol,
    resource: Resource,
) -> Result<SdkTracerProvider, RuntimeError> {
    let exporter = match protocol {
        Protocol::Grpc => SpanExporter::builder()
            .with_tonic()
            .with_protocol(protocol)
            .build(),
        Protocol::HttpBinary | Protocol::HttpJson => SpanExporter::builder()
            .with_http()
            .with_protocol(protocol)
            .build(),
    }
    .map_err(|e| RuntimeError::Telemetry(format!("build OTLP span exporter: {e}")))?;

    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build())
}

fn build_logger_provider(
    protocol: Protocol,
    resource: Resource,
) -> Result<SdkLoggerProvider, RuntimeError> {
    let exporter = match protocol {
        Protocol::Grpc => LogExporter::builder()
            .with_tonic()
            .with_protocol(protocol)
            .build(),
        Protocol::HttpBinary | Protocol::HttpJson => LogExporter::builder()
            .with_http()
            .with_protocol(protocol)
            .build(),
    }
    .map_err(|e| RuntimeError::Telemetry(format!("build OTLP log exporter: {e}")))?;

    Ok(SdkLoggerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::panic)]

    use super::*;

    #[test]
    fn test_resolve_protocol_defaults_to_grpc() {
        assert_eq!(resolve_protocol(None).unwrap(), Protocol::Grpc);
        assert_eq!(resolve_protocol(Some("")).unwrap(), Protocol::Grpc);
        assert_eq!(resolve_protocol(Some("grpc")).unwrap(), Protocol::Grpc);
    }

    #[test]
    fn test_resolve_protocol_http_protobuf() {
        assert_eq!(
            resolve_protocol(Some("http/protobuf")).unwrap(),
            Protocol::HttpBinary
        );
        assert_eq!(
            resolve_protocol(Some("  http/protobuf  ")).unwrap(),
            Protocol::HttpBinary,
            "leading/trailing whitespace should be tolerated"
        );
    }

    #[test]
    fn test_resolve_protocol_rejects_unknown() {
        let err = resolve_protocol(Some("smoke-signal")).unwrap_err();
        match err {
            RuntimeError::Telemetry(msg) => {
                assert!(msg.contains("smoke-signal"), "got: {msg}");
                assert!(msg.contains("grpc"), "should list valid options: {msg}");
            }
            other => panic!("expected Telemetry error, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_protocol_rejects_http_json() {
        // http/json is not supported by opentelemetry-otlp 0.31, so the
        // caller must get a clear error rather than a silent fallback.
        let err = resolve_protocol(Some("http/json")).unwrap_err();
        assert!(matches!(err, RuntimeError::Telemetry(_)));
    }

    #[test]
    fn test_build_resource_sets_service_attributes() {
        let resource = build_resource("research-agent");
        let name = resource
            .get(&opentelemetry::Key::from_static_str("service.name"))
            .map(|v| v.to_string());
        assert_eq!(name.as_deref(), Some("research-agent"));
        let version = resource
            .get(&opentelemetry::Key::from_static_str("service.version"))
            .map(|v| v.to_string());
        assert_eq!(version.as_deref(), Some(env!("CARGO_PKG_VERSION")));
    }
}
