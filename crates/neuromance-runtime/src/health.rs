//! Health and readiness HTTP endpoints.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{Router, http::StatusCode, routing::get};
use tower_http::trace::TraceLayer;
use tracing::{Level, Span, field, info_span};

/// Atomic flag flipped to `true` when the runtime has finished startup
/// (LLM client built, tools registered, server bound) and to `false`
/// when shutdown begins. Polled by `/readyz`.
#[derive(Debug, Default)]
pub struct ReadinessGate {
    ready: AtomicBool,
}

impl ReadinessGate {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ready: AtomicBool::new(false),
        }
    }

    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

pub fn router(readiness: Arc<ReadinessGate>) -> Router {
    // Probe endpoints fire constantly; only log non-2xx responses.
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|req: &axum::http::Request<_>| {
            info_span!(
                "http_request",
                method = %req.method(),
                path = %req.uri().path(),
                status = field::Empty,
            )
        })
        .on_request(())
        .on_response(
            |res: &axum::http::Response<_>, latency: std::time::Duration, span: &Span| {
                let status = res.status();
                if status.is_success() {
                    return;
                }
                span.record("status", status.as_u16());
                let latency_ms = u64::try_from(latency.as_millis()).unwrap_or(u64::MAX);
                if status.is_server_error() {
                    tracing::event!(parent: span, Level::ERROR, latency_ms, "http response");
                } else {
                    tracing::event!(parent: span, Level::WARN, latency_ms, "http response");
                }
            },
        );
    Router::new()
        .route("/healthz", get(|| async { (StatusCode::OK, "ok") }))
        .route(
            "/readyz",
            get(move || {
                let readiness = Arc::clone(&readiness);
                async move {
                    if readiness.is_ready() {
                        (StatusCode::OK, "ready")
                    } else {
                        (StatusCode::SERVICE_UNAVAILABLE, "not ready")
                    }
                }
            }),
        )
        .layer(trace_layer)
}
