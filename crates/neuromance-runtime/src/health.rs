//! Health and readiness HTTP endpoints.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{Router, http::StatusCode, routing::get};

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
}
