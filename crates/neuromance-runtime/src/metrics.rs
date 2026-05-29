//! Prometheus metrics endpoint.
//!
//! Installs a global `metrics` recorder backed by
//! [`metrics-exporter-prometheus`](metrics_exporter_prometheus) and exposes
//! a `GET /metrics` route that renders the current snapshot in Prometheus
//! exposition format. The route is mounted on the management port
//! (`runtime.health_addr`), not the task port, so scraping never competes
//! with task traffic.
//!
//! See [`crate::serve`] and `crates/neuromance/src/core.rs` for the
//! instrumentation sites that emit values into the recorder.

use axum::{Router, response::IntoResponse, routing::get};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use crate::error::RuntimeError;

/// Install the global Prometheus recorder.
///
/// Must be called exactly once at startup, before any `metrics::counter!`
/// or similar call would otherwise initialise a no-op recorder.
///
/// # Errors
/// Returns [`RuntimeError::Metrics`] if installing the recorder fails
/// (typically because one was already installed).
pub fn init() -> Result<PrometheusHandle, RuntimeError> {
    PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| RuntimeError::Metrics(format!("install prometheus recorder: {e}")))
}

/// Build the `/metrics` router that renders the recorder snapshot.
pub fn router(handle: PrometheusHandle) -> Router {
    Router::new().route("/metrics", get(move || render(handle.clone())))
}

#[allow(clippy::unused_async)]
async fn render(handle: PrometheusHandle) -> impl IntoResponse {
    handle.render()
}
