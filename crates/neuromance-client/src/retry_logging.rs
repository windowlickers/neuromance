//! Logging middleware that records each `reqwest-retry` attempt.
//!
//! `reqwest-retry` silently retries transient failures, so users have no
//! visibility into how many attempts a request actually took. This middleware
//! is registered after `RetryTransientMiddleware` so it sits innermost in the
//! chain: the retry loop re-invokes it on every attempt, letting it log each
//! re-entry plus the eventual outcome.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use reqwest::{Request, Response};
use reqwest_middleware::{Middleware, Next, Result as MwResult};
use tracing::warn;

/// Per-request attempt counter shared between middleware invocations.
///
/// Stored in the request's `http::Extensions`, which `reqwest-retry` threads
/// through every retry, so each attempt shares one counter even though
/// `Middleware::handle` is called fresh each time.
struct AttemptCounter(AtomicU32);

/// Middleware that logs every retry attempt and the final outcome.
#[derive(Debug, Default, Clone, Copy)]
pub struct RetryLoggingMiddleware;

#[async_trait]
impl Middleware for RetryLoggingMiddleware {
    async fn handle(
        &self,
        req: Request,
        ext: &mut http::Extensions,
        next: Next<'_>,
    ) -> MwResult<Response> {
        let counter = ext
            .get::<Arc<AttemptCounter>>()
            .cloned()
            .unwrap_or_else(|| {
                let c = Arc::new(AttemptCounter(AtomicU32::new(0)));
                ext.insert(Arc::clone(&c));
                c
            });
        let attempt = counter.0.fetch_add(1, Ordering::Relaxed) + 1;

        let method = req.method().clone();
        let url = req.url().clone();
        let host = url.host_str().unwrap_or("").to_string();
        let path = url.path().to_string();

        let result = next.run(req, ext).await;

        if attempt > 1 {
            match &result {
                Ok(resp) => warn!(
                    attempt,
                    method = %method,
                    host = %host,
                    path = %path,
                    status = resp.status().as_u16(),
                    "llm request retry returned",
                ),
                Err(e) => warn!(
                    attempt,
                    method = %method,
                    host = %host,
                    path = %path,
                    error = %e,
                    "llm request retry failed",
                ),
            }
        }
        result
    }
}
