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

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use tracing::Subscriber;
    use tracing::field::{Field, Visit};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    /// One captured `warn!` from the middleware.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Captured {
        message: String,
        attempt: Option<u32>,
        path: String,
    }

    /// Collects the middleware's own events so a test can assert on them.
    #[derive(Debug, Default, Clone)]
    struct CaptureLayer(Arc<Mutex<Vec<Captured>>>);

    impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for CaptureLayer {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            // The attempt number is what identifies this middleware's events;
            // matching on the module path would break if the module moved.
            if event.metadata().fields().field("attempt").is_none() {
                return;
            }
            let mut captured = Captured {
                message: String::new(),
                attempt: None,
                path: String::new(),
            };
            event.record(&mut captured);
            if let Ok(mut events) = self.0.lock() {
                events.push(captured);
            }
        }
    }

    impl Visit for Captured {
        fn record_u64(&mut self, field: &Field, value: u64) {
            if field.name() == "attempt" {
                self.attempt = u32::try_from(value).ok();
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            match field.name() {
                "message" => self.message = format!("{value:?}"),
                "path" => self.path = format!("{value:?}"),
                _ => {}
            }
        }
    }

    /// The capture buffer, fed by a subscriber installed for the whole binary.
    ///
    /// A thread-local subscriber (`tracing::subscriber::with_default`) is not
    /// enough here. `tracing` caches callsite interest *globally*: the
    /// middleware's `warn!` is first reached by other tests in this binary,
    /// while no subscriber is installed, so it gets cached as "never" and stays
    /// dark until some unrelated dispatcher change rebuilds the cache. Events
    /// then go missing depending on test interleaving. A global subscriber that
    /// outlives every test keeps the callsite unconditionally enabled, and each
    /// test filters the shared buffer by its own request path so the tests can
    /// still run in parallel with each other and with the rest of the suite.
    fn capture_buffer() -> &'static Arc<Mutex<Vec<Captured>>> {
        static CAPTURED: OnceLock<Arc<Mutex<Vec<Captured>>>> = OnceLock::new();
        CAPTURED.get_or_init(|| {
            let events = Arc::new(Mutex::new(Vec::new()));
            let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&events)));
            tracing::subscriber::set_global_default(subscriber)
                .expect("no other test may install a global subscriber");
            events
        })
    }

    /// A client with the middleware under test, retrying twice with no backoff.
    ///
    /// Installing the capture subscriber here, rather than where the events are
    /// read, is what makes the tests deterministic: the subscriber has to be in
    /// place before the first request, or the attempts made in the meantime are
    /// logged into the void.
    fn retrying_client() -> reqwest_middleware::ClientWithMiddleware {
        capture_buffer();
        reqwest_middleware::ClientBuilder::new(reqwest::Client::new())
            .with(reqwest_retry::RetryTransientMiddleware::new_with_policy(
                reqwest_retry::policies::ExponentialBackoff::builder()
                    .retry_bounds(Duration::from_millis(1), Duration::from_millis(1))
                    .build_with_max_retries(2),
            ))
            .with(RetryLoggingMiddleware)
            .build()
    }

    /// What the middleware logged for `request_path`, oldest first.
    fn events_for(request_path: &str) -> Vec<Captured> {
        capture_buffer()
            .lock()
            .expect("capture lock")
            .iter()
            .filter(|e| e.path == request_path)
            .cloned()
            .collect()
    }

    #[tokio::test]
    async fn test_only_retried_attempts_are_logged() {
        let mock_server = MockServer::start().await;
        // Fail twice, then succeed: attempts 2 and 3 should be logged, and the
        // counter must survive across retries rather than restarting at 1.
        Mock::given(method("GET"))
            .and(path("/retried"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&mock_server)
            .await;
        Mock::given(method("GET"))
            .and(path("/retried"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;
        Mock::given(method("GET"))
            .and(path("/first-try"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let client = retrying_client();
        drop(
            client
                .get(format!("{}/retried", mock_server.uri()))
                .send()
                .await,
        );
        drop(
            client
                .get(format!("{}/first-try", mock_server.uri()))
                .send()
                .await,
        );

        let retried = events_for("/retried");
        assert_eq!(
            retried.iter().map(|e| e.attempt).collect::<Vec<_>>(),
            vec![Some(2), Some(3)],
            "the first attempt is silent and the counter is shared: {retried:?}"
        );
        assert!(
            retried
                .iter()
                .all(|e| e.message.contains("llm request retry returned")),
            "a retry that got an HTTP response is a 'returned', not a failure: {retried:?}"
        );
        // Asserted in the same run as the case above, so an empty result means
        // "nothing was logged", not "capture is broken".
        assert_eq!(
            events_for("/first-try"),
            Vec::new(),
            "a request that succeeded first try is not worth a warning"
        );
    }

    #[tokio::test]
    async fn test_retried_transport_failures_log_the_error() {
        // Bind then drop, so the port is known-dead rather than merely unlikely.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        drop(
            retrying_client()
                .get(format!("http://127.0.0.1:{port}/refused"))
                .send()
                .await,
        );

        let events = events_for("/refused");
        assert_eq!(
            events.iter().map(|e| e.attempt).collect::<Vec<_>>(),
            vec![Some(2), Some(3)],
            "a connection failure is transient, so it is retried and logged: {events:?}"
        );
        assert!(
            events
                .iter()
                .all(|e| e.message.contains("llm request retry failed")),
            "an attempt that never got a response is a failure: {events:?}"
        );
    }
}
