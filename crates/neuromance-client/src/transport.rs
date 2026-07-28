use std::time::Duration;

use neuromance_common::client::ProxyConfig;
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;
use tracing::{error, trace, warn};

use crate::error::{ClientError, ErrorResponse};

/// Abstraction over request builder types that support setting headers.
///
/// Both `reqwest::RequestBuilder` and `reqwest_middleware::RequestBuilder` expose
/// a `.header()` method but as inherent methods, not via a shared trait.
/// This trait unifies them so `add_proxy_headers` can be a generic function.
pub trait WithHeader: Sized {
    fn header(self, name: &str, value: &str) -> Self;
}

impl WithHeader for reqwest::RequestBuilder {
    fn header(self, name: &str, value: &str) -> Self {
        Self::header(self, name, value)
    }
}

impl WithHeader for reqwest_middleware::RequestBuilder {
    fn header(self, name: &str, value: &str) -> Self {
        Self::header(self, name, value)
    }
}

/// Adds the sealed-token header to a request when proxy mode is active.
///
/// The proxy URL is configured on the underlying `reqwest::Client` so the
/// transport itself routes requests through the proxy in forward-proxy
/// (absolute-form) mode; the upstream target host therefore travels in the
/// request URL, not a side-band header. This function only needs to attach
/// the sealed-token header so the proxy can decrypt and inject the real
/// upstream credential.
pub fn add_proxy_headers<B: WithHeader>(
    builder: B,
    proxy_config: Option<&ProxyConfig>,
    api_key: &SecretString,
) -> B {
    if let Some(proxy) = proxy_config {
        builder.header(&proxy.token_header, api_key.expose_secret())
    } else {
        builder
    }
}

/// Parse a `Retry-After` header into a [`Duration`].
///
/// Only the delta-seconds form (RFC 9110 §10.2.3) is recognised; the HTTP-date
/// form yields `None` rather than a misleading zero.
#[must_use]
pub fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Map an HTTP error response (status + headers + body) to a typed [`ClientError`].
///
/// Tries to parse `body` as a structured [`ErrorResponse`], falling back to the
/// raw body text, then to `HTTP {status}` when the body is empty. The status
/// code selects the variant. `headers` supplies `Retry-After` for the 429 arm.
/// This is the single canonical mapping shared by the streaming
/// ([`crate::streaming`]) and non-streaming ([`send_json`]) paths, so both agree
/// that any `5xx` is a retryable [`ClientError::ServiceUnavailable`].
#[must_use]
pub fn map_http_error(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> ClientError {
    let message = match serde_json::from_str::<ErrorResponse>(body) {
        Ok(parsed) => parsed.error.message,
        Err(_) if body.is_empty() => format!("HTTP {status}"),
        Err(_) => body.to_string(),
    };

    match status.as_u16() {
        401 => ClientError::AuthenticationError(message),
        429 => ClientError::RateLimitError {
            message,
            retry_after: parse_retry_after(headers),
        },
        500..=599 => ClientError::ServiceUnavailable(message),
        _ => ClientError::RequestError(message),
    }
}

/// Map a provider error code carried *inside* a response body to a typed
/// [`ClientError`].
///
/// In-band stream `error` events arrive over an already-successful HTTP
/// response, so [`map_http_error`]'s status code is unavailable and the
/// provider's own `type`/`code` string is the only signal. Without this, an
/// `overloaded_error` delivered mid-stream became a permanent
/// [`ClientError::RequestError`] while the same overload delivered as HTTP 529
/// became a retryable [`ClientError::ServiceUnavailable`].
///
/// `error_type` is checked first, then `code`, so providers that populate
/// either field classify the same way. Unrecognised codes stay
/// [`ClientError::RequestError`] — terminal, because retrying an error we
/// cannot identify risks looping on a permanent failure.
///
/// `retry_after` is always `None` for rate limits here: an in-band event has no
/// `Retry-After` header, so callers fall back to their own backoff schedule.
#[must_use]
pub fn classify_provider_error(
    error_type: &str,
    code: Option<&str>,
    message: String,
) -> ClientError {
    let discriminator = match code {
        Some(code) if !PROVIDER_ERROR_CODES.contains(&error_type) => code,
        _ => error_type,
    };

    match discriminator {
        "rate_limit_error" | "rate_limit_exceeded" => ClientError::RateLimitError {
            message,
            retry_after: None,
        },
        "overloaded_error" | "api_error" | "server_error" => {
            ClientError::ServiceUnavailable(message)
        }
        "authentication_error" => ClientError::AuthenticationError(message),
        _ => ClientError::RequestError(message),
    }
}

/// Every provider error code [`classify_provider_error`] recognises.
///
/// Used to decide whether `error_type` carries the classification or whether to
/// defer to `code`: `OpenAI` sends a generic `type` (`invalid_request_error`)
/// alongside a specific `code` (`rate_limit_exceeded`), while Anthropic puts
/// the specific value in `type` and has no `code` field at all.
const PROVIDER_ERROR_CODES: [&str; 6] = [
    "rate_limit_error",
    "rate_limit_exceeded",
    "overloaded_error",
    "api_error",
    "server_error",
    "authentication_error",
];

/// Map a transport-level [`reqwest::Error`] to a typed [`ClientError`].
///
/// Separates a timeout from every other connection failure, which is what makes
/// [`ClientError::TimeoutError`] reachable at all — before this, every timeout
/// arrived as a [`ClientError::NetworkError`]. Both are retryable, so this
/// sharpens diagnostics rather than changing retry behaviour: an operator
/// seeing timeouts should raise `timeout_seconds`, not chase a network fault.
#[must_use]
pub fn map_transport_error(err: reqwest::Error) -> ClientError {
    if err.is_timeout() {
        ClientError::TimeoutError
    } else {
        ClientError::NetworkError(err)
    }
}

/// Does this link in an error chain hold a timed-out request?
///
/// `RetryTransientMiddleware` never lets the original [`reqwest::Error`] through
/// untouched: it re-wraps every outcome in a [`reqwest_retry::RetryError`], and
/// both of that enum's variants hide the [`reqwest_middleware::Error`] behind
/// `#[error(transparent)]`, so the plain source chain skips straight past it.
/// Unwrapping the retry error explicitly is the only way to recover the
/// distinction. The or-pattern is deliberately exhaustive: a new `RetryError`
/// variant should break the build here rather than silently downgrade timeouts.
fn cause_is_timeout(cause: &(dyn std::error::Error + 'static)) -> bool {
    let inner = match cause.downcast_ref::<reqwest_retry::RetryError>() {
        Some(
            reqwest_retry::RetryError::WithRetries { err, .. }
            | reqwest_retry::RetryError::Error(err),
        ) => err,
        None => match cause.downcast_ref::<reqwest_middleware::Error>() {
            Some(err) => err,
            None => return false,
        },
    };
    matches!(inner, reqwest_middleware::Error::Reqwest(e) if e.is_timeout())
}

/// Map a send failure from the middleware stack to a typed [`ClientError`].
///
/// A timeout does not arrive as [`reqwest_middleware::Error::Reqwest`]: the
/// retry middleware reports it as a `Middleware` error, so recovering it means
/// walking the source chain (see [`cause_is_timeout`]). Both outcomes are
/// retryable, so this sharpens the diagnostic rather than changing behaviour —
/// an operator seeing timeouts should raise `timeout_seconds`, not chase a
/// network fault.
#[must_use]
pub fn map_middleware_error(err: reqwest_middleware::Error) -> ClientError {
    let timed_out = match &err {
        reqwest_middleware::Error::Reqwest(e) => e.is_timeout(),
        reqwest_middleware::Error::Middleware(e) => e.chain().any(cause_is_timeout),
    };

    if timed_out {
        return ClientError::TimeoutError;
    }
    match err {
        reqwest_middleware::Error::Reqwest(e) => ClientError::NetworkError(e),
        other @ reqwest_middleware::Error::Middleware(_) => ClientError::MiddlewareError(other),
    }
}

/// Send a fully-built request and deserialize its JSON success body into `T`.
///
/// Owns the shared non-streaming transport tail: send, HTTP-status error mapping
/// (via [`map_http_error`]), and success-body deserialization. Callers build the
/// request — URL, headers, auth, proxy headers, and serialized body — the same
/// way they build the [`reqwest::RequestBuilder`] handed to
/// [`crate::streaming::run_sse_stream`] for the streaming path.
///
/// # Errors
///
/// - The variant selected by [`map_middleware_error`] if the request fails to
///   send, or by [`map_transport_error`] if a response body cannot be read.
/// - The variant selected by [`map_http_error`] on a non-success status.
/// - [`ClientError::SerializationError`] if a success body is not valid `T`.
pub async fn send_json<T: DeserializeOwned>(
    request: reqwest_middleware::RequestBuilder,
) -> Result<T, ClientError> {
    let response = request.send().await.map_err(map_middleware_error)?;

    if !response.status().is_success() {
        let status = response.status();
        let headers = response.headers().clone();
        let error_text = response.text().await.map_err(|e| {
            warn!("Failed to read error response body: {e}");
            map_transport_error(e)
        })?;
        let error = map_http_error(status, &headers, &error_text);
        error!(
            "API request failed with status {}: {error}",
            status.as_u16()
        );
        return Err(error);
    }

    let response_text = response.text().await.map_err(map_transport_error)?;
    trace!(target: "neuromance::wire", body = %response_text, "raw API response");

    serde_json::from_str(&response_text).map_err(ClientError::SerializationError)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    #![allow(clippy::panic)]

    use super::*;
    use reqwest::StatusCode;
    use reqwest::header::{HeaderMap, HeaderValue};

    fn no_headers() -> HeaderMap {
        HeaderMap::new()
    }

    fn retry_after(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            HeaderValue::from_str(value).expect("valid header value"),
        );
        headers
    }

    #[test]
    fn structured_body_uses_error_message() {
        let err = map_http_error(
            StatusCode::BAD_REQUEST,
            &no_headers(),
            r#"{"error":{"message":"bad tool schema"}}"#,
        );
        assert!(matches!(err, ClientError::RequestError(m) if m == "bad tool schema"));
    }

    #[test]
    fn non_json_body_is_used_verbatim() {
        let err = map_http_error(StatusCode::BAD_REQUEST, &no_headers(), "upstream exploded");
        assert!(matches!(err, ClientError::RequestError(m) if m == "upstream exploded"));
    }

    #[test]
    fn empty_body_falls_back_to_status_line() {
        let err = map_http_error(StatusCode::BAD_REQUEST, &no_headers(), "");
        assert!(matches!(err, ClientError::RequestError(m) if m == "HTTP 400 Bad Request"));
    }

    #[test]
    fn maps_401_to_authentication_error() {
        let err = map_http_error(StatusCode::UNAUTHORIZED, &no_headers(), "");
        assert!(matches!(err, ClientError::AuthenticationError(_)));
    }

    #[test]
    fn maps_429_to_rate_limit_error() {
        let err = map_http_error(StatusCode::TOO_MANY_REQUESTS, &no_headers(), "");
        assert!(matches!(
            err,
            ClientError::RateLimitError {
                retry_after: None,
                ..
            }
        ));
    }

    #[test]
    fn rate_limit_error_keeps_the_provider_message() {
        let err = map_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            &no_headers(),
            r#"{"error":{"message":"Provider returned error"}}"#,
        );
        match err {
            ClientError::RateLimitError { message, .. } => {
                assert_eq!(message, "Provider returned error");
            }
            other => panic!("expected RateLimitError, got {other:?}"),
        }
    }

    #[test]
    fn rate_limit_error_falls_back_to_status_line_on_empty_body() {
        let err = map_http_error(StatusCode::TOO_MANY_REQUESTS, &no_headers(), "");
        match err {
            ClientError::RateLimitError { message, .. } => {
                assert_eq!(message, "HTTP 429 Too Many Requests");
            }
            other => panic!("expected RateLimitError, got {other:?}"),
        }
    }

    #[test]
    fn rate_limit_error_reads_retry_after_seconds() {
        let err = map_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            &retry_after("12"),
            "slow down",
        );
        assert_eq!(err.retry_after(), Some(Duration::from_secs(12)));
    }

    #[test]
    fn retry_after_http_date_form_is_ignored() {
        let headers = retry_after("Wed, 21 Oct 2026 07:28:00 GMT");
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn retry_after_is_only_read_for_rate_limits() {
        let err = map_http_error(StatusCode::SERVICE_UNAVAILABLE, &retry_after("5"), "down");
        assert_eq!(err.retry_after(), None);
    }

    #[test]
    fn maps_all_5xx_to_service_unavailable() {
        for code in [500u16, 503, 529, 599] {
            let status = StatusCode::from_u16(code).expect("valid status");
            let err = map_http_error(status, &no_headers(), "overloaded");
            assert!(
                matches!(err, ClientError::ServiceUnavailable(_)),
                "status {code} should map to ServiceUnavailable"
            );
            assert!(err.is_retryable(), "status {code} should be retryable");
        }
    }

    #[test]
    fn maps_other_4xx_to_request_error() {
        for code in [400u16, 403, 418] {
            let status = StatusCode::from_u16(code).expect("valid status");
            let err = map_http_error(status, &no_headers(), "nope");
            assert!(
                matches!(err, ClientError::RequestError(_)),
                "status {code} should map to RequestError"
            );
        }
    }

    #[test]
    fn provider_overload_codes_are_retryable() {
        for code in ["overloaded_error", "api_error", "server_error"] {
            let err = classify_provider_error(code, None, "upstream is busy".to_string());
            assert!(
                matches!(err, ClientError::ServiceUnavailable(ref m) if m == "upstream is busy"),
                "{code} should map to ServiceUnavailable, got {err:?}"
            );
            assert!(err.is_retryable(), "{code} should be retryable");
        }
    }

    #[test]
    fn provider_rate_limit_codes_are_retryable() {
        for code in ["rate_limit_error", "rate_limit_exceeded"] {
            let err = classify_provider_error(code, None, "slow down".to_string());
            assert!(
                matches!(err, ClientError::RateLimitError { ref message, retry_after: None }
                    if message == "slow down"),
                "{code} should map to RateLimitError, got {err:?}"
            );
            assert!(err.is_retryable(), "{code} should be retryable");
        }
    }

    #[test]
    fn provider_authentication_code_is_terminal() {
        let err = classify_provider_error("authentication_error", None, "bad key".to_string());
        assert!(matches!(err, ClientError::AuthenticationError(ref m) if m == "bad key"));
        assert!(!err.is_retryable());
    }

    #[test]
    fn unrecognised_provider_code_stays_terminal() {
        let err = classify_provider_error("invalid_request_error", None, "bad tool".to_string());
        assert!(matches!(err, ClientError::RequestError(ref m) if m == "bad tool"));
        assert!(!err.is_retryable());
    }

    /// `OpenAI` carries the specific failure in `code` under a generic `type`.
    #[test]
    fn code_classifies_when_error_type_is_generic() {
        let err = classify_provider_error(
            "invalid_request_error",
            Some("rate_limit_exceeded"),
            "quota".to_string(),
        );
        assert!(matches!(err, ClientError::RateLimitError { .. }));
        assert!(err.is_retryable());
    }

    /// A recognised `type` wins over a `code` that would classify differently.
    #[test]
    fn error_type_wins_over_code_when_both_are_recognised() {
        let err = classify_provider_error(
            "overloaded_error",
            Some("authentication_error"),
            "busy".to_string(),
        );
        assert!(matches!(err, ClientError::ServiceUnavailable(_)));
    }

    /// Build the error shape `reqwest-retry` produces once retries are spent.
    /// A raw timeout — no middleware in the way — is recognised directly.
    #[tokio::test]
    async fn unwrapped_timeout_maps_to_timeout_error() {
        let err = map_middleware_error(reqwest_middleware::Error::Reqwest(timeout_error().await));
        assert!(matches!(err, ClientError::TimeoutError), "got {err:?}");
    }

    /// Pins the chain-walk against `reqwest-retry`'s wrapping, in both the
    /// retried and never-retried shapes. If either stops nesting a
    /// `reqwest_middleware::Error`, this fails loudly instead of silently
    /// degrading every timeout back to `MiddlewareError`.
    #[tokio::test]
    async fn timeout_survives_both_retry_wrappers() {
        let shapes = [
            reqwest_retry::RetryError::WithRetries {
                retries: 3,
                err: reqwest_middleware::Error::Reqwest(timeout_error().await),
            },
            reqwest_retry::RetryError::Error(reqwest_middleware::Error::Reqwest(
                timeout_error().await,
            )),
        ];

        for shape in shapes {
            let label = shape.to_string();
            let err = map_middleware_error(reqwest_middleware::Error::Middleware(shape.into()));
            assert!(
                matches!(err, ClientError::TimeoutError),
                "'{label}' should unwrap to TimeoutError, got {err:?}"
            );
        }
    }

    /// A middleware failure that is not a timeout keeps its own variant.
    #[test]
    fn non_timeout_middleware_failure_stays_a_middleware_error() {
        let err = map_middleware_error(reqwest_middleware::Error::Middleware(anyhow::anyhow!(
            "signing middleware failed"
        )));
        assert!(
            matches!(err, ClientError::MiddlewareError(_)),
            "got {err:?}"
        );
        assert!(err.is_retryable());
    }

    /// Produce a genuine `reqwest::Error` whose `is_timeout()` is true.
    ///
    /// `reqwest::Error` has no public constructor, so the only way to get one is
    /// to time a real request out against a socket that never answers.
    async fn timeout_error() -> reqwest::Error {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");

        reqwest::Client::builder()
            .timeout(Duration::from_millis(50))
            .build()
            .expect("build client")
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect_err("a socket that never accepts must time out")
    }
}
