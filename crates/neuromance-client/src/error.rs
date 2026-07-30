//! Error types for the client library.

use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;

/// Error response from the API.
///
/// Wraps the detailed error information returned by LLM providers.
#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    /// The error detail object from the API.
    pub error: ErrorDetail,
}

/// Detailed error information from the API.
///
/// Contains the specific error message returned by the provider.
#[derive(Debug, Deserialize)]
pub struct ErrorDetail {
    /// The error message text describing what went wrong.
    pub message: String,
}

/// Errors that can occur when interacting with LLM APIs.
///
/// This enum covers all error conditions from network failures to API-specific
/// errors like rate limiting and content filtering.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientError {
    /// Network or HTTP request failure.
    ///
    /// Indicates issues like DNS resolution, connection failures, or socket errors.
    /// These errors are typically retryable.
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// Middleware layer error.
    ///
    /// Errors from request/response middleware such as retry logic or logging.
    #[error("Middleware error: {0}")]
    MiddlewareError(#[from] reqwest_middleware::Error),

    /// JSON serialization or deserialization error.
    ///
    /// Occurs when request/response JSON cannot be properly encoded or decoded.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// API authentication failure (HTTP 401).
    ///
    /// The API key is missing, invalid, or revoked. Check your credentials.
    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    /// SSE event source error.
    #[error("EventSource error: {0}")]
    EventSourceError(#[from] reqwest_eventsource::Error),

    /// Rate limit exceeded (HTTP 429).
    ///
    /// Too many requests sent in a given time period. Wait and retry.
    ///
    /// `message` carries the provider's own explanation verbatim — a 429 can mean
    /// an exhausted credit balance, a per-key request rate, or upstream provider
    /// capacity, and only the body distinguishes them.
    #[error("Rate limit exceeded: {message} (retry after {retry_after:?})")]
    RateLimitError {
        /// The provider's error message, or an `HTTP {status}` fallback when the
        /// response body is empty.
        message: String,
        /// Suggested wait time before retrying, from the `Retry-After` header.
        retry_after: Option<Duration>,
    },

    /// API request rejected by the server.
    ///
    /// The server returned an error status (e.g., 400, 403) that isn't covered
    /// by a more specific variant. Not retryable.
    #[error("Request error: {0}")]
    RequestError(String),

    /// Client configuration issue.
    ///
    /// Invalid base URL, missing required fields, or incompatible settings.
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Request timeout.
    ///
    /// The request took longer than the configured timeout. Consider increasing
    /// the timeout or reducing request complexity.
    #[error("Timeout error")]
    TimeoutError,

    /// Malformed request.
    ///
    /// The request structure is invalid or missing required parameters.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Unexpected or malformed API response.
    ///
    /// The API returned data that doesn't match the expected format.
    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    /// Tools requested but not supported by this model.
    ///
    /// The model or provider doesn't support function calling.
    #[error("Tool execution not supported")]
    ToolsNotSupported,

    /// Streaming requested but not supported.
    ///
    /// The model or provider doesn't support streaming responses.
    #[error("Streaming not supported")]
    StreamingNotSupported,

    /// Token limit exceeded for this model.
    ///
    /// The input plus requested output exceeds the model's context window.
    #[error("Context length exceeded: {current_tokens} > {max_tokens}")]
    ContextLengthExceeded {
        /// Current number of tokens in the request.
        current_tokens: usize,
        /// Maximum tokens allowed by the model.
        max_tokens: usize,
    },

    /// Content blocked by safety filter.
    ///
    /// The content violates the provider's usage policies.
    #[error("Content filtered: {reason}")]
    ContentFiltered {
        /// Reason for filtering (e.g., "violence", "hate speech").
        reason: String,
    },

    /// API service unavailable (5xx errors).
    ///
    /// The provider's servers are experiencing issues. Retry with backoff.
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Temperature parameter out of valid range.
    ///
    /// Temperature must be between 0.0 and 2.0.
    #[error("Temperature must be between 0.0 & 2.0")]
    InvalidTemperature,

    /// `top_p` parameter out of valid range.
    ///
    /// `top_p` must be between 0.0 and 1.0.
    #[error("TopP must be between 0.0 & 1.0")]
    InvalidTopP,

    /// `frequency_penalty` parameter out of valid range.
    ///
    /// `frequency_penalty` must be between -2.0 and 2.0.
    #[error("FrequencyPenalty must be between -2.0 & 2.0")]
    InvalidFrequencyPenalty,

    /// Embedding operation error.
    ///
    /// An error occurred during embedding generation.
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Embeddings not supported by this provider.
    ///
    /// The provider or model doesn't support embedding generation.
    #[error("Embeddings not supported")]
    EmbeddingsNotSupported,
}

impl ClientError {
    /// Check if this error is potentially retryable.
    ///
    /// Returns `true` for network errors, timeouts, rate limits, and service unavailable errors.
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::NetworkError(_)
                | Self::MiddlewareError(_)
                | Self::TimeoutError
                | Self::RateLimitError { .. }
                | Self::ServiceUnavailable(_)
        )
    }

    /// Check if this is an authentication error.
    pub const fn is_authentication_error(&self) -> bool {
        matches!(self, Self::AuthenticationError(_))
    }

    /// Check if this is a rate limit error.
    pub const fn is_rate_limit_error(&self) -> bool {
        matches!(self, Self::RateLimitError { .. })
    }

    /// A stable, low-cardinality slug naming why the request failed.
    ///
    /// Intended for the `reason` label on failure metrics, so the value set is
    /// fixed and small: several variants deliberately share a slug (the three
    /// parameter-range errors are all `invalid_parameter`). Never derive this
    /// from an error message — the inner strings carry provider text and would
    /// blow up label cardinality.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_client::ClientError;
    ///
    /// assert_eq!(ClientError::TimeoutError.reason(), "timeout");
    /// assert_eq!(ClientError::InvalidTopP.reason(), "invalid_parameter");
    /// ```
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::NetworkError(_) => "network",
            Self::MiddlewareError(_) => "middleware",
            Self::SerializationError(_) => "serialization",
            Self::AuthenticationError(_) => "auth",
            Self::EventSourceError(_) => "stream_aborted",
            Self::RateLimitError { .. } => "rate_limited",
            Self::RequestError(_) => "provider_error",
            Self::ConfigurationError(_) => "configuration",
            Self::TimeoutError => "timeout",
            Self::InvalidRequest(_) => "invalid_request",
            Self::InvalidResponse(_) => "invalid_response",
            Self::ToolsNotSupported
            | Self::StreamingNotSupported
            | Self::EmbeddingsNotSupported => "unsupported",
            Self::ContextLengthExceeded { .. } => "context_length_exceeded",
            Self::ContentFiltered { .. } => "content_filtered",
            Self::ServiceUnavailable(_) => "service_unavailable",
            Self::InvalidTemperature | Self::InvalidTopP | Self::InvalidFrequencyPenalty => {
                "invalid_parameter"
            }
            Self::EmbeddingError(_) => "embedding",
        }
    }

    /// Get the retry-after duration if this is a rate limit error.
    ///
    /// Returns the suggested wait time before retrying the request.
    pub const fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimitError { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    /// Every variant of [`ClientError`], paired with whether a caller should
    /// retry it and the metric slug it reports.
    ///
    /// Retryability is what drives `Core::stream_with_retry` and the transport
    /// middleware, so a variant landing on the wrong side of this line either
    /// hammers a provider that will never succeed or gives up on one that would.
    ///
    /// The slug reaches Prometheus as a label value, so it is pinned here: a
    /// variant that starts reporting a different slug silently splits a metric
    /// series in two.
    fn error_table(network_error: reqwest::Error) -> Vec<(ClientError, bool, &'static str)> {
        let bad_json = serde_json::from_str::<i32>("nope").expect_err("not an integer");
        vec![
            (ClientError::NetworkError(network_error), true, "network"),
            (
                ClientError::MiddlewareError(reqwest_middleware::Error::Middleware(
                    anyhow::anyhow!("middleware blew up"),
                )),
                true,
                "middleware",
            ),
            (ClientError::TimeoutError, true, "timeout"),
            (
                ClientError::RateLimitError {
                    message: "slow down".to_string(),
                    retry_after: Some(Duration::from_secs(3)),
                },
                true,
                "rate_limited",
            ),
            (
                ClientError::ServiceUnavailable("busy".to_string()),
                true,
                "service_unavailable",
            ),
            (
                ClientError::SerializationError(bad_json),
                false,
                "serialization",
            ),
            (
                ClientError::AuthenticationError("bad key".to_string()),
                false,
                "auth",
            ),
            (
                ClientError::EventSourceError(reqwest_eventsource::Error::InvalidLastEventId(
                    "\n".to_string(),
                )),
                false,
                "stream_aborted",
            ),
            (
                ClientError::RequestError("bad tool".to_string()),
                false,
                "provider_error",
            ),
            (
                ClientError::ConfigurationError("no key".to_string()),
                false,
                "configuration",
            ),
            (
                ClientError::InvalidRequest("no messages".to_string()),
                false,
                "invalid_request",
            ),
            (
                ClientError::InvalidResponse("not SSE".to_string()),
                false,
                "invalid_response",
            ),
            (ClientError::ToolsNotSupported, false, "unsupported"),
            (ClientError::StreamingNotSupported, false, "unsupported"),
            (
                ClientError::ContextLengthExceeded {
                    current_tokens: 9,
                    max_tokens: 8,
                },
                false,
                "context_length_exceeded",
            ),
            (
                ClientError::ContentFiltered {
                    reason: "violence".to_string(),
                },
                false,
                "content_filtered",
            ),
            (ClientError::InvalidTemperature, false, "invalid_parameter"),
            (ClientError::InvalidTopP, false, "invalid_parameter"),
            (
                ClientError::InvalidFrequencyPenalty,
                false,
                "invalid_parameter",
            ),
            (
                ClientError::EmbeddingError("bad base64".to_string()),
                false,
                "embedding",
            ),
            (ClientError::EmbeddingsNotSupported, false, "unsupported"),
        ]
    }

    /// Fails to compile when a variant is added without a retryability decision.
    ///
    /// `#[non_exhaustive]` does not apply inside the defining crate, so this
    /// match really is exhaustive — the census is the thing that keeps
    /// [`retryability_table`] from silently going out of date.
    #[expect(
        clippy::match_same_arms,
        reason = "one arm per variant is the point; collapsing them defeats the census"
    )]
    fn assert_variant_is_in_the_table(error: &ClientError) {
        match error {
            ClientError::NetworkError(_) => (),
            ClientError::MiddlewareError(_) => (),
            ClientError::SerializationError(_) => (),
            ClientError::AuthenticationError(_) => (),
            ClientError::EventSourceError(_) => (),
            ClientError::RateLimitError { .. } => (),
            ClientError::RequestError(_) => (),
            ClientError::ConfigurationError(_) => (),
            ClientError::TimeoutError => (),
            ClientError::InvalidRequest(_) => (),
            ClientError::InvalidResponse(_) => (),
            ClientError::ToolsNotSupported => (),
            ClientError::StreamingNotSupported => (),
            ClientError::ContextLengthExceeded { .. } => (),
            ClientError::ContentFiltered { .. } => (),
            ClientError::ServiceUnavailable(_) => (),
            ClientError::InvalidTemperature => (),
            ClientError::InvalidTopP => (),
            ClientError::InvalidFrequencyPenalty => (),
            ClientError::EmbeddingError(_) => (),
            ClientError::EmbeddingsNotSupported => (),
        }
    }

    #[tokio::test]
    async fn test_every_variant_has_the_expected_retryability_and_reason() {
        let table = error_table(connection_refused().await);
        assert_eq!(
            table.len(),
            21,
            "a variant was added or removed without updating the table"
        );

        for (error, retryable, reason) in table {
            assert_variant_is_in_the_table(&error);
            assert_eq!(
                error.is_retryable(),
                retryable,
                "{error:?} should{} be retryable",
                if retryable { "" } else { " not" }
            );
            assert_eq!(error.reason(), reason, "wrong metric slug for {error:?}");
        }
    }

    /// The slug set must stay small enough to be a Prometheus label. Distinct
    /// variants sharing a slug is intentional; the count is the guard rail.
    #[tokio::test]
    async fn test_reason_slugs_stay_low_cardinality() {
        let slugs: std::collections::BTreeSet<_> = error_table(connection_refused().await)
            .iter()
            .map(|(error, _, _)| error.reason())
            .collect();

        assert!(
            slugs.len() <= 20,
            "reason() grew to {} distinct slugs: {slugs:?}",
            slugs.len()
        );
        assert!(
            slugs.iter().all(|s| !s.is_empty()),
            "an empty slug would produce an unlabelled metric series"
        );
    }

    #[test]
    fn test_retry_after_is_only_carried_by_rate_limits() {
        let rate_limited = ClientError::RateLimitError {
            message: "slow down".to_string(),
            retry_after: Some(Duration::from_secs(7)),
        };
        assert_eq!(rate_limited.retry_after(), Some(Duration::from_secs(7)));
        assert!(rate_limited.is_rate_limit_error());

        // A 503 is just as retryable but carries no schedule, so callers must
        // fall back to their own backoff rather than waiting forever.
        let unavailable = ClientError::ServiceUnavailable("busy".to_string());
        assert_eq!(unavailable.retry_after(), None);
        assert!(!unavailable.is_rate_limit_error());
    }

    #[test]
    fn test_only_authentication_errors_report_as_such() {
        assert!(ClientError::AuthenticationError("bad key".to_string()).is_authentication_error());
        assert!(!ClientError::RequestError("bad key".to_string()).is_authentication_error());
    }

    /// Produce a real `reqwest::Error` by dialing a port nothing is listening on.
    async fn connection_refused() -> reqwest::Error {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        drop(listener);

        reqwest::Client::new()
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect_err("nothing is listening on a closed port")
    }
}
