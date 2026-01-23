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
    #[error("Rate limit exceeded: {retry_after:?}")]
    RateLimitError {
        /// Suggested wait time before retrying, if provided by the API.
        retry_after: Option<Duration>,
    },

    /// Model-specific error from the API.
    ///
    /// The model encountered an error during generation.
    #[error("Model error: {0}")]
    ModelError(String),

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

    /// Get the retry-after duration if this is a rate limit error.
    ///
    /// Returns the suggested wait time before retrying the request.
    pub const fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimitError { retry_after } => *retry_after,
            _ => None,
        }
    }
}
