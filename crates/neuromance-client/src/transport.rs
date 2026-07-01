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

/// Map an HTTP error response (status + body) to a typed [`ClientError`].
///
/// Tries to parse `body` as a structured [`ErrorResponse`], falling back to the
/// raw body text, then to `HTTP {status}` when the body is empty. The status
/// code selects the variant. This is the single canonical mapping shared by the
/// streaming ([`crate::streaming`]) and non-streaming ([`send_json`]) paths, so
/// both agree that any `5xx` is a retryable [`ClientError::ServiceUnavailable`].
#[must_use]
pub fn map_http_error(status: reqwest::StatusCode, body: &str) -> ClientError {
    let message = match serde_json::from_str::<ErrorResponse>(body) {
        Ok(parsed) => parsed.error.message,
        Err(_) if body.is_empty() => format!("HTTP {status}"),
        Err(_) => body.to_string(),
    };

    match status.as_u16() {
        401 => ClientError::AuthenticationError(message),
        429 => ClientError::RateLimitError { retry_after: None },
        500..=599 => ClientError::ServiceUnavailable(message),
        _ => ClientError::RequestError(message),
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
/// - [`ClientError::MiddlewareError`] if the request fails to send.
/// - [`ClientError::NetworkError`] if a response body cannot be read.
/// - The variant selected by [`map_http_error`] on a non-success status.
/// - [`ClientError::SerializationError`] if a success body is not valid `T`.
pub async fn send_json<T: DeserializeOwned>(
    request: reqwest_middleware::RequestBuilder,
) -> Result<T, ClientError> {
    let response = request.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.map_err(|e| {
            warn!("Failed to read error response body: {e}");
            ClientError::NetworkError(e)
        })?;
        let error = map_http_error(status, &error_text);
        error!(
            "API request failed with status {}: {error}",
            status.as_u16()
        );
        return Err(error);
    }

    let response_text = response.text().await?;
    trace!(target: "neuromance::wire", body = %response_text, "raw API response");

    serde_json::from_str(&response_text).map_err(ClientError::SerializationError)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use reqwest::StatusCode;

    #[test]
    fn structured_body_uses_error_message() {
        let err = map_http_error(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"bad tool schema"}}"#,
        );
        assert!(matches!(err, ClientError::RequestError(m) if m == "bad tool schema"));
    }

    #[test]
    fn non_json_body_is_used_verbatim() {
        let err = map_http_error(StatusCode::BAD_REQUEST, "upstream exploded");
        assert!(matches!(err, ClientError::RequestError(m) if m == "upstream exploded"));
    }

    #[test]
    fn empty_body_falls_back_to_status_line() {
        let err = map_http_error(StatusCode::BAD_REQUEST, "");
        assert!(matches!(err, ClientError::RequestError(m) if m == "HTTP 400 Bad Request"));
    }

    #[test]
    fn maps_401_to_authentication_error() {
        let err = map_http_error(StatusCode::UNAUTHORIZED, "");
        assert!(matches!(err, ClientError::AuthenticationError(_)));
    }

    #[test]
    fn maps_429_to_rate_limit_error() {
        let err = map_http_error(StatusCode::TOO_MANY_REQUESTS, "");
        assert!(matches!(
            err,
            ClientError::RateLimitError { retry_after: None }
        ));
    }

    #[test]
    fn maps_all_5xx_to_service_unavailable() {
        for code in [500u16, 503, 529, 599] {
            let status = StatusCode::from_u16(code).expect("valid status");
            let err = map_http_error(status, "overloaded");
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
            let err = map_http_error(status, "nope");
            assert!(
                matches!(err, ClientError::RequestError(_)),
                "status {code} should map to RequestError"
            );
        }
    }
}
