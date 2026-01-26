//! Proxy-aware HTTP client for tools.
//!
//! This module provides an HTTP client that routes requests through a tokenizer proxy,
//! allowing tools to make authenticated requests without having access to real credentials.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use reqwest::{Client, Method, Request, Response};
use secrecy::{ExposeSecret, SecretString};

/// Default connection timeout (30 seconds).
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default total request timeout (60 seconds).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Configuration for a tokenizer proxy used by tools.
///
/// Similar to [`neuromance_common::ProxyConfig`] but designed for use in tool implementations
/// where the sealed token is provided directly rather than via a Config struct.
///
/// # Examples
///
/// ```
/// use neuromance_tools::proxy::ToolProxyConfig;
/// use secrecy::SecretString;
///
/// let config = ToolProxyConfig {
///     proxy_url: "http://tokenizer.internal:8080".to_string(),
///     token_header: "X-Tokenizer-Token".to_string(),
///     sealed_token: SecretString::new("sealed.abc123xyz".to_string().into()),
///     target_host_header: Some("X-Target-Host".to_string()),
///     connect_timeout: None,  // Uses default (30s)
///     timeout: None,          // Uses default (60s)
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ToolProxyConfig {
    /// URL of the tokenizer proxy.
    pub proxy_url: String,
    /// Header name for the sealed token.
    pub token_header: String,
    /// The sealed token value (stored securely).
    pub sealed_token: SecretString,
    /// Optional header name for the target host.
    pub target_host_header: Option<String>,
    /// Connection timeout (defaults to 30 seconds).
    pub connect_timeout: Option<Duration>,
    /// Total request timeout (defaults to 60 seconds).
    pub timeout: Option<Duration>,
}

/// HTTP client that routes requests through a tokenizer proxy.
///
/// This client rewrites URLs to route through the proxy and adds the necessary
/// headers for the proxy to inject real credentials before forwarding to the
/// target API.
///
/// The client is cheaply cloneable and can be shared across multiple tasks.
///
/// # Examples
///
/// ```no_run
/// use neuromance_tools::proxy::{ProxyAwareClient, ToolProxyConfig};
/// use secrecy::SecretString;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = ToolProxyConfig {
///     proxy_url: "http://tokenizer.internal:8080".to_string(),
///     token_header: "X-Tokenizer-Token".to_string(),
///     sealed_token: SecretString::new("sealed.abc123xyz".to_string().into()),
///     target_host_header: Some("X-Target-Host".to_string()),
///     connect_timeout: None,
///     timeout: None,
/// };
///
/// let client = ProxyAwareClient::new(config)?;
///
/// // Request to https://api.github.com/repos/owner/repo gets rewritten to:
/// // http://tokenizer.internal:8080/repos/owner/repo
/// // with headers: X-Tokenizer-Token: sealed.abc123xyz, X-Target-Host: api.github.com
/// let response = client.get("https://api.github.com/repos/owner/repo").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ProxyAwareClient {
    inner: Client,
    config: Arc<ToolProxyConfig>,
}

impl ProxyAwareClient {
    /// Creates a new proxy-aware HTTP client.
    ///
    /// Uses configured timeouts or defaults (30s connect, 60s total).
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the tokenizer proxy
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying HTTP client fails to build.
    pub fn new(config: ToolProxyConfig) -> Result<Self> {
        let connect_timeout = config.connect_timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT);
        let timeout = config.timeout.unwrap_or(DEFAULT_TIMEOUT);

        let client = Client::builder()
            .connect_timeout(connect_timeout)
            .timeout(timeout)
            .build()?;

        Ok(Self {
            inner: client,
            config: Arc::new(config),
        })
    }

    /// Makes a GET request through the proxy.
    ///
    /// # Arguments
    ///
    /// * `url` - The target URL (will be rewritten to route through proxy)
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn get(&self, url: &str) -> Result<Response> {
        let request = self.build_proxied_request(Method::GET, url, None)?;
        Ok(self.inner.execute(request).await?)
    }

    /// Makes a POST request through the proxy.
    ///
    /// # Arguments
    ///
    /// * `url` - The target URL (will be rewritten to route through proxy)
    /// * `body` - The request body as a string
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn post(&self, url: &str, body: String) -> Result<Response> {
        let request = self.build_proxied_request(Method::POST, url, Some(body))?;
        Ok(self.inner.execute(request).await?)
    }

    /// Makes a PUT request through the proxy.
    ///
    /// # Arguments
    ///
    /// * `url` - The target URL (will be rewritten to route through proxy)
    /// * `body` - The request body as a string
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn put(&self, url: &str, body: String) -> Result<Response> {
        let request = self.build_proxied_request(Method::PUT, url, Some(body))?;
        Ok(self.inner.execute(request).await?)
    }

    /// Makes a DELETE request through the proxy.
    ///
    /// # Arguments
    ///
    /// * `url` - The target URL (will be rewritten to route through proxy)
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn delete(&self, url: &str) -> Result<Response> {
        let request = self.build_proxied_request(Method::DELETE, url, None)?;
        Ok(self.inner.execute(request).await?)
    }

    /// Makes a PATCH request through the proxy.
    ///
    /// # Arguments
    ///
    /// * `url` - The target URL (will be rewritten to route through proxy)
    /// * `body` - The request body as a string
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn patch(&self, url: &str, body: String) -> Result<Response> {
        let request = self.build_proxied_request(Method::PATCH, url, Some(body))?;
        Ok(self.inner.execute(request).await?)
    }

    /// Builds a request that routes through the proxy.
    ///
    /// This method:
    /// 1. Rewrites the URL to route through the proxy
    /// 2. Adds the sealed token header
    /// 3. Optionally adds the target host header
    /// 4. Sets `Content-Type: application/json` only when a body is present
    fn build_proxied_request(
        &self,
        method: Method,
        original_url: &str,
        body: Option<String>,
    ) -> Result<Request> {
        let parsed_url: url::Url = original_url.parse()?;

        // Rewrite URL to go through proxy, preserving path and query
        let proxy_url = format!(
            "{}{}{}",
            self.config.proxy_url.trim_end_matches('/'),
            parsed_url.path(),
            parsed_url
                .query()
                .map(|q| format!("?{q}"))
                .unwrap_or_default()
        );

        let mut builder = self.inner.request(method, &proxy_url).header(
            &self.config.token_header,
            self.config.sealed_token.expose_secret(),
        );

        // Add target host header if configured
        if let Some(ref host_header) = self.config.target_host_header
            && let Some(host) = parsed_url.host_str()
        {
            builder = builder.header(host_header, host);
        }

        // Only set Content-Type when there's a body
        if let Some(body) = body {
            builder = builder
                .header("Content-Type", "application/json")
                .body(body);
        }

        Ok(builder.build()?)
    }
}

impl std::fmt::Debug for ProxyAwareClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProxyAwareClient")
            .field("proxy_url", &self.config.proxy_url)
            .field("token_header", &self.config.token_header)
            .field("sealed_token", &"[REDACTED]")
            .field("target_host_header", &self.config.target_host_header)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_url_rewrite_preserves_path() {
        let config = ToolProxyConfig {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Token".to_string(),
            sealed_token: SecretString::new("test-token".to_string().into()),
            target_host_header: Some("X-Target".to_string()),
            connect_timeout: None,
            timeout: None,
        };

        let client = ProxyAwareClient::new(config).unwrap();
        let request = client
            .build_proxied_request(Method::GET, "https://api.example.com/v1/users/123", None)
            .unwrap();

        assert_eq!(
            request.url().as_str(),
            "http://proxy.local:8080/v1/users/123"
        );
    }

    #[test]
    fn test_url_rewrite_preserves_query() {
        let config = ToolProxyConfig {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Token".to_string(),
            sealed_token: SecretString::new("test-token".to_string().into()),
            target_host_header: None,
            connect_timeout: None,
            timeout: None,
        };

        let client = ProxyAwareClient::new(config).unwrap();
        let request = client
            .build_proxied_request(
                Method::GET,
                "https://api.example.com/search?q=test&page=1",
                None,
            )
            .unwrap();

        assert_eq!(
            request.url().as_str(),
            "http://proxy.local:8080/search?q=test&page=1"
        );
    }

    #[test]
    fn test_token_header_added() {
        let config = ToolProxyConfig {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Tokenizer-Token".to_string(),
            sealed_token: SecretString::new("sealed.abc123".to_string().into()),
            target_host_header: None,
            connect_timeout: None,
            timeout: None,
        };

        let client = ProxyAwareClient::new(config).unwrap();
        let request = client
            .build_proxied_request(Method::GET, "https://api.example.com/test", None)
            .unwrap();

        assert_eq!(
            request
                .headers()
                .get("X-Tokenizer-Token")
                .map(|v| v.to_str().unwrap()),
            Some("sealed.abc123")
        );
    }

    #[test]
    fn test_target_host_header_added() {
        let config = ToolProxyConfig {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Token".to_string(),
            sealed_token: SecretString::new("token".to_string().into()),
            target_host_header: Some("X-Target-Host".to_string()),
            connect_timeout: None,
            timeout: None,
        };

        let client = ProxyAwareClient::new(config).unwrap();
        let request = client
            .build_proxied_request(Method::GET, "https://api.github.com/repos", None)
            .unwrap();

        assert_eq!(
            request
                .headers()
                .get("X-Target-Host")
                .map(|v| v.to_str().unwrap()),
            Some("api.github.com")
        );
    }

    #[test]
    fn test_debug_redacts_token() {
        let config = ToolProxyConfig {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Token".to_string(),
            sealed_token: SecretString::new("super-secret".to_string().into()),
            target_host_header: None,
            connect_timeout: None,
            timeout: None,
        };

        let client = ProxyAwareClient::new(config).unwrap();
        let debug_str = format!("{client:?}");

        assert!(debug_str.contains("[REDACTED]"));
        assert!(!debug_str.contains("super-secret"));
    }

    #[test]
    fn test_content_type_only_with_body() {
        let config = ToolProxyConfig {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Token".to_string(),
            sealed_token: SecretString::new("token".to_string().into()),
            target_host_header: None,
            connect_timeout: None,
            timeout: None,
        };

        let client = ProxyAwareClient::new(config).unwrap();

        // GET without body should not have Content-Type
        let get_request = client
            .build_proxied_request(Method::GET, "https://api.example.com/test", None)
            .unwrap();
        assert!(get_request.headers().get("Content-Type").is_none());

        // DELETE without body should not have Content-Type
        let delete_request = client
            .build_proxied_request(Method::DELETE, "https://api.example.com/test", None)
            .unwrap();
        assert!(delete_request.headers().get("Content-Type").is_none());

        // POST with body should have Content-Type
        let post_request = client
            .build_proxied_request(
                Method::POST,
                "https://api.example.com/test",
                Some(r#"{"key": "value"}"#.to_string()),
            )
            .unwrap();
        assert_eq!(
            post_request
                .headers()
                .get("Content-Type")
                .map(|v| v.to_str().unwrap()),
            Some("application/json")
        );
    }

    #[test]
    fn test_client_is_clone() {
        let config = ToolProxyConfig {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Token".to_string(),
            sealed_token: SecretString::new("token".to_string().into()),
            target_host_header: None,
            connect_timeout: None,
            timeout: None,
        };

        let client = ProxyAwareClient::new(config).unwrap();
        // Clone the client and use both to verify Clone works
        let cloned = client.clone();
        assert_eq!(cloned.config.proxy_url, client.config.proxy_url);
    }

    #[test]
    fn test_custom_timeouts() {
        let config = ToolProxyConfig {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Token".to_string(),
            sealed_token: SecretString::new("token".to_string().into()),
            target_host_header: None,
            connect_timeout: Some(Duration::from_secs(10)),
            timeout: Some(Duration::from_secs(120)),
        };

        // Just verify it builds successfully with custom timeouts
        let _client = ProxyAwareClient::new(config).unwrap();
    }
}
