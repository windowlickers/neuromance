use neuromance_common::client::ProxyConfig;
use secrecy::{ExposeSecret, SecretString};

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
