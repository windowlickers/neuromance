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

/// Adds proxy headers to a request builder if proxy is configured.
pub fn add_proxy_headers<B: WithHeader>(
    mut builder: B,
    proxy_config: Option<&ProxyConfig>,
    api_key: &SecretString,
    target_host: &str,
) -> B {
    if let Some(proxy) = proxy_config {
        builder = builder.header(&proxy.token_header, api_key.expose_secret());
        if let Some(ref host_header) = proxy.target_host_header {
            builder = builder.header(host_header, target_host);
        }
    }
    builder
}
