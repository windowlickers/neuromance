use std::collections::HashMap;
use std::time::Duration;

use log::warn;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

/// Configuration for exponential backoff retry behavior.
///
/// This struct controls how failed requests are retried with increasing delays
/// between attempts. Supports optional jitter to avoid thundering herd problems.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use neuromance_common::client::RetryConfig;
///
/// // Conservative retry policy
/// let config = RetryConfig {
///     max_retries: 5,
///     initial_delay: Duration::from_millis(500),
///     max_delay: Duration::from_secs(60),
///     backoff_multiplier: 2.0,
///     jitter: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts before failing.
    pub max_retries: u32,
    /// Initial delay before the first retry attempt.
    pub initial_delay: Duration,
    /// Maximum delay between retry attempts (caps exponential growth).
    pub max_delay: Duration,
    /// Multiplier for exponential backoff (typically 2.0 for doubling).
    pub backoff_multiplier: f64,
    /// Whether to add random jitter to retry delays to prevent thundering herd.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Configuration for routing requests through a tokenizer proxy.
///
/// A tokenizer proxy intercepts requests and injects real credentials,
/// allowing agents to use sealed tokens instead of raw API keys.
/// This provides an additional security layer where real credentials
/// never leave the proxy.
///
/// # Examples
///
/// Using the validated constructor:
///
/// ```
/// use neuromance_common::ProxyConfig;
///
/// let proxy = ProxyConfig::new("http://tokenizer.internal:8080")
///     .expect("Invalid proxy URL");
/// ```
///
/// Or with all options:
///
/// ```
/// use neuromance_common::ProxyConfig;
///
/// let proxy = ProxyConfig::with_options(
///     "http://tokenizer.internal:8080",
///     "X-Tokenizer-Token",
///     Some("X-Target-Host"),
/// ).expect("Invalid proxy URL");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    /// URL of the tokenizer proxy (e.g., `"http://localhost:5000"`).
    pub proxy_url: String,

    /// Header name for the sealed token.
    ///
    /// Defaults to `"X-Tokenizer-Token"` when deserialized.
    #[serde(default = "default_token_header")]
    pub token_header: String,

    /// Optional header name for forwarding the original target host.
    ///
    /// When set, the proxy can use this to route requests to the correct
    /// backend API (e.g., `"X-Target-Host"` with value `"api.anthropic.com"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_host_header: Option<String>,
}

impl ProxyConfig {
    /// Creates a new proxy configuration with the given URL.
    ///
    /// Uses the default token header (`X-Tokenizer-Token`) and no target host header.
    ///
    /// # Arguments
    ///
    /// * `proxy_url` - URL of the tokenizer proxy
    ///
    /// # Errors
    ///
    /// Returns an error if the proxy URL is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::ProxyConfig;
    ///
    /// let proxy = ProxyConfig::new("http://localhost:8080").unwrap();
    /// assert_eq!(proxy.token_header, "X-Tokenizer-Token");
    /// ```
    pub fn new(proxy_url: impl Into<String>) -> anyhow::Result<Self> {
        let proxy_url = proxy_url.into();
        Self::validate_url(&proxy_url)?;
        Ok(Self {
            proxy_url,
            token_header: default_token_header(),
            target_host_header: None,
        })
    }

    /// Creates a new proxy configuration with all options.
    ///
    /// # Arguments
    ///
    /// * `proxy_url` - URL of the tokenizer proxy
    /// * `token_header` - Header name for the sealed token
    /// * `target_host_header` - Optional header name for the target host
    ///
    /// # Errors
    ///
    /// Returns an error if the proxy URL is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::ProxyConfig;
    ///
    /// let proxy = ProxyConfig::with_options(
    ///     "http://tokenizer.internal:8080",
    ///     "X-My-Token",
    ///     Some("X-Target-Host"),
    /// ).unwrap();
    /// ```
    pub fn with_options(
        proxy_url: impl Into<String>,
        token_header: impl Into<String>,
        target_host_header: Option<impl Into<String>>,
    ) -> anyhow::Result<Self> {
        let proxy_url = proxy_url.into();
        Self::validate_url(&proxy_url)?;
        Ok(Self {
            proxy_url,
            token_header: token_header.into(),
            target_host_header: target_host_header.map(Into::into),
        })
    }

    /// Validates this proxy configuration.
    ///
    /// Checks that the `proxy_url` is a valid URL with a scheme (http/https)
    /// and a host.
    ///
    /// # Errors
    ///
    /// Returns an error if the proxy URL is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::ProxyConfig;
    ///
    /// // Valid configuration
    /// let valid = ProxyConfig {
    ///     proxy_url: "http://localhost:8080".to_string(),
    ///     token_header: "X-Token".to_string(),
    ///     target_host_header: None,
    /// };
    /// assert!(valid.validate().is_ok());
    ///
    /// // Invalid configuration
    /// let invalid = ProxyConfig {
    ///     proxy_url: "not-a-valid-url".to_string(),
    ///     token_header: "X-Token".to_string(),
    ///     target_host_header: None,
    /// };
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> anyhow::Result<()> {
        Self::validate_url(&self.proxy_url)
    }

    /// Validates that a URL string is a valid proxy URL.
    fn validate_url(url: &str) -> anyhow::Result<()> {
        let parsed: url::Url = url
            .parse()
            .map_err(|e: url::ParseError| anyhow::anyhow!("Invalid proxy URL '{url}': {e}"))?;

        // Ensure the URL has a valid scheme
        let scheme = parsed.scheme();
        if scheme != "http" && scheme != "https" {
            anyhow::bail!(
                "Proxy URL must use http or https scheme, \
                 got '{scheme}' in '{url}'"
            );
        }

        // Warn about non-HTTPS proxy URLs
        if scheme == "http" {
            warn!(
                "Proxy URL '{url}' uses plain HTTP. \
                 Sealed tokens will be transmitted unencrypted. \
                 Consider using HTTPS in production."
            );
        }

        // Ensure the URL has a host
        if parsed.host_str().is_none() {
            anyhow::bail!("Proxy URL must have a host: '{url}'");
        }

        Ok(())
    }
}

fn default_token_header() -> String {
    "X-Tokenizer-Token".to_string()
}

/// Configuration for an LLM client.
///
/// This struct holds both connection details (API keys, URLs) and default
/// generation parameters that will be applied to all requests unless overridden.
///
/// # Security
///
/// The `api_key` field uses `SecretString` to prevent accidental logging or
/// display of sensitive credentials.
///
/// # Examples
///
/// ```
/// use neuromance_common::Config;
///
/// let config = Config::new("openai", "gpt-4")
///     .with_api_key("sk-...")
///     .with_temperature(0.7)
///     .with_max_tokens(1000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// The LLM provider name (e.g., "openai", "anthropic").
    pub provider: String,
    /// The default model identifier to use.
    pub model: String,
    /// Optional custom base URL for API requests.
    ///
    /// Override this for self-hosted deployments or custom endpoints.
    pub base_url: Option<String>,
    /// API key for authentication (stored securely).
    ///
    /// Will not be serialized to prevent accidental exposure.
    /// When using a tokenizer proxy, this should contain the sealed token
    /// instead of the real API key.
    #[serde(skip_serializing, default)]
    pub api_key: Option<SecretString>,
    /// Optional organization identifier.
    pub organization: Option<String>,
    /// Request timeout in seconds.
    pub timeout_seconds: Option<u64>,
    /// Configuration for retry behavior with exponential backoff.
    #[serde(skip)]
    pub retry_config: RetryConfig,
    /// Default sampling temperature (0.0 to 2.0).
    pub temperature: Option<f32>,
    /// Default maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Default nucleus sampling threshold (0.0 to 1.0).
    pub top_p: Option<f32>,
    /// Default frequency penalty (-2.0 to 2.0).
    pub frequency_penalty: Option<f32>,
    /// Default presence penalty (-2.0 to 2.0).
    pub presence_penalty: Option<f32>,
    /// Default stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Additional metadata to attach to all requests.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Tokenizer proxy configuration (optional).
    ///
    /// When configured, requests are routed through the proxy which
    /// injects real credentials. The `api_key` field should contain
    /// a sealed token instead of the real API key.
    ///
    /// Note: This field is serialized (unlike `api_key`) because it contains
    /// only URLs and header names, not secrets. The actual sealed token
    /// is stored in `api_key`.
    #[serde(default)]
    pub proxy: Option<ProxyConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "gpt-oss:20b".to_string(),
            base_url: None,
            api_key: None,
            organization: None,
            timeout_seconds: None,
            retry_config: RetryConfig::default(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            metadata: HashMap::new(),
            proxy: None,
        }
    }
}

impl Config {
    /// Creates a new configuration with the specified provider and model.
    ///
    /// All optional fields are initialized to their defaults.
    ///
    /// # Arguments
    ///
    /// * `provider` - The LLM provider name
    /// * `model` - The model identifier
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::Config;
    ///
    /// let config = Config::new("openai", "gpt-4");
    /// ```
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Sets a custom base URL for API requests.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL for the API
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the API key for authentication.
    ///
    /// The key is stored securely using `SecretString`.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::new(api_key.into().into()));
        self
    }

    /// Sets the organization identifier.
    ///
    /// # Arguments
    ///
    /// * `organization` - The organization ID
    #[must_use]
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Sets the request timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout_seconds` - Timeout in seconds
    #[must_use]
    pub const fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = Some(timeout_seconds);
        self
    }

    /// Sets the default sampling temperature.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Value between 0.0 and 2.0
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the default maximum tokens to generate.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens in responses
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the default nucleus sampling threshold.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Value between 0.0 and 1.0
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the default frequency penalty.
    ///
    /// # Arguments
    ///
    /// * `frequency_penalty` - Value between -2.0 and 2.0
    #[must_use]
    pub const fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Sets the default presence penalty.
    ///
    /// # Arguments
    ///
    /// * `presence_penalty` - Value between -2.0 and 2.0
    #[must_use]
    pub const fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Sets the default stop sequences.
    ///
    /// # Arguments
    ///
    /// * `stop_sequences` - An iterable of stop sequences
    #[must_use]
    pub fn with_stop_sequences(
        mut self,
        stop_sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.stop_sequences = Some(stop_sequences.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the default metadata.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Key-value pairs of metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the retry configuration.
    ///
    /// # Arguments
    ///
    /// * `retry_config` - The retry configuration
    #[must_use]
    pub const fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    /// Sets the tokenizer proxy configuration.
    ///
    /// When configured, requests are routed through the proxy which
    /// injects real credentials. The `api_key` field should contain
    /// a sealed token instead of the real API key.
    ///
    /// # Arguments
    ///
    /// * `proxy` - The proxy configuration
    ///
    /// # Examples
    ///
    /// Using the validated constructor (recommended):
    ///
    /// ```
    /// use neuromance_common::{Config, ProxyConfig};
    ///
    /// let proxy = ProxyConfig::new("http://tokenizer.internal:8080").unwrap();
    /// let config = Config::new("anthropic", "claude-sonnet-4-20250514")
    ///     .with_api_key("sealed.abc123xyz...")
    ///     .with_proxy(proxy);
    /// ```
    ///
    /// Or with struct literal (validation via `Config::validate()`):
    ///
    /// ```
    /// use neuromance_common::{Config, ProxyConfig};
    ///
    /// let config = Config::new("anthropic", "claude-sonnet-4-20250514")
    ///     .with_api_key("sealed.abc123xyz...")
    ///     .with_proxy(ProxyConfig {
    ///         proxy_url: "http://tokenizer.internal:8080".to_string(),
    ///         token_header: "X-Tokenizer-Token".to_string(),
    ///         target_host_header: Some("X-Target-Host".to_string()),
    ///     });
    /// // Validation will fail later if proxy_url is invalid
    /// config.validate().unwrap();
    /// ```
    #[must_use]
    pub fn with_proxy(mut self, proxy: ProxyConfig) -> Self {
        self.proxy = Some(proxy);
        self
    }

    /// Validates the configuration parameters.
    ///
    /// Checks that all numeric parameters are within their valid ranges
    /// and that the proxy configuration (if present) is valid.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of range:
    /// - `temperature` must be between 0.0 and 2.0
    /// - `top_p` must be between 0.0 and 1.0
    /// - `frequency_penalty` must be between -2.0 and 2.0
    /// - `presence_penalty` must be between -2.0 and 2.0
    /// - `proxy.proxy_url` must be a valid http/https URL
    pub fn validate(&self) -> anyhow::Result<()> {
        super::request::validate_sampling_params(
            self.temperature,
            self.top_p,
            self.frequency_penalty,
            self.presence_penalty,
        )?;

        if let Some(ref proxy) = self.proxy {
            proxy.validate()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn temperature_validation(temp in -10.0f32..10.0f32) {
            let config = Config::new("openai", "gpt-4").with_temperature(temp);
            let is_valid = (0.0..=2.0).contains(&temp);
            assert_eq!(config.validate().is_ok(), is_valid);
        }

        #[test]
        fn top_p_validation(top_p in -5.0f32..5.0f32) {
            let config = Config::new("openai", "gpt-4").with_top_p(top_p);
            let is_valid = (0.0..=1.0).contains(&top_p);
            assert_eq!(config.validate().is_ok(), is_valid);
        }

        #[test]
        fn frequency_penalty_validation(penalty in -10.0f32..10.0f32) {
            let config = Config::new("openai", "gpt-4").with_frequency_penalty(penalty);
            let is_valid = (-2.0..=2.0).contains(&penalty);
            assert_eq!(config.validate().is_ok(), is_valid);
        }

        #[test]
        fn presence_penalty_validation(penalty in -10.0f32..10.0f32) {
            let config = Config::new("openai", "gpt-4").with_presence_penalty(penalty);
            let is_valid = (-2.0..=2.0).contains(&penalty);
            assert_eq!(config.validate().is_ok(), is_valid);
        }

        #[test]
        fn max_tokens_validation(tokens in 0u32..1_000_000_u32) {
            let config = Config::new("openai", "gpt-4").with_max_tokens(tokens);
            // max_tokens can be 0 (infinite) or any positive value
            assert!(config.validate().is_ok());
        }

        #[test]
        fn config_builder_with_string_slice(
            provider in ".*",
            model in ".*",
            base_url in ".*",
        ) {
            let config = Config::new(provider.as_str(), model.as_str())
                .with_base_url(base_url.as_str());

            // Should compile and work with &str
            assert_eq!(config.provider, provider);
            assert_eq!(config.model, model);
            assert_eq!(config.base_url, Some(base_url));
        }

        #[test]
        fn config_builder_with_owned_string(
            provider in ".*",
            model in ".*",
        ) {
            let config = Config::new(provider.clone(), model.clone());

            // Should compile and work with String
            assert_eq!(config.provider, provider);
            assert_eq!(config.model, model);
        }

        #[test]
        fn stop_sequences_accepts_various_types(
            sequences in prop::collection::vec(".*", 0..10),
        ) {
            // Test with Vec<String>
            let config1 = Config::new("openai", "gpt-4")
                .with_stop_sequences(sequences.clone());
            assert_eq!(config1.stop_sequences, Some(sequences.clone()));

            // Test with Vec<&str>
            let str_refs: Vec<&str> = sequences.iter().map(std::string::String::as_str).collect();
            let config2 = Config::new("openai", "gpt-4")
                .with_stop_sequences(str_refs);
            assert_eq!(config2.stop_sequences, Some(sequences.clone()));

            // Test with array of &str
            if sequences.len() <= 3 {
                let arr: Vec<&str> = sequences.iter().map(std::string::String::as_str).collect();
                let config3 = Config::new("openai", "gpt-4")
                    .with_stop_sequences(arr);
                assert_eq!(config3.stop_sequences, Some(sequences));
            }
        }

        #[test]
        fn builder_chain_preserves_all_values(
            provider in ".*",
            model in ".*",
            temp in 0.0f32..2.0f32,
            max_tokens in 0u32..100_000_u32,
        ) {
            let config = Config::new(provider.as_str(), model.as_str())
                .with_temperature(temp)
                .with_max_tokens(max_tokens);

            assert_eq!(config.provider, provider);
            assert_eq!(config.model, model);
            assert_eq!(config.temperature, Some(temp));
            assert_eq!(config.max_tokens, Some(max_tokens));
            assert!(config.validate().is_ok());
        }
    }
}
