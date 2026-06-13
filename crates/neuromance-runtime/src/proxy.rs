//! Construction of [`neuromance::Config`] from a [`ProviderConfig`].
//!
//! A provider supplies the credential via exactly one of two paths:
//!
//! - **Env var** (`provider.api_key_env`): read the raw provider key from the
//!   environment, pass it as `Config::api_key`. The client sends it as
//!   `Authorization: Bearer <key>` to the provider directly.
//! - **Tokenizer proxy** (`[providers.proxy]`): read a sealed token from
//!   `proxy.token_file`, pass it as `Config::api_key`, and attach a
//!   [`ProxyConfig`] so the client routes through the tokenizer proxy as an
//!   HTTP forward proxy. The sealed token is sent under `proxy.token_header`;
//!   the upstream target travels in the request URL (absolute-form
//!   request URI), not a side-band header. The proxy decrypts the sealed
//!   token server-side and injects the real provider credential. The
//!   agent pod never sees the plaintext.
//!
//! In proxy mode, `Config::base_url` and `ProxyConfig::proxy_url` carry
//! distinct URLs and must not be conflated. `proxy.base_url` is the
//! tokenizer-proxy destination — reqwest dials it as the forward proxy.
//! `config.base_url` is the *upstream* provider URL (provider default
//! from [`Config::from_model`], or `provider.base_url` if pinned), and it
//! is what appears in the absolute-form request URI sent through the proxy.
//! Overwriting `config.base_url` with the proxy URL breaks that routing.
//!
//! The config validator already enforces that exactly one credential path
//! is configured per provider.

use neuromance::{Config, ProxyConfig};
use secrecy::SecretString;
use tracing::warn;

use crate::{ProviderConfig, error::RuntimeError};

/// Build the LLM [`Config`] for a provider and effective model.
///
/// The credential path (env var vs tokenizer proxy) and upstream endpoint come
/// from the `provider`; the `model` is the caller's already-resolved effective
/// model (an agent or subagent override, or the provider's default).
///
/// # Errors
/// - [`RuntimeError::Config`] when the model string fails to resolve to a
///   provider, when the proxy URL is invalid, or when the provider has no
///   credential configured.
/// - [`RuntimeError::MissingEnv`] when the env-var path is taken and the
///   referenced variable is unset.
/// - [`RuntimeError::ProxyTokenRead`] when the proxy path cannot read the
///   sealed-token file.
pub fn build_provider_config(
    provider: &ProviderConfig,
    model: &str,
) -> Result<Config, RuntimeError> {
    let mut llm_config = Config::from_model(model)
        .map_err(|e| RuntimeError::Config(format!("model '{model}': {e}")))?;

    match (&provider.proxy, &provider.api_key_env) {
        (Some(proxy), _) => {
            let token = read_token_file(&proxy.token_file)?;
            let proxy_config =
                ProxyConfig::with_token_header(proxy.base_url.clone(), proxy.token_header.clone())
                    .map_err(|e| {
                        RuntimeError::Config(format!("provider '{}' proxy: {e}", provider.name))
                    })?;

            if provider.base_url.as_deref() == Some(proxy.base_url.as_str()) {
                warn!(
                    provider = %provider.name,
                    base_url = %proxy.base_url,
                    "provider base_url matches proxy.base_url; these fields have distinct \
                     roles — base_url is the upstream LLM endpoint (appears in the \
                     absolute-form request URI) and proxy.base_url is the forward proxy itself",
                );
            }

            // The sealed token replaces the raw provider key. neuromance-client's
            // `add_proxy_headers` reads `Config::api_key` and emits it under
            // `proxy.token_header` for the tokenizer proxy to unseal.
            //
            // Note: `config.base_url` stays as the upstream provider URL — either
            // the default from `Config::from_model` or `provider.base_url` below.
            // The proxy URL lives only on `ProxyConfig::proxy_url` and is
            // installed as the reqwest forward proxy in `build_client_resources`.
            llm_config.api_key = Some(token);
            llm_config = llm_config.with_proxy(proxy_config);
        }
        (None, Some(env_var)) => {
            let api_key =
                std::env::var(env_var).map_err(|_| RuntimeError::MissingEnv(env_var.clone()))?;
            llm_config = llm_config.with_api_key(api_key);
        }
        (None, None) => {
            return Err(RuntimeError::Config(format!(
                "provider '{}' has no credential: set api_key_env or proxy",
                provider.name
            )));
        }
    }

    if let Some(url) = &provider.base_url {
        llm_config = llm_config.with_base_url(url);
    }

    Ok(llm_config)
}

fn read_token_file(path: &std::path::Path) -> Result<SecretString, RuntimeError> {
    let raw = std::fs::read_to_string(path).map_err(|source| RuntimeError::ProxyTokenRead {
        path: path.to_path_buf(),
        source,
    })?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(RuntimeError::Config(format!(
            "proxy.token_file '{}' is empty",
            path.display()
        )));
    }
    Ok(SecretString::from(trimmed.to_owned()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::ProviderProxyConfig;
    use secrecy::ExposeSecret;
    use std::io::Write;
    use std::path::PathBuf;

    /// Provider using the env-var credential path, optionally pinning a
    /// `base_url`. Model defaults to `openai:gpt-4o`.
    fn env_provider(api_key_env: &str, base_url: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            name: "primary".to_string(),
            model: Some("openai:gpt-4o".to_string()),
            base_url: base_url.map(str::to_string),
            api_key_env: Some(api_key_env.to_string()),
            proxy: None,
        }
    }

    /// Provider using the tokenizer-proxy credential path, optionally pinning a
    /// `base_url`.
    fn proxy_provider(proxy: ProviderProxyConfig, base_url: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            name: "primary".to_string(),
            model: Some("openai:gpt-4o".to_string()),
            base_url: base_url.map(str::to_string),
            api_key_env: None,
            proxy: Some(proxy),
        }
    }

    fn proxy_at(base_url: &str, token_file: PathBuf) -> ProviderProxyConfig {
        ProviderProxyConfig {
            base_url: base_url.to_string(),
            token_file,
            token_header: "X-Tokenizer-Token".to_string(),
        }
    }

    fn write_token_file(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    /// A name that no sane CI environment will set. Setting env vars from a
    /// test is forbidden by `unsafe_code`, so the env-var branch is tested
    /// only against an *unset* variable; the populated path is covered by the
    /// existing integration setup.
    const UNSET_ENV_VAR: &str = "NEUROMANCE_RUNTIME_PROXY_TESTS_UNSET_KEY_8f3a";

    #[test]
    fn test_env_var_branch_missing_env_returns_missing_env_error() {
        // Guard against the impossible-but-let's-be-safe case where another
        // test or the surrounding process did set our chosen sentinel.
        assert!(
            std::env::var(UNSET_ENV_VAR).is_err(),
            "test sentinel {UNSET_ENV_VAR} must not be set in the environment",
        );
        let provider = env_provider(UNSET_ENV_VAR, None);
        let err = build_provider_config(&provider, "openai:gpt-4o").unwrap_err();
        assert!(matches!(err, RuntimeError::MissingEnv(ref v) if v == UNSET_ENV_VAR));
    }

    #[test]
    fn test_proxy_branch_reads_token_from_file_and_sets_proxy() {
        let token = write_token_file("sealed-token-blob\n");
        let provider = proxy_provider(
            proxy_at(
                "http://tokenizer-proxy.svc:8080",
                token.path().to_path_buf(),
            ),
            None,
        );
        let llm = build_provider_config(&provider, "openai:gpt-4o").unwrap();
        let api_key = llm.api_key.as_ref().expect("api_key set");
        assert_eq!(api_key.expose_secret(), "sealed-token-blob");
        let proxy = llm.proxy.as_ref().expect("proxy set");
        assert_eq!(proxy.proxy_url, "http://tokenizer-proxy.svc:8080");
        assert_eq!(proxy.token_header, "X-Tokenizer-Token");
        // base_url stays as the provider default from Config::from_model
        // ("openai:gpt-4o" -> api.openai.com). The proxy URL lives on
        // proxy.proxy_url and must not clobber the upstream — reqwest uses
        // it as the forward proxy in build_client_resources.
        assert_eq!(llm.base_url.as_deref(), Some("https://api.openai.com/v1"));
    }

    #[test]
    fn test_proxy_branch_preserves_provider_base_url_as_upstream() {
        let token = write_token_file("blob");
        let provider = proxy_provider(
            proxy_at(
                "http://tokenizer-proxy.svc:8080",
                token.path().to_path_buf(),
            ),
            Some("https://openrouter.ai/api/v1"),
        );
        let llm = build_provider_config(&provider, "openai:gpt-4o").unwrap();
        // provider.base_url is the upstream the proxy routes to; it must be
        // preserved because it ends up in the absolute-form request URI
        // sent through the proxy.
        assert_eq!(
            llm.base_url.as_deref(),
            Some("https://openrouter.ai/api/v1")
        );
        let proxy = llm.proxy.as_ref().expect("proxy set");
        assert_eq!(proxy.proxy_url, "http://tokenizer-proxy.svc:8080");
    }

    #[test]
    fn test_proxy_branch_warns_when_provider_and_proxy_urls_match() {
        // Same value in both fields is almost certainly a misconfiguration —
        // the runtime emits a tracing::warn! but still builds the config. We
        // don't wire a tracing-capture writer here; asserting on the log line
        // would need a tracing_subscriber::fmt::test writer.
        let token = write_token_file("blob");
        let provider = proxy_provider(
            proxy_at(
                "http://tokenizer-proxy.svc:8080",
                token.path().to_path_buf(),
            ),
            Some("http://tokenizer-proxy.svc:8080"),
        );
        let llm = build_provider_config(&provider, "openai:gpt-4o").unwrap();
        assert_eq!(
            llm.base_url.as_deref(),
            Some("http://tokenizer-proxy.svc:8080")
        );
        assert!(llm.proxy.is_some());
    }

    #[test]
    fn test_proxy_branch_missing_token_file_returns_proxy_token_read() {
        let provider = proxy_provider(
            proxy_at(
                "http://tokenizer-proxy.svc:8080",
                PathBuf::from("/nonexistent/path/to/token"),
            ),
            None,
        );
        let err = build_provider_config(&provider, "openai:gpt-4o").unwrap_err();
        assert!(matches!(err, RuntimeError::ProxyTokenRead { ref path, .. }
                if path == &PathBuf::from("/nonexistent/path/to/token")));
    }

    #[test]
    fn test_proxy_branch_empty_token_file_rejected() {
        let token = write_token_file("   \n\n");
        let provider = proxy_provider(
            proxy_at(
                "http://tokenizer-proxy.svc:8080",
                token.path().to_path_buf(),
            ),
            None,
        );
        let err = build_provider_config(&provider, "openai:gpt-4o").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("empty"),
            "expected empty-token error, got: {msg}"
        );
    }

    #[test]
    fn test_proxy_branch_invalid_url_returns_config_error() {
        let token = write_token_file("blob");
        let provider = proxy_provider(proxy_at("not-a-url", token.path().to_path_buf()), None);
        let err = build_provider_config(&provider, "openai:gpt-4o").unwrap_err();
        assert!(matches!(err, RuntimeError::Config(ref msg) if msg.contains("proxy")));
    }

    #[test]
    fn test_trailing_newline_trimmed_from_token() {
        let token = write_token_file("blob-with-newline\n");
        let provider = proxy_provider(
            proxy_at(
                "http://tokenizer-proxy.svc:8080",
                token.path().to_path_buf(),
            ),
            None,
        );
        let llm = build_provider_config(&provider, "openai:gpt-4o").unwrap();
        assert_eq!(
            llm.api_key.expect("api_key set").expose_secret(),
            "blob-with-newline",
        );
    }

    #[test]
    fn test_no_credential_returns_config_error() {
        let provider = ProviderConfig {
            name: "broken".to_string(),
            model: Some("openai:gpt-4o".to_string()),
            base_url: None,
            api_key_env: None,
            proxy: None,
        };
        let err = build_provider_config(&provider, "openai:gpt-4o").unwrap_err();
        assert!(matches!(err, RuntimeError::Config(ref msg)
                if msg.contains("no credential")));
    }
}
