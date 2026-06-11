//! Runtime configuration parsed from a TOML file.
//!
//! The runtime loads `$NEUROMANCE_CONFIG` (default `/etc/neuromance/config.toml`)
//! at startup. API keys are *not* embedded in this file. Credentials reach the
//! runtime via one of two paths:
//!
//! - **Env var (legacy)** — `agent.api_key_env` names an environment variable
//!   whose value is the raw provider API key, read at startup.
//! - **Tokenizer proxy** — `[proxy]` points at a sealed-token file on disk
//!   (typically a projected k8s `Secret` volume). The runtime forwards LLM
//!   requests through the tokenizer proxy, which injects the real credential
//!   server-side. The agent pod never holds the plaintext.
//!
//! Exactly one of these paths must be configured.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use neuromance_tools::ToolConfig;

use crate::error::RuntimeError;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Oneshot,
    Serve,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    #[default]
    Auto,
    Async,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeConfig {
    pub mode: Mode,
    pub agent: AgentConfig,
    #[serde(default)]
    pub runtime: RuntimeSettings,
    #[serde(default)]
    pub approval: ApprovalConfig,
    #[serde(default)]
    pub tools: Vec<ToolConfig>,
    #[serde(default)]
    pub oneshot: Option<OneshotConfig>,
    /// When set, the runtime routes outbound LLM requests through a tokenizer
    /// proxy and reads its sealed token from `proxy.token_file` instead
    /// of from `agent.api_key_env`.
    #[serde(default)]
    pub proxy: Option<ProxyTomlConfig>,
    /// When set, conversation history is written through to postgres as
    /// tasks run. In-memory state stays authoritative for serving.
    #[serde(default)]
    pub database: Option<DatabaseSettings>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentConfig {
    pub id: String,
    pub model: String,
    /// Environment variable holding the raw provider API key. Required unless
    /// `[proxy]` is set, in which case the sealed token from the proxy section
    /// replaces it and this field should be omitted.
    #[serde(default)]
    pub api_key_env: Option<String>,
    pub system_prompt: String,
    /// Override the upstream LLM endpoint, e.g. `http://llama-server.windowlickers.svc:8080/v1`
    /// for an in-cluster `OpenAI`-compatible server. Required when `model` uses the
    /// generic `chat_completions:` or `responses:` prefixes, which have no default
    /// base URL.
    ///
    /// When `[proxy]` is set, this field is still the *upstream* the proxy
    /// routes to — `proxy.base_url` is the forward proxy itself,
    /// not the upstream. The upstream URL travels in the absolute-form
    /// request URI sent through the proxy. The two fields have distinct roles
    /// and can both be set.
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub max_turns: Option<u32>,
    #[serde(default)]
    pub streaming: bool,
}

/// Tokenizer-proxy settings.
///
/// When present, the runtime sends every outbound LLM request to `base_url`
/// with the sealed token from `token_file` carried in the `token_header`. The
/// proxy decrypts the sealed token server-side and injects the real provider
/// credential. The agent pod never sees the plaintext.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProxyTomlConfig {
    /// URL of the tokenizer-proxy Service (e.g.
    /// `http://tokenizer-proxy.windowlickers.svc.cluster.local:8080`). The
    /// runtime installs this as the HTTP forward proxy on the reqwest
    /// client; the upstream target host travels in the request URL
    /// (absolute form), not a side-band header.
    pub base_url: String,
    /// Absolute path to a file holding the sealed token. Typically a projected
    /// k8s `Secret` volume mount such as `/var/run/neuromance/tokens/llm`.
    pub token_file: PathBuf,
    /// Header name under which the sealed token is sent. Defaults to
    /// `X-Tokenizer-Token`.
    #[serde(default = "default_token_header")]
    pub token_header: String,
}

fn default_token_header() -> String {
    "X-Tokenizer-Token".to_string()
}

/// Postgres persistence settings.
///
/// The connection URL is a credential (it usually embeds a password), so it
/// is never written in the TOML file — `url_env` names the environment
/// variable that holds it, the same policy as `agent.api_key_env`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatabaseSettings {
    /// Environment variable holding the postgres connection URL
    /// (e.g. `postgres://user:pass@host:5432/neuromance`).
    pub url_env: String,
    /// Maximum connections in the pool.
    #[serde(default = "default_db_max_connections")]
    pub max_connections: u32,
    /// Cap on how long a persistence call may wait for a pool connection.
    /// Bounds the stall a sick database can add to an agent turn.
    #[serde(default = "default_db_acquire_timeout")]
    pub acquire_timeout_seconds: u64,
}

const fn default_db_max_connections() -> u32 {
    5
}
const fn default_db_acquire_timeout() -> u64 {
    5
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeSettings {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
    #[serde(default = "default_health_addr")]
    pub health_addr: String,
    #[serde(default = "default_shutdown_grace")]
    pub shutdown_grace_seconds: u64,
    /// Maximum number of tasks buffered for the worker. `POST /tasks/new`
    /// returns `429` once the queue is at capacity. Sized for a single
    /// serial worker — buffering far beyond a handful is dishonest about
    /// what the runtime can actually do.
    #[serde(default = "default_max_queue_depth")]
    pub max_queue_depth: usize,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
            health_addr: default_health_addr(),
            shutdown_grace_seconds: default_shutdown_grace(),
            max_queue_depth: default_max_queue_depth(),
        }
    }
}

fn default_listen_addr() -> String {
    "127.0.0.1:8080".to_string()
}
fn default_health_addr() -> String {
    "127.0.0.1:8081".to_string()
}
const fn default_shutdown_grace() -> u64 {
    30
}
const fn default_max_queue_depth() -> usize {
    8
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ApprovalConfig {
    #[serde(default)]
    pub mode: ApprovalMode,
    pub webhook_url: Option<String>,
    #[serde(default = "default_approval_timeout")]
    pub timeout_seconds: u64,
    /// Skip the startup safety gate that refuses to start when
    /// `mode = "auto"` is paired with tools whose `is_auto_approved()`
    /// returns false (`bash`, `edit`, `write`). The agentic loop already
    /// approves every tool under `Auto` — this flag opts out of the
    /// duplicate startup check, intended for deployments where the pod
    /// boundary itself provides isolation (e.g. kata containers).
    #[serde(default)]
    pub allow_unsafe_tools: bool,
}

const fn default_approval_timeout() -> u64 {
    60
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OneshotConfig {
    pub input: String,
    #[serde(default)]
    pub output_path: Option<PathBuf>,
}

impl RuntimeConfig {
    /// Load from `$NEUROMANCE_CONFIG`, defaulting to `/etc/neuromance/config.toml`.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] if the file is missing, unreadable, or malformed.
    pub fn load_default() -> Result<Self, RuntimeError> {
        let path = std::env::var("NEUROMANCE_CONFIG")
            .unwrap_or_else(|_| "/etc/neuromance/config.toml".to_string());
        Self::load(&path)
    }

    /// Load and parse a config file at `path`.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] if the file cannot be read or parsed.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, RuntimeError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .map_err(|e| RuntimeError::Config(format!("read {}: {e}", path.display())))?;
        let config: Self = toml::from_str(&contents)
            .map_err(|e| RuntimeError::Config(format!("parse {}: {e}", path.display())))?;
        config.validate()?;
        Ok(config)
    }

    /// Cross-field validation that serde alone cannot express.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] when:
    /// - `mode = "oneshot"` but no `[oneshot]` section is present
    /// - `approval.mode = "async"` but `approval.webhook_url` is unset
    /// - `approval.webhook_url` is set to a URL whose scheme is not
    ///   `http` or `https`
    /// - neither `agent.api_key_env` nor `[proxy]` is set, or both are set
    /// - `[proxy].base_url` is not an `http(s)` URL or `[proxy].token_file`
    ///   is not an absolute path
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if matches!(self.mode, Mode::Oneshot) && self.oneshot.is_none() {
            return Err(RuntimeError::Config(
                "oneshot mode requires [oneshot] section".to_string(),
            ));
        }
        if matches!(self.approval.mode, ApprovalMode::Async) && self.approval.webhook_url.is_none()
        {
            return Err(RuntimeError::Config(
                "approval.mode = \"async\" requires approval.webhook_url".to_string(),
            ));
        }
        if let Some(url) = &self.approval.webhook_url {
            validate_http_url(url, "approval.webhook_url")?;
        }
        if let Some(url) = &self.agent.base_url {
            validate_http_url(url, "agent.base_url")?;
        }

        match (&self.agent.api_key_env, &self.proxy) {
            (Some(_), Some(_)) => {
                return Err(RuntimeError::Config(
                    "agent.api_key_env and [proxy] are mutually exclusive: set one or the other"
                        .to_string(),
                ));
            }
            (None, None) => {
                return Err(RuntimeError::Config(
                    "credentials must come from either agent.api_key_env or [proxy]".to_string(),
                ));
            }
            _ => {}
        }

        if let Some(proxy) = &self.proxy {
            validate_http_url(&proxy.base_url, "proxy.base_url")?;
            if !proxy.token_file.is_absolute() {
                return Err(RuntimeError::Config(format!(
                    "proxy.token_file must be an absolute path, got '{}'",
                    proxy.token_file.display()
                )));
            }
            if proxy.token_header.trim().is_empty() {
                return Err(RuntimeError::Config(
                    "proxy.token_header must not be empty".to_string(),
                ));
            }
        }

        if self.runtime.max_queue_depth == 0 {
            return Err(RuntimeError::Config(
                "runtime.max_queue_depth must be at least 1".to_string(),
            ));
        }

        if let Some(database) = &self.database {
            if database.url_env.trim().is_empty() {
                return Err(RuntimeError::Config(
                    "database.url_env must not be empty".to_string(),
                ));
            }
            if database.max_connections == 0 {
                return Err(RuntimeError::Config(
                    "database.max_connections must be at least 1".to_string(),
                ));
            }
        }
        Ok(())
    }
}

fn validate_http_url(url: &str, field: &str) -> Result<(), RuntimeError> {
    let parsed = url::Url::parse(url)
        .map_err(|e| RuntimeError::Config(format!("{field} is not a valid URL: {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(RuntimeError::Config(format!(
            "{field} must use http or https, got scheme '{}'",
            parsed.scheme()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    fn minimal_oneshot_toml() -> &'static str {
        r#"
            mode = "oneshot"

            [agent]
            id = "research"
            model = "openai:gpt-4o"
            api_key_env = "OPENAI_API_KEY"
            system_prompt = "Be helpful."

            [oneshot]
            input = "Hello, world."
        "#
    }

    #[test]
    fn test_minimal_oneshot_parses_with_defaults() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert_eq!(config.mode, Mode::Oneshot);
        assert_eq!(config.agent.id, "research");
        assert_eq!(config.runtime.listen_addr, "127.0.0.1:8080");
        assert_eq!(config.runtime.shutdown_grace_seconds, 30);
        assert_eq!(config.runtime.max_queue_depth, 8);
        assert_eq!(config.approval.mode, ApprovalMode::Auto);
        assert!(config.tools.is_empty());
        assert!(config.oneshot.is_some());
    }

    #[test]
    fn test_max_queue_depth_round_trips_when_set() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "manager"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "be helpful"
            [runtime]
            max_queue_depth = 64
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(config.runtime.max_queue_depth, 64);
    }

    #[test]
    fn test_oneshot_without_section_fails_validation() {
        let toml_str = r#"
            mode = "oneshot"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("oneshot mode requires"));
    }

    #[test]
    fn test_zero_max_queue_depth_fails_validation() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "be helpful"
            [runtime]
            max_queue_depth = 0
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("max_queue_depth"));
    }

    #[test]
    fn test_async_approval_without_webhook_fails_validation() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "be helpful"
            [approval]
            mode = "async"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("webhook_url"));
    }

    #[test]
    fn test_webhook_url_must_be_http_or_https() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "be helpful"
            [approval]
            mode = "async"
            webhook_url = "file:///etc/passwd"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        let msg = format!("{err}");
        assert!(
            msg.contains("http or https"),
            "expected scheme rejection, got: {msg}"
        );
    }

    #[test]
    fn test_webhook_url_https_validates() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "be helpful"
            [approval]
            mode = "async"
            webhook_url = "https://approve.example.com/decide"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn test_allow_unsafe_tools_defaults_to_false_when_omitted() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        assert!(!config.approval.allow_unsafe_tools);
    }

    #[test]
    fn test_allow_unsafe_tools_round_trips_when_set() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "manager"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "be helpful"
            [approval]
            mode = "auto"
            allow_unsafe_tools = true
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert!(config.approval.allow_unsafe_tools);
    }

    #[test]
    fn test_serve_mode_does_not_require_oneshot() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(config.mode, Mode::Serve);
    }

    #[test]
    fn test_base_url_round_trips_from_toml() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "manager"
            model = "chat_completions:qwen"
            api_key_env = "OPENAI_API_KEY"
            system_prompt = "be helpful"
            base_url = "http://llama-server.windowlickers.svc.cluster.local:8080/v1"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(
            config.agent.base_url.as_deref(),
            Some("http://llama-server.windowlickers.svc.cluster.local:8080/v1"),
        );
    }

    #[test]
    fn test_base_url_rejects_non_http_scheme() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "x"
            base_url = "file:///etc/passwd"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("http or https"));
    }

    #[test]
    fn test_proxy_section_parses_with_required_fields() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            system_prompt = "x"
            [proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/neuromance/tokens/llm"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        let proxy = config.proxy.expect("proxy section");
        assert_eq!(proxy.base_url, "http://tokenizer-proxy.svc:8080");
        assert_eq!(
            proxy.token_file,
            PathBuf::from("/var/run/neuromance/tokens/llm")
        );
        // Default applied.
        assert_eq!(proxy.token_header, "X-Tokenizer-Token");
        assert!(config.agent.api_key_env.is_none());
    }

    #[test]
    fn test_proxy_section_round_trips_custom_token_header() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            system_prompt = "x"
            [proxy]
            base_url = "https://tokenizer-proxy.example.com"
            token_file = "/var/run/tokens/llm"
            token_header = "X-Token"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        let proxy = config.proxy.expect("proxy section");
        assert_eq!(proxy.token_header, "X-Token");
    }

    #[test]
    fn test_proxy_and_api_key_env_are_mutually_exclusive() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "x"
            [proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/tokens/llm"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("mutually exclusive"),
            "expected exclusivity error, got: {err}",
        );
    }

    #[test]
    fn test_missing_both_api_key_env_and_proxy_fails() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("credentials must come from"),
            "expected missing-credentials error, got: {err}",
        );
    }

    #[test]
    fn test_proxy_token_file_must_be_absolute() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            system_prompt = "x"
            [proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "relative/path/token"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("absolute path"),
            "expected absolute-path error, got: {err}",
        );
    }

    #[test]
    fn test_proxy_base_url_rejects_non_http_scheme() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            system_prompt = "x"
            [proxy]
            base_url = "file:///etc/passwd"
            token_file = "/var/run/tokens/llm"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("http or https"));
    }

    #[test]
    fn test_proxy_token_header_must_not_be_empty() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            system_prompt = "x"
            [proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/tokens/llm"
            token_header = "   "
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("token_header"));
    }

    #[test]
    fn test_database_section_absent_means_none() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert!(config.database.is_none());
    }

    #[test]
    fn test_database_section_applies_defaults() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "x"
            [database]
            url_env = "DATABASE_URL"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        let database = config.database.expect("database section");
        assert_eq!(database.url_env, "DATABASE_URL");
        assert_eq!(database.max_connections, 5);
        assert_eq!(database.acquire_timeout_seconds, 5);
    }

    #[test]
    fn test_database_section_round_trips_custom_values() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "x"
            [database]
            url_env = "PG_URL"
            max_connections = 12
            acquire_timeout_seconds = 30
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        let database = config.database.expect("database section");
        assert_eq!(database.url_env, "PG_URL");
        assert_eq!(database.max_connections, 12);
        assert_eq!(database.acquire_timeout_seconds, 30);
    }

    #[test]
    fn test_database_empty_url_env_fails_validation() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "x"
            [database]
            url_env = "  "
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("database.url_env"));
    }

    #[test]
    fn test_database_zero_max_connections_fails_validation() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "x"
            [database]
            url_env = "DATABASE_URL"
            max_connections = 0
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("database.max_connections"));
    }

    #[test]
    fn test_tools_section_parses() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "x"

            [[tools]]
            name = "read"

            [[tools]]
            name = "bash"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(config.tools.len(), 2);
        assert_eq!(config.tools[0].name, "read");
        assert_eq!(config.tools[1].name, "bash");
    }
}
