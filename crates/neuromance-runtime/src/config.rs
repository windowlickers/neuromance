//! Runtime configuration parsed from a TOML file.
//!
//! The runtime loads `$NEUROMANCE_CONFIG` (default `/etc/neuromance/config.toml`)
//! at startup. API keys are *not* embedded in this file — they are referenced
//! by environment variable name (`agent.api_key_env`) and read at startup.

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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentConfig {
    pub id: String,
    pub model: String,
    pub api_key_env: String,
    pub system_prompt: String,
    #[serde(default)]
    pub max_turns: Option<u32>,
    #[serde(default)]
    pub streaming: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeSettings {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
    #[serde(default = "default_health_addr")]
    pub health_addr: String,
    #[serde(default = "default_shutdown_grace")]
    pub shutdown_grace_seconds: u64,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
            health_addr: default_health_addr(),
            shutdown_grace_seconds: default_shutdown_grace(),
        }
    }
}

fn default_listen_addr() -> String {
    "0.0.0.0:8080".to_string()
}
fn default_health_addr() -> String {
    "0.0.0.0:8081".to_string()
}
const fn default_shutdown_grace() -> u64 {
    30
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ApprovalConfig {
    #[serde(default)]
    pub mode: ApprovalMode,
    pub webhook_url: Option<String>,
    #[serde(default = "default_approval_timeout")]
    pub timeout_seconds: u64,
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
        let contents = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::Config(format!("read {}: {e}", path.display()))
        })?;
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
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if matches!(self.mode, Mode::Oneshot) && self.oneshot.is_none() {
            return Err(RuntimeError::Config(
                "oneshot mode requires [oneshot] section".to_string(),
            ));
        }
        if matches!(self.approval.mode, ApprovalMode::Async)
            && self.approval.webhook_url.is_none()
        {
            return Err(RuntimeError::Config(
                "approval.mode = \"async\" requires approval.webhook_url".to_string(),
            ));
        }
        Ok(())
    }
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
        assert_eq!(config.runtime.listen_addr, "0.0.0.0:8080");
        assert_eq!(config.runtime.shutdown_grace_seconds, 30);
        assert_eq!(config.approval.mode, ApprovalMode::Auto);
        assert!(config.tools.is_empty());
        assert!(config.oneshot.is_some());
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
    fn test_tools_section_parses() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            model = "openai:gpt-4o"
            api_key_env = "K"
            system_prompt = "x"

            [[tools]]
            name = "think"

            [[tools]]
            name = "todos"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(config.tools.len(), 2);
        assert_eq!(config.tools[0].name, "think");
        assert_eq!(config.tools[1].name, "todos");
    }
}
