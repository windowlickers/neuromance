//! Daemon configuration and model profile management.
//!
//! Configuration is loaded from `~/.config/neuromance/config.toml`.
//!
//! ## Example Configuration
//!
//! ```toml
//! active_model = "sonnet"
//!
//! [[models]]
//! nickname = "sonnet"
//! provider = "anthropic"
//! model = "claude-sonnet-4-5-20250929"
//! api_key_env = "ANTHROPIC_API_KEY"
//!
//! [[models]]
//! nickname = "gpt4"
//! provider = "openai"
//! model = "gpt-4o"
//! api_key_env = "OPENAI_API_KEY"
//!
//! [settings]
//! auto_approve_tools = false
//! max_turns = 10
//! thinking_budget = 10000
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use neuromance_common::ModelProfile;
use neuromance_tools::mcp::config::{McpServerConfig, McpSettings};
use serde::{Deserialize, Serialize};

use crate::error::{DaemonError, Result};

/// Daemon configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// The active model nickname (default if not specified in requests)
    pub active_model: String,

    /// Available model profiles
    pub models: Vec<ModelProfile>,

    /// Optional daemon settings
    #[serde(default)]
    pub settings: Settings,
}

/// Optional daemon settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Whether to auto-approve tool calls (default: false)
    #[serde(default)]
    pub auto_approve_tools: bool,

    /// Maximum conversation turns before stopping (default: 20)
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,

    /// Default thinking budget in tokens (default: 10000)
    #[serde(default = "default_thinking_budget")]
    pub thinking_budget: u32,

    /// Inactivity timeout in seconds (default: 900 = 15 minutes)
    #[serde(default = "default_inactivity_timeout")]
    pub inactivity_timeout: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            auto_approve_tools: false,
            max_turns: DEFAULT_MAX_TURNS,
            thinking_budget: default_thinking_budget(),
            inactivity_timeout: default_inactivity_timeout(),
        }
    }
}

const DEFAULT_MAX_TURNS: u32 = 20;

const fn default_max_turns() -> u32 {
    DEFAULT_MAX_TURNS
}

const fn default_thinking_budget() -> u32 {
    10_000
}

const fn default_inactivity_timeout() -> u64 {
    900 // 15 minutes
}

impl DaemonConfig {
    /// Loads configuration from the default location.
    ///
    /// Reads from `~/.config/neuromance/config.toml`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The config directory cannot be determined
    /// - The file doesn't exist
    /// - Deserialization fails
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;

        if !path.exists() {
            return Err(DaemonError::Config(format!(
                "Configuration file not found: {}",
                path.display()
            )));
        }

        let contents = fs::read_to_string(&path)
            .map_err(|e| DaemonError::Config(format!("Failed to read config file: {e}")))?;

        let config: Self = toml::from_str(&contents)?;
        config.validate()?;

        Ok(config)
    }

    /// Returns the default configuration file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the config directory cannot be determined.
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = neuromance_daemon::paths::neuromance_config_dir().ok_or_else(|| {
            DaemonError::Config("Failed to determine config directory".to_string())
        })?;

        Ok(config_dir.join("config.toml"))
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No models are defined
    /// - The active model doesn't exist
    /// - Duplicate model nicknames are found
    pub fn validate(&self) -> Result<()> {
        if self.models.is_empty() {
            return Err(DaemonError::Config(
                "No models defined in configuration".to_string(),
            ));
        }

        // Check active model exists
        if !self.models.iter().any(|m| m.nickname == self.active_model) {
            return Err(DaemonError::Config(format!(
                "Active model '{}' not found in model profiles",
                self.active_model
            )));
        }

        // Check for duplicate nicknames
        let mut seen = HashMap::new();
        for model in &self.models {
            if let Some(existing) = seen.insert(&model.nickname, &model.provider) {
                return Err(DaemonError::Config(format!(
                    "Duplicate model nickname '{}' (providers: {existing}, {})",
                    model.nickname, model.provider
                )));
            }
        }

        Ok(())
    }

    /// Gets a model profile by nickname.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found.
    pub fn get_model(&self, nickname: &str) -> Result<&ModelProfile> {
        self.models
            .iter()
            .find(|m| m.nickname == nickname)
            .ok_or_else(|| DaemonError::ModelNotFound(nickname.to_string()))
    }
}

/// Tools configuration loaded from `~/.config/neuromance/tools.toml`.
///
/// Combines built-in tool toggles with MCP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    /// Built-in tool toggles
    #[serde(default)]
    pub builtin: BuiltinToolsConfig,

    /// MCP global settings
    #[serde(default)]
    pub settings: McpSettings,

    /// MCP server configurations
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

/// Configuration for built-in tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinToolsConfig {
    /// Enable `get_current_time` tool (safe, read-only)
    #[serde(default = "default_true")]
    pub current_time: bool,

    /// Enable calculator tool (demo-quality implementation)
    #[serde(default)]
    pub calculator: bool,
}

impl Default for BuiltinToolsConfig {
    fn default() -> Self {
        Self {
            current_time: true,
            calculator: false,
        }
    }
}

const fn default_true() -> bool {
    true
}

impl ToolsConfig {
    /// Loads tools configuration from `~/.config/neuromance/tools.toml`.
    ///
    /// Returns `None` if the file doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or parsed.
    pub fn load() -> Result<Option<Self>> {
        let path = Self::config_path()?;

        if !path.exists() {
            return Ok(None);
        }

        let contents = fs::read_to_string(&path)
            .map_err(|e| DaemonError::Config(format!("Failed to read tools config: {e}")))?;

        let config: Self = toml::from_str(&contents)?;
        Ok(Some(config))
    }

    /// Returns the tools configuration file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the config directory cannot be determined.
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = neuromance_daemon::paths::neuromance_config_dir().ok_or_else(|| {
            DaemonError::Config("Failed to determine config directory".to_string())
        })?;

        Ok(config_dir.join("tools.toml"))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    fn sample_config_toml() -> &'static str {
        r#"
active_model = "sonnet"

[[models]]
nickname = "sonnet"
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
api_key_env = "ANTHROPIC_API_KEY"

[[models]]
nickname = "gpt4"
provider = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"

[settings]
auto_approve_tools = true
max_turns = 15
thinking_budget = 20000
        "#
    }

    #[test]
    fn test_parse_config() {
        let config: DaemonConfig = toml::from_str(sample_config_toml()).unwrap();

        assert_eq!(config.active_model, "sonnet");
        assert_eq!(config.models.len(), 2);
        assert!(config.settings.auto_approve_tools);
        assert_eq!(config.settings.max_turns, 15);
        assert_eq!(config.settings.thinking_budget, 20_000);
    }

    #[test]
    fn test_get_model() {
        let config: DaemonConfig = toml::from_str(sample_config_toml()).unwrap();

        let model = config.get_model("sonnet").unwrap();
        assert_eq!(model.provider, "anthropic");
        assert_eq!(model.model, "claude-sonnet-4-5-20250929");

        assert!(config.get_model("nonexistent").is_err());
    }

    #[test]
    fn test_validate_missing_active_model() {
        let toml = r#"
active_model = "nonexistent"

[[models]]
nickname = "gpt4"
provider = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
        "#;

        let config: DaemonConfig = toml::from_str(toml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_duplicate_nicknames() {
        let toml = r#"
active_model = "gpt4"

[[models]]
nickname = "gpt4"
provider = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"

[[models]]
nickname = "gpt4"
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
api_key_env = "ANTHROPIC_API_KEY"
        "#;

        let config: DaemonConfig = toml::from_str(toml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_default_settings() {
        let toml = r#"
active_model = "gpt4"

[[models]]
nickname = "gpt4"
provider = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
        "#;

        let config: DaemonConfig = toml::from_str(toml).unwrap();
        assert!(!config.settings.auto_approve_tools);
        assert_eq!(config.settings.max_turns, 20);
        assert_eq!(config.settings.thinking_budget, 10_000);
        assert_eq!(config.settings.inactivity_timeout, 900);
    }

    #[test]
    fn test_parse_tools_config() {
        let toml = r#"
[builtin]
current_time = true
calculator = true

[settings]
max_retries = 5

[[servers]]
id = "filesystem"
name = "Local Filesystem"
protocol = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
auto_approve = false
        "#;

        let config: ToolsConfig = toml::from_str(toml).unwrap();
        assert!(config.builtin.current_time);
        assert!(config.builtin.calculator);
        assert_eq!(config.settings.max_retries, 5);
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].id, "filesystem");
    }

    #[test]
    fn test_tools_config_defaults() {
        let config: ToolsConfig = toml::from_str("").unwrap();
        assert!(config.builtin.current_time);
        assert!(!config.builtin.calculator);
        assert!(config.servers.is_empty());
    }

    #[test]
    fn test_builtin_only_tools_config() {
        let toml = r"
[builtin]
calculator = true
        ";

        let config: ToolsConfig = toml::from_str(toml).unwrap();
        assert!(config.builtin.current_time);
        assert!(config.builtin.calculator);
        assert!(config.servers.is_empty());
    }
}
