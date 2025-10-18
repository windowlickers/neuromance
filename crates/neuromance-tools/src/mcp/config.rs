//! MCP (Model Context Protocol) configuration types.
//!
//! This module provides configuration structures for connecting to MCP servers,
//! which expose external tools that LLMs can use (filesystem, databases, web APIs, etc.).
//!
//! ## Configuration File Formats
//!
//! MCP configuration can be loaded from TOML, YAML, or JSON files using [`McpConfig::from_file`].
//!
//! ## Example TOML Configuration
//!
//! ```toml
//! # Global settings
//! [settings]
//! max_retries = 3
//! debug = false
//!
//! # Filesystem server via stdio
//! [[servers]]
//! id = "filesystem"
//! name = "Local Filesystem"
//! description = "Access local files"
//! protocol = "stdio"
//! command = "npx"
//! args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
//! auto_approve = false
//!
//! # Web search via SSE
//! [[servers]]
//! id = "search"
//! name = "Web Search"
//! protocol = "sse"
//! url = "https://example.com/mcp"
//! auto_approve = true
//! ```
//!
//! ## Loading Configuration
//!
//! ```rust,ignore
//! use neuromance_tools::mcp::McpConfig;
//! use std::path::Path;
//!
//! // Load from file (auto-detects format from extension)
//! let config = McpConfig::from_file(Path::new("mcp_config.toml"))?;
//!
//! // Or load from specific format
//! let config = McpConfig::from_toml_file(Path::new("config.toml"))?;
//! let config = McpConfig::from_yaml_file(Path::new("config.yaml"))?;
//! let config = McpConfig::from_json_file(Path::new("config.json"))?;
//! ```
//!
//! ## Transport Protocols
//!
//! MCP supports three transport mechanisms via [`McpTransportConfig`]:
//!
//! - **Stdio**: Spawn a subprocess and communicate via stdin/stdout
//! - **SSE**: Connect via Server-Sent Events over HTTP
//! - **HTTP**: Connect via HTTP streaming
//!
//! ## Auto-Approval
//!
//! Servers can be configured with `auto_approve = true` to automatically execute
//! their tools without user confirmation. Use this carefully for trusted, read-only tools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Configuration for MCP servers.
///
/// Contains a list of server configurations and global settings for MCP client behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// List of MCP server configurations
    pub servers: Vec<McpServerConfig>,
    /// Global settings for MCP client
    #[serde(default)]
    pub settings: McpSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique identifier for this server
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Server connection configuration
    #[serde(flatten)]
    pub transport: McpTransportConfig,
    /// Optional description
    pub description: Option<String>,
    /// Whether this server should be auto-approved for tool calls
    #[serde(default)]
    pub auto_approve: bool,
    /// Working directory for the server process (if applicable)
    pub working_directory: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "protocol", rename_all = "lowercase")]
pub enum McpTransportConfig {
    /// Connect via stdio (spawning a process)
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
    /// Connect via Server-Sent Events
    Sse { url: String },
    /// Connect via HTTP streaming
    Http { url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpSettings {
    /// Maximum retries for failed connections
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    /// Whether to enable debug logging for MCP
    #[serde(default)]
    pub debug: bool,
}

const fn default_max_retries() -> usize {
    3
}

impl McpConfig {
    /// Load configuration from a YAML file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_yaml_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a JSON file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_json_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from TOML file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_toml_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Load from file based on extension
    ///
    /// # Errors
    /// Returns an error if the file cannot be read, parsed, or has an unsupported extension.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("yaml" | "yml") => Self::from_yaml_file(path),
            Some("json") => Self::from_json_file(path),
            Some("toml") => Self::from_toml_file(path),
            _ => Err(anyhow::anyhow!(
                "Unsupported config file format. Use .yaml, .yml, .json, or .toml"
            )),
        }
    }

    /// Validate the configuration
    fn validate(&self) -> anyhow::Result<()> {
        // Check for duplicate server IDs
        let mut seen_ids = std::collections::HashSet::new();
        for server in &self.servers {
            if !seen_ids.insert(&server.id) {
                return Err(anyhow::anyhow!("Duplicate server ID found: {}", server.id));
            }
        }
        Ok(())
    }
}
