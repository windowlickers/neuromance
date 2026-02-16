//! Neuromance Daemon
//!
//! Long-running server that manages multiple LLM conversations and provides
//! a lightweight client interface over Unix domain sockets.

mod config;
mod conversation_manager;
mod error;
mod server;
mod storage;

use std::sync::Arc;

use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook_tokio::Signals;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

use neuromance_tools::ToolImplementation;
use neuromance_tools::generic::{CalculatorTool, CurrentTimeTool};
use neuromance_tools::mcp::{McpConfig, McpManager};

use crate::config::{BuiltinToolsConfig, DaemonConfig, ToolsConfig};
use crate::conversation_manager::ConversationManager;
use crate::error::Result;
use crate::server::Server;
use crate::storage::Storage;

/// Initializes structured logging with tracing.
///
/// Supports two output formats via `NEUROMANCE_LOG_FORMAT` environment variable:
/// - `json`: Machine-readable JSON logs (default for production)
/// - `pretty`: Human-readable formatted logs (default for development)
///
/// Log level is controlled via `RUST_LOG` environment variable.
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};

    let format = std::env::var("NEUROMANCE_LOG_FORMAT")
        .unwrap_or_else(|_| "pretty".to_string())
        .to_lowercase();

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("neuromance_daemon=info,neuromance=info"));

    match format.as_str() {
        "json" => {
            fmt()
                .json()
                .with_env_filter(filter)
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .init();
        }
        _ => {
            fmt()
                .with_env_filter(filter)
                .with_target(true)
                .with_thread_ids(false)
                .with_file(false)
                .with_line_number(false)
                .init();
        }
    }
}

/// Collects enabled built-in tools based on configuration.
fn collect_builtin_tools(config: &BuiltinToolsConfig) -> Vec<Arc<dyn ToolImplementation>> {
    let mut tools: Vec<Arc<dyn ToolImplementation>> = Vec::new();

    if config.current_time {
        tools.push(Arc::new(CurrentTimeTool));
    }
    if config.calculator {
        tools.push(Arc::new(CalculatorTool));
    }

    tools
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging
    init_tracing();

    info!("Starting Neuromance daemon");

    // Load configuration
    let config = match DaemonConfig::load() {
        Ok(config) => Arc::new(config),
        Err(e) => {
            error!("Failed to load configuration: {e}");
            error!("Expected config at: {:?}", DaemonConfig::config_path());
            return Err(e);
        }
    };

    info!("Loaded configuration with {} models", config.models.len());

    // Initialize storage
    let storage = match Storage::new() {
        Ok(storage) => Arc::new(storage),
        Err(e) => {
            error!("Failed to initialize storage: {e}");
            return Err(e);
        }
    };

    info!("Storage initialized at {}", storage.socket_path().display());

    // Load tools configuration
    let tools_config = match ToolsConfig::load() {
        Ok(Some(tc)) => {
            info!("Loaded tools config from {:?}", ToolsConfig::config_path());
            Some(tc)
        }
        Ok(None) => {
            info!("No tools config found, starting without tools");
            None
        }
        Err(e) => {
            warn!("Failed to load tools config: {e}");
            None
        }
    };

    // Initialize MCP manager if servers are configured
    let mcp_manager = if let Some(ref tc) = tools_config
        && !tc.servers.is_empty()
    {
        let mcp_config = McpConfig {
            servers: tc.servers.clone(),
            settings: tc.settings.clone(),
        };
        match McpManager::new(mcp_config).await {
            Ok(manager) => {
                info!("MCP manager initialized");
                Some(Arc::new(manager))
            }
            Err(e) => {
                warn!("Failed to initialize MCP manager: {e}");
                None
            }
        }
    } else {
        None
    };

    // Collect built-in tools
    let builtin_config = tools_config
        .as_ref()
        .map_or_else(BuiltinToolsConfig::default, |tc| tc.builtin.clone());
    let builtin_tools = collect_builtin_tools(&builtin_config);

    if !builtin_tools.is_empty() {
        info!("Enabled {} built-in tool(s)", builtin_tools.len());
    }

    // Create conversation manager
    let manager = Arc::new(ConversationManager::new(
        Arc::clone(&storage),
        Arc::clone(&config),
        mcp_manager,
        builtin_tools,
    ));

    // Create shutdown channel
    let (shutdown_tx, _) = broadcast::channel(1);

    // Set up signal handlers
    let mut signals = Signals::new([SIGTERM, SIGINT])?;
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        use futures::stream::StreamExt;
        while let Some(signal) = signals.next().await {
            match signal {
                SIGTERM => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                    let _ = shutdown_tx_clone.send(());
                    break;
                }
                SIGINT => {
                    info!("Received SIGINT, initiating graceful shutdown");
                    let _ = shutdown_tx_clone.send(());
                    break;
                }
                _ => {}
            }
        }
    });

    // Create and run server
    let server = Arc::new(Server::new(manager, storage, config, shutdown_tx));

    info!("Daemon ready");

    server.run().await?;

    info!("Daemon shutdown complete");

    Ok(())
}
