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

use log::{error, info};
use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook_tokio::Signals;
use tokio::sync::broadcast;

use crate::config::DaemonConfig;
use crate::conversation_manager::ConversationManager;
use crate::error::Result;
use crate::server::Server;
use crate::storage::Storage;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();

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

    // Create conversation manager
    let manager = Arc::new(ConversationManager::new(
        Arc::clone(&storage),
        Arc::clone(&config),
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
