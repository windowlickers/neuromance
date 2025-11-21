use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::adapter::McpToolAdapter;
use super::client::McpClientWrapper;
use super::config::McpConfig;
use crate::ToolImplementation;

/// Manages multiple MCP server connections
pub struct McpManager {
    config: McpConfig,
    clients: Arc<RwLock<HashMap<String, Arc<McpClientWrapper>>>>,
}

impl McpManager {
    /// Create a new MCP manager from a config file
    ///
    /// # Errors
    /// Returns an error if the config file cannot be read or connections fail.
    pub async fn from_config_file(path: &Path) -> Result<Self> {
        let config = McpConfig::from_file(path)?;
        Self::new(config).await
    }

    /// Create a new MCP manager with the given configuration
    ///
    /// # Errors
    /// Returns an error if connections fail.
    pub async fn new(config: McpConfig) -> Result<Self> {
        let manager = Self {
            config,
            clients: Arc::new(RwLock::new(HashMap::new())),
        };

        // Connect to all configured servers
        manager.connect_all().await?;

        Ok(manager)
    }

    /// Connect to all configured MCP servers
    ///
    /// # Errors
    /// Returns an error if no servers can be connected.
    pub async fn connect_all(&self) -> Result<()> {
        let mut clients = self.clients.write().await;

        for server_config in &self.config.servers {
            log::info!("Connecting to MCP server '{}'...", server_config.id);

            match McpClientWrapper::connect(server_config.clone()).await {
                Ok(client) => {
                    let tools_count = client.get_tools().await.len();
                    log::info!(
                        "Successfully connected to '{}' with {} tools available",
                        server_config.id,
                        tools_count
                    );
                    clients.insert(server_config.id.clone(), Arc::new(client));
                }
                Err(e) => {
                    log::error!(
                        "Failed to connect to MCP server '{}': {}",
                        server_config.id,
                        e
                    );
                    if self.config.settings.max_retries > 0 {
                        // Try reconnecting with retries
                        for attempt in 1..=self.config.settings.max_retries {
                            log::info!(
                                "Retrying connection to '{}' (attempt {}/{})",
                                server_config.id,
                                attempt,
                                self.config.settings.max_retries
                            );

                            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

                            if let Ok(client) =
                                McpClientWrapper::connect(server_config.clone()).await
                            {
                                let tools_count = client.get_tools().await.len();
                                log::info!(
                                    "Successfully connected to '{}' on retry with {} tools",
                                    server_config.id,
                                    tools_count
                                );
                                clients.insert(server_config.id.clone(), Arc::new(client));
                                break;
                            }
                        }
                    }
                }
            }
        }

        if clients.is_empty() {
            return Err(anyhow::anyhow!("Failed to connect to any MCP servers"));
        }
        drop(clients);

        Ok(())
    }

    /// Connect to a specific MCP server
    ///
    /// # Errors
    /// Returns an error if the server is not found or connection fails.
    pub async fn connect_server(&self, server_id: &str) -> Result<()> {
        let server_config = self
            .config
            .servers
            .iter()
            .find(|s| s.id == server_id)
            .ok_or_else(|| anyhow::anyhow!("Server '{server_id}' not found in configuration"))?
            .clone();

        let client = McpClientWrapper::connect(server_config).await?;

        self.clients
            .write()
            .await
            .insert(server_id.to_string(), Arc::new(client));

        Ok(())
    }

    /// Disconnect from a specific MCP server
    ///
    /// # Errors
    /// Returns an error if shutdown fails.
    pub async fn disconnect_server(&self, server_id: &str) -> Result<()> {
        let client = self.clients.write().await.remove(server_id);
        if let Some(client) = client {
            // Shut down the client gracefully
            if let Ok(client) = Arc::try_unwrap(client) {
                client.shutdown().await?;
            }
        }
        Ok(())
    }

    /// Get all available tools from all connected MCP servers
    ///
    /// # Errors
    /// This function currently cannot fail but returns Result for API consistency.
    pub async fn get_all_tools(&self) -> Result<Vec<Arc<dyn ToolImplementation>>> {
        let mut tools: Vec<Arc<dyn ToolImplementation>> = Vec::new();

        for (server_id, client) in self.clients.read().await.iter() {
            let mcp_tools = client.get_tools().await;

            for mcp_tool in mcp_tools {
                let adapter = McpToolAdapter::new(server_id.clone(), client.clone(), mcp_tool);
                tools.push(Arc::new(adapter) as Arc<dyn ToolImplementation>);
            }
        }

        Ok(tools)
    }

    /// Get a specific tool by name
    ///
    /// # Errors
    /// Returns an error if the tool name format is invalid, server is not connected, or tool not found.
    pub async fn get_tool(&self, full_name: &str) -> Result<Arc<dyn ToolImplementation>> {
        // Parse full name (format: server_id.tool_name)
        let parts: Vec<&str> = full_name.splitn(2, '.').collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!(
                "Invalid tool name format. Expected: server_id.tool_name"
            ));
        }

        let server_id = parts[0];
        let tool_name = parts[1];

        let client = self
            .clients
            .read()
            .await
            .get(server_id)
            .ok_or_else(|| anyhow::anyhow!("Server '{server_id}' not connected"))?
            .clone();

        let mcp_tool = client
            .tools
            .read()
            .await
            .get(tool_name)
            .ok_or_else(|| anyhow::anyhow!("Tool '{tool_name}' not found on server '{server_id}'"))?
            .clone();

        let adapter = McpToolAdapter::new(server_id.to_string(), client, mcp_tool);

        Ok(Arc::new(adapter) as Arc<dyn ToolImplementation>)
    }

    /// Refresh tools for all connected servers
    ///
    /// # Errors
    /// Returns an error if refreshing tools fails on any server.
    pub async fn refresh_all_tools(&self) -> Result<()> {
        for (server_id, client) in self.clients.read().await.iter() {
            log::info!("Refreshing tools for server '{server_id}'...");
            client.refresh_tools().await?;
        }

        Ok(())
    }

    /// Get the status of all MCP servers
    pub async fn get_status(&self) -> HashMap<String, ServerStatus> {
        let mut status_map = HashMap::new();

        for server_config in &self.config.servers {
            let status = if let Some(client) = self.clients.read().await.get(&server_config.id) {
                let tools_count = client.get_tools().await.len();
                let server_info = client.service.peer_info().map_or_else(
                    || "Unknown".to_string(),
                    |info| info.server_info.name.clone(),
                );
                ServerStatus::Connected {
                    tools_count,
                    server_name: server_info,
                }
            } else {
                ServerStatus::Disconnected
            };
            status_map.insert(server_config.id.clone(), status);
        }

        status_map
    }

    /// Shutdown all connections
    ///
    /// # Errors
    /// Returns an error if shutting down a client fails.
    pub async fn shutdown(self) -> Result<()> {
        for (server_id, client) in self.clients.write().await.drain() {
            log::info!("Shutting down connection to '{server_id}'");
            if let Ok(client) = Arc::try_unwrap(client) {
                client.shutdown().await?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ServerStatus {
    Connected {
        tools_count: usize,
        server_name: String,
    },
    Disconnected,
}
