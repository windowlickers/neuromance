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
    pub async fn from_config_file(path: &Path) -> Result<Self> {
        let config = McpConfig::from_file(path)?;
        Self::new(config).await
    }

    /// Create a new MCP manager with the given configuration
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

        Ok(())
    }

    /// Connect to a specific MCP server
    pub async fn connect_server(&self, server_id: &str) -> Result<()> {
        let server_config = self
            .config
            .servers
            .iter()
            .find(|s| s.id == server_id)
            .ok_or_else(|| anyhow::anyhow!("Server '{}' not found in configuration", server_id))?
            .clone();

        let client = McpClientWrapper::connect(server_config).await?;

        let mut clients = self.clients.write().await;
        clients.insert(server_id.to_string(), Arc::new(client));

        Ok(())
    }

    /// Disconnect from a specific MCP server
    pub async fn disconnect_server(&self, server_id: &str) -> Result<()> {
        let mut clients = self.clients.write().await;
        if let Some(client) = clients.remove(server_id) {
            // Shut down the client gracefully
            if let Ok(client) = Arc::try_unwrap(client) {
                client.shutdown().await?;
            }
        }
        Ok(())
    }

    /// Get all available tools from all connected MCP servers
    pub async fn get_all_tools(&self) -> Result<Vec<Arc<dyn ToolImplementation>>> {
        let mut tools: Vec<Arc<dyn ToolImplementation>> = Vec::new();
        let clients = self.clients.read().await;

        for (server_id, client) in clients.iter() {
            let mcp_tools = client.get_tools().await;

            for mcp_tool in mcp_tools {
                let adapter = McpToolAdapter::new(server_id.clone(), client.clone(), mcp_tool);
                tools.push(Arc::new(adapter) as Arc<dyn ToolImplementation>);
            }
        }

        Ok(tools)
    }

    /// Get a specific tool by name
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

        let clients = self.clients.read().await;
        let client = clients
            .get(server_id)
            .ok_or_else(|| anyhow::anyhow!("Server '{}' not connected", server_id))?;

        let tools = client.tools.read().await;
        let mcp_tool = tools
            .get(tool_name)
            .ok_or_else(|| {
                anyhow::anyhow!("Tool '{}' not found on server '{}'", tool_name, server_id)
            })?
            .clone();

        let adapter = McpToolAdapter::new(server_id.to_string(), client.clone(), mcp_tool);

        Ok(Arc::new(adapter) as Arc<dyn ToolImplementation>)
    }

    /// Refresh tools for all connected servers
    pub async fn refresh_all_tools(&self) -> Result<()> {
        let clients = self.clients.read().await;

        for (server_id, client) in clients.iter() {
            log::info!("Refreshing tools for server '{}'...", server_id);
            client.refresh_tools().await?;
        }

        Ok(())
    }

    /// Get the status of all MCP servers
    pub async fn get_status(&self) -> HashMap<String, ServerStatus> {
        let mut status_map = HashMap::new();
        let clients = self.clients.read().await;

        for server_config in &self.config.servers {
            let status = if let Some(client) = clients.get(&server_config.id) {
                let tools_count = client.get_tools().await.len();
                let server_info = client
                    .service
                    .peer_info()
                    .map(|info| info.server_info.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());
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
    pub async fn shutdown(self) -> Result<()> {
        let mut clients = self.clients.write().await;
        for (server_id, client) in clients.drain() {
            log::info!("Shutting down connection to '{}'", server_id);
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
