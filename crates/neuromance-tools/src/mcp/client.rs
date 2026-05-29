use anyhow::Result;
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, CallToolResult, Tool as McpTool},
    service::{RunningService, ServerSink},
    transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess},
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::config::{McpServerConfig, McpTransportConfig};

/// Wrapper around an MCP client connection
pub struct McpClientWrapper {
    pub server_config: McpServerConfig,
    pub service: RunningService<RoleClient, ()>,
    pub tools: Arc<RwLock<HashMap<String, McpTool>>>,
}

impl McpClientWrapper {
    /// Connect to an MCP server using the provided configuration
    ///
    /// # Errors
    /// Returns an error if the connection fails.
    pub async fn connect(config: McpServerConfig) -> Result<Self> {
        info!(server = %config.id, "connecting to MCP server");

        // Create the appropriate transport based on configuration
        let service = match &config.transport {
            McpTransportConfig::Stdio { command, args, env } => {
                let cmd = tokio::process::Command::new(command);
                let cmd = cmd.configure(|c| {
                    c.args(args).envs(env.clone());

                    if let Some(ref cwd) = config.working_directory {
                        c.current_dir(cwd);
                    }
                });

                let transport = TokioChildProcess::new(cmd)?;
                ().serve(transport).await?
            }
            McpTransportConfig::Sse { url } | McpTransportConfig::Http { url } => {
                let transport = StreamableHttpClientTransport::from_uri(url.clone());
                ().serve(transport).await?
            }
        };

        // Get server info
        if let Some(peer_info) = service.peer_info() {
            info!(
                server = %config.id,
                server_name = %peer_info.server_info.name,
                "connected to MCP server",
            );
        }

        // Fetch available tools
        let tools_list = service.list_all_tools().await?;

        info!(
            server = %config.id,
            tools = tools_list.len(),
            "MCP server tools listed",
        );

        let mut tools_map = HashMap::new();
        for tool in tools_list {
            debug!(server = %config.id, tool = %tool.name, "MCP tool available");
            tools_map.insert(tool.name.clone().to_string(), tool);
        }

        Ok(Self {
            server_config: config,
            service,
            tools: Arc::new(RwLock::new(tools_map)),
        })
    }

    /// Get the server sink for making calls
    #[must_use]
    pub fn peer(&self) -> ServerSink {
        self.service.peer().clone()
    }

    /// Call a tool on this MCP server
    ///
    /// # Errors
    /// Returns an error if the tool call fails.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult> {
        debug!(
            server = %self.server_config.id,
            tool = %name,
            "calling MCP tool",
        );

        let arguments = match arguments {
            serde_json::Value::Object(map) => Some(map),
            _ => None,
        };

        let request = CallToolRequestParams {
            meta: None,
            name: name.to_owned().into(),
            arguments,
            task: None,
        };

        let result = self.service.peer().call_tool(request).await?;
        Ok(result)
    }

    /// Refresh the list of available tools
    ///
    /// # Errors
    /// Returns an error if fetching tools fails.
    pub async fn refresh_tools(&self) -> Result<()> {
        debug!(server = %self.server_config.id, "refreshing MCP tools");

        let tools_list = self.service.list_all_tools().await?;

        let mut tools = self.tools.write().await;
        tools.clear();
        for tool in tools_list {
            tools.insert(tool.name.clone().to_string(), tool);
        }
        drop(tools);

        Ok(())
    }

    /// Get all available tools
    pub async fn get_tools(&self) -> Vec<McpTool> {
        let tools = self.tools.read().await;
        tools.values().cloned().collect()
    }

    /// Check if a specific tool is available
    pub async fn has_tool(&self, name: &str) -> bool {
        let tools = self.tools.read().await;
        tools.contains_key(name)
    }

    /// Shutdown the client
    ///
    /// # Errors
    /// Returns an error if shutdown fails.
    pub async fn shutdown(self) -> Result<()> {
        self.service.cancel().await?;
        Ok(())
    }
}
