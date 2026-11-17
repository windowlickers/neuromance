//! Connect the `[[mcp_servers]]` that supply the agent's tools.
//!
//! Servers connect once at startup. Every agent build — the shared agent, a
//! per-task model override, each subagent run — reuses the same adapters, so a
//! stdio server is spawned once per process, not once per task.

use std::sync::Arc;
use std::time::Duration;

use neuromance_tools::ToolImplementation;
use neuromance_tools::mcp::McpServerConfig;
use neuromance_tools::mcp::adapter::McpToolAdapter;
use neuromance_tools::mcp::client::McpClientWrapper;
use tracing::{info, warn};

use crate::error::RuntimeError;

/// Connection attempts per server. A sidecar server in the same pod can start
/// after the runtime, so a refused connection is retried before startup fails.
const ATTEMPTS: u32 = 10;
const BACKOFF: Duration = Duration::from_millis(500);

/// Connect every configured server and return one tool adapter per tool it
/// advertises.
///
/// # Errors
/// Returns [`RuntimeError::Mcp`] naming the first server that stays
/// unreachable across all attempts.
pub async fn connect(
    servers: &[McpServerConfig],
) -> Result<Vec<Arc<dyn ToolImplementation>>, RuntimeError> {
    let mut tools = Vec::new();
    for server in servers {
        tools.extend(connect_server(server).await?);
    }
    Ok(tools)
}

async fn connect_server(
    server: &McpServerConfig,
) -> Result<Vec<Arc<dyn ToolImplementation>>, RuntimeError> {
    let client = Arc::new(connect_with_retry(server).await?);
    let tools: Vec<Arc<dyn ToolImplementation>> = client
        .get_tools()
        .await
        .into_iter()
        .map(|tool| {
            Arc::new(McpToolAdapter::new(
                server.id.clone(),
                Arc::clone(&client),
                tool,
            )) as Arc<dyn ToolImplementation>
        })
        .collect();
    info!(server = %server.id, tools = tools.len(), "MCP server connected");
    Ok(tools)
}

async fn connect_with_retry(server: &McpServerConfig) -> Result<McpClientWrapper, RuntimeError> {
    let mut attempt = 1;
    loop {
        match McpClientWrapper::connect(server.clone()).await {
            Ok(client) => return Ok(client),
            Err(source) if attempt >= ATTEMPTS => {
                return Err(RuntimeError::Mcp {
                    server: server.id.clone(),
                    source,
                });
            }
            Err(e) => {
                warn!(server = %server.id, attempt, error = %e, "MCP server not ready; retrying");
                attempt += 1;
                tokio::time::sleep(BACKOFF).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::collections::HashMap;

    use neuromance_tools::mcp::McpTransportConfig;

    use super::*;

    #[tokio::test]
    async fn test_connect_without_servers_yields_no_tools() {
        assert!(connect(&[]).await.unwrap().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn test_connect_names_unreachable_server_after_retries() {
        let server = McpServerConfig {
            id: "ghost".to_string(),
            name: "Ghost".to_string(),
            transport: McpTransportConfig::Stdio {
                command: "/nonexistent/neuromance-mcp-test".to_string(),
                args: Vec::new(),
                env: HashMap::new(),
            },
            description: None,
            auto_approve: false,
            working_directory: None,
        };
        let err = connect(&[server]).await.err().expect("connect must fail");
        assert!(
            matches!(err, RuntimeError::Mcp { ref server, .. } if server == "ghost"),
            "{err}"
        );
    }
}
