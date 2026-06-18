//! Orchestrator-side [`ToolImplementation`] that forwards execution to the
//! sandbox over gRPC.
//!
//! Mirrors `neuromance_tools::mcp::McpToolAdapter`: each adapter advertises a
//! sandbox-hosted tool's definition and auto-approval flag, and its `execute`
//! sends the call over the gRPC channel. Approval is decided by the
//! orchestrator's `Core` before `execute` is ever called.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use neuromance_common::delegation;
use neuromance_common::tools::{Function, Tool};
use neuromance_tools::{ToolError, ToolImplementation};

use super::client::SandboxClient;
use super::proto::ToolDefinition;
use crate::error::RuntimeError;

/// The session key for the current agent run: the ambient task id when one is in
/// scope, else empty. Serve mode seeds a fresh task id per task (one
/// `execute_with_history` call), so `execute_python` gets a per-run interpreter
/// whose state never bleeds across tasks; the orchestrator frees it with
/// [`SandboxClient::close_session`] once the run completes. Stateless tools
/// ignore the session entirely. Empty outside any task scope (e.g. oneshot),
/// which the sandbox maps to its single default interpreter.
fn run_session_id() -> String {
    delegation::current()
        .task_id
        .map(|id| id.to_string())
        .unwrap_or_default()
}

/// A sandbox-hosted tool, executed remotely over gRPC.
pub struct RemoteToolAdapter {
    client: SandboxClient,
    definition: Tool,
    auto_approved: bool,
}

impl RemoteToolAdapter {
    /// Build an adapter from a [`ToolDefinition`] advertised by the sandbox.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] if the advertised `parameters_json` is
    /// not valid JSON.
    pub fn new(client: SandboxClient, def: ToolDefinition) -> Result<Self, RuntimeError> {
        let parameters: Value = serde_json::from_str(&def.parameters_json).map_err(|e| {
            RuntimeError::Config(format!(
                "sandbox tool '{}' advertised invalid parameters: {e}",
                def.name
            ))
        })?;
        let definition = Tool::builder()
            .function(Function {
                name: def.name,
                description: def.description,
                parameters,
            })
            .build();
        Ok(Self {
            client,
            definition,
            auto_approved: def.auto_approved,
        })
    }
}

#[async_trait]
impl ToolImplementation for RemoteToolAdapter {
    fn get_definition(&self) -> Tool {
        self.definition.clone()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        let arguments_json =
            serde_json::to_string(args).map_err(|e| ToolError::Execution(e.into()))?;
        let name = self.definition.function.name.clone();

        // A transport/connection failure is not the tool's fault; surface it as
        // a tool error so the loop feeds it back to the LLM rather than crashing.
        let response = self
            .client
            .execute_tool(name.clone(), arguments_json, run_session_id())
            .await
            .map_err(|status| {
                tracing::warn!(tool = %name, %status, "sandbox transport error");
                ToolError::execution(format!("sandbox unavailable: {status}"))
            })?;

        if response.is_error {
            Err(ToolError::execution(response.content))
        } else {
            Ok(response.content)
        }
    }

    fn is_auto_approved(&self) -> bool {
        self.auto_approved
    }
}

/// Build adapters for the sandbox's tools over `client`, retrying briefly so a
/// sandbox container that is still starting does not crash-loop the
/// orchestrator.
///
/// # Errors
/// Returns [`RuntimeError`] if the sandbox stays unreachable across all retries.
pub async fn connect_tools(
    client: &SandboxClient,
) -> Result<Vec<Arc<dyn ToolImplementation>>, RuntimeError> {
    const ATTEMPTS: u32 = 10;
    const BACKOFF: std::time::Duration = std::time::Duration::from_millis(500);

    let mut last_err = None;
    for attempt in 1..=ATTEMPTS {
        match remote_tools(client).await {
            Ok(tools) => return Ok(tools),
            Err(e) => {
                tracing::warn!(attempt, error = %e, "sandbox not ready; retrying");
                last_err = Some(e);
                tokio::time::sleep(BACKOFF).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| RuntimeError::Config("sandbox is unreachable".to_string())))
}

/// Connect to the sandbox and build a [`ToolImplementation`] adapter for each
/// tool it hosts.
///
/// # Errors
/// Returns [`RuntimeError`] if the sandbox cannot be reached or advertises an
/// invalid tool.
pub async fn remote_tools(
    client: &SandboxClient,
) -> Result<Vec<Arc<dyn ToolImplementation>>, RuntimeError> {
    let definitions = client
        .list_tools()
        .await
        .map_err(|status| RuntimeError::Other(anyhow::anyhow!("sandbox ListTools: {status}")))?;

    definitions
        .into_iter()
        .map(|def| {
            RemoteToolAdapter::new(client.clone(), def)
                .map(|adapter| Arc::new(adapter) as Arc<dyn ToolImplementation>)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::time::Duration;

    use serde_json::{Value, json};
    use tokio_util::sync::CancellationToken;

    use neuromance_tools::ToolConfig;

    use super::*;
    use crate::sandbox::server::build_sandbox_toolset;

    fn tool(name: &str) -> ToolConfig {
        ToolConfig {
            name: name.to_string(),
            config: Value::Null,
        }
    }

    /// Spawn a sandbox server on a loopback port and return a connected client
    /// plus a cancel handle. Retries the first RPC until the server is ready.
    async fn spawn_sandbox(tools: &[ToolConfig]) -> (SandboxClient, CancellationToken) {
        let toolset = Arc::new(build_sandbox_toolset(tools).unwrap());
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        drop(listener);

        let cancel = CancellationToken::new();
        let serve_cancel = cancel.clone();
        tokio::spawn(
            async move { crate::sandbox::server::serve(toolset, addr, serve_cancel).await },
        );

        let client = SandboxClient::connect(&format!("http://{addr}")).unwrap();
        // Wait for the server to bind.
        for _ in 0..50 {
            if client.list_tools().await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        (client, cancel)
    }

    /// An adapter advertises the sandbox tool's name and parameters, and
    /// executing it round-trips the result back.
    #[tokio::test]
    async fn test_adapter_round_trips_definition_and_execution() {
        let (client, cancel) = spawn_sandbox(&[tool("bash")]).await;
        let tools = remote_tools(&client).await.unwrap();

        let bash = tools
            .iter()
            .find(|t| t.get_definition().function.name == "bash")
            .expect("bash adapter");
        assert!(bash.get_definition().function.parameters.is_object());

        let out = bash
            .execute(&json!({ "command": "echo via-adapter" }))
            .await
            .unwrap();
        assert!(out.contains("via-adapter"), "{out}");
        cancel.cancel();
    }

    /// The adapter reports each tool's remote auto-approval flag faithfully:
    /// `ls` is auto-approved, `bash` is not. (Guards against the adapter
    /// hard-coding a single value, which would bypass approval.)
    #[tokio::test]
    async fn test_adapter_preserves_auto_approval() {
        let (client, cancel) = spawn_sandbox(&[tool("ls"), tool("bash")]).await;
        let tools = remote_tools(&client).await.unwrap();
        let approved: HashMap<String, bool> = tools
            .iter()
            .map(|t| (t.get_definition().function.name, t.is_auto_approved()))
            .collect();

        assert!(approved["ls"], "ls is read-only and auto-approved");
        assert!(!approved["bash"], "bash requires approval");
        cancel.cancel();
    }

    /// A tool that runs but fails surfaces as an Err whose message carries the
    /// cause, not a silent success.
    #[tokio::test]
    async fn test_adapter_maps_execution_error() {
        let (client, cancel) = spawn_sandbox(&[tool("bash")]).await;
        let tools = remote_tools(&client).await.unwrap();
        let bash = tools.into_iter().next().unwrap();

        // Missing the required "command" argument -> InvalidArguments.
        let err = bash.execute(&json!({})).await.unwrap_err();
        assert!(err.to_string().to_lowercase().contains("command"), "{err}");
        cancel.cancel();
    }

    /// `execute_python` runs in the sandbox over gRPC, and a result larger than
    /// tonic's 4 MiB default still round-trips (the message-size limit is
    /// raised on both ends).
    #[cfg(feature = "python-repl")]
    #[tokio::test]
    #[serial_test::serial]
    async fn test_execute_python_large_output_round_trips() {
        let (client, cancel) = spawn_sandbox(&[tool("execute_python")]).await;
        let tools = remote_tools(&client).await.unwrap();
        let python = tools
            .iter()
            .find(|t| t.get_definition().function.name == "execute_python")
            .expect("execute_python adapter");

        // Print ~5 MiB, past the 4 MiB default decode limit.
        let out = python
            .execute(&json!({ "code": "print('x' * (5 * 1024 * 1024))" }))
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["status"], "success", "{out:.200}");
        assert!(parsed["stdout"].as_str().unwrap().len() >= 5 * 1024 * 1024);
        cancel.cancel();
    }

    /// `execute_python` isolation is driven by the ambient task id: two runs
    /// under different task scopes get independent interpreters, and releasing a
    /// run's session frees its interpreter so a later run with the same id
    /// starts fresh. Exercises the production path — the session id is derived
    /// from the delegation context, not hand-passed.
    #[cfg(feature = "python-repl")]
    #[tokio::test]
    #[serial_test::serial]
    async fn test_execute_python_isolated_per_task_scope() {
        use uuid::Uuid;

        use neuromance_common::delegation::{self, DelegationContext};

        async fn run_in(python: &Arc<dyn ToolImplementation>, task: Uuid, code: &str) -> Value {
            let ctx = DelegationContext {
                conversation_id: None,
                task_id: Some(task),
            };
            let out = delegation::scope(ctx, python.execute(&json!({ "code": code })))
                .await
                .unwrap();
            serde_json::from_str(&out).unwrap()
        }

        let (client, cancel) = spawn_sandbox(&[tool("execute_python")]).await;
        let tools = remote_tools(&client).await.unwrap();
        let python = tools
            .iter()
            .find(|t| t.get_definition().function.name == "execute_python")
            .expect("execute_python adapter");

        let task_a = Uuid::new_v4();
        let task_b = Uuid::new_v4();

        // Task A defines a variable and reads it back within its own scope.
        assert_eq!(
            run_in(python, task_a, "stash = 7").await["status"],
            "success"
        );
        let read = run_in(python, task_a, "print(stash)").await;
        assert!(read["stdout"].as_str().unwrap().contains('7'), "{read}");

        // Task B gets a separate interpreter: it cannot see Task A's variable.
        let other = run_in(python, task_b, "print(stash)").await;
        assert_eq!(other["status"], "error", "task b leaked task a state");

        // Releasing Task A's session frees its interpreter; the id starts fresh.
        client.close_session(task_a.to_string()).await.unwrap();
        let after = run_in(python, task_a, "print(stash)").await;
        assert_eq!(after["status"], "error", "closed session retained state");

        cancel.cancel();
    }

    /// When the sandbox is unreachable, execute returns a transport error
    /// rather than panicking.
    #[tokio::test]
    async fn test_adapter_transport_error_when_sandbox_down() {
        // A loopback port with nothing listening.
        let client = SandboxClient::connect("http://127.0.0.1:1").unwrap();
        let adapter = RemoteToolAdapter::new(
            client,
            ToolDefinition {
                name: "bash".to_string(),
                description: "bash".to_string(),
                parameters_json: "{}".to_string(),
                auto_approved: false,
            },
        )
        .unwrap();

        let err = adapter
            .execute(&json!({ "command": "echo hi" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("sandbox unavailable"), "{err}");
    }
}
