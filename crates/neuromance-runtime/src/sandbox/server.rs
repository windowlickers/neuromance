//! Sandbox-side gRPC server: executes capability tools on behalf of the
//! orchestrator.
//!
//! [`SandboxToolset`] holds the stateless capability tools (built from the same
//! `[[tools]]` factory path the orchestrator uses) plus, under the
//! `python-repl` feature, session-scoped Python interpreters. The orchestrator
//! has already applied approval before calling [`SandboxToolService::execute_tool`];
//! the sandbox just runs the tool and returns its result.

use std::net::SocketAddr;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

use neuromance_common::tools::Tool;
use neuromance_tools::{ToolConfig, ToolExecutor, ToolExecutorError, ToolFactoryRegistry};

use super::proto::sandbox_tool_service_server::{SandboxToolService, SandboxToolServiceServer};
use super::proto::{
    CloseSessionRequest, CloseSessionResponse, ExecuteToolRequest, ExecuteToolResponse,
    ListToolsRequest, ListToolsResponse, ToolDefinition,
};
use super::{EXECUTE_PYTHON, MAX_MESSAGE_SIZE};
use crate::error::RuntimeError;

/// The tools the sandbox hosts and executes.
pub struct SandboxToolset {
    /// Stateless capability tools (`bash`, file tools, `grep`/`find`/`ls`).
    executor: ToolExecutor,
    /// Session-scoped Python interpreters, present when `execute_python` is
    /// configured and the `python-repl` feature is enabled.
    #[cfg(feature = "python-repl")]
    python: Option<python::PythonSessions>,
}

impl SandboxToolset {
    /// Definitions of every hosted tool paired with its auto-approval flag, for
    /// the orchestrator to advertise and to mirror approval requirements.
    fn tool_definitions(&self) -> Vec<(Tool, bool)> {
        let defs: Vec<(Tool, bool)> = self
            .executor
            .get_all_tools()
            .into_iter()
            .map(|tool| {
                let approved = self.executor.is_tool_auto_approved(&tool.function.name);
                (tool, approved)
            })
            .collect();

        #[cfg(feature = "python-repl")]
        let defs = {
            let mut defs = defs;
            if let Some(py) = &self.python {
                defs.push((py.definition.clone(), py.auto_approved));
            }
            defs
        };

        defs
    }

    /// Execute a tool by name. `session_id` scopes stateful interpreters and is
    /// ignored by stateless tools.
    #[cfg_attr(not(feature = "python-repl"), allow(unused_variables))]
    async fn execute(
        &self,
        name: &str,
        arguments_json: &str,
        session_id: &str,
    ) -> Result<String, ToolExecutorError> {
        #[cfg(feature = "python-repl")]
        if name == EXECUTE_PYTHON && let Some(py) = &self.python {
            let session = py.get_or_create(session_id).map_err(|e| {
                ToolExecutorError::Tool(neuromance_tools::ToolError::execution(e.to_string()))
            })?;
            return session.execute_named(name, arguments_json).await;
        }
        self.executor.execute_named(name, arguments_json).await
    }
}

/// Build the sandbox toolset from the `[[tools]]` config.
///
/// `execute_python` is handled separately (session-scoped, see [`SandboxToolset`])
/// and is excluded from the stateless factory build.
///
/// # Errors
/// Returns [`RuntimeError`] if a tool factory fails, or if `execute_python` is
/// configured without the `python-repl` feature.
pub fn build_sandbox_toolset(tools: &[ToolConfig]) -> Result<SandboxToolset, RuntimeError> {
    let stateless: Vec<ToolConfig> = tools
        .iter()
        .filter(|t| t.name != EXECUTE_PYTHON)
        .cloned()
        .collect();
    let factories = ToolFactoryRegistry::with_builtin();
    let executor = ToolExecutor::from_registry(factories.build_all(&stateless)?);

    #[cfg(not(feature = "python-repl"))]
    if tools.iter().any(|t| t.name == EXECUTE_PYTHON) {
        return Err(RuntimeError::Config(
            "execute_python is configured but the python-repl feature is not enabled".to_string(),
        ));
    }

    Ok(SandboxToolset {
        executor,
        #[cfg(feature = "python-repl")]
        python: python::PythonSessions::from_config(tools)?,
    })
}

/// gRPC service over a [`SandboxToolset`].
pub struct SandboxToolServer {
    toolset: Arc<SandboxToolset>,
}

impl SandboxToolServer {
    #[must_use]
    pub const fn new(toolset: Arc<SandboxToolset>) -> Self {
        Self { toolset }
    }
}

#[tonic::async_trait]
impl SandboxToolService for SandboxToolServer {
    async fn list_tools(
        &self,
        _request: Request<ListToolsRequest>,
    ) -> Result<Response<ListToolsResponse>, Status> {
        let tools = self
            .toolset
            .tool_definitions()
            .into_iter()
            .map(|(tool, auto_approved)| {
                let parameters_json = serde_json::to_string(&tool.function.parameters)
                    .map_err(|e| Status::internal(format!("serialize tool parameters: {e}")))?;
                Ok(ToolDefinition {
                    name: tool.function.name,
                    description: tool.function.description,
                    parameters_json,
                    auto_approved,
                })
            })
            .collect::<Result<Vec<_>, Status>>()?;

        Ok(Response::new(ListToolsResponse { tools }))
    }

    async fn execute_tool(
        &self,
        request: Request<ExecuteToolRequest>,
    ) -> Result<Response<ExecuteToolResponse>, Status> {
        let req = request.into_inner();
        // A tool that runs but fails is not a transport error: report it as a
        // result with is_error set, matching the in-process contract where the
        // loop turns a ToolError into a tool message for the LLM.
        let response = match self
            .toolset
            .execute(&req.name, &req.arguments_json, &req.session_id)
            .await
        {
            Ok(content) => ExecuteToolResponse {
                content,
                is_error: false,
            },
            Err(e) => ExecuteToolResponse {
                content: e.to_string(),
                is_error: true,
            },
        };
        Ok(Response::new(response))
    }

    async fn close_session(
        &self,
        request: Request<CloseSessionRequest>,
    ) -> Result<Response<CloseSessionResponse>, Status> {
        let _session_id = request.into_inner().session_id;
        #[cfg(feature = "python-repl")]
        if let Some(py) = &self.toolset.python {
            py.sessions.remove(&_session_id);
        }
        Ok(Response::new(CloseSessionResponse {}))
    }
}

/// Serve the sandbox tool service on `addr` until `cancel` fires.
///
/// # Errors
/// Returns [`RuntimeError`] if the server fails to bind or serve.
pub async fn serve(
    toolset: Arc<SandboxToolset>,
    addr: SocketAddr,
    cancel: CancellationToken,
) -> Result<(), RuntimeError> {
    let service = SandboxToolServiceServer::new(SandboxToolServer::new(toolset))
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE);

    tracing::info!(%addr, "sandbox tool service listening");
    tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_shutdown(addr, cancel.cancelled())
        .await
        .map_err(|e| RuntimeError::Other(anyhow::anyhow!("sandbox gRPC server: {e}")))
}

#[cfg(feature = "python-repl")]
mod python {
    //! Session-scoped Python interpreters.
    //!
    //! Each session id maps to its own interpreter so state never bleeds across
    //! agent runs (or concurrent sibling runs in a `spawn_agents` fan-out). The
    //! orchestrator mints a session id per run and releases it via `CloseSession`.

    use std::sync::Arc;

    use dashmap::DashMap;

    use neuromance_common::tools::Tool;
    use neuromance_repl::python::{InteractivePythonRepl, PythonRepl, PythonReplTool};
    use neuromance_tools::ToolExecutor;
    use serde_json::Value;

    use super::{EXECUTE_PYTHON, ToolConfig};
    use crate::error::RuntimeError;

    pub(super) struct PythonSessions {
        restricted: bool,
        pub(super) definition: Tool,
        pub(super) auto_approved: bool,
        pub(super) sessions: DashMap<String, Arc<ToolExecutor>>,
    }

    impl PythonSessions {
        /// Build the session manager when `execute_python` is configured.
        ///
        /// Eagerly builds one interpreter (the default `""` session) so the
        /// service fails fast if Python is unavailable and so the tool
        /// definition can be captured without re-deriving it.
        pub(super) fn from_config(tools: &[ToolConfig]) -> Result<Option<Self>, RuntimeError> {
            let Some(entry) = tools.iter().find(|t| t.name == EXECUTE_PYTHON) else {
                return Ok(None);
            };
            let restricted = entry.config.get("restricted") != Some(&Value::Bool(false));

            let default = Arc::new(build_executor(restricted)?);
            let definition = default
                .get_all_tools()
                .into_iter()
                .find(|t| t.function.name == EXECUTE_PYTHON)
                .ok_or_else(|| {
                    RuntimeError::Config("python tool did not advertise execute_python".to_string())
                })?;
            let auto_approved = default.is_tool_auto_approved(EXECUTE_PYTHON);

            let sessions = DashMap::new();
            sessions.insert(String::new(), default);

            Ok(Some(Self {
                restricted,
                definition,
                auto_approved,
                sessions,
            }))
        }

        /// Return the interpreter for `session_id`, building a fresh one on
        /// first use.
        pub(super) fn get_or_create(
            &self,
            session_id: &str,
        ) -> Result<Arc<ToolExecutor>, RuntimeError> {
            if let Some(existing) = self.sessions.get(session_id) {
                return Ok(Arc::clone(existing.value()));
            }
            // Build outside the map lock; a concurrent first-use of the same id
            // (rare — calls within a run are sequential) discards one build.
            let built = Arc::new(build_executor(self.restricted)?);
            Ok(Arc::clone(
                self.sessions
                    .entry(session_id.to_string())
                    .or_insert(built)
                    .value(),
            ))
        }
    }

    /// A one-tool executor wrapping a fresh Python interpreter, so the shared
    /// argument-parsing path (`execute_named`) is reused.
    fn build_executor(restricted: bool) -> Result<ToolExecutor, RuntimeError> {
        let tool = if restricted {
            PythonReplTool::new(Arc::new(PythonRepl::new().map_err(|e| {
                RuntimeError::Config(format!("build restricted python interpreter: {e}"))
            })?))
        } else {
            PythonReplTool::with_interactive(Arc::new(InteractivePythonRepl::new().map_err(
                |e| RuntimeError::Config(format!("build python interpreter: {e}")),
            )?))
        };
        let mut executor = ToolExecutor::new();
        executor.add_tool(tool);
        Ok(executor)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::collections::HashMap;

    use serde_json::{Value, json};

    use super::*;

    fn tool(name: &str) -> ToolConfig {
        ToolConfig {
            name: name.to_string(),
            config: Value::Null,
        }
    }

    fn server_with(tools: &[ToolConfig]) -> SandboxToolServer {
        let toolset = build_sandbox_toolset(tools).unwrap();
        SandboxToolServer::new(Arc::new(toolset))
    }

    /// `ListTools` advertises every hosted tool and mirrors each tool's
    /// auto-approval flag: `ls` is auto-approved, `bash` is not.
    #[tokio::test]
    async fn test_list_tools_reports_definitions_and_approval() {
        let server = server_with(&[tool("ls"), tool("bash")]);
        let resp = server
            .list_tools(Request::new(ListToolsRequest {}))
            .await
            .unwrap()
            .into_inner();

        let by_name: HashMap<String, ToolDefinition> =
            resp.tools.into_iter().map(|t| (t.name.clone(), t)).collect();

        assert!(by_name["ls"].auto_approved, "ls should be auto-approved");
        assert!(
            !by_name["bash"].auto_approved,
            "bash should not be auto-approved"
        );
        // The advertised parameters round-trip back to JSON.
        let params: Value = serde_json::from_str(&by_name["bash"].parameters_json).unwrap();
        assert!(params.is_object());
    }

    /// A tool that executes successfully returns its output with `is_error` unset.
    #[tokio::test]
    async fn test_execute_tool_runs_bash() {
        let server = server_with(&[tool("bash")]);
        let resp = server
            .execute_tool(Request::new(ExecuteToolRequest {
                name: "bash".to_string(),
                arguments_json: json!({ "command": "echo sandbox-ok" }).to_string(),
                session_id: String::new(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert!(!resp.is_error, "got error: {}", resp.content);
        assert!(resp.content.contains("sandbox-ok"), "{}", resp.content);
    }

    /// An unknown tool is reported as a result-level error, not a transport
    /// failure, so the orchestrator can feed it back to the LLM.
    #[tokio::test]
    async fn test_execute_unknown_tool_is_result_error() {
        let server = server_with(&[tool("ls")]);
        let resp = server
            .execute_tool(Request::new(ExecuteToolRequest {
                name: "does-not-exist".to_string(),
                arguments_json: "{}".to_string(),
                session_id: String::new(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert!(resp.is_error);
        assert!(resp.content.contains("does-not-exist"), "{}", resp.content);
    }
}
