//! Tool implementation for executing Python code via LLMs.
//!
//! This module provides a [`ToolImplementation`] for the Python REPL, allowing
//! LLMs to execute Python code as a tool call.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use neuromance_common::tools::{Function, Parameters, Property, Tool};
use neuromance_tools::{ToolError, ToolFactory, ToolImplementation, ToolRegistry};

use crate::{ReplError, ReplResult};

use super::{InteractivePythonRepl, PythonRepl};

/// Backing REPL for [`PythonReplTool`].
#[derive(Debug)]
enum Backend {
    /// Restricted-builtins REPL with a Rust-backed import allowlist.
    Restricted(Arc<PythonRepl>),
    /// Unrestricted REPL — full builtins, unfiltered imports. Suitable when the
    /// surrounding container is the security boundary.
    Unrestricted(Arc<InteractivePythonRepl>),
}

impl Backend {
    async fn execute(&self, code: &str) -> Result<ReplResult, ReplError> {
        match self {
            Self::Restricted(repl) => repl.execute(code).await,
            Self::Unrestricted(repl) => repl.execute(code).await,
        }
    }

    async fn reset(&self) -> Result<(), ReplError> {
        match self {
            Self::Restricted(repl) => repl.reset().await,
            Self::Unrestricted(repl) => repl.reset().await,
        }
    }
}

/// Tool implementation for executing Python code in a REPL environment.
///
/// This tool wraps a [`PythonRepl`] instance and provides it as a tool that can be
/// called by LLMs. The REPL state persists between tool calls, allowing for
/// stateful computations.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use neuromance_repl::python::{PythonRepl, PythonReplTool};
/// use neuromance_tools::ToolExecutor;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let repl = Arc::new(PythonRepl::new()?);
/// let tool = PythonReplTool::new(repl);
///
/// let mut executor = ToolExecutor::new();
/// executor.add_tool(tool);
///
/// let tools = executor.get_all_tools();
/// // Pass tools to LLM client...
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PythonReplTool {
    backend: Backend,
}

impl PythonReplTool {
    /// Create a new Python REPL tool backed by a restricted [`PythonRepl`].
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Arc is not const-constructible
    pub fn new(repl: Arc<PythonRepl>) -> Self {
        Self {
            backend: Backend::Restricted(repl),
        }
    }

    /// Create a new Python REPL tool backed by an unrestricted
    /// [`InteractivePythonRepl`].
    ///
    /// Pick this only when the surrounding container provides the security
    /// boundary — the in-process builtin allowlist and import filter are
    /// disabled in this mode.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Arc is not const-constructible
    pub fn with_interactive(repl: Arc<InteractivePythonRepl>) -> Self {
        Self {
            backend: Backend::Unrestricted(repl),
        }
    }

    /// Clear the interpreter's user namespace.
    ///
    /// Drops variables, functions, classes, and imports the executed code
    /// defined (these bind into the REPL's locals), while keeping the
    /// configured baseline — allowlisted builtins, configured modules, and any
    /// injected Rust callbacks — which live in globals. Use this to keep one
    /// task's state from bleeding into the next when a single tool instance is
    /// reused across runs.
    ///
    /// # Errors
    ///
    /// Returns [`ReplError`] if the reset fails (e.g. the state mutex is
    /// poisoned).
    pub async fn reset(&self) -> Result<(), ReplError> {
        self.backend.reset().await
    }
}

#[async_trait]
impl ToolImplementation for PythonReplTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "code".to_string(),
            Property::string(
                "The Python code to execute. Can include multiple \
                 statements and definitions. Use print() to see output.",
            ),
        );

        let function = Function {
            name: "execute_python".to_string(),
            description: "Execute Python code in a persistent REPL environment. \
                 State persists between calls - variables and functions defined \
                 in previous executions are available. \
                 Use this to perform calculations, data processing, or any \
                 computational task. \
                 IMPORTANT: To see results, you MUST use print() statements. \
                 Expression values are NOT automatically displayed - assign to variables \
                 or print() them explicitly. Returns stdout, stderr, and execution status."
                .to_string(),
            parameters: Parameters::new(properties, vec!["code".into()]).into(),
        };

        Tool::builder().function(function).build()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        let obj = args
            .as_object()
            .ok_or_else(|| ToolError::InvalidArguments("Expected object arguments".into()))?;

        let code = obj.get("code").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidArguments("Missing or invalid 'code' argument".into())
        })?;

        tracing::debug!(code, "executing python code");

        let result = self
            .backend
            .execute(code)
            .await
            .map_err(|e| ToolError::Execution(e.into()))?;

        if result.success {
            tracing::debug!(
                duration_ms = result.execution_time_ms,
                "python execution succeeded",
            );
            if !result.stdout.is_empty() {
                tracing::debug!(stdout = %result.stdout, "python stdout");
            }
            if let Some(ref return_value) = result.return_value {
                tracing::debug!(return_value = %return_value, "python return value");
            }
        } else {
            tracing::warn!(
                duration_ms = result.execution_time_ms,
                "python execution failed",
            );
            if !result.stderr.is_empty() {
                tracing::warn!(stderr = %result.stderr, "python stderr");
            }
        }

        // Format tool response
        let response = if result.success {
            json!({
                "status": "success",
                "stdout": result.stdout,
                "return_value": result.return_value,
                "execution_time_ms": result.execution_time_ms
            })
            .to_string()
        } else {
            json!({
                "status": "error",
                "stderr": result.stderr,
                "execution_time_ms": result.execution_time_ms
            })
            .to_string()
        };

        Ok(response)
    }

    fn is_auto_approved(&self) -> bool {
        // Restricted mode sandboxes via the builtin allowlist and import filter.
        // Unrestricted mode relies on the surrounding container as the security
        // boundary; either way per-call approval is unnecessary in this runtime.
        true
    }
}

/// Tool config for [`PythonReplToolFactory`].
///
/// Deserialized from the `[tools.config]` block of a `[[tools]]` entry whose
/// `name = "execute_python"`. Defaults to restricted mode.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PythonReplToolConfig {
    /// When `true` (default), use the sandboxed [`PythonRepl`] with restricted
    /// builtins and a Rust-backed import allowlist. When `false`, use the full
    /// [`InteractivePythonRepl`] — file I/O, networking, and arbitrary imports
    /// are permitted.
    restricted: bool,
}

impl Default for PythonReplToolConfig {
    fn default() -> Self {
        Self { restricted: true }
    }
}

/// Factory that registers [`PythonReplTool`] under the name `execute_python`.
///
/// Reads an optional `restricted` boolean from the tool's config block:
///
/// ```toml
/// [[tools]]
/// name = "execute_python"
/// [tools.config]
/// restricted = false  # default: true
/// ```
///
/// State persists across tool calls within a runtime instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PythonReplToolFactory;

impl PythonReplToolFactory {
    /// Parse the `execute_python` config block into its restricted-mode flag.
    ///
    /// Applies the same `deny_unknown_fields` validation as [`build_tool`], so
    /// callers that build their own interpreters (e.g. the sandbox server)
    /// interpret the config identically and reject the same malformed blocks
    /// instead of silently defaulting to restricted.
    ///
    /// [`build_tool`]: PythonReplToolFactory::build_tool
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the config block is malformed (unknown field,
    /// wrong-typed value).
    pub fn parse_restricted(config: &Value) -> Result<bool, ToolError> {
        let cfg: PythonReplToolConfig = if config.is_null() {
            PythonReplToolConfig::default()
        } else {
            serde_json::from_value(config.clone())
                .map_err(|e| ToolError::execution(format!("invalid execute_python config: {e}")))?
        };
        Ok(cfg.restricted)
    }

    /// Construct the `execute_python` tool from its config block.
    ///
    /// Returns the typed `Arc` so callers that need a handle on the tool (e.g.
    /// to [`reset`](PythonReplTool::reset) it between runs) can keep one,
    /// rather than only registering it behind `dyn ToolImplementation`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the config block is malformed or the
    /// interpreter fails to initialize.
    pub fn build_tool(config: &Value) -> Result<Arc<PythonReplTool>, ToolError> {
        let restricted = Self::parse_restricted(config)?;

        let tool = if restricted {
            PythonReplTool::new(Arc::new(PythonRepl::new()?))
        } else {
            tracing::warn!(
                "execute_python registered in unrestricted mode: full builtins and unfiltered imports"
            );
            PythonReplTool::with_interactive(Arc::new(InteractivePythonRepl::new()?))
        };
        Ok(Arc::new(tool))
    }
}

impl ToolFactory for PythonReplToolFactory {
    fn name(&self) -> &'static str {
        "execute_python"
    }

    fn build(&self, config: &Value, registry: &ToolRegistry) -> Result<(), ToolError> {
        registry.register(Self::build_tool(config)?);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use serial_test::serial;

    fn make_tool() -> PythonReplTool {
        PythonReplTool::new(Arc::new(PythonRepl::new().unwrap()))
    }

    #[test]
    fn test_get_definition_function_name_and_required() {
        let tool = make_tool();
        let def = tool.get_definition();

        assert_eq!(def.function.name, "execute_python");

        let required = def
            .function
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .expect("parameters.required must be an array");
        let required_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            required_strs.contains(&"code"),
            "required missing 'code': {required_strs:?}"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_rejects_non_object_args() {
        let tool = make_tool();
        let err = tool.execute(&json!("not an object")).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(msg) if msg.contains("object")));
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_rejects_missing_code_arg() {
        let tool = make_tool();
        let err = tool.execute(&json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(msg) if msg.contains("code")));
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_rejects_non_string_code_arg() {
        let tool = make_tool();
        let err = tool.execute(&json!({ "code": 42 })).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(msg) if msg.contains("code")));
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_returns_success_json_shape() {
        let tool = make_tool();
        let response = tool
            .execute(&json!({ "code": "print('hello')" }))
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["status"], "success");
        assert!(parsed["stdout"].as_str().unwrap().contains("hello"));
        assert!(parsed["execution_time_ms"].is_number());
        assert!(parsed.get("return_value").is_some());
        assert!(parsed.get("stderr").is_none());
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_returns_error_json_shape() {
        let tool = make_tool();
        let response = tool.execute(&json!({ "code": "1 / 0" })).await.unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["status"], "error");
        assert!(
            parsed["stderr"]
                .as_str()
                .unwrap()
                .contains("ZeroDivisionError")
        );
        assert!(parsed["execution_time_ms"].is_number());
        assert!(parsed.get("stdout").is_none());
        assert!(parsed.get("return_value").is_none());
    }

    #[tokio::test]
    #[serial]
    async fn test_reset_clears_user_state() {
        let tool = make_tool();

        tool.execute(&json!({ "code": "marker = 41" }))
            .await
            .unwrap();
        let before: Value = serde_json::from_str(
            &tool
                .execute(&json!({ "code": "print(marker)" }))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(before["status"], "success");
        assert!(before["stdout"].as_str().unwrap().contains("41"));

        tool.reset().await.unwrap();

        let after: Value = serde_json::from_str(
            &tool
                .execute(&json!({ "code": "print(marker)" }))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(after["status"], "error");
        assert!(after["stderr"].as_str().unwrap().contains("NameError"));
    }

    #[test]
    fn test_factory_default_config_is_restricted() {
        let cfg = PythonReplToolConfig::default();
        assert!(cfg.restricted);
    }

    #[test]
    fn test_factory_config_rejects_unknown_fields() {
        let err = serde_json::from_value::<PythonReplToolConfig>(json!({
            "restricted": false,
            "bogus": 1,
        }))
        .unwrap_err();
        assert!(err.to_string().contains("bogus"));
    }

    #[tokio::test]
    #[serial]
    async fn test_unrestricted_tool_allows_open_builtin() {
        let repl = Arc::new(InteractivePythonRepl::new().unwrap());
        let tool = PythonReplTool::with_interactive(repl);
        let response = tool
            .execute(&json!({ "code": "print(callable(open))" }))
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["status"], "success");
        assert!(parsed["stdout"].as_str().unwrap().contains("True"));
    }

    #[tokio::test]
    #[serial]
    async fn test_restricted_tool_blocks_open_builtin() {
        let tool = make_tool();
        let response = tool
            .execute(&json!({ "code": "open('/etc/passwd')" }))
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["status"], "error");
        assert!(parsed["stderr"].as_str().unwrap().contains("NameError"));
    }
}
