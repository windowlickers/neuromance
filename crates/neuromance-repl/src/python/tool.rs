//! Tool implementation for executing Python code via LLMs.
//!
//! This module provides a [`ToolImplementation`] for the Python REPL, allowing
//! LLMs to execute Python code as a tool call.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{Value, json};

use neuromance_common::tools::{Function, Parameters, Property, Tool};
use neuromance_tools::{ToolError, ToolFactory, ToolImplementation, ToolRegistry};

use super::PythonRepl;

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
pub struct PythonReplTool {
    repl: Arc<PythonRepl>,
}

impl PythonReplTool {
    /// Create a new Python REPL tool with the given REPL instance.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Arc is not const-constructible
    pub fn new(repl: Arc<PythonRepl>) -> Self {
        Self { repl }
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

        log::debug!("Executing Python code:\n```python\n{code}\n```");

        // Execute in REPL
        let result = self
            .repl
            .execute(code)
            .await
            .map_err(|e| ToolError::Execution(e.into()))?;

        if result.success {
            log::debug!(
                "Python execution succeeded in {}ms",
                result.execution_time_ms
            );
            if !result.stdout.is_empty() {
                log::debug!("stdout:\n{}", result.stdout);
            }
            if let Some(ref return_value) = result.return_value {
                log::debug!("Return value: {return_value}");
            }
        } else {
            log::warn!("Python execution failed in {}ms", result.execution_time_ms);
            if !result.stderr.is_empty() {
                log::warn!("stderr:\n{}", result.stderr);
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
        // The REPL sandboxes execution in-process via restricted builtins and a
        // Rust-backed import allowlist, so per-call approval is unnecessary.
        true
    }
}

/// Factory that registers [`PythonReplTool`] under the name `execute_python`.
///
/// Constructs a default [`PythonRepl`] (state persists across tool calls within
/// a runtime instance). Takes no configuration.
pub struct PythonReplToolFactory;

impl ToolFactory for PythonReplToolFactory {
    fn name(&self) -> &'static str {
        "execute_python"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<()> {
        let repl = PythonRepl::new().context("init PythonRepl")?;
        registry.register(Arc::new(PythonReplTool::new(Arc::new(repl))));
        Ok(())
    }
}
