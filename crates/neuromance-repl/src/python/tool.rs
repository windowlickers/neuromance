//! Tool implementation for executing Python code via LLMs.
//!
//! This module provides a [`ToolImplementation`] for the Python REPL, allowing
//! LLMs to execute Python code as a tool call.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use neuromance_common::tools::{Function, Tool};
use neuromance_tools::ToolImplementation;

use super::PythonRepl;
use crate::ReplEnvironment;

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
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Can include multiple statements and definitions. Use print() to see output."
                    }
                },
                "required": ["code"]
            }),
        };

        Tool::builder().function(function).build()
    }

    async fn execute(&self, args: &Value) -> Result<String> {
        let code = args["code"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'code' argument"))?;

        println!(">>> Executing Python code:");
        println!("```python");
        println!("{code}");
        println!("```\n");

        // Execute in REPL
        let result = self.repl.execute(code).await?;

        println!("Result:");
        if result.success {
            println!("✓ Success");
            if !result.stdout.is_empty() {
                let stdout = &result.stdout;
                println!("Output:\n{stdout}");
            }
            if let Some(ref return_value) = result.return_value {
                println!("Return value: {return_value}");
            }
        } else {
            println!("✗ Error");
            if !result.stderr.is_empty() {
                let stderr = &result.stderr;
                println!("Error:\n{stderr}");
            }
        }
        let ms = result.execution_time_ms;
        println!("Execution time: {ms}ms\n");

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
        false
    }
}
