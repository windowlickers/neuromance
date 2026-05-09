//! REPL (Read-Eval-Print-Loop) environments for LLM tool execution.
//!
//! Provides Python execution with restricted builtins, persistent state, configurable modules,
//! and stdout/stderr capture.
//!
//! # Features
//!
//! - `python`: Python REPL via `PyO3` with restricted builtins
//! - `tools`: LLM tool integration via `PythonReplTool`
//!
//! # Example
//!
//! ```rust,no_run
//! # async fn example() -> anyhow::Result<()> {
//! #[cfg(feature = "python")]
//! {
//!     use neuromance_repl::python::{PythonRepl, PythonReplConfig};
//!
//!     // Configure available modules
//!     let config = PythonReplConfig::default().with_modules(["numpy"]);
//!
//!     let repl = PythonRepl::with_config(config)?;
//!
//!     // State persists between executions
//!     repl.execute("x = 10").await?;
//!     let result = repl.execute("print(x + 5)").await?; // Use print() for output
//!     assert!(result.stdout.contains("15"));
//! }
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

pub mod error;
pub use error::ReplError;

#[cfg(feature = "python")]
pub mod python;

// Re-export commonly used types for convenience
#[cfg(feature = "python")]
pub use python::{InteractivePythonRepl, PythonCallback, PythonRepl, PythonReplConfig};

#[cfg(all(feature = "python", feature = "tools"))]
pub use python::PythonReplTool;

/// Result of executing code in a REPL environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct ReplResult {
    /// Standard output captured during execution
    pub stdout: String,

    /// Standard error captured during execution
    pub stderr: String,

    /// Whether execution was successful
    pub success: bool,

    /// Optional return value from the execution
    pub return_value: Option<String>,

    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

impl ReplResult {
    /// Create a successful result.
    pub const fn success(stdout: String) -> Self {
        Self {
            stdout,
            stderr: String::new(),
            success: true,
            return_value: None,
            execution_time_ms: 0,
        }
    }

    /// Create an error result.
    pub const fn error(stderr: String) -> Self {
        Self {
            stdout: String::new(),
            stderr,
            success: false,
            return_value: None,
            execution_time_ms: 0,
        }
    }

    /// Set the return value.
    pub fn with_return_value(mut self, value: impl Into<String>) -> Self {
        self.return_value = Some(value.into());
        self
    }

    /// Set the execution time.
    pub const fn with_execution_time(mut self, ms: u64) -> Self {
        self.execution_time_ms = ms;
        self
    }
}

impl std::fmt::Display for ReplResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.success {
            write!(f, "Success")?;
            if !self.stdout.is_empty() {
                write!(f, "\nStdout:\n{}", self.stdout)?;
            }
            if let Some(ref val) = self.return_value {
                write!(f, "\nReturn value: {val}")?;
            }
        } else {
            write!(f, "Error")?;
            if !self.stderr.is_empty() {
                write!(f, "\nStderr:\n{}", self.stderr)?;
            }
        }
        let ms = self.execution_time_ms;
        write!(f, "\nExecution time: {ms}ms")?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_result_display() {
        let success = ReplResult::success("Hello, World!".to_string())
            .with_return_value("42".to_string())
            .with_execution_time(123);

        let display = format!("{success}");
        assert!(display.contains("Success"));
        assert!(display.contains("Hello, World!"));
        assert!(display.contains("42"));
        assert!(display.contains("123ms"));

        let error = ReplResult::error("SyntaxError: invalid syntax".to_string());
        let display = format!("{error}");
        assert!(display.contains("Error"));
        assert!(display.contains("SyntaxError"));
    }
}
