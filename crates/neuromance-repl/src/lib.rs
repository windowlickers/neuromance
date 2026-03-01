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
//!     let mut config = PythonReplConfig::default();
//!     config.python_modules.push("numpy".to_string());
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
use std::time::Duration;
use thiserror::Error;

#[cfg(feature = "python")]
pub mod python;

// Re-export commonly used types for convenience
#[cfg(feature = "python")]
pub use python::{InteractivePythonRepl, PythonCallback, PythonRepl, PythonReplConfig};

#[cfg(all(feature = "python", feature = "tools"))]
pub use python::PythonReplTool;

/// Errors that can occur during REPL operations.
#[derive(Error, Debug)]
pub enum ReplError {
    /// Code execution failed
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Timeout during execution
    #[error("Execution timeout after {0:?}")]
    Timeout(Duration),

    /// Environment initialization failed
    #[error("Initialization error: {0}")]
    InitializationError(String),
}

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
    pub fn with_return_value(mut self, value: String) -> Self {
        self.return_value = Some(value);
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

/// Configuration for REPL execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplConfig {
    /// Maximum execution time per code block
    pub timeout: Duration,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
        }
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

    #[test]
    fn test_repl_config_default() {
        let config = ReplConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
    }
}
