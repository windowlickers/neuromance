use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during REPL operations.
#[derive(Error, Debug)]
pub enum ReplError {
    /// Code execution failed
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Input rejected before execution (e.g. interior NUL byte)
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Timeout during execution
    #[error("Execution timeout after {0:?}")]
    Timeout(Duration),

    /// Python infrastructure error (import, attr access, method call, capture
    /// setup, …). `operation` is a short, source-literal tag identifying which
    /// step failed so the operator has a Rust-side breadcrumb beyond the bare
    /// `PyErr` message.
    #[cfg(feature = "python")]
    #[error("Python error during {operation}: {source}")]
    Python {
        operation: &'static str,
        #[source]
        source: pyo3::PyErr,
    },

    /// Python error raised by user code executed via `py.run` /
    /// `console.push`. Transparent so `ReplResult.stderr` shows the raw
    /// traceback unchanged.
    #[cfg(feature = "python")]
    #[error(transparent)]
    PythonExec(pyo3::PyErr),

    /// Tokio task join failure
    #[error("Task join error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),

    /// REPL state mutex was poisoned by a prior panic
    #[error("REPL state poisoned by a prior panic: {0}")]
    StatePoisoned(String),
}

#[cfg(feature = "tools")]
impl From<ReplError> for neuromance_tools::ToolError {
    fn from(err: ReplError) -> Self {
        Self::Execution(Box::new(err))
    }
}

/// Attach an operation tag to a [`pyo3::PyResult`], producing
/// [`ReplError::Python`] on failure. Use a short, literal tag like
/// `"import io"` or `"setattr sys.stdout"`.
#[cfg(feature = "python")]
pub(crate) trait PyResultExt<T> {
    fn at(self, operation: &'static str) -> Result<T, ReplError>;
}

#[cfg(feature = "python")]
impl<T> PyResultExt<T> for pyo3::PyResult<T> {
    fn at(self, operation: &'static str) -> Result<T, ReplError> {
        self.map_err(|source| ReplError::Python { operation, source })
    }
}
