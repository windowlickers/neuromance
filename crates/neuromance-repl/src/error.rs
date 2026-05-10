use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during REPL operations.
#[derive(Error, Debug)]
pub enum ReplError {
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
    PythonInfra {
        operation: &'static str,
        #[source]
        source: pyo3::PyErr,
    },

    /// Python error while accessing a named variable in the REPL's locals
    /// dict. `operation` identifies the step (e.g. `"get_item"`,
    /// `"set_item"`, `"extract -> String"`); `name` is the variable the
    /// operator tried to access.
    #[cfg(feature = "python")]
    #[error("Python error during {operation} for variable {name:?}: {source}")]
    PythonVariable {
        operation: &'static str,
        name: String,
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

/// Attach an operation tag to a [`pyo3::PyResult`].
///
/// [`Self::at`] produces [`ReplError::PythonInfra`] for general infrastructure
/// failures; use a short, literal tag like `"import io"` or
/// `"setattr sys.stdout"`. [`Self::at_var`] produces [`ReplError::PythonVariable`]
/// for failures tied to a specific named variable in the REPL locals dict, so
/// the operator sees which assignment or lookup triggered the error.
#[cfg(feature = "python")]
pub(crate) trait PyResultExt<T> {
    fn at(self, operation: &'static str) -> Result<T, ReplError>;
    fn at_var(self, operation: &'static str, name: &str) -> Result<T, ReplError>;
}

#[cfg(feature = "python")]
impl<T> PyResultExt<T> for pyo3::PyResult<T> {
    fn at(self, operation: &'static str) -> Result<T, ReplError> {
        self.map_err(|source| ReplError::PythonInfra { operation, source })
    }

    fn at_var(self, operation: &'static str, name: &str) -> Result<T, ReplError> {
        self.map_err(|source| ReplError::PythonVariable {
            operation,
            name: name.to_string(),
            source,
        })
    }
}
