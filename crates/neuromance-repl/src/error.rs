use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during REPL operations.
#[derive(Error, Debug)]
pub enum ReplError {
    /// Code execution failed
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Timeout during execution
    #[error("Execution timeout after {0:?}")]
    Timeout(Duration),

    /// Python runtime error
    #[cfg(feature = "python")]
    #[error("Python error: {0}")]
    Python(#[from] pyo3::PyErr),

    /// Tokio task join failure
    #[error("Task join error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),

    /// REPL state mutex was poisoned by a prior panic
    #[error("REPL state poisoned by a prior panic")]
    StatePoisoned,
}
