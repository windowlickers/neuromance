use std::error::Error;
use std::fmt;

use thiserror::Error;

/// Errors from [`ToolImplementation::execute()`].
#[derive(Error, Debug)]
pub enum ToolError {
    /// Missing or malformed arguments.
    #[error("{0}")]
    InvalidArguments(String),

    /// Runtime failure during tool execution.
    ///
    /// Wraps the source error for downcasting (e.g., to `ReplError`).
    #[error(transparent)]
    Execution(Box<dyn Error + Send + Sync>),
}

impl ToolError {
    /// Wrap a display-able message as an [`Execution`](Self::Execution) error.
    pub fn execution(msg: impl fmt::Display) -> Self {
        Self::Execution(Box::new(StringError(msg.to_string())))
    }
}

/// Newtype so a plain `String` can implement `std::error::Error`.
#[derive(Debug)]
struct StringError(String);

impl fmt::Display for StringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for StringError {}

/// Errors from [`ToolExecutor::execute_tool()`].
#[derive(Error, Debug)]
pub enum ToolExecutorError {
    /// The requested tool was not found in the registry.
    #[error("Unknown tool: '{0}'")]
    UnknownTool(String),

    /// Forwarded from the tool's `execute()` method.
    #[error(transparent)]
    Tool(#[from] ToolError),
}
