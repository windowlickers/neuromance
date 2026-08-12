use std::error::Error;
use std::fmt;

use thiserror::Error;

/// Errors from [`ToolImplementation::execute()`] and [`ToolFactory::build()`].
#[derive(Error, Debug)]
pub enum ToolError {
    /// Missing or malformed arguments.
    #[error("{0}")]
    InvalidArguments(String),

    /// Runtime failure during tool construction or execution.
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

impl ToolExecutorError {
    /// A stable, low-cardinality slug naming why the tool call failed.
    ///
    /// Reported as the `error.type` attribute on the `execute_tool` span.
    /// Never derive this from [`Display`](fmt::Display) output — the inner
    /// strings carry tool names, arguments, and provider text, any of which
    /// would blow up cardinality in a metrics backend.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::UnknownTool(_) => "unknown_tool",
            Self::Tool(ToolError::InvalidArguments(_)) => "invalid_arguments",
            Self::Tool(ToolError::Execution(_)) => "tool_error",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reason_separates_a_missing_tool_from_a_failing_one() {
        assert_eq!(
            ToolExecutorError::UnknownTool("grep".to_string()).reason(),
            "unknown_tool"
        );
        assert_eq!(
            ToolExecutorError::Tool(ToolError::InvalidArguments("no path".to_string())).reason(),
            "invalid_arguments"
        );
        assert_eq!(
            ToolExecutorError::Tool(ToolError::execution("boom")).reason(),
            "tool_error"
        );
    }
}
