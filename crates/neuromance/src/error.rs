use neuromance_client::ClientError;
use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreError {
    #[error(transparent)]
    Client(#[from] ClientError),

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Maximum turns exceeded: {0}")]
    MaxTurnsExceeded(String),

    #[error("User quit: {0}")]
    UserQuit(String),

    #[error("Cancelled: {0}")]
    Cancelled(String),

    #[error("No response: {0}")]
    NoResponse(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Hook {hook} failed: {source}")]
    Hook {
        /// Name of the hook that failed.
        hook: String,
        /// The underlying error returned by the hook.
        #[source]
        source: anyhow::Error,
    },

    #[error("Context compaction error: {0}")]
    CompactionError(String),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}
