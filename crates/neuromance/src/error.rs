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

    #[error("No response: {0}")]
    NoResponse(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Turn callback error: {0}")]
    TurnCallback(Box<dyn std::error::Error + Send + Sync>),
}
