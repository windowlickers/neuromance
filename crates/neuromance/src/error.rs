use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Maximum turns exceeded: {0}")]
    MaxTurnsExceeded(String),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}
