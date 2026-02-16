//! Error types for the Neuromance daemon.

use thiserror::Error;

/// Errors that can occur in the daemon.
#[derive(Debug, Error)]
pub enum DaemonError {
    /// I/O error (file operations, socket communication).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// TOML deserialization error.
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    /// Conversation not found.
    #[error("Conversation not found: {0}")]
    ConversationNotFound(String),

    /// Model not found in configuration.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Bookmark not found.
    #[error("Bookmark not found: {0}")]
    BookmarkNotFound(String),

    /// Bookmark already exists.
    #[error("Bookmark already exists: {0}")]
    BookmarkExists(String),

    /// No active conversation.
    #[error("No active conversation set")]
    NoActiveConversation,

    /// Invalid conversation ID format.
    #[error("Invalid conversation ID: {0}")]
    InvalidConversationId(String),

    /// Core orchestration error.
    #[error("Core error: {0}")]
    Core(String),

    /// Client error.
    #[error("Client error: {0}")]
    Client(String),

    /// Tool execution error.
    #[error("Tool error: {0}")]
    Tool(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Storage error.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Generic error with context.
    #[error("{0}")]
    Other(String),
}

/// Result type alias using `DaemonError`.
pub type Result<T> = std::result::Result<T, DaemonError>;

impl From<anyhow::Error> for DaemonError {
    fn from(err: anyhow::Error) -> Self {
        Self::Other(format!("{err:?}"))
    }
}

impl From<DaemonError> for neuromance_common::DaemonResponse {
    fn from(err: DaemonError) -> Self {
        use neuromance_common::ErrorCode;

        let code = match &err {
            DaemonError::ConversationNotFound(_) => ErrorCode::ConversationNotFound,
            DaemonError::ModelNotFound(_) => ErrorCode::ModelNotFound,
            DaemonError::BookmarkNotFound(_) => ErrorCode::BookmarkNotFound,
            DaemonError::BookmarkExists(_) => ErrorCode::BookmarkExists,
            DaemonError::NoActiveConversation => ErrorCode::NoActiveConversation,
            DaemonError::InvalidConversationId(_) => ErrorCode::InvalidConversationId,
            DaemonError::Core(_) | DaemonError::Client(_) | DaemonError::Tool(_) => {
                ErrorCode::LlmError
            }
            DaemonError::Config(_) | DaemonError::Toml(_) => ErrorCode::ConfigError,
            DaemonError::Storage(_) => ErrorCode::StorageError,
            DaemonError::Io(_) | DaemonError::Json(_) | DaemonError::Other(_) => {
                ErrorCode::Internal
            }
        };

        Self::Error {
            code,
            message: err.to_string(),
        }
    }
}
