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
        Self::Other(err.to_string())
    }
}

impl From<DaemonError> for neuromance_common::DaemonResponse {
    fn from(err: DaemonError) -> Self {
        Self::Error {
            message: err.to_string(),
        }
    }
}
