//! Typed errors for database operations.

use uuid::Uuid;

/// Errors returned by conversation persistence operations.
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    /// A query or connection-level failure from sqlx.
    #[error("database query failed: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// Running embedded migrations failed.
    #[error("database migration failed: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),

    /// A `role` column held a value that is not a known [`MessageRole`].
    ///
    /// [`MessageRole`]: neuromance_common::chat::MessageRole
    #[error("unknown message role '{value}' in message row {message_id}")]
    UnknownRole {
        /// The unrecognized role string read from the database.
        value: String,
        /// The id of the message row holding it.
        message_id: Uuid,
    },

    /// A `status` column held a value that is not a known [`ConversationStatus`].
    ///
    /// [`ConversationStatus`]: neuromance_common::chat::ConversationStatus
    #[error("unknown conversation status '{value}' in conversation row {conversation_id}")]
    UnknownStatus {
        /// The unrecognized status string read from the database.
        value: String,
        /// The id of the conversation row holding it.
        conversation_id: Uuid,
    },

    /// Serializing a field to JSON for a JSONB column failed.
    #[error("failed to encode {column} for row {id} as JSON: {source}")]
    Encode {
        /// The destination column.
        column: &'static str,
        /// The id of the row being written.
        id: Uuid,
        /// The underlying serialization error.
        #[source]
        source: serde_json::Error,
    },

    /// Deserializing a JSONB column back into its Rust type failed.
    #[error("invalid JSON in column {column} of row {id}: {source}")]
    Decode {
        /// The source column.
        column: &'static str,
        /// The id of the row being read.
        id: Uuid,
        /// The underlying deserialization error.
        #[source]
        source: serde_json::Error,
    },
}
