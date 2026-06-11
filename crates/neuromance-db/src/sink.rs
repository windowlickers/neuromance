//! The narrow write-side interface consumed by orchestration code.

use neuromance_common::chat::Message;
use uuid::Uuid;

use crate::error::DbError;

/// A sink that durably records conversation messages.
///
/// This is the only interface `neuromance` (Core) depends on, so callers can
/// substitute test doubles without touching sqlx. Implementations must be
/// **idempotent per [`Message::id`]**: re-sending an already-persisted message
/// is a no-op, which lets callers safely retry whole history snapshots after
/// a failed write.
#[async_trait::async_trait]
pub trait ConversationSink: Send + Sync {
    /// Inserts any messages not yet persisted (deduplicated by [`Message::id`]),
    /// creating a minimal conversation row if one does not exist.
    ///
    /// Messages are appended in slice order after the conversation's current
    /// maximum sequence number. Returns the number of rows actually inserted.
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the transaction fails or a message cannot be
    /// serialized for storage.
    #[must_use = "the inserted-message count drives the persistence metrics counter"]
    async fn append_messages(
        &self,
        conversation_id: Uuid,
        messages: &[Message],
    ) -> Result<u64, DbError>;
}
