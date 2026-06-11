//! Postgres-backed conversation store.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use neuromance_common::chat::{Conversation, ConversationStatus, Message};
use serde_json::Value;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use tracing::warn;
use uuid::Uuid;

use crate::error::{DbError, SqlxResultExt};
use crate::rows::{MessageRow, message_to_columns, status_from_str, status_to_string};
use crate::sink::ConversationSink;

/// A lightweight listing entry for a stored conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationSummary {
    /// Conversation id.
    pub id: Uuid,
    /// Optional human-readable title.
    pub title: Option<String>,
    /// Current lifecycle status.
    pub status: ConversationStatus,
    /// When the conversation row was created.
    pub created_at: DateTime<Utc>,
    /// When the conversation was last written to.
    pub updated_at: DateTime<Utc>,
    /// Number of persisted messages.
    pub message_count: i64,
}

/// Postgres-backed store for conversations and messages.
///
/// Messages are stored as an **append-only log**: every distinct message ever
/// observed for a conversation gets a row, ordered by a per-conversation `seq`
/// assigned at insert time. Context compaction never deletes or renumbers
/// rows — summary messages simply append after the existing history, so the
/// database always holds the full record even when the live in-memory history
/// has been rewritten.
#[derive(Debug, Clone)]
pub struct PgConversationStore {
    pool: PgPool,
}

impl PgConversationStore {
    /// Wraps an existing connection pool.
    #[must_use]
    pub const fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Connects to postgres at `url` with the given pool limits.
    ///
    /// `acquire_timeout` bounds how long any persistence call can stall on a
    /// saturated or unreachable database.
    ///
    /// # Errors
    ///
    /// Returns [`DbError::Sqlx`] if the connection cannot be established.
    pub async fn connect(
        url: &str,
        max_connections: u32,
        acquire_timeout: Duration,
    ) -> Result<Self, DbError> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .acquire_timeout(acquire_timeout)
            .connect(url)
            .await
            .op("connect to database")?;
        Ok(Self::new(pool))
    }

    /// Runs the embedded schema migrations.
    ///
    /// # Errors
    ///
    /// Returns [`DbError::Migrate`] if a migration fails to apply.
    pub async fn migrate(&self) -> Result<(), DbError> {
        sqlx::migrate!("./migrations").run(&self.pool).await?;
        Ok(())
    }

    /// Inserts or updates a conversation row (metadata, title, status — not
    /// its messages; those go through [`ConversationSink::append_messages`]).
    ///
    /// `created_at` is preserved on update.
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the write fails or a field cannot be serialized.
    pub async fn upsert_conversation(&self, conversation: &Conversation) -> Result<(), DbError> {
        let status = status_to_string(&conversation.status, conversation.id)?;
        let metadata =
            serde_json::to_value(&conversation.metadata).map_err(|source| DbError::Encode {
                table: "conversations",
                column: "metadata",
                id: conversation.id,
                source,
            })?;
        sqlx::query!(
            r#"
            INSERT INTO conversations (id, title, description, status, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                title       = EXCLUDED.title,
                description = EXCLUDED.description,
                status      = EXCLUDED.status,
                metadata    = EXCLUDED.metadata,
                updated_at  = EXCLUDED.updated_at
            "#,
            conversation.id,
            conversation.title,
            conversation.description,
            status,
            metadata,
            conversation.created_at,
            conversation.updated_at,
        )
        .execute(&self.pool)
        .await
        .op("upsert conversation")?;
        Ok(())
    }

    /// Sets the lifecycle status of a conversation.
    ///
    /// A missing conversation is a no-op (the status of a conversation that
    /// was never persisted is not meaningful).
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the update fails.
    pub async fn set_conversation_status(
        &self,
        id: Uuid,
        status: ConversationStatus,
    ) -> Result<(), DbError> {
        let status = status_to_string(&status, id)?;
        sqlx::query!(
            "UPDATE conversations SET status = $2, updated_at = now() WHERE id = $1",
            id,
            status,
        )
        .execute(&self.pool)
        .await
        .op("set conversation status")?;
        Ok(())
    }

    /// Loads a conversation and its full message log, ordered by `seq`.
    ///
    /// Returns `None` if no conversation row exists.
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the query fails or a stored row cannot be
    /// mapped back to its Rust type.
    pub async fn get_conversation(&self, id: Uuid) -> Result<Option<Conversation>, DbError> {
        let Some(row) = sqlx::query!(
            r#"
            SELECT id, title, description, status, metadata, created_at, updated_at
            FROM conversations WHERE id = $1
            "#,
            id,
        )
        .fetch_optional(&self.pool)
        .await
        .op("select conversation")?
        else {
            return Ok(None);
        };

        let messages = sqlx::query_as!(
            MessageRow,
            r#"
            SELECT id, conversation_id, role, content, tool_calls, tool_call_id,
                   name, reasoning, metadata, timestamp
            FROM messages WHERE conversation_id = $1 ORDER BY seq
            "#,
            id,
        )
        .fetch_all(&self.pool)
        .await
        .op("select conversation messages")?
        .into_iter()
        .map(MessageRow::into_message)
        .collect::<Result<Vec<_>, _>>()?;

        let metadata = serde_json::from_value(row.metadata).map_err(|source| DbError::Decode {
            table: "conversations",
            column: "metadata",
            id,
            source,
        })?;

        Ok(Some(Conversation {
            id: row.id,
            title: row.title,
            description: row.description,
            created_at: row.created_at,
            updated_at: row.updated_at,
            metadata,
            status: status_from_str(&row.status, id)?,
            messages: Arc::new(messages),
        }))
    }

    /// Lists stored conversations, most recently updated first.
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the query fails or a stored status is unknown.
    pub async fn list_conversations(
        &self,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ConversationSummary>, DbError> {
        let rows = sqlx::query!(
            r#"
            SELECT c.id, c.title, c.status, c.created_at, c.updated_at,
                   COUNT(m.id) AS "message_count!"
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT $1 OFFSET $2
            "#,
            limit,
            offset,
        )
        .fetch_all(&self.pool)
        .await
        .op("list conversations")?;

        rows.into_iter()
            .map(|row| {
                Ok(ConversationSummary {
                    id: row.id,
                    title: row.title,
                    status: status_from_str(&row.status, row.id)?,
                    created_at: row.created_at,
                    updated_at: row.updated_at,
                    message_count: row.message_count,
                })
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl ConversationSink for PgConversationStore {
    async fn append_messages(
        &self,
        conversation_id: Uuid,
        messages: &[Message],
    ) -> Result<u64, DbError> {
        let candidates: Vec<&Message> = messages
            .iter()
            .filter(|m| {
                if m.conversation_id == conversation_id {
                    return true;
                }
                warn!(
                    message_id = %m.id,
                    expected = %conversation_id,
                    actual = %m.conversation_id,
                    "skipping message with mismatched conversation_id"
                );
                false
            })
            .collect();
        if candidates.is_empty() {
            return Ok(0);
        }

        let mut tx = self.pool.begin().await.op("begin transaction")?;

        // Ensure the FK target exists; other columns take their defaults.
        sqlx::query!(
            "INSERT INTO conversations (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
            conversation_id,
        )
        .execute(&mut *tx)
        .await
        .op("insert conversation row")?;

        // Serialize writers per conversation so the max-seq read below is
        // race-free across concurrent agents sharing this conversation.
        sqlx::query!(
            "SELECT id FROM conversations WHERE id = $1 FOR UPDATE",
            conversation_id,
        )
        .fetch_one(&mut *tx)
        .await
        .op("lock conversation for update")?;

        let mut next_seq = sqlx::query_scalar!(
            r#"SELECT COALESCE(MAX(seq) + 1, 0) AS "base!" FROM messages WHERE conversation_id = $1"#,
            conversation_id,
        )
        .fetch_one(&mut *tx)
        .await
        .op("select next message seq")?;

        let ids: Vec<Uuid> = candidates.iter().map(|m| m.id).collect();
        let existing: HashSet<Uuid> =
            sqlx::query_scalar!("SELECT id FROM messages WHERE id = ANY($1)", &ids)
                .fetch_all(&mut *tx)
                .await
                .op("select existing message ids")?
                .into_iter()
                .collect();

        let mut inserted = 0u64;
        for message in candidates {
            if existing.contains(&message.id) {
                continue;
            }
            let columns = message_to_columns(message)?;
            let result = sqlx::query!(
                r#"
                INSERT INTO messages (id, conversation_id, seq, role, content, tool_calls,
                                      tool_call_id, name, reasoning, metadata, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (id) DO NOTHING
                "#,
                message.id,
                conversation_id,
                next_seq,
                columns.role,
                message.content,
                columns.tool_calls,
                message.tool_call_id,
                message.name,
                columns.reasoning as Option<Value>,
                columns.metadata,
                message.timestamp,
            )
            .execute(&mut *tx)
            .await
            .op("insert message")?;
            inserted += result.rows_affected();
            next_seq += 1;
        }

        sqlx::query!(
            "UPDATE conversations SET updated_at = now() WHERE id = $1",
            conversation_id,
        )
        .execute(&mut *tx)
        .await
        .op("touch conversation updated_at")?;

        tx.commit().await.op("commit transaction")?;
        Ok(inserted)
    }
}
