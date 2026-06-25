//! Postgres-backed conversation store.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use neuromance_common::chat::{Conversation, ConversationStatus, Message, TaskStatus};
use serde::Serialize;
use serde_json::Value;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use tracing::warn;
use uuid::Uuid;

use crate::error::{DbError, SqlxResultExt};
use crate::rows::{
    MessageRow, message_to_columns, status_from_str, status_to_string, task_status_from_str,
    task_status_to_string,
};
use crate::sink::ConversationSink;

/// A lightweight listing entry for a stored conversation.
#[derive(Debug, Clone, PartialEq, Serialize)]
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
    pub message_count: u64,
    /// Conversation that spawned this one, or `None` for a root conversation.
    pub parent_conversation_id: Option<Uuid>,
    /// Assistant message in the parent that emitted the spawning tool call.
    pub parent_message_id: Option<Uuid>,
    /// Id of the specific tool call within [`Self::parent_message_id`].
    pub parent_tool_call_id: Option<String>,
}

/// A durable task row: lifecycle status plus the fields `GET /tasks/{id}` serializes.
///
/// Mirrors the runtime's in-memory task record so a replica that never owned
/// the task can still answer a poll.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StoredTask {
    /// Task id.
    pub id: Uuid,
    /// Conversation the task ran against.
    pub conversation_id: Uuid,
    /// Current lifecycle status.
    pub status: TaskStatus,
    /// Final assistant output, present once the task succeeds.
    pub output: Option<String>,
    /// Failure or cancellation reason, present on a non-success terminal state.
    pub error: Option<String>,
    /// Tasks already buffered when this task was accepted, frozen at submit time.
    pub queue_depth_at_enqueue: i64,
    /// When the task was first recorded (at enqueue).
    pub created_at: DateTime<Utc>,
    /// When the status was last written.
    pub updated_at: DateTime<Utc>,
}

/// The status fields written through to the durable task row at enqueue,
/// dequeue, and terminal. Bundled so [`PgConversationStore::record_task_status`]
/// stays within the positional-argument budget.
#[derive(Debug, Clone)]
pub struct TaskStatusUpdate {
    /// Task id (the upsert key).
    pub id: Uuid,
    /// Conversation the task runs against (the FK target).
    pub conversation_id: Uuid,
    /// Status to write.
    pub status: TaskStatus,
    /// Output to write, if any.
    pub output: Option<String>,
    /// Error to write, if any.
    pub error: Option<String>,
    /// Queue depth at enqueue, frozen at submit time.
    pub queue_depth_at_enqueue: i64,
    /// Creation timestamp, preserved across status updates.
    pub created_at: DateTime<Utc>,
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

    /// Returns whether a conversation row exists.
    ///
    /// Used by the serve loop's admission check to resolve a continuation
    /// against durable state when the in-process cache misses (the request
    /// reached a replica that did not handle earlier turns).
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the query fails.
    pub async fn conversation_exists(&self, id: Uuid) -> Result<bool, DbError> {
        let exists = sqlx::query_scalar!(
            "SELECT EXISTS(SELECT 1 FROM conversations WHERE id = $1)",
            id,
        )
        .fetch_one(&self.pool)
        .await
        .op("check conversation exists")?;
        Ok(exists.unwrap_or(false))
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
            SELECT id, title, description, status, metadata, created_at, updated_at,
                   parent_conversation_id, parent_message_id, parent_tool_call_id
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
                   name, reasoning, metadata, timestamp, model, provider, usage
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
            parent_conversation_id: row.parent_conversation_id,
            parent_message_id: row.parent_message_id,
            parent_tool_call_id: row.parent_tool_call_id,
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
        limit: u32,
        offset: u32,
    ) -> Result<Vec<ConversationSummary>, DbError> {
        let rows = sqlx::query!(
            r#"
            SELECT c.id, c.title, c.status, c.created_at, c.updated_at,
                   c.parent_conversation_id, c.parent_message_id, c.parent_tool_call_id,
                   COUNT(m.id) AS "message_count!"
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT $1 OFFSET $2
            "#,
            i64::from(limit),
            i64::from(offset),
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
                    message_count: u64::try_from(row.message_count).unwrap_or(0),
                    parent_conversation_id: row.parent_conversation_id,
                    parent_message_id: row.parent_message_id,
                    parent_tool_call_id: row.parent_tool_call_id,
                })
            })
            .collect()
    }

    /// Lists the child conversations of `parent_id`, most recently updated first.
    ///
    /// Children are conversations a delegating run spawned (e.g. subagents),
    /// linked via [`ConversationSink::set_conversation_parent`].
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the query fails or a stored status is unknown.
    pub async fn list_child_conversations(
        &self,
        parent_id: Uuid,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<ConversationSummary>, DbError> {
        let rows = sqlx::query!(
            r#"
            SELECT c.id, c.title, c.status, c.created_at, c.updated_at,
                   c.parent_conversation_id, c.parent_message_id, c.parent_tool_call_id,
                   COUNT(m.id) AS "message_count!"
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.parent_conversation_id = $1
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT $2 OFFSET $3
            "#,
            parent_id,
            i64::from(limit),
            i64::from(offset),
        )
        .fetch_all(&self.pool)
        .await
        .op("list child conversations")?;

        rows.into_iter()
            .map(|row| {
                Ok(ConversationSummary {
                    id: row.id,
                    title: row.title,
                    status: status_from_str(&row.status, row.id)?,
                    created_at: row.created_at,
                    updated_at: row.updated_at,
                    message_count: u64::try_from(row.message_count).unwrap_or(0),
                    parent_conversation_id: row.parent_conversation_id,
                    parent_message_id: row.parent_message_id,
                    parent_tool_call_id: row.parent_tool_call_id,
                })
            })
            .collect()
    }

    /// Highest `seq` currently stored for a conversation, or `None` when it has
    /// no messages yet.
    ///
    /// Callers bracket a unit of work by reading this before and after to learn
    /// the `seq` range that work contributed (see [`Self::record_task`]).
    ///
    /// # Errors
    ///
    /// Returns [`DbError::Sqlx`] if the query fails.
    pub async fn max_seq(&self, conversation_id: Uuid) -> Result<Option<i64>, DbError> {
        let max = sqlx::query_scalar!(
            "SELECT MAX(seq) FROM messages WHERE conversation_id = $1",
            conversation_id,
        )
        .fetch_one(&self.pool)
        .await
        .op("select max message seq")?;
        Ok(max)
    }

    /// Records the `seq` range a task contributed to a conversation.
    ///
    /// Idempotent per `task_id`: re-recording updates the stored range, so a
    /// retried task converges on its final span rather than duplicating.
    ///
    /// # Errors
    ///
    /// Returns [`DbError::Sqlx`] if the insert fails.
    pub async fn record_task(
        &self,
        task_id: Uuid,
        conversation_id: Uuid,
        start_seq: i64,
        end_seq: i64,
    ) -> Result<(), DbError> {
        sqlx::query!(
            r#"
            INSERT INTO tasks (id, conversation_id, start_seq, end_seq)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO UPDATE
                SET start_seq = EXCLUDED.start_seq,
                    end_seq = EXCLUDED.end_seq
            "#,
            task_id,
            conversation_id,
            start_seq,
            end_seq,
        )
        .execute(&self.pool)
        .await
        .op("insert task provenance")?;
        Ok(())
    }

    /// Upserts a task's status, output, error and queue depth, keyed by id.
    ///
    /// Idempotent per id and column-scoped: the conflict clause updates only the
    /// status columns, never the provenance columns (`start_seq`/`end_seq`) that
    /// [`Self::record_task`] writes, so the two upserts compose regardless of
    /// order. `created_at` is preserved across updates.
    ///
    /// The conversation row is pre-inserted (`ON CONFLICT DO NOTHING`) so this
    /// write is self-sufficient against the `tasks.conversation_id` foreign key
    /// even when the conversation row write is still in flight.
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the write fails or the status cannot be encoded.
    pub async fn record_task_status(&self, update: &TaskStatusUpdate) -> Result<(), DbError> {
        let status = task_status_to_string(update.status, update.id)?;
        let mut tx = self.pool.begin().await.op("begin transaction")?;

        sqlx::query!(
            "INSERT INTO conversations (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
            update.conversation_id,
        )
        .execute(&mut *tx)
        .await
        .op("insert conversation row")?;

        sqlx::query!(
            r#"
            INSERT INTO tasks
                (id, conversation_id, status, output, error,
                 queue_depth_at_enqueue, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, now())
            ON CONFLICT (id) DO UPDATE SET
                status                 = EXCLUDED.status,
                output                 = EXCLUDED.output,
                error                  = EXCLUDED.error,
                queue_depth_at_enqueue = EXCLUDED.queue_depth_at_enqueue,
                updated_at             = now()
            "#,
            update.id,
            update.conversation_id,
            status,
            update.output,
            update.error,
            update.queue_depth_at_enqueue,
            update.created_at,
        )
        .execute(&mut *tx)
        .await
        .op("upsert task status")?;

        tx.commit().await.op("commit transaction")?;
        Ok(())
    }

    /// Deletes a task's durable row.
    ///
    /// Used to roll back an enqueue that persisted a `pending` row but then
    /// failed to hand the job to the worker, so a rejected task never lingers
    /// in [`Self::list_active_tasks`]. A missing row is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`DbError::Sqlx`] if the delete fails.
    pub async fn delete_task(&self, task_id: Uuid) -> Result<(), DbError> {
        sqlx::query!("DELETE FROM tasks WHERE id = $1", task_id)
            .execute(&self.pool)
            .await
            .op("delete task")?;
        Ok(())
    }

    /// Loads the durable status row for a task, or `None` if none was recorded.
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the query fails or the stored status is unknown.
    pub async fn get_task(&self, task_id: Uuid) -> Result<Option<StoredTask>, DbError> {
        let Some(row) = sqlx::query!(
            r#"
            SELECT id, conversation_id, status, output, error,
                   queue_depth_at_enqueue, created_at, updated_at
            FROM tasks WHERE id = $1
            "#,
            task_id,
        )
        .fetch_optional(&self.pool)
        .await
        .op("select task")?
        else {
            return Ok(None);
        };
        Ok(Some(StoredTask {
            id: row.id,
            conversation_id: row.conversation_id,
            status: task_status_from_str(&row.status, row.id)?,
            output: row.output,
            error: row.error,
            queue_depth_at_enqueue: row.queue_depth_at_enqueue,
            created_at: row.created_at,
            updated_at: row.updated_at,
        }))
    }

    /// Lists tasks that are still `pending` or `running`, oldest first.
    ///
    /// Backs `GET /tasks`: a caller's index in the returned vec is their queue
    /// position. Reads from postgres so any replica returns the same view.
    ///
    /// # Errors
    ///
    /// Returns [`DbError`] if the query fails or a stored status is unknown.
    pub async fn list_active_tasks(&self) -> Result<Vec<StoredTask>, DbError> {
        let rows = sqlx::query!(
            r#"
            SELECT id, conversation_id, status, output, error,
                   queue_depth_at_enqueue, created_at, updated_at
            FROM tasks
            WHERE status IN ('pending', 'running')
            ORDER BY created_at
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .op("list active tasks")?;

        rows.into_iter()
            .map(|row| {
                Ok(StoredTask {
                    id: row.id,
                    conversation_id: row.conversation_id,
                    status: task_status_from_str(&row.status, row.id)?,
                    output: row.output,
                    error: row.error,
                    queue_depth_at_enqueue: row.queue_depth_at_enqueue,
                    created_at: row.created_at,
                    updated_at: row.updated_at,
                })
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl ConversationSink for PgConversationStore {
    async fn set_conversation_parent(
        &self,
        child: Uuid,
        parent: Uuid,
        parent_task_id: Option<Uuid>,
        parent_message_id: Option<Uuid>,
        parent_tool_call_id: Option<&str>,
    ) -> Result<(), DbError> {
        // Upsert so the link records regardless of whether the child row has
        // been created by the first message append yet.
        sqlx::query!(
            r#"
            INSERT INTO conversations
                (id, parent_conversation_id, parent_task_id,
                 parent_message_id, parent_tool_call_id)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (id) DO UPDATE
                SET parent_conversation_id = EXCLUDED.parent_conversation_id,
                    parent_task_id = EXCLUDED.parent_task_id,
                    parent_message_id = EXCLUDED.parent_message_id,
                    parent_tool_call_id = EXCLUDED.parent_tool_call_id
            "#,
            child,
            parent,
            parent_task_id,
            parent_message_id,
            parent_tool_call_id,
        )
        .execute(&self.pool)
        .await
        .op("link conversation parent")?;
        Ok(())
    }

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
                                      tool_call_id, name, reasoning, metadata, timestamp,
                                      model, provider, usage)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
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
                columns.model,
                columns.provider,
                columns.usage as Option<Value>,
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
