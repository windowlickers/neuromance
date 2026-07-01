//! Storage seam for serve mode.
//!
//! Serve keeps task and conversation state in two places: an in-memory working
//! set (fast, per-replica) and, when `[database]` is configured, a durable
//! postgres store that lets any replica answer a poll for any task. Rather than
//! branch on `Option<store>` at every handler and worker step, both speak to one
//! [`TaskStore`] interface with two adapters:
//!
//! - [`InMemoryTaskStore`] — working set only (tests, no-database serve).
//! - [`PostgresTaskStore`] — the same working set, plus write-through to postgres
//!   and postgres-authoritative reads for cross-replica polling.
//!
//! Durable-write policy lives here, encoded in the method contracts:
//! [`TaskStore::insert_pending`] is **fail-closed** (a returned task id must be
//! durably pollable, so it returns `Result`); every other durable write is
//! best-effort — it logs and continues, matching the log-and-continue policy of
//! the rest of the persistence layer.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::Serialize;
use tracing::warn;
use uuid::Uuid;

use neuromance_common::chat::{Message, MessageRole, TaskStatus};
use neuromance_db::{
    ConversationSummary as DbConversationSummary, DbError, PgConversationStore, StoredTask,
    TaskStatusUpdate,
};

/// A single task's in-memory working record.
///
/// This is the mutable state the worker advances through a task's lifecycle. The
/// durable/serialized view is [`StoredTask`]; `TaskRecord` converts into it for
/// reads (see [`impl From<&TaskRecord> for StoredTask`](StoredTask)).
#[derive(Debug, Clone, Serialize)]
pub struct TaskRecord {
    pub id: Uuid,
    pub status: TaskStatus,
    pub conversation_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub output: Option<String>,
    pub error: Option<String>,
    /// Number of tasks already buffered when this task was accepted.
    /// Frozen at submit time so postmortems can answer
    /// "how deep was the queue when this landed?" after the task has run.
    pub queue_depth_at_enqueue: usize,
}

/// Full conversation history, stored across many tasks.
///
/// The first message is always a system message stamped at conversation
/// creation. Each `POST /tasks/new` referencing this conversation appends a
/// user message immediately; on a successful turn, the canonical history is
/// replaced with the vec returned by `Agent::execute_with_history`, which
/// includes every intermediate assistant and tool message.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationRecord {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Number of user messages submitted against this conversation.
    pub turn_count: usize,
    pub messages: Vec<Message>,
}

/// Summary view used by `GET /conversations` — omits the message vec so
/// list responses don't grow unboundedly with conversation length.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationSummary {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub turn_count: usize,
    pub message_count: usize,
}

impl From<&ConversationRecord> for ConversationSummary {
    fn from(record: &ConversationRecord) -> Self {
        Self {
            id: record.id,
            created_at: record.created_at,
            updated_at: record.updated_at,
            turn_count: record.turn_count,
            message_count: record.messages.len(),
        }
    }
}

impl From<&TaskRecord> for StoredTask {
    fn from(record: &TaskRecord) -> Self {
        Self {
            id: record.id,
            conversation_id: record.conversation_id,
            status: record.status,
            output: record.output.clone(),
            error: record.error.clone(),
            queue_depth_at_enqueue: i64::try_from(record.queue_depth_at_enqueue)
                .unwrap_or(i64::MAX),
            created_at: record.created_at,
            updated_at: record.updated_at,
        }
    }
}

/// Snapshots the durable status fields of an in-memory task record.
fn status_update(record: &TaskRecord) -> TaskStatusUpdate {
    TaskStatusUpdate {
        id: record.id,
        conversation_id: record.conversation_id,
        status: record.status,
        output: record.output.clone(),
        error: record.error.clone(),
        queue_depth_at_enqueue: i64::try_from(record.queue_depth_at_enqueue).unwrap_or(i64::MAX),
        created_at: record.created_at,
    }
}

/// Timing fields the worker reads when a task starts running, to record queue
/// wait latency.
#[derive(Debug, Clone, Copy)]
pub struct TaskTiming {
    pub created_at: DateTime<Utc>,
    pub queue_depth_at_enqueue: usize,
}

/// Why a turn's input history could not be assembled. Each variant maps to the
/// message the worker logs and records on the failed task.
#[derive(Debug, Clone, Copy)]
pub enum TurnInputError {
    HistoryLoadFailed,
    ConversationNotFound,
    RecordMissing,
}

impl TurnInputError {
    /// The failure reason recorded on the task and surfaced in logs.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::HistoryLoadFailed => "failed to load conversation history",
            Self::ConversationNotFound => "conversation not found",
            Self::RecordMissing => "conversation record missing",
        }
    }
}

/// Summary of what happened to in-flight tasks at shutdown.
#[derive(Debug, Default, Clone, Copy)]
pub struct ShutdownSummary {
    pub pending_dropped: usize,
    pub in_flight_cancelled: usize,
    pub succeeded: usize,
    pub failed: usize,
}

/// The in-memory working set shared by both adapters: the authoritative store
/// for a no-database deployment, and the worker's fast local cache when postgres
/// backs the deployment.
#[derive(Default)]
struct WorkingState {
    tasks: DashMap<Uuid, TaskRecord>,
    conversations: DashMap<Uuid, ConversationRecord>,
}

impl WorkingState {
    fn mark_running(&self, id: Uuid) -> Option<TaskTiming> {
        let mut entry = self.tasks.get_mut(&id)?;
        entry.status = TaskStatus::Running;
        entry.updated_at = Utc::now();
        Some(TaskTiming {
            created_at: entry.created_at,
            queue_depth_at_enqueue: entry.queue_depth_at_enqueue,
        })
    }

    fn mark_succeeded(&self, id: Uuid, output: String) {
        if let Some(mut entry) = self.tasks.get_mut(&id) {
            entry.status = TaskStatus::Succeeded;
            entry.output = Some(output);
            entry.updated_at = Utc::now();
        }
    }

    fn mark_terminal(&self, id: Uuid, status: TaskStatus, reason: &str) {
        if let Some(mut entry) = self.tasks.get_mut(&id) {
            entry.status = status;
            entry.error = Some(reason.to_string());
            entry.updated_at = Utc::now();
        }
    }

    fn get_task(&self, id: Uuid) -> Option<StoredTask> {
        self.tasks.get(&id).map(|rec| StoredTask::from(&*rec))
    }

    fn active_tasks(&self) -> Vec<StoredTask> {
        let mut active: Vec<TaskRecord> = self
            .tasks
            .iter()
            .filter(|e| matches!(e.value().status, TaskStatus::Pending | TaskStatus::Running))
            .map(|e| e.value().clone())
            .collect();
        active.sort_by_key(|r| r.created_at);
        active.iter().map(StoredTask::from).collect()
    }

    /// Cache path for turn assembly: snapshot history, append the user message,
    /// and advance the record in place.
    fn build_turn_input(
        &self,
        conversation_id: Uuid,
        user_msg: Message,
    ) -> Result<Vec<Message>, TurnInputError> {
        let Some(mut entry) = self.conversations.get_mut(&conversation_id) else {
            return Err(TurnInputError::RecordMissing);
        };
        let mut snapshot = entry.messages.clone();
        snapshot.push(user_msg.clone());
        entry.messages.push(user_msg);
        entry.turn_count = entry.turn_count.saturating_add(1);
        entry.updated_at = Utc::now();
        Ok(snapshot)
    }

    fn refresh_conversation(&self, id: Uuid, full_history: Vec<Message>) {
        if let Some(mut entry) = self.conversations.get_mut(&id) {
            entry.messages = full_history;
            entry.updated_at = Utc::now();
        }
    }

    fn conversation_summaries(&self) -> Vec<ConversationSummary> {
        let mut summaries: Vec<ConversationSummary> = self
            .conversations
            .iter()
            .map(|e| ConversationSummary::from(e.value()))
            .collect();
        summaries.sort_by_key(|s| s.created_at);
        summaries
    }

    /// Mark pending tasks cancelled and return the summary plus the records that
    /// were cancelled, so a durable adapter can mirror them.
    fn drain(&self) -> (ShutdownSummary, Vec<TaskRecord>) {
        let mut summary = ShutdownSummary::default();
        let now = Utc::now();
        let mut cancelled = Vec::new();
        for mut entry in self.tasks.iter_mut() {
            match entry.status {
                TaskStatus::Pending => {
                    let queue_age_ms = (now - entry.created_at).num_milliseconds().max(0);
                    warn!(task_id = %entry.id, queue_age_ms, "task dropped at shutdown");
                    entry.status = TaskStatus::Cancelled;
                    entry.error = Some("dropped at shutdown".to_string());
                    entry.updated_at = now;
                    cancelled.push(entry.value().clone());
                    summary.pending_dropped = summary.pending_dropped.saturating_add(1);
                }
                TaskStatus::Running => {
                    summary.in_flight_cancelled = summary.in_flight_cancelled.saturating_add(1);
                }
                TaskStatus::Succeeded => {
                    summary.succeeded = summary.succeeded.saturating_add(1);
                }
                TaskStatus::Failed | TaskStatus::Cancelled => {
                    summary.failed = summary.failed.saturating_add(1);
                }
            }
        }
        (summary, cancelled)
    }
}

/// Derive a conversation's turn count from a durable message log (the number of
/// user messages), so a store-authoritative read reports the same `turn_count`
/// the in-memory record tracks.
fn turn_count_of(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|m| m.role == MessageRole::User)
        .count()
}

/// The storage operations serve's handlers and worker perform, abstracted over
/// the in-memory and postgres-write-through backends.
#[async_trait]
pub trait TaskStore: Send + Sync {
    // --- Task lifecycle: writes ---

    /// Record a freshly-minted pending task. **Fail-closed**: when durable, the
    /// pending row is written synchronously and a write failure leaves no working
    /// record and returns `Err`, so a returned task id is always durably pollable.
    async fn insert_pending(&self, task: &TaskRecord) -> Result<(), DbError>;

    /// Best-effort removal of a task (enqueue rollback): drops the working record
    /// and, when durable, deletes the row.
    async fn remove_task(&self, id: Uuid);

    /// Transition a task to `Running`, returning its timing, or `None` if the
    /// record vanished. Writes through when durable.
    async fn mark_running(&self, id: Uuid) -> Option<TaskTiming>;
    async fn mark_succeeded(&self, id: Uuid, output: String);
    async fn mark_failed(&self, id: Uuid, reason: &str);
    async fn mark_cancelled(&self, id: Uuid, reason: &str);

    // --- Task lifecycle: reads (postgres-authoritative when durable) ---

    async fn get_task(&self, id: Uuid) -> Result<Option<StoredTask>, DbError>;
    async fn list_active_tasks(&self) -> Result<Vec<StoredTask>, DbError>;

    // --- Provenance: best-effort, no-op without a store ---

    /// Read the `seq` a task's first message will occupy, to bracket its run.
    async fn begin_provenance(&self, conversation_id: Uuid) -> Option<i64>;
    /// Record the `[start_seq, end_seq]` range a task contributed.
    async fn record_provenance(&self, task_id: Uuid, conversation_id: Uuid, start_seq: Option<i64>);

    // --- Conversation resolution / turn assembly ---

    /// Insert a freshly-seeded conversation into the working set. The durable row
    /// is created lazily (by `insert_pending`'s pending write and by Core's
    /// message persistence), so this only establishes the in-memory record.
    fn seed_conversation(&self, record: ConversationRecord);
    async fn conversation_exists(&self, id: Uuid) -> Result<bool, DbError>;
    async fn remove_conversation(&self, id: Uuid);
    async fn build_turn_input(
        &self,
        conversation_id: Uuid,
        seeded: bool,
        user_msg: Message,
    ) -> Result<Vec<Message>, TurnInputError>;
    /// Refresh the working-set cache with a run's full history. A no-op when this
    /// replica holds no local record (it only continued from the store).
    async fn refresh_conversation(&self, id: Uuid, full_history: Vec<Message>);

    // --- Conversation reads (handlers) ---

    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>, DbError>;
    async fn get_conversation_view(&self, id: Uuid) -> Result<Option<ConversationRecord>, DbError>;
    /// Child (delegation) conversations. `Ok(None)` means lineage is unavailable
    /// (no durable store) and the handler should answer `503`.
    async fn list_conversation_children(
        &self,
        id: Uuid,
        limit: u32,
        offset: u32,
    ) -> Result<Option<Vec<DbConversationSummary>>, DbError>;

    // --- Shutdown ---

    async fn drain_pending(&self) -> ShutdownSummary;
}

/// Working-set-only adapter: authoritative for a no-database deployment and used
/// by tests. Every method operates on the in-memory maps and never fails.
#[derive(Default)]
pub struct InMemoryTaskStore {
    state: WorkingState,
}

impl InMemoryTaskStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of task records held.
    #[cfg(test)]
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.state.tasks.len()
    }

    /// All task records held, in arbitrary order.
    #[cfg(test)]
    #[must_use]
    pub fn all_tasks(&self) -> Vec<StoredTask> {
        self.state
            .tasks
            .iter()
            .map(|e| StoredTask::from(e.value()))
            .collect()
    }

    /// Number of conversation records held.
    #[cfg(test)]
    #[must_use]
    pub fn conversation_count(&self) -> usize {
        self.state.conversations.len()
    }

    /// Snapshot of a conversation's messages, or `None` if absent.
    #[cfg(test)]
    #[must_use]
    pub fn conversation_messages(&self, id: Uuid) -> Option<Vec<Message>> {
        self.state
            .conversations
            .get(&id)
            .map(|r| r.messages.clone())
    }

    /// A conversation's turn count, or `None` if absent.
    #[cfg(test)]
    #[must_use]
    pub fn conversation_turn_count(&self, id: Uuid) -> Option<usize> {
        self.state.conversations.get(&id).map(|r| r.turn_count)
    }
}

#[async_trait]
impl TaskStore for InMemoryTaskStore {
    async fn insert_pending(&self, task: &TaskRecord) -> Result<(), DbError> {
        self.state.tasks.insert(task.id, task.clone());
        Ok(())
    }

    async fn remove_task(&self, id: Uuid) {
        self.state.tasks.remove(&id);
    }

    async fn mark_running(&self, id: Uuid) -> Option<TaskTiming> {
        self.state.mark_running(id)
    }

    async fn mark_succeeded(&self, id: Uuid, output: String) {
        self.state.mark_succeeded(id, output);
    }

    async fn mark_failed(&self, id: Uuid, reason: &str) {
        self.state.mark_terminal(id, TaskStatus::Failed, reason);
    }

    async fn mark_cancelled(&self, id: Uuid, reason: &str) {
        self.state.mark_terminal(id, TaskStatus::Cancelled, reason);
    }

    async fn get_task(&self, id: Uuid) -> Result<Option<StoredTask>, DbError> {
        Ok(self.state.get_task(id))
    }

    async fn list_active_tasks(&self) -> Result<Vec<StoredTask>, DbError> {
        Ok(self.state.active_tasks())
    }

    async fn begin_provenance(&self, _conversation_id: Uuid) -> Option<i64> {
        None
    }

    async fn record_provenance(
        &self,
        _task_id: Uuid,
        _conversation_id: Uuid,
        _start_seq: Option<i64>,
    ) {
    }

    fn seed_conversation(&self, record: ConversationRecord) {
        self.state.conversations.insert(record.id, record);
    }

    async fn conversation_exists(&self, id: Uuid) -> Result<bool, DbError> {
        Ok(self.state.conversations.contains_key(&id))
    }

    async fn remove_conversation(&self, id: Uuid) {
        self.state.conversations.remove(&id);
    }

    async fn build_turn_input(
        &self,
        conversation_id: Uuid,
        _seeded: bool,
        user_msg: Message,
    ) -> Result<Vec<Message>, TurnInputError> {
        self.state.build_turn_input(conversation_id, user_msg)
    }

    async fn refresh_conversation(&self, id: Uuid, full_history: Vec<Message>) {
        self.state.refresh_conversation(id, full_history);
    }

    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>, DbError> {
        Ok(self.state.conversation_summaries())
    }

    async fn get_conversation_view(&self, id: Uuid) -> Result<Option<ConversationRecord>, DbError> {
        Ok(self.state.conversations.get(&id).map(|r| r.clone()))
    }

    async fn list_conversation_children(
        &self,
        _id: Uuid,
        _limit: u32,
        _offset: u32,
    ) -> Result<Option<Vec<DbConversationSummary>>, DbError> {
        // Lineage is durable-only; subagent conversations never enter the
        // in-memory working set.
        Ok(None)
    }

    async fn drain_pending(&self) -> ShutdownSummary {
        self.state.drain().0
    }
}

/// Postgres-write-through adapter.
///
/// Keeps the same working set as the worker's fast cache, writes task state
/// through to postgres, and answers reads from postgres so any replica behind a
/// shared Service can poll any task.
pub struct PostgresTaskStore {
    state: WorkingState,
    store: Arc<PgConversationStore>,
}

impl PostgresTaskStore {
    #[must_use]
    pub fn new(store: Arc<PgConversationStore>) -> Self {
        Self {
            state: WorkingState::default(),
            store,
        }
    }

    /// Snapshot of a working-set conversation's messages, or `None` if this
    /// replica holds no local record for it.
    #[cfg(test)]
    #[must_use]
    pub fn conversation_messages(&self, id: Uuid) -> Option<Vec<Message>> {
        self.state
            .conversations
            .get(&id)
            .map(|r| r.messages.clone())
    }

    /// Whether this replica holds a working-set record for `id`.
    #[cfg(test)]
    #[must_use]
    pub fn has_cached_conversation(&self, id: Uuid) -> bool {
        self.state.conversations.contains_key(&id)
    }

    /// The underlying durable store, for tests that seed history directly.
    #[cfg(test)]
    #[must_use]
    pub fn durable_store(&self) -> Arc<PgConversationStore> {
        Arc::clone(&self.store)
    }

    /// Seed the working-set cache directly, bypassing the durable write.
    #[cfg(test)]
    pub fn seed_cache(&self, record: ConversationRecord) {
        self.state.conversations.insert(record.id, record);
    }

    /// Insert a task into the working set directly, bypassing the durable write.
    #[cfg(test)]
    pub fn seed_task(&self, task: TaskRecord) {
        self.state.tasks.insert(task.id, task);
    }

    /// Best-effort write-through of a task's current working state.
    async fn write_through(&self, id: Uuid) {
        let Some(update) = self.state.tasks.get(&id).map(|rec| status_update(&rec)) else {
            return;
        };
        if let Err(e) = self.store.record_task_status(&update).await {
            warn!(task_id = %id, error = %e, "record task status failed");
        }
    }
}

#[async_trait]
impl TaskStore for PostgresTaskStore {
    async fn insert_pending(&self, task: &TaskRecord) -> Result<(), DbError> {
        self.state.tasks.insert(task.id, task.clone());
        // Persist the pending row synchronously and fail-closed: a returned
        // task id must be durably pollable from any replica.
        if let Err(e) = self.store.record_task_status(&status_update(task)).await {
            self.state.tasks.remove(&task.id);
            return Err(e);
        }
        Ok(())
    }

    async fn remove_task(&self, id: Uuid) {
        self.state.tasks.remove(&id);
        if let Err(e) = self.store.delete_task(id).await {
            warn!(task_id = %id, error = %e, "rollback of durable task row failed");
        }
    }

    async fn mark_running(&self, id: Uuid) -> Option<TaskTiming> {
        let timing = self.state.mark_running(id);
        self.write_through(id).await;
        timing
    }

    async fn mark_succeeded(&self, id: Uuid, output: String) {
        self.state.mark_succeeded(id, output);
        self.write_through(id).await;
    }

    async fn mark_failed(&self, id: Uuid, reason: &str) {
        self.state.mark_terminal(id, TaskStatus::Failed, reason);
        self.write_through(id).await;
    }

    async fn mark_cancelled(&self, id: Uuid, reason: &str) {
        self.state.mark_terminal(id, TaskStatus::Cancelled, reason);
        self.write_through(id).await;
    }

    async fn get_task(&self, id: Uuid) -> Result<Option<StoredTask>, DbError> {
        self.store.get_task(id).await
    }

    async fn list_active_tasks(&self) -> Result<Vec<StoredTask>, DbError> {
        self.store.list_active_tasks().await
    }

    async fn begin_provenance(&self, conversation_id: Uuid) -> Option<i64> {
        match self.store.max_seq(conversation_id).await {
            Ok(max) => Some(max.map_or(0, |m| m + 1)),
            Err(e) => {
                warn!(%conversation_id, error = %e, "max_seq before run failed; task provenance skipped");
                None
            }
        }
    }

    async fn record_provenance(
        &self,
        task_id: Uuid,
        conversation_id: Uuid,
        start_seq: Option<i64>,
    ) {
        let Some(start) = start_seq else {
            return;
        };
        let end = match self.store.max_seq(conversation_id).await {
            Ok(Some(end)) => end,
            Ok(None) => return,
            Err(e) => {
                warn!(%conversation_id, error = %e, "max_seq after run failed; task provenance skipped");
                return;
            }
        };
        if end < start {
            return;
        }
        if let Err(e) = self
            .store
            .record_task(task_id, conversation_id, start, end)
            .await
        {
            warn!(%task_id, error = %e, "record task provenance failed");
        }
    }

    fn seed_conversation(&self, record: ConversationRecord) {
        self.state.conversations.insert(record.id, record);
    }

    async fn conversation_exists(&self, id: Uuid) -> Result<bool, DbError> {
        if self.state.conversations.contains_key(&id) {
            return Ok(true);
        }
        // Cold replica: the conversation may have been seeded on a sibling.
        self.store.conversation_exists(id).await
    }

    async fn remove_conversation(&self, id: Uuid) {
        self.state.conversations.remove(&id);
    }

    async fn build_turn_input(
        &self,
        conversation_id: Uuid,
        seeded: bool,
        user_msg: Message,
    ) -> Result<Vec<Message>, TurnInputError> {
        // A continuation reads history from postgres so a turn landing on a
        // replica without the local cache (or with a stale one) still sees every
        // prior turn — the store is the single source of truth. A new
        // conversation carries its just-minted seed only in the local record.
        if !seeded {
            let conv = self
                .store
                .get_conversation(conversation_id)
                .await
                .map_err(|e| {
                    warn!(error = %e, "load conversation history failed");
                    TurnInputError::HistoryLoadFailed
                })?
                .ok_or(TurnInputError::ConversationNotFound)?;
            let mut messages = conv.messages.as_ref().clone();
            messages.push(user_msg);
            return Ok(messages);
        }
        self.state.build_turn_input(conversation_id, user_msg)
    }

    async fn refresh_conversation(&self, id: Uuid, full_history: Vec<Message>) {
        self.state.refresh_conversation(id, full_history);
    }

    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>, DbError> {
        let summaries = self
            .store
            .list_conversations(CONVERSATIONS_PAGE_LIMIT, 0)
            .await?;
        Ok(summaries
            .into_iter()
            .map(|s| ConversationSummary {
                id: s.id,
                created_at: s.created_at,
                updated_at: s.updated_at,
                turn_count: usize::try_from(s.turn_count).unwrap_or(usize::MAX),
                message_count: usize::try_from(s.message_count).unwrap_or(usize::MAX),
            })
            .collect())
    }

    async fn get_conversation_view(&self, id: Uuid) -> Result<Option<ConversationRecord>, DbError> {
        // Postgres-authoritative so a cold replica answers, falling back to the
        // local record for a freshly-seeded conversation not yet persisted.
        if let Some(conv) = self.store.get_conversation(id).await? {
            let messages = conv.messages.as_ref().clone();
            return Ok(Some(ConversationRecord {
                id: conv.id,
                created_at: conv.created_at,
                updated_at: conv.updated_at,
                turn_count: turn_count_of(&messages),
                messages,
            }));
        }
        Ok(self.state.conversations.get(&id).map(|r| r.clone()))
    }

    async fn list_conversation_children(
        &self,
        id: Uuid,
        limit: u32,
        offset: u32,
    ) -> Result<Option<Vec<DbConversationSummary>>, DbError> {
        self.store
            .list_child_conversations(id, limit, offset)
            .await
            .map(Some)
    }

    async fn drain_pending(&self) -> ShutdownSummary {
        let (summary, cancelled) = self.state.drain();
        // Mirror the shutdown cancellations so a killed replica's dropped tasks
        // read `cancelled` rather than a stuck `pending`. Best-effort: shutdown
        // should not block on the database.
        for record in &cancelled {
            if let Err(e) = self.store.record_task_status(&status_update(record)).await {
                warn!(task_id = %record.id, error = %e, "record shutdown cancellation failed");
            }
        }
        summary
    }
}

/// Page size for a store-backed `GET /conversations`. Roots are bounded in
/// practice; this caps a single response without pagination params, matching the
/// child-listing cap in serve.
const CONVERSATIONS_PAGE_LIMIT: u32 = 100;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    fn task(id: Uuid, status: TaskStatus, created_at: DateTime<Utc>) -> TaskRecord {
        TaskRecord {
            id,
            status,
            conversation_id: Uuid::new_v4(),
            created_at,
            updated_at: created_at,
            output: None,
            error: None,
            queue_depth_at_enqueue: 0,
        }
    }

    fn seeded_conversation(id: Uuid) -> ConversationRecord {
        ConversationRecord {
            id,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            turn_count: 0,
            messages: vec![Message::system(id, "system")],
        }
    }

    #[tokio::test]
    async fn test_insert_pending_is_recorded_and_pollable() {
        let store = InMemoryTaskStore::new();
        let id = Uuid::new_v4();
        store
            .insert_pending(&task(id, TaskStatus::Pending, Utc::now()))
            .await
            .unwrap();
        assert_eq!(store.task_count(), 1);
        let got = store.get_task(id).await.unwrap().expect("task pollable");
        assert_eq!(got.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn test_lifecycle_transitions_advance_status() {
        let store = InMemoryTaskStore::new();
        let id = Uuid::new_v4();
        store
            .insert_pending(&task(id, TaskStatus::Pending, Utc::now()))
            .await
            .unwrap();

        let timing = store.mark_running(id).await.expect("running timing");
        assert_eq!(timing.queue_depth_at_enqueue, 0);
        store.mark_succeeded(id, "done".to_string()).await;

        let got = store.get_task(id).await.unwrap().expect("task exists");
        assert_eq!(got.status, TaskStatus::Succeeded);
        assert_eq!(got.output.as_deref(), Some("done"));
    }

    #[tokio::test]
    async fn test_mark_running_missing_task_returns_none() {
        let store = InMemoryTaskStore::new();
        assert!(store.mark_running(Uuid::new_v4()).await.is_none());
    }

    #[tokio::test]
    async fn test_list_active_tasks_excludes_finished_and_sorts_by_created_at() {
        let store = InMemoryTaskStore::new();
        let base = Utc::now();
        let inserts = [
            (TaskStatus::Succeeded, base),
            (TaskStatus::Running, base + chrono::Duration::seconds(1)),
            (TaskStatus::Pending, base + chrono::Duration::seconds(3)),
            (TaskStatus::Failed, base + chrono::Duration::seconds(2)),
            (TaskStatus::Pending, base + chrono::Duration::seconds(4)),
        ];
        for (status, created_at) in inserts {
            store
                .insert_pending(&task(Uuid::new_v4(), status, created_at))
                .await
                .unwrap();
        }
        let active = store.list_active_tasks().await.unwrap();
        let statuses: Vec<TaskStatus> = active.iter().map(|t| t.status).collect();
        assert_eq!(
            statuses,
            vec![
                TaskStatus::Running,
                TaskStatus::Pending,
                TaskStatus::Pending
            ]
        );
        assert!(
            active
                .windows(2)
                .all(|w| w[0].created_at <= w[1].created_at)
        );
    }

    #[tokio::test]
    async fn test_conversation_exists_reflects_seed_and_removal() {
        let store = InMemoryTaskStore::new();
        let id = Uuid::new_v4();
        assert!(!store.conversation_exists(id).await.unwrap());
        store.seed_conversation(seeded_conversation(id));
        assert!(store.conversation_exists(id).await.unwrap());
        store.remove_conversation(id).await;
        assert!(!store.conversation_exists(id).await.unwrap());
    }

    #[tokio::test]
    async fn test_build_turn_input_appends_user_and_advances_record() {
        let store = InMemoryTaskStore::new();
        let id = Uuid::new_v4();
        store.seed_conversation(seeded_conversation(id));

        let snapshot = store
            .build_turn_input(id, true, Message::user(id, "hi"))
            .await
            .unwrap();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[1].content, "hi");
        assert_eq!(store.conversation_turn_count(id), Some(1));
        // The user message is retained for the next turn's snapshot.
        assert_eq!(store.conversation_messages(id).unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_build_turn_input_missing_record_errors() {
        let store = InMemoryTaskStore::new();
        let err = store
            .build_turn_input(Uuid::new_v4(), true, Message::user(Uuid::new_v4(), "hi"))
            .await
            .expect_err("missing record must error");
        assert_eq!(err.reason(), "conversation record missing");
    }

    #[tokio::test]
    async fn test_children_unavailable_without_store() {
        let store = InMemoryTaskStore::new();
        assert!(
            store
                .list_conversation_children(Uuid::new_v4(), 10, 0)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_drain_cancels_pending_only() {
        let store = InMemoryTaskStore::new();
        let now = Utc::now();
        let pending = Uuid::new_v4();
        store
            .insert_pending(&task(pending, TaskStatus::Pending, now))
            .await
            .unwrap();
        let done = Uuid::new_v4();
        store
            .insert_pending(&task(done, TaskStatus::Succeeded, now))
            .await
            .unwrap();

        let summary = store.drain_pending().await;
        assert_eq!(summary.pending_dropped, 1);
        assert_eq!(summary.succeeded, 1);
        assert_eq!(
            store.get_task(pending).await.unwrap().unwrap().status,
            TaskStatus::Cancelled
        );
    }
}
