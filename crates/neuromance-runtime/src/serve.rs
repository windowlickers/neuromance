//! Serve mode: long-lived HTTP intake for agent task work.
//!
//! - `POST /tasks/new` enqueues a task and returns a UUID and the queue
//!   depth at enqueue time. Returns `429` when the queue is at capacity.
//! - `GET /tasks` lists the active queue (pending + running), sorted by
//!   submit time — a caller's index in the array is their queue position.
//! - `GET /tasks/{id}` returns the current state of a single task.
//!
//! Liveness/readiness live on a separate server at `runtime.health_addr`
//! (default `127.0.0.1:8081`) — see `health.rs`. Do not point readiness
//! probes at the task port.
//!
//! Tasks are processed sequentially by a single worker that owns the
//! agent. Because that one agent is reused across tasks, the worker resets
//! its `execute_python` interpreter after each run — the sandbox session
//! when tools run remotely, the in-process interpreter otherwise — so one
//! task's interpreter state never bleeds into the next. In-memory state is
//! authoritative for serving; restarts lose
//! pending and completed tasks. When `[database]` is configured,
//! conversation history is additionally written through to postgres as a
//! durable record: Core persists messages incrementally during each run,
//! and this module records conversation metadata on creation and soft
//! deletes (the database keeps the history when a conversation is
//! `DELETE`d here).

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use metrics::{counter, gauge, histogram};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::{Level, Span, error, field, info, info_span, warn};
use uuid::Uuid;

use neuromance::error::CoreError;
use neuromance_agent::Agent;
use neuromance_client::LLMClient;
use neuromance_common::chat::{Conversation, ConversationStatus, Message};
use neuromance_db::PgConversationStore;

use crate::SessionReset;
use crate::config::RuntimeConfig;
use crate::sandbox::{EXECUTE_PYTHON, SandboxClient};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

enum JobOutcome {
    Succeeded,
    Failed,
    Cancelled,
}

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

#[derive(Debug, Deserialize)]
pub struct CreateTaskRequest {
    pub user: String,
    /// Continue an existing conversation. When omitted, the server mints a
    /// fresh conversation seeded with the configured system prompt and
    /// returns its id in the response. When supplied, the conversation must
    /// already exist — unknown ids return 404 rather than auto-creating.
    #[serde(default)]
    pub conversation_id: Option<Uuid>,
    /// Override the configured system prompt for a freshly-seeded
    /// conversation. Falls back to the runtime's configured prompt when
    /// omitted. Must be non-empty; an empty or whitespace-only value is
    /// rejected with 400. Supplying it alongside an existing `conversation_id` is
    /// rejected with 400, since that conversation already holds its system
    /// message; an unknown `conversation_id` still returns 404.
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateTaskResponse {
    pub task_id: Uuid,
    pub conversation_id: Uuid,
    pub status: TaskStatus,
    pub queue_depth_at_enqueue: usize,
}

#[derive(Debug)]
enum EnqueueError {
    QueueFull { depth: usize, max: usize },
    WorkerShutdown,
    ConversationNotFound(Uuid),
    SystemPromptOnExisting(Uuid),
    EmptySystemPrompt,
}

impl IntoResponse for EnqueueError {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            Self::QueueFull { depth, max } => (
                StatusCode::TOO_MANY_REQUESTS,
                serde_json::json!({
                    "error": "queue full",
                    "queue_depth": depth,
                    "max_queue_depth": max,
                }),
            ),
            Self::WorkerShutdown => (
                StatusCode::SERVICE_UNAVAILABLE,
                serde_json::json!({"error": "worker shutting down"}),
            ),
            Self::ConversationNotFound(id) => (
                StatusCode::NOT_FOUND,
                serde_json::json!({
                    "error": "conversation not found",
                    "conversation_id": id,
                }),
            ),
            Self::SystemPromptOnExisting(id) => (
                StatusCode::BAD_REQUEST,
                serde_json::json!({
                    "error": "system_prompt cannot be set when continuing an existing conversation",
                    "conversation_id": id,
                }),
            ),
            Self::EmptySystemPrompt => (
                StatusCode::BAD_REQUEST,
                serde_json::json!({
                    "error": "system_prompt must not be empty or whitespace-only",
                }),
            ),
        };
        (status, Json(body)).into_response()
    }
}

struct WorkerJob {
    task_id: Uuid,
    conversation_id: Uuid,
    user: String,
}

#[derive(Clone)]
pub struct ServeState {
    tasks: Arc<DashMap<Uuid, TaskRecord>>,
    conversations: Arc<DashMap<Uuid, ConversationRecord>>,
    work_tx: mpsc::Sender<WorkerJob>,
    system_prompt: Arc<str>,
    agent_id: Arc<str>,
    /// Durable conversation record; `None` when `[database]` is not configured.
    store: Option<Arc<PgConversationStore>>,
}

/// Cap on `POST /tasks` request bodies. Task input is a single user prompt;
/// 64 KiB is generous and prevents memory amplification from oversized intake.
const MAX_TASK_BODY_BYTES: usize = 64 * 1024;

/// Page size for `GET /conversations/{id}/children`. A delegation fan-out is
/// bounded in practice; this caps a single response without pagination params.
const CHILDREN_PAGE_LIMIT: u32 = 100;

pub fn router(state: ServeState) -> Router {
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|req: &axum::http::Request<_>| {
            info_span!(
                "http_request",
                method = %req.method(),
                path = %req.uri().path(),
                status = field::Empty,
            )
        })
        .on_response(
            |res: &axum::http::Response<_>, latency: std::time::Duration, span: &Span| {
                let status = res.status();
                span.record("status", status.as_u16());
                let latency_ms = u64::try_from(latency.as_millis()).unwrap_or(u64::MAX);
                // Access logs are high-volume — clients poll /tasks/{id} on a tight
                // loop — so successful responses log at DEBUG. Task lifecycle events
                // ("task starting"/"task succeeded") carry the real signal at INFO.
                // Client and server errors stay at WARN/ERROR so they remain visible
                // at the default level.
                if status.is_server_error() {
                    tracing::event!(parent: span, Level::ERROR, latency_ms, "http response");
                } else if status.is_client_error() {
                    tracing::event!(parent: span, Level::WARN, latency_ms, "http response");
                } else {
                    tracing::event!(parent: span, Level::DEBUG, latency_ms, "http response");
                }
            },
        );
    Router::new()
        .route("/tasks", get(list_tasks))
        .route("/tasks/new", post(create_task))
        .route("/tasks/{id}", get(get_task))
        .route("/conversations", get(list_conversations))
        .route(
            "/conversations/{id}",
            get(get_conversation).delete(delete_conversation),
        )
        .route("/conversations/{id}/children", get(list_conversation_children))
        .layer(DefaultBodyLimit::max(MAX_TASK_BODY_BYTES))
        .layer(trace_layer)
        .with_state(state)
}

fn queue_depth(tx: &mpsc::Sender<WorkerJob>) -> usize {
    tx.max_capacity().saturating_sub(tx.capacity())
}

/// Mint a fresh conversation seeded with a system prompt, preferring the
/// caller's override and falling back to the configured prompt.
fn seed_new_conversation(state: &ServeState, system_prompt: Option<&str>) -> Uuid {
    let id = Uuid::new_v4();
    let now = Utc::now();
    let prompt = system_prompt.unwrap_or_else(|| state.system_prompt.as_ref());
    state.conversations.insert(
        id,
        ConversationRecord {
            id,
            created_at: now,
            updated_at: now,
            turn_count: 0,
            messages: vec![Message::system(id, prompt)],
        },
    );
    // Record the conversation row up front so it carries the agent identity;
    // the messages themselves are persisted by Core during the run.
    if let Some(store) = &state.store {
        let store = Arc::clone(store);
        let mut conversation = Conversation::new();
        conversation.id = id;
        conversation.metadata.insert(
            "agent_id".to_string(),
            serde_json::json!(state.agent_id.as_ref()),
        );
        tokio::spawn(async move {
            if let Err(e) = store.upsert_conversation(&conversation).await {
                warn!(error = %e, conversation_id = %id, "failed to record conversation row");
            }
        });
    }
    id
}

/// Resolve a request's `conversation_id`: reuse if it exists, mint and seed
/// a fresh record (honoring the `system_prompt` override) if omitted, reject
/// unknown ids. A `system_prompt` supplied for an existing conversation is
/// rejected, since that conversation already holds its system message.
///
/// Returns the id and whether a fresh conversation was seeded in this call, so
/// the caller can roll the seed back if the task ultimately fails to enqueue.
fn resolve_conversation(
    state: &ServeState,
    requested: Option<Uuid>,
    system_prompt: Option<&str>,
) -> Result<(Uuid, bool), EnqueueError> {
    if system_prompt.is_some_and(|p| p.trim().is_empty()) {
        return Err(EnqueueError::EmptySystemPrompt);
    }
    requested.map_or_else(
        || Ok((seed_new_conversation(state, system_prompt), true)),
        |id| {
            if !state.conversations.contains_key(&id) {
                Err(EnqueueError::ConversationNotFound(id))
            } else if system_prompt.is_some() {
                Err(EnqueueError::SystemPromptOnExisting(id))
            } else {
                Ok((id, false))
            }
        },
    )
}

fn try_enqueue(
    state: &ServeState,
    user: String,
    conversation_id: Option<Uuid>,
    system_prompt: Option<&str>,
) -> Result<TaskRecord, EnqueueError> {
    let (conversation_id, seeded) = resolve_conversation(state, conversation_id, system_prompt)?;
    let task_id = Uuid::new_v4();
    let now = Utc::now();
    let depth = queue_depth(&state.work_tx);
    let record = TaskRecord {
        id: task_id,
        status: TaskStatus::Pending,
        conversation_id,
        created_at: now,
        updated_at: now,
        output: None,
        error: None,
        queue_depth_at_enqueue: depth,
    };
    state.tasks.insert(task_id, record.clone());

    match state.work_tx.try_send(WorkerJob {
        task_id,
        conversation_id,
        user,
    }) {
        Ok(()) => {
            #[allow(clippy::cast_precision_loss)]
            gauge!("neuromance_queue_depth").set(queue_depth(&state.work_tx) as f64);
            Ok(record)
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            state.tasks.remove(&task_id);
            if seeded {
                state.conversations.remove(&conversation_id);
            }
            counter!("neuromance_enqueue_rejections_total", "reason" => "queue_full").increment(1);
            Err(EnqueueError::QueueFull {
                depth,
                max: state.work_tx.max_capacity(),
            })
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            if let Some(mut entry) = state.tasks.get_mut(&task_id) {
                entry.status = TaskStatus::Failed;
                entry.error = Some("worker shutting down".to_string());
                entry.updated_at = Utc::now();
            }
            if seeded {
                state.conversations.remove(&conversation_id);
            }
            counter!("neuromance_enqueue_rejections_total", "reason" => "worker_shutdown")
                .increment(1);
            Err(EnqueueError::WorkerShutdown)
        }
    }
}

async fn create_task(
    State(state): State<ServeState>,
    Json(req): Json<CreateTaskRequest>,
) -> impl IntoResponse {
    match try_enqueue(
        &state,
        req.user,
        req.conversation_id,
        req.system_prompt.as_deref(),
    ) {
        Ok(record) => (
            StatusCode::ACCEPTED,
            Json(CreateTaskResponse {
                task_id: record.id,
                conversation_id: record.conversation_id,
                status: record.status,
                queue_depth_at_enqueue: record.queue_depth_at_enqueue,
            }),
        )
            .into_response(),
        Err(err) => err.into_response(),
    }
}

fn active_tasks_sorted(tasks: &DashMap<Uuid, TaskRecord>) -> Vec<TaskRecord> {
    let mut active: Vec<TaskRecord> = tasks
        .iter()
        .filter(|e| matches!(e.value().status, TaskStatus::Pending | TaskStatus::Running))
        .map(|e| e.value().clone())
        .collect();
    active.sort_by_key(|r| r.created_at);
    active
}

async fn list_tasks(State(state): State<ServeState>) -> impl IntoResponse {
    (StatusCode::OK, Json(active_tasks_sorted(&state.tasks))).into_response()
}

async fn get_task(State(state): State<ServeState>, Path(id): Path<Uuid>) -> impl IntoResponse {
    state.tasks.get(&id).map_or_else(
        || {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "task not found"})),
            )
                .into_response()
        },
        |rec| (StatusCode::OK, Json(rec.clone())).into_response(),
    )
}

fn conversations_sorted(
    conversations: &DashMap<Uuid, ConversationRecord>,
) -> Vec<ConversationSummary> {
    let mut summaries: Vec<ConversationSummary> = conversations
        .iter()
        .map(|e| ConversationSummary::from(e.value()))
        .collect();
    summaries.sort_by_key(|s| s.created_at);
    summaries
}

async fn list_conversations(State(state): State<ServeState>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(conversations_sorted(&state.conversations)),
    )
        .into_response()
}

async fn get_conversation(
    State(state): State<ServeState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    state.conversations.get(&id).map_or_else(
        || {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "conversation not found"})),
            )
                .into_response()
        },
        |rec| (StatusCode::OK, Json(rec.clone())).into_response(),
    )
}

/// Lists the child conversations (e.g. subagent delegations) of a conversation.
///
/// Lineage is only durable in postgres — subagent conversations never enter the
/// in-memory serving map — so this returns `503` when no `[database]` is
/// configured.
async fn list_conversation_children(
    State(state): State<ServeState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    let Some(store) = &state.store else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "conversation lineage requires a configured database",
            })),
        )
            .into_response();
    };
    match store
        .list_child_conversations(id, CHILDREN_PAGE_LIMIT, 0)
        .await
    {
        Ok(children) => (StatusCode::OK, Json(children)).into_response(),
        Err(e) => {
            warn!(error = %e, conversation_id = %id, "failed to list conversation children");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "failed to list conversation children"})),
            )
                .into_response()
        }
    }
}

async fn delete_conversation(
    State(state): State<ServeState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if state.conversations.remove(&id).is_some() {
        // Soft delete in the durable record: the history stays queryable,
        // only the lifecycle status changes.
        if let Some(store) = &state.store {
            let store = Arc::clone(store);
            tokio::spawn(async move {
                if let Err(e) = store
                    .set_conversation_status(id, ConversationStatus::Deleted)
                    .await
                {
                    warn!(error = %e, conversation_id = %id, "failed to soft-delete conversation");
                }
            });
        }
        StatusCode::NO_CONTENT.into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "conversation not found"})),
        )
            .into_response()
    }
}

/// Shared handles the worker needs to run a job and record its provenance.
/// Bundled so [`worker_loop`] and [`run`] stay within the positional-argument
/// budget as fields accrue.
struct WorkerCtx<C: LLMClient + Send + Sync> {
    tasks: Arc<DashMap<Uuid, TaskRecord>>,
    conversations: Arc<DashMap<Uuid, ConversationRecord>>,
    agent: Arc<Mutex<Agent<C>>>,
    /// Durable conversation store; `None` when `[database]` is not configured.
    store: Option<Arc<PgConversationStore>>,
    /// Sandbox handle used to release a task's `execute_python` interpreter once
    /// the run completes. `Some` only when the sandbox hosts `execute_python`;
    /// the interpreter is keyed by task id (see `RemoteToolAdapter`).
    sandbox: Option<SandboxClient>,
    /// Reset handle for the main agent's in-process `execute_python`
    /// interpreter, called after each task so user state does not bleed into
    /// the next. `Some` only when `execute_python` runs locally (not in the
    /// sandbox) and the `python-repl` feature is built.
    local_python: Option<SessionReset>,
}

async fn worker_loop<C: LLMClient + Send + Sync + 'static>(
    mut rx: mpsc::Receiver<WorkerJob>,
    ctx: WorkerCtx<C>,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            () = cancel.cancelled() => {
                info!("worker received shutdown signal");
                return;
            }
            maybe_job = rx.recv() => {
                let Some(job) = maybe_job else {
                    info!("worker channel closed");
                    return;
                };
                #[allow(clippy::cast_precision_loss)]
                gauge!("neuromance_queue_depth").set(rx.len() as f64);
                // Bracket the run by the conversation's seq high-water mark so we
                // can record which messages this task contributed. Core persists
                // the messages themselves during the run; this only reads the
                // boundaries. Best-effort, matching the log-and-continue policy.
                let (task_id, conversation_id) = (job.task_id, job.conversation_id);
                let start_seq = next_seq_before_run(ctx.store.as_ref(), conversation_id).await;
                let _ = process_job(
                    &ctx.tasks,
                    &ctx.conversations,
                    &ctx.agent,
                    job,
                    cancel.clone(),
                )
                .await;
                release_sandbox_session(ctx.sandbox.as_ref(), task_id).await;
                reset_local_python(ctx.local_python.as_ref()).await;
                record_task_provenance(ctx.store.as_ref(), task_id, conversation_id, start_seq)
                    .await;
            }
        }
    }
}

/// Reads the `seq` a task's first message will occupy, or `None` when
/// persistence is off or the read fails. A failed read drops this task's
/// provenance without failing the task.
async fn next_seq_before_run(
    store: Option<&Arc<PgConversationStore>>,
    conversation_id: Uuid,
) -> Option<i64> {
    let store = store?;
    match store.max_seq(conversation_id).await {
        Ok(max) => Some(max.map_or(0, |m| m + 1)),
        Err(e) => {
            warn!(%conversation_id, error = %e, "max_seq before run failed; task provenance skipped");
            None
        }
    }
}

/// Records the `[start_seq, end_seq]` range a task contributed. Skips silently
/// when persistence is off, the boundary read fails, or the run persisted no
/// new messages (`end < start`).
async fn record_task_provenance(
    store: Option<&Arc<PgConversationStore>>,
    task_id: Uuid,
    conversation_id: Uuid,
    start_seq: Option<i64>,
) {
    let (Some(store), Some(start)) = (store, start_seq) else {
        return;
    };
    let end = match store.max_seq(conversation_id).await {
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
    if let Err(e) = store
        .record_task(task_id, conversation_id, start, end)
        .await
    {
        warn!(%task_id, error = %e, "record task provenance failed");
    }
}

/// Free the sandbox `execute_python` interpreter a task used. The interpreter
/// is keyed by task id, so closing it after each run keeps state from bleeding
/// into the next task and stops interpreters accumulating in a long-lived serve
/// process. Best-effort: a failed close leaks one interpreter but never fails
/// the task. No-op when the sandbox is absent or hosts no `execute_python`.
async fn release_sandbox_session(sandbox: Option<&SandboxClient>, task_id: Uuid) {
    let Some(client) = sandbox else {
        return;
    };
    if let Err(e) = client.close_session(task_id.to_string()).await {
        warn!(%task_id, error = %e, "failed to release sandbox interpreter session");
    }
}

/// Clear the main agent's in-process `execute_python` interpreter after a task.
/// The single agent is reused across tasks, so without this a task's variables,
/// imports, and definitions would still be visible to the next one. The
/// in-sandbox interpreter is handled separately by [`release_sandbox_session`];
/// this is the local counterpart. No-op when there is no local interpreter to
/// reset; the handle itself logs on failure.
async fn reset_local_python(local_python: Option<&SessionReset>) {
    if let Some(reset) = local_python {
        reset().await;
    }
}

/// Mark a task `Failed` with `reason` and emit the failure metric. Pulled out
/// because both the early "conversation deleted" exit and the late
/// agent-error path share this shape.
fn fail_task(tasks: &DashMap<Uuid, TaskRecord>, task_id: Uuid, reason: &str) {
    if let Some(mut entry) = tasks.get_mut(&task_id) {
        entry.status = TaskStatus::Failed;
        entry.error = Some(reason.to_string());
        entry.updated_at = Utc::now();
    }
    counter!("neuromance_tasks_total", "outcome" => "failed").increment(1);
}

#[allow(clippy::significant_drop_tightening)]
#[tracing::instrument(
    name = "task",
    skip_all,
    fields(
        task_id = %job.task_id,
        agent_id = field::Empty,
        conversation_id = %job.conversation_id,
    ),
)]
async fn process_job<C: LLMClient + Send + Sync>(
    tasks: &Arc<DashMap<Uuid, TaskRecord>>,
    conversations: &Arc<DashMap<Uuid, ConversationRecord>>,
    agent: &Arc<Mutex<Agent<C>>>,
    job: WorkerJob,
    cancel: CancellationToken,
) -> JobOutcome {
    let dequeued_at = Utc::now();
    let run_start = Instant::now();
    let (created_at, depth_at_enqueue) = if let Some(mut entry) = tasks.get_mut(&job.task_id) {
        entry.status = TaskStatus::Running;
        entry.updated_at = dequeued_at;
        (entry.created_at, entry.queue_depth_at_enqueue)
    } else {
        (dequeued_at, 0)
    };

    let queue_wait_ms = (dequeued_at - created_at).num_milliseconds().max(0);
    #[allow(clippy::cast_precision_loss)]
    histogram!("neuromance_queue_wait_seconds").record(queue_wait_ms as f64 / 1000.0);
    info!(
        queue_wait_ms,
        depth_at_enqueue,
        user_bytes = job.user.len(),
        "task starting",
    );

    // Snapshot + append-user inside a single get_mut so a racing DELETE
    // either commits the user message or fails the task cleanly, never
    // both.
    let user_msg = Message::user(job.conversation_id, &job.user);
    let input_messages = if let Some(mut entry) = conversations.get_mut(&job.conversation_id) {
        let mut snapshot = entry.messages.clone();
        snapshot.push(user_msg.clone());
        entry.messages.push(user_msg);
        entry.turn_count = entry.turn_count.saturating_add(1);
        entry.updated_at = Utc::now();
        snapshot
    } else {
        warn!("conversation deleted before task dequeue");
        fail_task(tasks, job.task_id, "conversation deleted");
        return JobOutcome::Failed;
    };

    let mut agent = agent.lock().await;
    let span = Span::current();
    span.record("agent_id", field::display(agent.id()));

    // Seed the runtime task id so subagent conversations spawned during this
    // run inherit it as their `parent_task_id`. The root conversation itself
    // has no parent, so no conversation id is seeded here.
    let exec_result = tokio::select! {
        biased;
        () = cancel.cancelled() => Err(CoreError::Cancelled("worker shutdown".to_string())),
        res = neuromance_agent::scope_task(
            Some(job.task_id),
            agent.execute_with_history(Some(input_messages), cancel.child_token()),
        ) => res,
    };

    let run_ms = u64::try_from(run_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    #[allow(clippy::cast_precision_loss)]
    histogram!("neuromance_task_duration_seconds").record(run_ms as f64 / 1000.0);
    match exec_result {
        Ok((response, full_history)) => {
            let output_bytes = response.content.content.len();
            info!(run_ms, queue_wait_ms, output_bytes, "task succeeded");
            // Replace stored messages with the full history (system + every
            // user/assistant/tool turn). If the conversation was DELETEd
            // mid-flight, silently drop the result — the deletion wins.
            if let Some(mut entry) = conversations.get_mut(&job.conversation_id) {
                entry.messages = full_history;
                entry.updated_at = Utc::now();
            }
            if let Some(mut entry) = tasks.get_mut(&job.task_id) {
                entry.status = TaskStatus::Succeeded;
                entry.output = Some(response.content.content);
                entry.updated_at = Utc::now();
            }
            counter!("neuromance_tasks_total", "outcome" => "succeeded").increment(1);
            JobOutcome::Succeeded
        }
        Err(CoreError::Cancelled(_)) => {
            warn!(run_ms, "task cancelled");
            if let Some(mut entry) = tasks.get_mut(&job.task_id) {
                entry.status = TaskStatus::Cancelled;
                entry.error = Some("cancelled".to_string());
                entry.updated_at = Utc::now();
            }
            counter!("neuromance_tasks_total", "outcome" => "cancelled").increment(1);
            JobOutcome::Cancelled
        }
        Err(e) => {
            error!(run_ms, error = %e, "task failed");
            fail_task(tasks, job.task_id, &e.to_string());
            JobOutcome::Failed
        }
    }
}

/// Bind the task server, spawn the worker, and run until `cancel` fires.
///
/// # Errors
/// Returns an error if `runtime.listen_addr` is invalid, the bind fails,
/// or the HTTP server returns an error during operation.
pub async fn run<C: LLMClient + Send + Sync + 'static>(
    config: &RuntimeConfig,
    agent: Agent<C>,
    store: Option<Arc<PgConversationStore>>,
    sandbox_client: Option<SandboxClient>,
    local_python: Option<SessionReset>,
    cancel: CancellationToken,
) -> Result<()> {
    let tasks: Arc<DashMap<Uuid, TaskRecord>> = Arc::new(DashMap::new());
    let conversations: Arc<DashMap<Uuid, ConversationRecord>> = Arc::new(DashMap::new());
    let agent = Arc::new(Mutex::new(agent));
    let (work_tx, work_rx) = mpsc::channel::<WorkerJob>(config.runtime.max_queue_depth);
    let system_prompt: Arc<str> = Arc::from(config.agent.system_prompt.as_str());

    // Per-task interpreter cleanup is only needed when the sandbox actually hosts
    // execute_python; stateless tools create no session to release.
    let session_closer =
        sandbox_client.filter(|_| config.tools.iter().any(|t| t.name == EXECUTE_PYTHON));

    let worker = tokio::spawn(worker_loop(
        work_rx,
        WorkerCtx {
            tasks: Arc::clone(&tasks),
            conversations: Arc::clone(&conversations),
            agent: Arc::clone(&agent),
            store: store.clone(),
            sandbox: session_closer,
            local_python,
        },
        cancel.clone(),
    ));

    let state = ServeState {
        tasks: Arc::clone(&tasks),
        conversations: Arc::clone(&conversations),
        work_tx,
        system_prompt,
        agent_id: Arc::from(config.agent.id.as_str()),
        store,
    };
    let app = router(state);
    let addr: std::net::SocketAddr = config
        .runtime
        .listen_addr
        .parse()
        .with_context(|| format!("invalid listen_addr: {}", config.runtime.listen_addr))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    info!(%addr, "task server listening");

    let serve_cancel = cancel.clone();
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(async move { serve_cancel.cancelled().await })
        .await;
    if let Err(e) = serve_result {
        warn!(error=%e, "task server exited with error");
    }

    if let Err(e) = worker.await {
        warn!(error=%e, "worker task panicked or was cancelled");
    }

    let summary = drain_pending_tasks(&tasks);
    info!(
        pending_dropped = summary.pending_dropped,
        in_flight_cancelled = summary.in_flight_cancelled,
        succeeded = summary.succeeded,
        failed = summary.failed,
        "serve mode shutdown",
    );
    Ok(())
}

#[derive(Debug, Default, Clone, Copy)]
struct ShutdownSummary {
    pending_dropped: usize,
    in_flight_cancelled: usize,
    succeeded: usize,
    failed: usize,
}

/// Mark any tasks still `Pending` as `Cancelled` and return a summary.
///
/// Without this, dropped tasks would stay `Pending` in the in-memory state
/// map forever — `GET /tasks/{id}` would return a misleading status after
/// the server stopped.
fn drain_pending_tasks(tasks: &DashMap<Uuid, TaskRecord>) -> ShutdownSummary {
    let mut summary = ShutdownSummary::default();
    let now = Utc::now();
    for mut entry in tasks.iter_mut() {
        match entry.status {
            TaskStatus::Pending => {
                let queue_age_ms = (now - entry.created_at).num_milliseconds().max(0);
                warn!(
                    task_id = %entry.id,
                    queue_age_ms,
                    "task dropped at shutdown",
                );
                entry.status = TaskStatus::Cancelled;
                entry.error = Some("dropped at shutdown".to_string());
                entry.updated_at = now;
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
    summary
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::pin::Pin;
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::Stream;
    use tokio::time::{sleep, timeout};

    use neuromance::Core;
    use neuromance_client::ClientError;
    use neuromance_common::chat::MessageRole;
    use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config};

    use super::*;

    /// `LLMClient` stub whose `chat_stream` sleeps long enough to outlast the test.
    /// The cancel-aware select inside `Core::run` should drop the future when
    /// the worker's cancellation token fires.
    struct SleepingClient {
        config: Config,
    }

    impl SleepingClient {
        fn new() -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
            }
        }
    }

    #[async_trait]
    impl LLMClient for SleepingClient {
        fn config(&self) -> &Config {
            &self.config
        }

        async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            std::future::pending().await
        }

        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
        {
            // Never yields a chunk — the cancel race in `Core::run`'s SSE
            // consumer fires when the worker's token is cancelled.
            Ok(Box::pin(futures::stream::pending()))
        }

        fn supports_tools(&self) -> bool {
            true
        }

        fn supports_streaming(&self) -> bool {
            true
        }
    }

    /// `LLMClient` stub that returns a deterministic assistant reply so tests
    /// can assert what landed in conversation history after a successful turn.
    struct EchoClient {
        config: Config,
        reply: String,
    }

    impl EchoClient {
        fn new(reply: &str) -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                reply: reply.to_string(),
            }
        }
    }

    #[async_trait]
    impl LLMClient for EchoClient {
        fn config(&self) -> &Config {
            &self.config
        }

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            let conv_id = request
                .messages
                .first()
                .map_or_else(Uuid::new_v4, |m| m.conversation_id);
            Ok(ChatResponse {
                message: Message::assistant(conv_id, &self.reply),
                model: "mock-model".to_string(),
                usage: None,
                finish_reason: None,
                created_at: Utc::now(),
                response_id: Some("test-response".to_string()),
                metadata: std::collections::HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
        {
            Ok(Box::pin(futures::stream::pending()))
        }

        fn supports_tools(&self) -> bool {
            true
        }

        fn supports_streaming(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn worker_cancels_in_flight_task() {
        let task_id = Uuid::new_v4();
        let conv_id = Uuid::new_v4();
        let now = Utc::now();
        let tasks: Arc<DashMap<Uuid, TaskRecord>> = Arc::new(DashMap::new());
        let conversations: Arc<DashMap<Uuid, ConversationRecord>> = Arc::new(DashMap::new());
        tasks.insert(
            task_id,
            TaskRecord {
                id: task_id,
                status: TaskStatus::Pending,
                conversation_id: conv_id,
                created_at: now,
                updated_at: now,
                output: None,
                error: None,
                queue_depth_at_enqueue: 0,
            },
        );
        conversations.insert(
            conv_id,
            ConversationRecord {
                id: conv_id,
                created_at: now,
                updated_at: now,
                turn_count: 0,
                messages: vec![Message::system(conv_id, "system")],
            },
        );

        let core = Core::new(SleepingClient::new()).with_streaming();
        let agent = Arc::new(Mutex::new(Agent::new("test".into(), core)));
        let (work_tx, work_rx) = mpsc::channel::<WorkerJob>(1);
        let cancel = CancellationToken::new();

        let worker = tokio::spawn(worker_loop(
            work_rx,
            WorkerCtx {
                tasks: Arc::clone(&tasks),
                conversations: Arc::clone(&conversations),
                agent: Arc::clone(&agent),
                store: None,
                sandbox: None,
                local_python: None,
            },
            cancel.clone(),
        ));

        work_tx
            .send(WorkerJob {
                task_id,
                conversation_id: conv_id,
                user: "hello".to_string(),
            })
            .await
            .unwrap();

        // Give the worker a moment to pick up the job and start the agent.
        sleep(Duration::from_millis(100)).await;
        cancel.cancel();

        timeout(Duration::from_secs(2), worker)
            .await
            .expect("worker did not exit within timeout")
            .unwrap();

        let (status, error) = {
            let entry = tasks.get(&task_id).expect("task record should exist");
            (entry.status, entry.error.clone())
        };
        assert_eq!(status, TaskStatus::Cancelled);
        assert_eq!(error.as_deref(), Some("cancelled"));
    }

    fn fresh_state(capacity: usize) -> (ServeState, mpsc::Receiver<WorkerJob>) {
        let tasks: Arc<DashMap<Uuid, TaskRecord>> = Arc::new(DashMap::new());
        let conversations: Arc<DashMap<Uuid, ConversationRecord>> = Arc::new(DashMap::new());
        let (work_tx, work_rx) = mpsc::channel::<WorkerJob>(capacity);
        (
            ServeState {
                tasks,
                conversations,
                work_tx,
                system_prompt: Arc::from("system"),
                agent_id: Arc::from("test-agent"),
                store: None,
            },
            work_rx,
        )
    }

    fn record(status: TaskStatus, created_at: DateTime<Utc>) -> TaskRecord {
        TaskRecord {
            id: Uuid::new_v4(),
            status,
            conversation_id: Uuid::new_v4(),
            created_at,
            updated_at: created_at,
            output: None,
            error: None,
            queue_depth_at_enqueue: 0,
        }
    }

    #[test]
    fn test_try_enqueue_records_queue_depth_zero_when_empty() {
        let (state, _rx) = fresh_state(4);
        let record =
            try_enqueue(&state, "hi".to_string(), None, None).expect("enqueue should succeed");
        assert_eq!(record.queue_depth_at_enqueue, 0);
        assert_eq!(record.status, TaskStatus::Pending);
        assert!(state.tasks.contains_key(&record.id));
    }

    #[test]
    fn test_try_enqueue_queue_depth_grows_as_channel_fills() {
        let (state, _rx) = fresh_state(4);
        let depths: Vec<usize> = (0..3)
            .map(|_| {
                try_enqueue(&state, "hi".to_string(), None, None)
                    .expect("enqueue should succeed")
                    .queue_depth_at_enqueue
            })
            .collect();
        assert_eq!(depths, vec![0, 1, 2]);
    }

    #[test]
    fn test_try_enqueue_returns_queue_full_at_capacity() {
        let (state, _rx) = fresh_state(2);
        let first = try_enqueue(&state, "a".to_string(), None, None).expect("first should fit");
        let second = try_enqueue(&state, "b".to_string(), None, None).expect("second should fit");

        let err =
            try_enqueue(&state, "c".to_string(), None, None).expect_err("third should reject");
        assert!(
            matches!(err, EnqueueError::QueueFull { depth: 2, max: 2 }),
            "got {err:?}"
        );

        assert!(state.tasks.contains_key(&first.id));
        assert!(state.tasks.contains_key(&second.id));
        assert_eq!(state.tasks.len(), 2, "rejected task must not linger");
        assert_eq!(
            state.conversations.len(),
            2,
            "rejected enqueue must not leak a seeded conversation"
        );
    }

    #[test]
    fn test_try_enqueue_returns_worker_shutdown_when_rx_dropped() {
        let (state, rx) = fresh_state(4);
        drop(rx);

        let err = try_enqueue(&state, "hi".to_string(), None, None).expect_err("send should fail");
        assert!(matches!(err, EnqueueError::WorkerShutdown), "got {err:?}");

        let task_id = state
            .tasks
            .iter()
            .next()
            .map(|e| *e.key())
            .expect("record should exist");
        let (status, error) = {
            let entry = state.tasks.get(&task_id).expect("record should exist");
            (entry.status, entry.error.clone())
        };
        assert_eq!(status, TaskStatus::Failed);
        assert_eq!(error.as_deref(), Some("worker shutting down"));
        assert!(
            state.conversations.is_empty(),
            "worker-shutdown rejection must not leak a seeded conversation"
        );
    }

    #[test]
    fn test_try_enqueue_seeds_fresh_conversation_with_system_prompt() {
        let (state, _rx) = fresh_state(4);
        let record =
            try_enqueue(&state, "hi".to_string(), None, None).expect("enqueue should succeed");

        let (turn_count, messages) = {
            let conv = state
                .conversations
                .get(&record.conversation_id)
                .expect("conversation should exist");
            (conv.turn_count, conv.messages.clone())
        };
        assert_eq!(turn_count, 0);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[0].content, "system");
        assert_eq!(messages[0].conversation_id, record.conversation_id);
    }

    #[test]
    fn test_try_enqueue_override_seeds_custom_system_prompt() {
        let (state, _rx) = fresh_state(4);
        let record = try_enqueue(&state, "hi".to_string(), None, Some("be terse"))
            .expect("enqueue should succeed");

        let messages = {
            let conv = state
                .conversations
                .get(&record.conversation_id)
                .expect("conversation should exist");
            conv.messages.clone()
        };
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[0].content, "be terse");
    }

    #[test]
    fn test_try_enqueue_falls_back_to_configured_prompt_when_override_omitted() {
        let (state, _rx) = fresh_state(4);
        let record =
            try_enqueue(&state, "hi".to_string(), None, None).expect("enqueue should succeed");

        let content = {
            let conv = state
                .conversations
                .get(&record.conversation_id)
                .expect("conversation should exist");
            conv.messages[0].content.clone()
        };
        assert_eq!(content, "system");
    }

    #[test]
    fn test_try_enqueue_override_on_existing_conversation_is_rejected() {
        let (state, _rx) = fresh_state(4);
        let first =
            try_enqueue(&state, "hi".to_string(), None, None).expect("first should succeed");

        let err = try_enqueue(
            &state,
            "again".to_string(),
            Some(first.conversation_id),
            Some("be terse"),
        )
        .expect_err("override on existing conversation should be rejected");
        assert!(
            matches!(err, EnqueueError::SystemPromptOnExisting(id) if id == first.conversation_id)
        );

        let content = {
            let conv = state
                .conversations
                .get(&first.conversation_id)
                .expect("conversation should exist");
            conv.messages[0].content.clone()
        };
        assert_eq!(
            content, "system",
            "rejected override must not mutate the prompt"
        );
    }

    #[test]
    fn test_try_enqueue_rejects_empty_system_prompt() {
        let (state, _rx) = fresh_state(4);

        for prompt in ["", "   \n\t"] {
            let err = try_enqueue(&state, "hi".to_string(), None, Some(prompt))
                .expect_err("blank override should be rejected");
            assert!(
                matches!(err, EnqueueError::EmptySystemPrompt),
                "got {err:?}"
            );
        }

        assert!(
            state.conversations.is_empty(),
            "rejected request must not seed a conversation"
        );
        assert!(state.tasks.is_empty(), "rejected task must not be recorded");
    }

    #[test]
    fn test_try_enqueue_reuses_existing_conversation() {
        let (state, _rx) = fresh_state(4);
        let first =
            try_enqueue(&state, "hi".to_string(), None, None).expect("first should succeed");
        let second = try_enqueue(
            &state,
            "again".to_string(),
            Some(first.conversation_id),
            None,
        )
        .expect("continuation should succeed");

        assert_eq!(first.conversation_id, second.conversation_id);
        assert_eq!(state.conversations.len(), 1);
    }

    #[test]
    fn test_try_enqueue_unknown_conversation_returns_not_found() {
        let (state, _rx) = fresh_state(4);
        let bogus = Uuid::new_v4();
        let err = try_enqueue(&state, "hi".to_string(), Some(bogus), None)
            .expect_err("unknown conv id should be rejected");
        assert!(matches!(err, EnqueueError::ConversationNotFound(id) if id == bogus));
        assert!(state.tasks.is_empty(), "rejected task must not be recorded");
        assert!(
            state.conversations.is_empty(),
            "conv must not be auto-created"
        );
    }

    #[tokio::test]
    async fn process_job_continues_existing_conversation() {
        let (state, _rx) = fresh_state(4);
        let first = try_enqueue(&state, "hello".to_string(), None, None)
            .expect("first enqueue should succeed");
        let conv_id = first.conversation_id;

        let core = Core::new(EchoClient::new("hi-1"));
        let agent = Arc::new(Mutex::new(Agent::new("test".into(), core)));

        process_job(
            &state.tasks,
            &state.conversations,
            &agent,
            WorkerJob {
                task_id: first.id,
                conversation_id: conv_id,
                user: "hello".to_string(),
            },
            CancellationToken::new(),
        )
        .await;

        // The first turn must have appended user + assistant messages.
        let after_first = state
            .conversations
            .get(&conv_id)
            .expect("conv exists")
            .messages
            .clone();
        assert_eq!(after_first.len(), 3, "expected system + user + assistant");
        assert_eq!(after_first[0].role, MessageRole::System);
        assert_eq!(after_first[1].role, MessageRole::User);
        assert_eq!(after_first[1].content, "hello");
        assert_eq!(after_first[2].role, MessageRole::Assistant);
        assert_eq!(after_first[2].content, "hi-1");
        assert_eq!(
            state
                .conversations
                .get(&conv_id)
                .expect("conv exists")
                .turn_count,
            1
        );

        let second = try_enqueue(&state, "again".to_string(), Some(conv_id), None)
            .expect("second enqueue should succeed");
        let core2 = Core::new(EchoClient::new("hi-2"));
        let agent2 = Arc::new(Mutex::new(Agent::new("test".into(), core2)));
        process_job(
            &state.tasks,
            &state.conversations,
            &agent2,
            WorkerJob {
                task_id: second.id,
                conversation_id: conv_id,
                user: "again".to_string(),
            },
            CancellationToken::new(),
        )
        .await;

        let after_second = state
            .conversations
            .get(&conv_id)
            .expect("conv exists")
            .messages
            .clone();
        assert_eq!(
            after_second.len(),
            5,
            "expected system + (user, assistant) x 2"
        );
        assert_eq!(after_second[3].role, MessageRole::User);
        assert_eq!(after_second[3].content, "again");
        assert_eq!(after_second[4].role, MessageRole::Assistant);
        assert_eq!(after_second[4].content, "hi-2");
        assert_eq!(
            state
                .conversations
                .get(&conv_id)
                .expect("conv exists")
                .turn_count,
            2
        );
    }

    #[tokio::test]
    async fn process_job_fails_cleanly_when_conversation_deleted() {
        let (state, _rx) = fresh_state(4);
        let task =
            try_enqueue(&state, "hi".to_string(), None, None).expect("enqueue should succeed");
        let conv_id = task.conversation_id;

        // DELETE before the worker runs.
        state.conversations.remove(&conv_id);

        let core = Core::new(EchoClient::new("never-runs"));
        let agent = Arc::new(Mutex::new(Agent::new("test".into(), core)));
        process_job(
            &state.tasks,
            &state.conversations,
            &agent,
            WorkerJob {
                task_id: task.id,
                conversation_id: conv_id,
                user: "hi".to_string(),
            },
            CancellationToken::new(),
        )
        .await;

        let (status, error) = {
            let entry = state.tasks.get(&task.id).expect("task record");
            (entry.status, entry.error.clone())
        };
        assert_eq!(status, TaskStatus::Failed);
        assert_eq!(error.as_deref(), Some("conversation deleted"));
    }

    #[test]
    fn test_conversations_sorted_by_created_at() {
        let conversations: DashMap<Uuid, ConversationRecord> = DashMap::new();
        let base = Utc::now();
        for offset in [3, 1, 2, 0] {
            let id = Uuid::new_v4();
            let ts = base + chrono::Duration::seconds(offset);
            conversations.insert(
                id,
                ConversationRecord {
                    id,
                    created_at: ts,
                    updated_at: ts,
                    turn_count: 0,
                    messages: vec![Message::system(id, "system")],
                },
            );
        }
        let listed = conversations_sorted(&conversations);
        assert!(
            listed
                .windows(2)
                .all(|w| w[0].created_at <= w[1].created_at)
        );
        assert_eq!(listed.len(), 4);
        assert!(listed.iter().all(|s| s.message_count == 1));
    }

    #[test]
    fn test_list_tasks_returns_active_sorted_by_created_at() {
        let tasks: DashMap<Uuid, TaskRecord> = DashMap::new();
        let base = Utc::now();
        let inserts = [
            (TaskStatus::Succeeded, base),
            (TaskStatus::Running, base + chrono::Duration::seconds(1)),
            (TaskStatus::Pending, base + chrono::Duration::seconds(3)),
            (TaskStatus::Failed, base + chrono::Duration::seconds(2)),
            (TaskStatus::Pending, base + chrono::Duration::seconds(4)),
            (TaskStatus::Cancelled, base + chrono::Duration::seconds(5)),
        ];
        for (status, created_at) in inserts {
            let rec = record(status, created_at);
            tasks.insert(rec.id, rec);
        }

        let active = active_tasks_sorted(&tasks);
        let statuses: Vec<TaskStatus> = active.iter().map(|r| r.status).collect();
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

    #[test]
    fn test_list_tasks_empty_when_only_finished() {
        let tasks: DashMap<Uuid, TaskRecord> = DashMap::new();
        let now = Utc::now();
        for status in [
            TaskStatus::Succeeded,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ] {
            let rec = record(status, now);
            tasks.insert(rec.id, rec);
        }
        assert!(active_tasks_sorted(&tasks).is_empty());
    }
}
