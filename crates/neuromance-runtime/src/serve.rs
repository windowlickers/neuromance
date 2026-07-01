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
//! task's interpreter state never bleeds into the next. Without `[database]`,
//! in-memory state is authoritative for serving and restarts lose pending and
//! completed tasks.
//!
//! When `[database]` is configured, task status is written through to postgres
//! at every transition, and `GET /tasks` and `GET /tasks/{id}` read from
//! postgres so any replica behind a shared Service answers a poll for any
//! task — not just the one it accepted. The enqueue `pending` write is
//! synchronous and fail-closed (a returned `task_id` is always durably
//! pollable); mid-run transitions are best-effort. Conversation history is
//! likewise durable: Core persists messages incrementally during each run, and
//! continuations are store-authoritative — an existing `conversation_id` is
//! resolved against postgres when this replica's cache misses, and the worker
//! reads the turn's history from postgres, so a conversation continues
//! correctly on any replica regardless of which one accepted earlier turns.

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
use chrono::Utc;
use metrics::{counter, gauge, histogram};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::{Level, Span, error, field, info, info_span, warn};
use uuid::Uuid;

use neuromance::error::CoreError;
use neuromance_agent::{Agent, AgentResponse};
use neuromance_client::LLMClient;
use neuromance_common::chat::{Message, TaskStatus};
use neuromance_common::client::Config;
use neuromance_db::PgConversationStore;

use crate::AgentBuilder;
use crate::SessionReset;
use crate::config::RuntimeConfig;
use crate::sandbox::{EXECUTE_PYTHON, SandboxClient};
use crate::task_store::{
    ConversationRecord, InMemoryTaskStore, PostgresTaskStore, TaskRecord, TaskStore,
};

/// The agent type the worker drives. Serve always boots a boxed client, and a
/// per-task override produces the same type, so both run paths share it.
type ServeAgent = Agent<Box<dyn LLMClient>>;

enum JobOutcome {
    Succeeded,
    Failed,
    Cancelled,
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
    /// Select a configured `[[providers]]` entry (its credential and endpoint)
    /// for this task by name. The runtime builds a one-off agent bound to that
    /// provider, so distinct providers let a single workflow run tasks against
    /// different credentials. Omitted uses the runtime's configured
    /// `agent.provider`. An unknown name is rejected with 400.
    #[serde(default)]
    pub provider: Option<String>,
    /// Override the model for this task as a raw `provider:model` string (e.g.
    /// `anthropic:claude-opus-4-8`). The runtime builds a one-off agent for the
    /// task bound to this model, using the selected provider's credential and
    /// endpoint — so the override must name a model that credential covers (pair
    /// it with `provider` to point at one that does). Omitted runs on the
    /// selected provider's default model. A malformed string is rejected with 400.
    #[serde(default)]
    pub model: Option<String>,
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
    QueueFull {
        depth: usize,
        max: usize,
    },
    WorkerShutdown,
    ConversationNotFound(Uuid),
    SystemPromptOnExisting(Uuid),
    EmptySystemPrompt,
    /// The `provider` override names no configured `[[providers]]` entry.
    UnknownProvider(String),
    /// The `model` override could not be parsed as a `provider:model` string.
    InvalidModel(String),
    /// The durable `pending` row could not be written, so the task was rejected
    /// rather than handed back an id no sibling replica could resolve.
    Persistence,
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
            Self::UnknownProvider(name) => (
                StatusCode::BAD_REQUEST,
                serde_json::json!({
                    "error": "unknown provider override",
                    "detail": format!("provider '{name}' does not match any configured provider"),
                }),
            ),
            Self::InvalidModel(detail) => (
                StatusCode::BAD_REQUEST,
                serde_json::json!({
                    "error": "invalid model override",
                    "detail": detail,
                }),
            ),
            Self::Persistence => (
                StatusCode::SERVICE_UNAVAILABLE,
                serde_json::json!({"error": "failed to persist task"}),
            ),
        };
        (status, Json(body)).into_response()
    }
}

struct WorkerJob {
    task_id: Uuid,
    conversation_id: Uuid,
    user: String,
    /// Whether this request minted a fresh conversation. A seeded conversation's
    /// system message lives only in the per-pod cache until Core persists it
    /// mid-run; a continuation (`false`) reads its history from the store.
    seeded: bool,
    /// Per-task provider override naming a configured `[[providers]]` entry;
    /// `None` uses the configured `agent.provider`. Validated at enqueue time,
    /// so the worker can trust it resolves.
    provider: Option<String>,
    /// Per-task model override (raw `provider:model`); `None` uses the selected
    /// provider's default model. Validated at enqueue time, so the worker can
    /// trust it parses. When both this and `provider` are `None`, the shared
    /// agent runs the task.
    model: Option<String>,
}

#[derive(Clone)]
pub struct ServeState {
    /// Task and conversation storage. Backed by an in-memory working set alone,
    /// or by postgres write-through when `[database]` is configured. The worker
    /// shares this exact instance so a conversation seeded by a handler is
    /// visible to the run that continues it.
    task_store: Arc<dyn TaskStore>,
    work_tx: mpsc::Sender<WorkerJob>,
    system_prompt: Arc<str>,
    /// File-oriented skills menu folded into each new conversation's system
    /// prompt; `None` when no skills are configured.
    skills_menu: Option<Arc<str>>,
    /// Names of the configured `[[providers]]` entries, used to reject an unknown
    /// per-task `provider` override at enqueue time (400) rather than failing the
    /// task mid-run.
    provider_names: Arc<[String]>,
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
        .route("/conversations/{id}", get(get_conversation))
        .route(
            "/conversations/{id}/children",
            get(list_conversation_children),
        )
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
    // Fold the skills menu into the seed system message so the conversation
    // opens as a clean [System(prompt+menu), …] — the menu lists on-disk paths
    // the agent reads, and round-trips through persisted history thereafter.
    let seed = state.skills_menu.as_ref().map_or_else(
        || Message::system(id, prompt),
        |menu| Message::system(id, format!("{prompt}\n\n{menu}")),
    );
    // The durable conversation row is created fail-closed by `record_task_status`
    // in `try_enqueue` (it pre-inserts the row in the same transaction as the
    // pending task), so seeding only has to establish the in-memory record here.
    state.task_store.seed_conversation(ConversationRecord {
        id,
        created_at: now,
        updated_at: now,
        turn_count: 0,
        messages: vec![seed],
    });
    id
}

/// Resolve a request's `conversation_id`: reuse if it exists, mint and seed
/// a fresh record (honoring the `system_prompt` override) if omitted, reject
/// unknown ids. A `system_prompt` supplied for an existing conversation is
/// rejected, since that conversation already holds its system message.
///
/// An existing id missing from this replica's cache is resolved against the
/// durable store, so a continuation that load-balances to a replica which did
/// not accept earlier turns still succeeds instead of 404ing.
///
/// Returns the id and whether a fresh conversation was seeded in this call, so
/// the caller can roll the seed back if the task ultimately fails to enqueue.
async fn resolve_conversation(
    state: &ServeState,
    requested: Option<Uuid>,
    system_prompt: Option<&str>,
) -> Result<(Uuid, bool), EnqueueError> {
    if system_prompt.is_some_and(|p| p.trim().is_empty()) {
        return Err(EnqueueError::EmptySystemPrompt);
    }
    let Some(id) = requested else {
        return Ok((seed_new_conversation(state, system_prompt), true));
    };
    if system_prompt.is_some() {
        return Err(EnqueueError::SystemPromptOnExisting(id));
    }
    // A cache miss falls through to the durable store (when configured), so a
    // continuation that load-balances to a replica which did not accept earlier
    // turns still resolves instead of 404ing on a local-cache miss.
    match state.task_store.conversation_exists(id).await {
        Ok(true) => Ok((id, false)),
        Ok(false) => Err(EnqueueError::ConversationNotFound(id)),
        Err(e) => {
            warn!(conversation_id = %id, error = %e, "conversation existence check failed");
            Err(EnqueueError::Persistence)
        }
    }
}

async fn try_enqueue(
    state: &ServeState,
    user: String,
    conversation_id: Option<Uuid>,
    system_prompt: Option<&str>,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<TaskRecord, EnqueueError> {
    // Reject an unknown provider override before minting a task, so the caller
    // gets a 400 at submit instead of a mid-run failure.
    if let Some(provider) = provider
        && !state.provider_names.iter().any(|p| p == provider)
    {
        return Err(EnqueueError::UnknownProvider(provider.to_owned()));
    }
    // Reject a malformed model override before minting a task, so the caller
    // gets a 400 at submit instead of a mid-run failure. Parsing also fixes the
    // client family the worker will build against.
    if let Some(model) = model
        && let Err(e) = Config::from_model(model)
    {
        return Err(EnqueueError::InvalidModel(format!("{model}: {e}")));
    }
    let (conversation_id, seeded) =
        resolve_conversation(state, conversation_id, system_prompt).await?;
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
    // Record the pending task BEFORE handing the job to the worker. Intake and
    // the worker are separate tasks, so persisting after `try_send` would let the
    // worker's `Running` write race ahead of (and be clobbered by) this one.
    // Fail-closed: a returned task_id must be durably pollable from any replica,
    // so a durable write failure rejects the enqueue. (Mid-run status writes stay
    // best-effort — see the `mark_*` methods.)
    if let Err(e) = state.task_store.insert_pending(&record).await {
        error!(%task_id, error = %e, "persist pending task failed; rejecting enqueue");
        if seeded {
            state.task_store.remove_conversation(conversation_id).await;
        }
        counter!("neuromance_enqueue_rejections_total", "reason" => "persistence").increment(1);
        return Err(EnqueueError::Persistence);
    }

    match state.work_tx.try_send(WorkerJob {
        task_id,
        conversation_id,
        user,
        seeded,
        provider: provider.map(str::to_owned),
        model: model.map(str::to_owned),
    }) {
        Ok(()) => {
            #[allow(clippy::cast_precision_loss)]
            gauge!("neuromance_queue_depth").set(queue_depth(&state.work_tx) as f64);
            Ok(record)
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            state.task_store.remove_task(task_id).await;
            if seeded {
                state.task_store.remove_conversation(conversation_id).await;
            }
            counter!("neuromance_enqueue_rejections_total", "reason" => "queue_full").increment(1);
            Err(EnqueueError::QueueFull {
                depth,
                max: state.work_tx.max_capacity(),
            })
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            state
                .task_store
                .mark_failed(task_id, "worker shutting down")
                .await;
            if seeded {
                state.task_store.remove_conversation(conversation_id).await;
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
        req.provider.as_deref(),
        req.model.as_deref(),
    )
    .await
    {
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

async fn list_tasks(State(state): State<ServeState>) -> impl IntoResponse {
    match state.task_store.list_active_tasks().await {
        Ok(tasks) => (StatusCode::OK, Json(tasks)).into_response(),
        Err(e) => {
            warn!(error = %e, "failed to list active tasks");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "failed to list tasks"})),
            )
                .into_response()
        }
    }
}

async fn get_task(State(state): State<ServeState>, Path(id): Path<Uuid>) -> impl IntoResponse {
    match state.task_store.get_task(id).await {
        Ok(Some(task)) => (StatusCode::OK, Json(task)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "task not found"})),
        )
            .into_response(),
        Err(e) => {
            warn!(task_id = %id, error = %e, "failed to load task status");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "failed to load task"})),
            )
                .into_response()
        }
    }
}

async fn list_conversations(State(state): State<ServeState>) -> impl IntoResponse {
    match state.task_store.list_conversations().await {
        Ok(summaries) => (StatusCode::OK, Json(summaries)).into_response(),
        Err(e) => {
            warn!(error = %e, "failed to list conversations");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "failed to list conversations"})),
            )
                .into_response()
        }
    }
}

async fn get_conversation(
    State(state): State<ServeState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.task_store.get_conversation_view(id).await {
        Ok(Some(record)) => (StatusCode::OK, Json(record)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "conversation not found"})),
        )
            .into_response(),
        Err(e) => {
            warn!(conversation_id = %id, error = %e, "failed to load conversation");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "failed to load conversation"})),
            )
                .into_response()
        }
    }
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
    match state
        .task_store
        .list_conversation_children(id, CHILDREN_PAGE_LIMIT, 0)
        .await
    {
        Ok(Some(children)) => (StatusCode::OK, Json(children)).into_response(),
        Ok(None) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "conversation lineage requires a configured database",
            })),
        )
            .into_response(),
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

/// Shared handles the worker needs to run a job and record its provenance.
/// Bundled so [`worker_loop`] and [`run`] stay within the positional-argument
/// budget as fields accrue.
struct WorkerCtx {
    /// Task and conversation storage — the same instance held by [`ServeState`].
    task_store: Arc<dyn TaskStore>,
    /// Shared agent reused by every task that does not override the model.
    agent: Arc<Mutex<ServeAgent>>,
    /// Builds a throwaway agent for a task that overrides the model.
    builder: Arc<dyn AgentBuilder>,
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

async fn worker_loop(mut rx: mpsc::Receiver<WorkerJob>, ctx: WorkerCtx, cancel: CancellationToken) {
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
                let start_seq = ctx.task_store.begin_provenance(conversation_id).await;
                let _ = process_job(&ctx, job, cancel.clone()).await;
                release_sandbox_session(ctx.sandbox.as_ref(), task_id).await;
                reset_local_python(ctx.local_python.as_ref()).await;
                ctx.task_store
                    .record_provenance(task_id, conversation_id, start_seq)
                    .await;
            }
        }
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
async fn fail_task(ctx: &WorkerCtx, task_id: Uuid, reason: &str) {
    ctx.task_store.mark_failed(task_id, reason).await;
    counter!("neuromance_tasks_total", "outcome" => "failed").increment(1);
}

/// Drives one agent turn over `input_messages`, racing it against `cancel` so a
/// shutdown abandons the run rather than stalling the drain. Records the running
/// agent's id on the current task span.
async fn run_turn(
    agent: &mut ServeAgent,
    task_id: Uuid,
    input_messages: Vec<Message>,
    cancel: &CancellationToken,
) -> Result<(AgentResponse, Vec<Message>), CoreError> {
    Span::current().record("agent_id", field::display(agent.id()));
    // Seed the runtime task id so subagent conversations spawned during this
    // run inherit it as their `parent_task_id`. The root conversation itself
    // has no parent, so no conversation id is seeded here.
    tokio::select! {
        biased;
        () = cancel.cancelled() => Err(CoreError::Cancelled("worker shutdown".to_string())),
        res = neuromance_agent::scope_task(
            Some(task_id),
            agent.execute_with_history(Some(input_messages), cancel.child_token()),
        ) => res,
    }
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
async fn process_job(ctx: &WorkerCtx, job: WorkerJob, cancel: CancellationToken) -> JobOutcome {
    let dequeued_at = Utc::now();
    let run_start = Instant::now();
    let (created_at, depth_at_enqueue) = ctx
        .task_store
        .mark_running(job.task_id)
        .await
        .map_or((dequeued_at, 0), |timing| {
            (timing.created_at, timing.queue_depth_at_enqueue)
        });

    let queue_wait_ms = (dequeued_at - created_at).num_milliseconds().max(0);
    #[allow(clippy::cast_precision_loss)]
    histogram!("neuromance_queue_wait_seconds").record(queue_wait_ms as f64 / 1000.0);
    info!(
        queue_wait_ms,
        depth_at_enqueue,
        user_bytes = job.user.len(),
        "task starting",
    );

    // Skill menu and `$mention` bodies are injected by the SkillsHook inside the
    // conversation loop, not assembled here.
    let user_msg = Message::user(job.conversation_id, &job.user);
    let input_messages = match ctx
        .task_store
        .build_turn_input(job.conversation_id, job.seeded, user_msg)
        .await
    {
        Ok(messages) => messages,
        Err(err) => {
            let reason = err.reason();
            warn!(conversation_id = %job.conversation_id, reason, "cannot build turn input");
            fail_task(ctx, job.task_id, reason).await;
            return JobOutcome::Failed;
        }
    };

    // A task with a provider and/or model override runs on a throwaway agent
    // built for it and dropped after the turn; everything else reuses the shared
    // agent. The override values were validated at enqueue, but the build can
    // still fail (e.g. a missing provider credential), which fails just this task.
    let exec_result = if job.provider.is_some() || job.model.is_some() {
        let provider = job.provider.as_deref();
        let model = job.model.as_deref();
        match ctx.builder.build(provider, model).await {
            Ok((mut agent, _local_python)) => {
                info!(
                    provider,
                    model, "task running on per-task provider/model override"
                );
                run_turn(&mut agent, job.task_id, input_messages, &cancel).await
            }
            Err(e) => {
                error!(provider, model, error = %e, "failed to build agent for override");
                fail_task(
                    ctx,
                    job.task_id,
                    &format!("task override (provider={provider:?}, model={model:?}): {e}"),
                )
                .await;
                return JobOutcome::Failed;
            }
        }
    } else {
        let mut agent = ctx.agent.lock().await;
        run_turn(&mut agent, job.task_id, input_messages, &cancel).await
    };

    let run_ms = u64::try_from(run_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    #[allow(clippy::cast_precision_loss)]
    histogram!("neuromance_task_duration_seconds").record(run_ms as f64 / 1000.0);
    match exec_result {
        Ok((response, full_history)) => {
            let output_bytes = response.content.content.len();
            info!(run_ms, queue_wait_ms, output_bytes, "task succeeded");
            // Refresh the per-pod cache with the full history (system + every
            // user/assistant/tool turn) when this replica holds it. A replica
            // that only continued from the store has no local entry; the durable
            // history written by Core stands on its own, so the refresh is a no-op.
            ctx.task_store
                .refresh_conversation(job.conversation_id, full_history)
                .await;
            ctx.task_store
                .mark_succeeded(job.task_id, response.content.content)
                .await;
            counter!("neuromance_tasks_total", "outcome" => "succeeded").increment(1);
            JobOutcome::Succeeded
        }
        Err(CoreError::Cancelled(_)) => {
            warn!(run_ms, "task cancelled");
            ctx.task_store
                .mark_cancelled(job.task_id, "cancelled")
                .await;
            counter!("neuromance_tasks_total", "outcome" => "cancelled").increment(1);
            JobOutcome::Cancelled
        }
        Err(e) => {
            error!(run_ms, error = %e, "task failed");
            fail_task(ctx, job.task_id, &e.to_string()).await;
            JobOutcome::Failed
        }
    }
}

/// Bind the task server, spawn the worker, and run until `cancel` fires.
///
/// # Errors
/// Returns an error if `runtime.listen_addr` is invalid, the bind fails,
/// or the HTTP server returns an error during operation.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    config: &RuntimeConfig,
    agent: ServeAgent,
    builder: Arc<dyn AgentBuilder>,
    store: Option<Arc<PgConversationStore>>,
    sandbox_client: Option<SandboxClient>,
    local_python: Option<SessionReset>,
    skills_menu: Option<Arc<str>>,
    cancel: CancellationToken,
) -> Result<()> {
    // One storage instance, shared by the worker and the handlers: a conversation
    // a handler seeds must be visible to the run that continues it. Postgres
    // deployment writes through and reads authoritatively; otherwise the working
    // set alone is authoritative.
    let task_store: Arc<dyn TaskStore> = match store {
        Some(store) => Arc::new(PostgresTaskStore::new(store)),
        None => Arc::new(InMemoryTaskStore::new()),
    };
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
            task_store: Arc::clone(&task_store),
            agent: Arc::clone(&agent),
            builder,
            sandbox: session_closer,
            local_python,
        },
        cancel.clone(),
    ));

    let provider_names: Arc<[String]> = config.providers.iter().map(|p| p.name.clone()).collect();
    let state = ServeState {
        task_store: Arc::clone(&task_store),
        work_tx,
        system_prompt,
        skills_menu,
        provider_names,
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

    let summary = task_store.drain_pending().await;
    info!(
        pending_dropped = summary.pending_dropped,
        in_flight_cancelled = summary.in_flight_cancelled,
        succeeded = summary.succeeded,
        failed = summary.failed,
        "serve mode shutdown",
    );
    Ok(())
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
    use neuromance_db::ConversationSink;
    use sqlx::PgPool;

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
        let store = Arc::new(InMemoryTaskStore::new());
        store
            .insert_pending(&TaskRecord {
                id: task_id,
                status: TaskStatus::Pending,
                conversation_id: conv_id,
                created_at: now,
                updated_at: now,
                output: None,
                error: None,
                queue_depth_at_enqueue: 0,
            })
            .await
            .unwrap();
        store.seed_conversation(ConversationRecord {
            id: conv_id,
            created_at: now,
            updated_at: now,
            turn_count: 0,
            messages: vec![Message::system(conv_id, "system")],
        });

        let core =
            Core::new(Box::new(SleepingClient::new()) as Box<dyn LLMClient>).with_streaming();
        let agent = Arc::new(Mutex::new(Agent::new("test".into(), core)));
        let (work_tx, work_rx) = mpsc::channel::<WorkerJob>(1);
        let cancel = CancellationToken::new();

        let worker = tokio::spawn(worker_loop(
            work_rx,
            WorkerCtx {
                task_store: Arc::clone(&store) as Arc<dyn TaskStore>,
                agent: Arc::clone(&agent),
                builder: stub_builder(),
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
                seeded: true,
                provider: None,
                model: None,
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

        let task = store.get_task(task_id).await.unwrap().expect("task record");
        assert_eq!(task.status, TaskStatus::Cancelled);
        assert_eq!(task.error.as_deref(), Some("cancelled"));
    }

    fn fresh_state(
        capacity: usize,
    ) -> (
        ServeState,
        Arc<InMemoryTaskStore>,
        mpsc::Receiver<WorkerJob>,
    ) {
        let store = Arc::new(InMemoryTaskStore::new());
        let (work_tx, work_rx) = mpsc::channel::<WorkerJob>(capacity);
        (
            ServeState {
                task_store: Arc::clone(&store) as Arc<dyn TaskStore>,
                work_tx,
                system_prompt: Arc::from("system"),
                skills_menu: None,
                provider_names: ["primary".to_owned(), "secondary".to_owned()].into(),
            },
            store,
            work_rx,
        )
    }

    #[test]
    fn test_seed_folds_skills_menu_into_single_system_message() {
        let (mut state, store, _rx) = fresh_state(4);
        state.skills_menu = Some(Arc::from(
            "<skills_instructions>\n- alpha: a (file: /tmp/x/SKILL.md)\n</skills_instructions>",
        ));

        let id = seed_new_conversation(&state, None);
        let messages = store
            .conversation_messages(id)
            .expect("conversation should be seeded");

        // Exactly one system message — the menu is folded in, not a second
        // System message after the user turn.
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::System);
        assert!(messages[0].content.starts_with("system"));
        assert!(messages[0].content.contains("<skills_instructions>"));
        assert!(messages[0].content.contains("file: /tmp/x/SKILL.md"));
    }

    #[test]
    fn test_seed_without_skills_menu_is_plain_prompt() {
        let (state, store, _rx) = fresh_state(4);
        let id = seed_new_conversation(&state, None);
        let messages = store
            .conversation_messages(id)
            .expect("conversation should be seeded");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "system");
    }

    /// Boxes a mock client into the agent type the worker drives.
    fn boxed_agent(client: impl LLMClient + 'static) -> Arc<Mutex<ServeAgent>> {
        let core = Core::new(Box::new(client) as Box<dyn LLMClient>);
        Arc::new(Mutex::new(Agent::new("test".into(), core)))
    }

    /// An [`AgentBuilder`] that yields an echo agent. The non-override tests
    /// never call it — the shared agent handles those — so a stub reply is fine;
    /// the override test points it at a distinct reply and asserts that reply
    /// lands, proving the per-task build path ran instead of the shared agent.
    struct TestBuilder {
        reply: String,
    }

    #[async_trait]
    impl AgentBuilder for TestBuilder {
        async fn build(
            &self,
            _provider_override: Option<&str>,
            _model_override: Option<&str>,
        ) -> Result<(ServeAgent, Option<SessionReset>), crate::RuntimeError> {
            let core = Core::new(Box::new(EchoClient::new(&self.reply)) as Box<dyn LLMClient>);
            Ok((Agent::new("override-agent".into(), core), None))
        }
    }

    fn stub_builder() -> Arc<dyn AgentBuilder> {
        Arc::new(TestBuilder {
            reply: "unused".to_string(),
        })
    }

    /// Builds a worker context sharing a state's task store, so `process_job`
    /// can be driven directly in tests.
    fn worker_ctx(state: &ServeState, agent: Arc<Mutex<ServeAgent>>) -> WorkerCtx {
        WorkerCtx {
            task_store: Arc::clone(&state.task_store),
            agent,
            builder: stub_builder(),
            sandbox: None,
            local_python: None,
        }
    }

    #[tokio::test]
    async fn test_try_enqueue_records_queue_depth_zero_when_empty() {
        let (state, store, _rx) = fresh_state(4);
        let record = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect("enqueue should succeed");
        assert_eq!(record.queue_depth_at_enqueue, 0);
        assert_eq!(record.status, TaskStatus::Pending);
        assert!(store.get_task(record.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_try_enqueue_queue_depth_grows_as_channel_fills() {
        let (state, _store, _rx) = fresh_state(4);
        let mut depths = Vec::new();
        for _ in 0..3 {
            depths.push(
                try_enqueue(&state, "hi".to_string(), None, None, None, None)
                    .await
                    .expect("enqueue should succeed")
                    .queue_depth_at_enqueue,
            );
        }
        assert_eq!(depths, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn test_try_enqueue_returns_queue_full_at_capacity() {
        let (state, store, _rx) = fresh_state(2);
        let first = try_enqueue(&state, "a".to_string(), None, None, None, None)
            .await
            .expect("first should fit");
        let second = try_enqueue(&state, "b".to_string(), None, None, None, None)
            .await
            .expect("second should fit");

        let err = try_enqueue(&state, "c".to_string(), None, None, None, None)
            .await
            .expect_err("third should reject");
        assert!(
            matches!(err, EnqueueError::QueueFull { depth: 2, max: 2 }),
            "got {err:?}"
        );

        assert!(store.get_task(first.id).await.unwrap().is_some());
        assert!(store.get_task(second.id).await.unwrap().is_some());
        assert_eq!(store.task_count(), 2, "rejected task must not linger");
        assert_eq!(
            store.conversation_count(),
            2,
            "rejected enqueue must not leak a seeded conversation"
        );
    }

    #[tokio::test]
    async fn test_try_enqueue_returns_worker_shutdown_when_rx_dropped() {
        let (state, store, rx) = fresh_state(4);
        drop(rx);

        let err = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect_err("send should fail");
        assert!(matches!(err, EnqueueError::WorkerShutdown), "got {err:?}");

        let tasks = store.all_tasks();
        assert_eq!(tasks.len(), 1, "record should exist");
        assert_eq!(tasks[0].status, TaskStatus::Failed);
        assert_eq!(tasks[0].error.as_deref(), Some("worker shutting down"));
        assert_eq!(
            store.conversation_count(),
            0,
            "worker-shutdown rejection must not leak a seeded conversation"
        );
    }

    #[tokio::test]
    async fn test_try_enqueue_seeds_fresh_conversation_with_system_prompt() {
        let (state, store, _rx) = fresh_state(4);
        let record = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect("enqueue should succeed");

        let turn_count = store
            .conversation_turn_count(record.conversation_id)
            .expect("conversation should exist");
        let messages = store
            .conversation_messages(record.conversation_id)
            .expect("conversation should exist");
        assert_eq!(turn_count, 0);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[0].content, "system");
        assert_eq!(messages[0].conversation_id, record.conversation_id);
    }

    #[tokio::test]
    async fn test_try_enqueue_override_seeds_custom_system_prompt() {
        let (state, store, _rx) = fresh_state(4);
        let record = try_enqueue(&state, "hi".to_string(), None, Some("be terse"), None, None)
            .await
            .expect("enqueue should succeed");

        let messages = store
            .conversation_messages(record.conversation_id)
            .expect("conversation should exist");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[0].content, "be terse");
    }

    #[tokio::test]
    async fn test_try_enqueue_falls_back_to_configured_prompt_when_override_omitted() {
        let (state, store, _rx) = fresh_state(4);
        let record = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect("enqueue should succeed");

        let messages = store
            .conversation_messages(record.conversation_id)
            .expect("conversation should exist");
        assert_eq!(messages[0].content, "system");
    }

    #[tokio::test]
    async fn test_try_enqueue_override_on_existing_conversation_is_rejected() {
        let (state, store, _rx) = fresh_state(4);
        let first = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect("first should succeed");

        let err = try_enqueue(
            &state,
            "again".to_string(),
            Some(first.conversation_id),
            Some("be terse"),
            None,
            None,
        )
        .await
        .expect_err("override on existing conversation should be rejected");
        assert!(
            matches!(err, EnqueueError::SystemPromptOnExisting(id) if id == first.conversation_id)
        );

        let messages = store
            .conversation_messages(first.conversation_id)
            .expect("conversation should exist");
        assert_eq!(
            messages[0].content, "system",
            "rejected override must not mutate the prompt"
        );
    }

    #[tokio::test]
    async fn test_try_enqueue_rejects_empty_system_prompt() {
        let (state, store, _rx) = fresh_state(4);

        for prompt in ["", "   \n\t"] {
            let err = try_enqueue(&state, "hi".to_string(), None, Some(prompt), None, None)
                .await
                .expect_err("blank override should be rejected");
            assert!(
                matches!(err, EnqueueError::EmptySystemPrompt),
                "got {err:?}"
            );
        }

        assert_eq!(
            store.conversation_count(),
            0,
            "rejected request must not seed a conversation"
        );
        assert_eq!(store.task_count(), 0, "rejected task must not be recorded");
    }

    #[tokio::test]
    async fn test_try_enqueue_reuses_existing_conversation() {
        let (state, store, _rx) = fresh_state(4);
        let first = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect("first should succeed");
        let second = try_enqueue(
            &state,
            "again".to_string(),
            Some(first.conversation_id),
            None,
            None,
            None,
        )
        .await
        .expect("continuation should succeed");

        assert_eq!(first.conversation_id, second.conversation_id);
        assert_eq!(store.conversation_count(), 1);
    }

    #[tokio::test]
    async fn test_try_enqueue_unknown_conversation_returns_not_found() {
        let (state, store, _rx) = fresh_state(4);
        let bogus = Uuid::new_v4();
        let err = try_enqueue(&state, "hi".to_string(), Some(bogus), None, None, None)
            .await
            .expect_err("unknown conv id should be rejected");
        assert!(matches!(err, EnqueueError::ConversationNotFound(id) if id == bogus));
        assert_eq!(store.task_count(), 0, "rejected task must not be recorded");
        assert_eq!(
            store.conversation_count(),
            0,
            "conv must not be auto-created"
        );
    }

    #[tokio::test]
    async fn process_job_continues_existing_conversation() {
        let (state, store, _rx) = fresh_state(4);
        let first = try_enqueue(&state, "hello".to_string(), None, None, None, None)
            .await
            .expect("first enqueue should succeed");
        let conv_id = first.conversation_id;

        let agent = boxed_agent(EchoClient::new("hi-1"));

        process_job(
            &worker_ctx(&state, agent),
            WorkerJob {
                task_id: first.id,
                conversation_id: conv_id,
                user: "hello".to_string(),
                seeded: true,
                provider: None,
                model: None,
            },
            CancellationToken::new(),
        )
        .await;

        // The first turn must have appended user + assistant messages.
        let after_first = store.conversation_messages(conv_id).expect("conv exists");
        assert_eq!(after_first.len(), 3, "expected system + user + assistant");
        assert_eq!(after_first[0].role, MessageRole::System);
        assert_eq!(after_first[1].role, MessageRole::User);
        assert_eq!(after_first[1].content, "hello");
        assert_eq!(after_first[2].role, MessageRole::Assistant);
        assert_eq!(after_first[2].content, "hi-1");
        assert_eq!(store.conversation_turn_count(conv_id), Some(1));

        let second = try_enqueue(&state, "again".to_string(), Some(conv_id), None, None, None)
            .await
            .expect("second enqueue should succeed");
        let agent2 = boxed_agent(EchoClient::new("hi-2"));
        process_job(
            &worker_ctx(&state, agent2),
            WorkerJob {
                task_id: second.id,
                conversation_id: conv_id,
                user: "again".to_string(),
                seeded: false,
                provider: None,
                model: None,
            },
            CancellationToken::new(),
        )
        .await;

        let after_second = store.conversation_messages(conv_id).expect("conv exists");
        assert_eq!(
            after_second.len(),
            5,
            "expected system + (user, assistant) x 2"
        );
        assert_eq!(after_second[3].role, MessageRole::User);
        assert_eq!(after_second[3].content, "again");
        assert_eq!(after_second[4].role, MessageRole::Assistant);
        assert_eq!(after_second[4].content, "hi-2");
        assert_eq!(store.conversation_turn_count(conv_id), Some(2));
    }

    #[tokio::test]
    async fn process_job_fails_cleanly_when_conversation_record_missing() {
        let (state, store, _rx) = fresh_state(4);
        let task = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect("enqueue should succeed");
        let conv_id = task.conversation_id;

        // Drop the per-pod record before the worker runs.
        store.remove_conversation(conv_id).await;

        let agent = boxed_agent(EchoClient::new("never-runs"));
        process_job(
            &worker_ctx(&state, agent),
            WorkerJob {
                task_id: task.id,
                conversation_id: conv_id,
                user: "hi".to_string(),
                seeded: true,
                provider: None,
                model: None,
            },
            CancellationToken::new(),
        )
        .await;

        let record = store.get_task(task.id).await.unwrap().expect("task record");
        assert_eq!(record.status, TaskStatus::Failed);
        assert_eq!(record.error.as_deref(), Some("conversation record missing"));
    }

    #[tokio::test]
    async fn test_try_enqueue_rejects_invalid_model() {
        let (state, store, _rx) = fresh_state(4);
        let err = try_enqueue(
            &state,
            "hi".to_string(),
            None,
            None,
            None,
            Some("not-a-valid-model"),
        )
        .await
        .expect_err("malformed model override should be rejected");
        assert!(matches!(err, EnqueueError::InvalidModel(_)), "got {err:?}");
        assert_eq!(store.task_count(), 0, "rejected task must not be recorded");
        assert_eq!(
            store.conversation_count(),
            0,
            "rejected request must not seed a conversation"
        );
    }

    #[tokio::test]
    async fn test_try_enqueue_rejects_unknown_provider() {
        // `fresh_state` configures providers "primary" and "secondary"; anything
        // else is rejected at enqueue with a 400-mapped error, before a task or
        // conversation is minted.
        let (state, store, _rx) = fresh_state(4);
        let err = try_enqueue(&state, "hi".to_string(), None, None, Some("ghost"), None)
            .await
            .expect_err("unknown provider override should be rejected");
        assert!(
            matches!(&err, EnqueueError::UnknownProvider(name) if name == "ghost"),
            "got {err:?}",
        );
        assert_eq!(store.task_count(), 0, "rejected task must not be recorded");
        assert_eq!(
            store.conversation_count(),
            0,
            "rejected request must not seed a conversation"
        );
    }

    #[tokio::test]
    async fn test_try_enqueue_accepts_known_provider() {
        let (state, store, _rx) = fresh_state(4);
        let record = try_enqueue(
            &state,
            "hi".to_string(),
            None,
            None,
            Some("secondary"),
            None,
        )
        .await
        .expect("a configured provider override should be accepted");
        assert_eq!(
            store
                .get_task(record.id)
                .await
                .unwrap()
                .expect("task record")
                .status,
            TaskStatus::Pending,
        );
    }

    #[tokio::test]
    async fn process_job_with_provider_override_runs_built_agent() {
        // A provider override with no model still routes through the per-task
        // builder rather than the shared agent.
        let (state, store, _rx) = fresh_state(4);
        let task = try_enqueue(
            &state,
            "hi".to_string(),
            None,
            None,
            Some("secondary"),
            None,
        )
        .await
        .expect("enqueue should succeed");
        let conv_id = task.conversation_id;

        let ctx = WorkerCtx {
            task_store: Arc::clone(&state.task_store),
            agent: boxed_agent(EchoClient::new("from-shared")),
            builder: Arc::new(TestBuilder {
                reply: "from-override".to_string(),
            }),
            sandbox: None,
            local_python: None,
        };
        process_job(
            &ctx,
            WorkerJob {
                task_id: task.id,
                conversation_id: conv_id,
                user: "hi".to_string(),
                seeded: true,
                provider: Some("secondary".to_string()),
                model: None,
            },
            CancellationToken::new(),
        )
        .await;

        let last = store
            .conversation_messages(conv_id)
            .expect("conv exists")
            .last()
            .expect("assistant reply")
            .content
            .clone();
        assert_eq!(last, "from-override", "the override agent should have run");
        assert_eq!(
            store
                .get_task(task.id)
                .await
                .unwrap()
                .expect("task record")
                .status,
            TaskStatus::Succeeded
        );
    }

    #[tokio::test]
    async fn process_job_with_model_override_runs_built_agent() {
        let (state, store, _rx) = fresh_state(4);
        let task = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect("enqueue should succeed");
        let conv_id = task.conversation_id;

        // The shared agent and the per-task builder return different replies, so
        // the assistant message that lands tells us which agent actually ran.
        let ctx = WorkerCtx {
            task_store: Arc::clone(&state.task_store),
            agent: boxed_agent(EchoClient::new("from-shared")),
            builder: Arc::new(TestBuilder {
                reply: "from-override".to_string(),
            }),
            sandbox: None,
            local_python: None,
        };
        process_job(
            &ctx,
            WorkerJob {
                task_id: task.id,
                conversation_id: conv_id,
                user: "hi".to_string(),
                seeded: true,
                provider: None,
                model: Some("anthropic:claude-haiku-4-5".to_string()),
            },
            CancellationToken::new(),
        )
        .await;

        let last = store
            .conversation_messages(conv_id)
            .expect("conv exists")
            .last()
            .expect("assistant reply")
            .content
            .clone();
        assert_eq!(last, "from-override", "the override agent should have run");
        assert_eq!(
            store
                .get_task(task.id)
                .await
                .unwrap()
                .expect("task record")
                .status,
            TaskStatus::Succeeded
        );
    }

    // The active-task filtering/sorting and conversation-summary ordering that
    // used to be tested through serve's free functions now live behind the
    // `TaskStore` seam — see `task_store::tests`.

    // --- Postgres-backed continuation tests --------------------------------
    //
    // These exercise the store-authoritative continuation path. They are
    // `#[ignore]`d because CI has no postgres; run locally with `DATABASE_URL`
    // set, matching `crates/neuromance-db/tests/store.rs`:
    //
    //   DATABASE_URL=postgres://postgres:pg@localhost:5432/neuromance \
    //       cargo test -p neuromance-runtime -- --ignored

    /// `fresh_state` backed by a postgres write-through store wrapping `pool`.
    fn state_with_store(
        capacity: usize,
        pool: PgPool,
    ) -> (
        ServeState,
        Arc<PostgresTaskStore>,
        mpsc::Receiver<WorkerJob>,
    ) {
        let store = Arc::new(PostgresTaskStore::new(Arc::new(PgConversationStore::new(
            pool,
        ))));
        let (work_tx, work_rx) = mpsc::channel::<WorkerJob>(capacity);
        (
            ServeState {
                task_store: Arc::clone(&store) as Arc<dyn TaskStore>,
                work_tx,
                system_prompt: Arc::from("system"),
                skills_menu: None,
                provider_names: ["primary".to_owned(), "secondary".to_owned()].into(),
            },
            store,
            work_rx,
        )
    }

    /// `worker_ctx` sharing the state's postgres-backed task store.
    fn worker_ctx_with_store(state: &ServeState, agent: Arc<Mutex<ServeAgent>>) -> WorkerCtx {
        WorkerCtx {
            task_store: Arc::clone(&state.task_store),
            agent,
            builder: stub_builder(),
            sandbox: None,
            local_python: None,
        }
    }

    #[sqlx::test(migrations = "../neuromance-db/migrations")]
    #[ignore = "requires postgres via DATABASE_URL"]
    async fn resolve_continuation_rehydrates_from_store_on_cold_cache(pool: PgPool) {
        let (state, task_store, _rx) = state_with_store(4, pool);
        let conv_id = Uuid::new_v4();
        // Seed durable history but leave the in-memory cache empty — the
        // cross-pod scenario where the request lands on a replica that never
        // accepted earlier turns.
        let store = task_store.durable_store();
        store
            .append_messages(
                conv_id,
                &[
                    Message::system(conv_id, "system"),
                    Message::user(conv_id, "first"),
                    Message::assistant(conv_id, "reply"),
                ],
            )
            .await
            .expect("seed durable history");

        let record = try_enqueue(&state, "again".to_string(), Some(conv_id), None, None, None)
            .await
            .expect("continuation against a store-only conversation should enqueue");

        assert_eq!(record.conversation_id, conv_id);
        assert!(
            !task_store.has_cached_conversation(conv_id),
            "resolution must not be gated on the local cache"
        );
    }

    #[sqlx::test(migrations = "../neuromance-db/migrations")]
    #[ignore = "requires postgres via DATABASE_URL"]
    async fn resolve_unknown_id_with_store_returns_not_found(pool: PgPool) {
        let (state, _task_store, _rx) = state_with_store(4, pool);
        let bogus = Uuid::new_v4();
        let err = try_enqueue(&state, "hi".to_string(), Some(bogus), None, None, None)
            .await
            .expect_err("id absent from both cache and store must 404");
        assert!(matches!(err, EnqueueError::ConversationNotFound(id) if id == bogus));
    }

    #[sqlx::test(migrations = "../neuromance-db/migrations")]
    #[ignore = "requires postgres via DATABASE_URL"]
    async fn process_job_reads_history_from_store_not_stale_cache(pool: PgPool) {
        let (state, task_store, _rx) = state_with_store(4, pool);
        let conv_id = Uuid::new_v4();
        let store = task_store.durable_store();

        // The store holds two completed turns...
        store
            .append_messages(
                conv_id,
                &[
                    Message::system(conv_id, "system"),
                    Message::user(conv_id, "turn-1"),
                    Message::assistant(conv_id, "reply-1"),
                    Message::user(conv_id, "turn-2"),
                    Message::assistant(conv_id, "reply-2"),
                ],
            )
            .await
            .expect("seed durable history");

        // ...but this replica's cache is stale at turn 1 only. If the worker
        // trusted the cache it would silently drop turn 2.
        task_store.seed_cache(ConversationRecord {
            id: conv_id,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            turn_count: 1,
            messages: vec![
                Message::system(conv_id, "system"),
                Message::user(conv_id, "turn-1"),
                Message::assistant(conv_id, "reply-1"),
            ],
        });

        let task_id = Uuid::new_v4();
        let now = Utc::now();
        task_store.seed_task(TaskRecord {
            id: task_id,
            status: TaskStatus::Pending,
            conversation_id: conv_id,
            created_at: now,
            updated_at: now,
            output: None,
            error: None,
            queue_depth_at_enqueue: 0,
        });

        let agent = boxed_agent(EchoClient::new("reply-3"));
        process_job(
            &worker_ctx_with_store(&state, agent),
            WorkerJob {
                task_id,
                conversation_id: conv_id,
                user: "turn-3".to_string(),
                seeded: false,
                provider: None,
                model: None,
            },
            CancellationToken::new(),
        )
        .await;

        // Run input came from the store (5 prior + user-3 + assistant-3 = 7),
        // not the 3-message stale cache (which would yield 5).
        let history = task_store
            .conversation_messages(conv_id)
            .expect("conv exists");
        assert_eq!(history.len(), 7, "history must include the store's turn 2");
        assert_eq!(history[3].content, "turn-2");
        assert_eq!(history[5].content, "turn-3");
        assert_eq!(history[6].content, "reply-3");
        assert_eq!(
            task_store
                .get_task(task_id)
                .await
                .unwrap()
                .expect("task record")
                .status,
            TaskStatus::Succeeded
        );
    }

    #[sqlx::test(migrations = "../neuromance-db/migrations")]
    #[ignore = "requires postgres via DATABASE_URL"]
    async fn conversation_reads_are_store_authoritative_on_cold_cache(pool: PgPool) {
        let (_state, task_store, _rx) = state_with_store(4, pool);
        let store = task_store.durable_store();

        // A root conversation with two user turns, seeded only in postgres.
        let conv_id = Uuid::new_v4();
        store
            .append_messages(
                conv_id,
                &[
                    Message::system(conv_id, "system"),
                    Message::user(conv_id, "turn-1"),
                    Message::assistant(conv_id, "reply-1"),
                    Message::user(conv_id, "turn-2"),
                    Message::assistant(conv_id, "reply-2"),
                ],
            )
            .await
            .expect("seed durable history");

        // A delegated child conversation, which the list endpoint must exclude.
        let child_id = Uuid::new_v4();
        store
            .append_messages(child_id, &[Message::user(child_id, "child")])
            .await
            .expect("seed child history");
        store
            .set_conversation_parent(child_id, conv_id, None, None, None)
            .await
            .expect("link child to parent");

        assert!(
            !task_store.has_cached_conversation(conv_id),
            "the replica must not have cached the conversation"
        );

        // get_conversation answers from postgres, deriving turn_count.
        let view = task_store
            .get_conversation_view(conv_id)
            .await
            .expect("read ok")
            .expect("conversation resolves from the store");
        assert_eq!(view.messages.len(), 5);
        assert_eq!(view.turn_count, 2);

        // list_conversations answers from postgres and lists roots only.
        let summaries = task_store.list_conversations().await.expect("list ok");
        assert_eq!(summaries.len(), 1, "child conversation must be excluded");
        assert_eq!(summaries[0].id, conv_id);
        assert_eq!(summaries[0].message_count, 5);
        assert_eq!(summaries[0].turn_count, 2);
    }

    /// A `TaskStore` whose `insert_pending` always fails, to exercise the
    /// fail-closed enqueue path in CI without a live database.
    struct FailingTaskStore {
        inner: InMemoryTaskStore,
    }

    #[async_trait]
    impl TaskStore for FailingTaskStore {
        async fn insert_pending(&self, _task: &TaskRecord) -> Result<(), neuromance_db::DbError> {
            Err(neuromance_db::DbError::UnknownTaskStatus {
                value: "insert_pending forced failure".to_string(),
                task_id: Uuid::nil(),
            })
        }
        async fn remove_task(&self, id: Uuid) {
            self.inner.remove_task(id).await;
        }
        async fn mark_running(&self, id: Uuid) -> Option<crate::task_store::TaskTiming> {
            self.inner.mark_running(id).await
        }
        async fn mark_succeeded(&self, id: Uuid, output: String) {
            self.inner.mark_succeeded(id, output).await;
        }
        async fn mark_failed(&self, id: Uuid, reason: &str) {
            self.inner.mark_failed(id, reason).await;
        }
        async fn mark_cancelled(&self, id: Uuid, reason: &str) {
            self.inner.mark_cancelled(id, reason).await;
        }
        async fn get_task(
            &self,
            id: Uuid,
        ) -> Result<Option<neuromance_db::StoredTask>, neuromance_db::DbError> {
            self.inner.get_task(id).await
        }
        async fn list_active_tasks(
            &self,
        ) -> Result<Vec<neuromance_db::StoredTask>, neuromance_db::DbError> {
            self.inner.list_active_tasks().await
        }
        async fn begin_provenance(&self, conversation_id: Uuid) -> Option<i64> {
            self.inner.begin_provenance(conversation_id).await
        }
        async fn record_provenance(
            &self,
            task_id: Uuid,
            conversation_id: Uuid,
            start: Option<i64>,
        ) {
            self.inner
                .record_provenance(task_id, conversation_id, start)
                .await;
        }
        fn seed_conversation(&self, record: ConversationRecord) {
            self.inner.seed_conversation(record);
        }
        async fn conversation_exists(&self, id: Uuid) -> Result<bool, neuromance_db::DbError> {
            self.inner.conversation_exists(id).await
        }
        async fn remove_conversation(&self, id: Uuid) {
            self.inner.remove_conversation(id).await;
        }
        async fn build_turn_input(
            &self,
            conversation_id: Uuid,
            seeded: bool,
            user_msg: Message,
        ) -> Result<Vec<Message>, crate::task_store::TurnInputError> {
            self.inner
                .build_turn_input(conversation_id, seeded, user_msg)
                .await
        }
        async fn refresh_conversation(&self, id: Uuid, full_history: Vec<Message>) {
            self.inner.refresh_conversation(id, full_history).await;
        }
        async fn list_conversations(
            &self,
        ) -> Result<Vec<crate::task_store::ConversationSummary>, neuromance_db::DbError> {
            self.inner.list_conversations().await
        }
        async fn get_conversation_view(
            &self,
            id: Uuid,
        ) -> Result<Option<ConversationRecord>, neuromance_db::DbError> {
            self.inner.get_conversation_view(id).await
        }
        async fn list_conversation_children(
            &self,
            id: Uuid,
            limit: u32,
            offset: u32,
        ) -> Result<Option<Vec<neuromance_db::ConversationSummary>>, neuromance_db::DbError>
        {
            self.inner
                .list_conversation_children(id, limit, offset)
                .await
        }
        async fn drain_pending(&self) -> crate::task_store::ShutdownSummary {
            self.inner.drain_pending().await
        }
    }

    #[tokio::test]
    async fn test_try_enqueue_fail_closed_when_pending_write_fails() {
        let store: Arc<dyn TaskStore> = Arc::new(FailingTaskStore {
            inner: InMemoryTaskStore::new(),
        });
        let (work_tx, _rx) = mpsc::channel::<WorkerJob>(4);
        let state = ServeState {
            task_store: store,
            work_tx,
            system_prompt: Arc::from("system"),
            skills_menu: None,
            provider_names: ["primary".to_owned()].into(),
        };

        let err = try_enqueue(&state, "hi".to_string(), None, None, None, None)
            .await
            .expect_err("a failed durable pending write must reject the enqueue");
        assert!(matches!(err, EnqueueError::Persistence), "got {err:?}");
    }
}
