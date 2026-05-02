//! Serve mode: long-lived HTTP intake for `AgentTask`-style work.
//!
//! `POST /tasks` enqueues a task and returns immediately with a UUID;
//! `GET /tasks/{id}` returns the current state. Tasks are processed
//! sequentially by a single worker that owns the agent.
//!
//! State is in-memory only; restarts lose pending and completed tasks.
//! Postgres-backed persistence is future work.

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use uuid::Uuid;

use neuromance_agent::{Agent, BaseAgent};
use neuromance_client::LLMClient;
use neuromance_common::chat::Message;

use crate::config::RuntimeConfig;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskRecord {
    pub id: Uuid,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateTaskRequest {
    pub user: String,
}

#[derive(Debug, Serialize)]
pub struct CreateTaskResponse {
    pub task_id: Uuid,
    pub status: TaskStatus,
}

struct WorkerJob {
    task_id: Uuid,
    user: String,
}

#[derive(Clone)]
pub struct ServeState {
    tasks: Arc<DashMap<Uuid, TaskRecord>>,
    work_tx: mpsc::Sender<WorkerJob>,
}

pub fn router(state: ServeState) -> Router {
    Router::new()
        .route("/tasks", post(create_task))
        .route("/tasks/{id}", get(get_task))
        .with_state(state)
}

async fn create_task(
    State(state): State<ServeState>,
    Json(req): Json<CreateTaskRequest>,
) -> impl IntoResponse {
    let task_id = Uuid::new_v4();
    let now = Utc::now();
    let record = TaskRecord {
        id: task_id,
        status: TaskStatus::Pending,
        created_at: now,
        updated_at: now,
        output: None,
        error: None,
    };
    state.tasks.insert(task_id, record);

    if state
        .work_tx
        .send(WorkerJob {
            task_id,
            user: req.user,
        })
        .await
        .is_err()
    {
        if let Some(mut entry) = state.tasks.get_mut(&task_id) {
            entry.status = TaskStatus::Failed;
            entry.error = Some("worker shutting down".to_string());
            entry.updated_at = Utc::now();
        }
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "worker shutting down"})),
        )
            .into_response();
    }

    (
        StatusCode::ACCEPTED,
        Json(CreateTaskResponse {
            task_id,
            status: TaskStatus::Pending,
        }),
    )
        .into_response()
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

async fn worker_loop<C: LLMClient + Send + Sync + 'static>(
    mut rx: mpsc::Receiver<WorkerJob>,
    tasks: Arc<DashMap<Uuid, TaskRecord>>,
    agent: Arc<Mutex<BaseAgent<C>>>,
    system_prompt: String,
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
                process_job(&tasks, &agent, &system_prompt, job).await;
            }
        }
    }
}

async fn process_job<C: LLMClient + Send + Sync>(
    tasks: &Arc<DashMap<Uuid, TaskRecord>>,
    agent: &Arc<Mutex<BaseAgent<C>>>,
    system_prompt: &str,
    job: WorkerJob,
) {
    if let Some(mut entry) = tasks.get_mut(&job.task_id) {
        entry.status = TaskStatus::Running;
        entry.updated_at = Utc::now();
    }

    let mut agent = agent.lock().await;
    let conversation_id = Uuid::new_v4();
    let messages = vec![
        Message::system(conversation_id, system_prompt),
        Message::user(conversation_id, &job.user),
    ];

    match agent.execute(Some(messages)).await {
        Ok(response) => {
            info!(task_id=%job.task_id, "task succeeded");
            if let Some(mut entry) = tasks.get_mut(&job.task_id) {
                entry.status = TaskStatus::Succeeded;
                entry.output = Some(response.content.content);
                entry.updated_at = Utc::now();
            }
        }
        Err(e) => {
            error!(task_id=%job.task_id, error=%e, "task failed");
            if let Some(mut entry) = tasks.get_mut(&job.task_id) {
                entry.status = TaskStatus::Failed;
                entry.error = Some(e.to_string());
                entry.updated_at = Utc::now();
            }
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
    agent: BaseAgent<C>,
    cancel: CancellationToken,
) -> Result<()> {
    let tasks: Arc<DashMap<Uuid, TaskRecord>> = Arc::new(DashMap::new());
    let agent = Arc::new(Mutex::new(agent));
    let (work_tx, work_rx) = mpsc::channel::<WorkerJob>(32);

    let worker = tokio::spawn(worker_loop(
        work_rx,
        Arc::clone(&tasks),
        Arc::clone(&agent),
        config.agent.system_prompt.clone(),
        cancel.clone(),
    ));

    let state = ServeState { tasks, work_tx };
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

    info!("serve mode shutdown complete");
    Ok(())
}
