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

use neuromance::error::CoreError;
use neuromance_agent::Agent;
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
    agent: Arc<Mutex<Agent<C>>>,
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
                let _ = process_job(&tasks, &agent, &system_prompt, job, cancel.clone()).await;
            }
        }
    }
}

#[allow(clippy::significant_drop_tightening)]
async fn process_job<C: LLMClient + Send + Sync>(
    tasks: &Arc<DashMap<Uuid, TaskRecord>>,
    agent: &Arc<Mutex<Agent<C>>>,
    system_prompt: &str,
    job: WorkerJob,
    cancel: CancellationToken,
) -> JobOutcome {
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

    let exec_result = tokio::select! {
        biased;
        () = cancel.cancelled() => Err(CoreError::Cancelled("worker shutdown".to_string())),
        res = agent.execute(Some(messages), cancel.child_token()) => res,
    };

    match exec_result {
        Ok(response) => {
            info!(task_id=%job.task_id, "task succeeded");
            if let Some(mut entry) = tasks.get_mut(&job.task_id) {
                entry.status = TaskStatus::Succeeded;
                entry.output = Some(response.content.content);
                entry.updated_at = Utc::now();
            }
            JobOutcome::Succeeded
        }
        Err(CoreError::Cancelled(_)) => {
            warn!(task_id=%job.task_id, "task cancelled");
            if let Some(mut entry) = tasks.get_mut(&job.task_id) {
                entry.status = TaskStatus::Cancelled;
                entry.error = Some("cancelled".to_string());
                entry.updated_at = Utc::now();
            }
            JobOutcome::Cancelled
        }
        Err(e) => {
            error!(task_id=%job.task_id, error=%e, "task failed");
            if let Some(mut entry) = tasks.get_mut(&job.task_id) {
                entry.status = TaskStatus::Failed;
                entry.error = Some(e.to_string());
                entry.updated_at = Utc::now();
            }
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

    #[tokio::test]
    async fn worker_cancels_in_flight_task() {
        let task_id = Uuid::new_v4();
        let now = Utc::now();
        let tasks: Arc<DashMap<Uuid, TaskRecord>> = Arc::new(DashMap::new());
        tasks.insert(
            task_id,
            TaskRecord {
                id: task_id,
                status: TaskStatus::Pending,
                created_at: now,
                updated_at: now,
                output: None,
                error: None,
            },
        );

        let core = Core::new(SleepingClient::new()).with_streaming();
        let agent = Arc::new(Mutex::new(Agent::new("test".into(), core)));
        let (work_tx, work_rx) = mpsc::channel::<WorkerJob>(1);
        let cancel = CancellationToken::new();

        let worker = tokio::spawn(worker_loop(
            work_rx,
            Arc::clone(&tasks),
            Arc::clone(&agent),
            "system".to_string(),
            cancel.clone(),
        ));

        work_tx
            .send(WorkerJob {
                task_id,
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
}
