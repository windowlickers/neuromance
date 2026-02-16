//! gRPC client for communicating with the daemon.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};
use fs2::FileExt;
use hyper_util::rt::TokioIo;
use neuromance_daemon::process::is_process_running;
use neuromance_proto::NeuromanceClient;
use neuromance_proto::proto;
use tokio::net::UnixStream;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;

/// Client for communicating with the Neuromance daemon over gRPC.
pub struct DaemonClient {
    inner: NeuromanceClient<Channel>,
}

/// A bidirectional chat stream session.
pub struct ChatSession {
    tx: tokio::sync::mpsc::Sender<proto::ChatClientMessage>,
    rx: tonic::Streaming<proto::ChatEvent>,
}

impl ChatSession {
    /// Reads the next event from the server.
    pub async fn next_event(&mut self) -> Result<Option<proto::ChatEvent>> {
        use tokio_stream::StreamExt;
        match self.rx.next().await {
            Some(Ok(event)) => Ok(Some(event)),
            Some(Err(e)) => Err(anyhow::anyhow!("Stream error: {e}")),
            None => Ok(None),
        }
    }

    /// Sends a tool approval response.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub async fn send_tool_approval(
        &mut self,
        conversation_id: String,
        tool_call_id: String,
        approval: neuromance_common::ToolApproval,
    ) -> Result<()> {
        let msg = proto::ChatClientMessage {
            message: Some(proto::chat_client_message::Message::ToolApproval(
                proto::ToolApprovalResponse {
                    conversation_id,
                    tool_call_id,
                    approval: Some(proto::ToolApprovalDecision::from(&approval)),
                },
            )),
        };

        self.tx
            .send(msg)
            .await
            .map_err(|_| anyhow::anyhow!("Stream closed"))
    }
}

impl DaemonClient {
    /// Waits for socket with exponential backoff.
    async fn wait_for_socket(socket_path: &Path, timeout_secs: u64) -> Result<()> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
        let mut delay = Duration::from_millis(50);
        let max_delay = Duration::from_millis(500);

        loop {
            if UnixStream::connect(socket_path).await.is_ok() {
                return Ok(());
            }

            if tokio::time::Instant::now() + delay > deadline {
                anyhow::bail!("Socket unavailable after {timeout_secs}s");
            }

            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(max_delay);
        }
    }

    /// Creates a gRPC channel connected to the Unix socket.
    async fn create_channel(socket_path: PathBuf) -> Result<Channel> {
        let channel = Endpoint::try_from("http://[::]:50051")
            .context("Invalid endpoint")?
            .connect_with_connector(service_fn(move |_: Uri| {
                let path = socket_path.clone();
                async move {
                    let stream = UnixStream::connect(path).await?;
                    Ok::<_, std::io::Error>(TokioIo::new(stream))
                }
            }))
            .await
            .context("Failed to connect to daemon")?;

        Ok(channel)
    }

    /// Connects to the daemon, auto-spawning if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails after spawn attempts.
    #[allow(clippy::significant_drop_in_scrutinee)]
    pub async fn connect() -> Result<Self> {
        let socket_path = Self::socket_path()?;
        let pid_file = Self::pid_file()?;
        let lock_file = Self::lock_file()?;

        // Try to connect to existing daemon
        if let Ok(channel) = Self::create_channel(socket_path.clone()).await {
            return Ok(Self {
                inner: NeuromanceClient::new(channel),
            });
        }

        // Check if daemon process exists via PID file
        if let Some(pid) = Self::read_pid(&pid_file)? {
            if is_process_running(pid) {
                Self::wait_for_socket(&socket_path, 10)
                    .await
                    .context(format!("Daemon (PID {pid}) but socket unavailable"))?;
                let channel = Self::create_channel(socket_path).await?;
                return Ok(Self {
                    inner: NeuromanceClient::new(channel),
                });
            }
            let _ = std::fs::remove_file(&pid_file);
        }

        // Acquire lock to prevent concurrent spawns
        let lock = Self::acquire_spawn_lock(&lock_file)?;

        // Double-check after acquiring lock
        if let Ok(channel) = Self::create_channel(socket_path.clone()).await {
            drop(lock);
            return Ok(Self {
                inner: NeuromanceClient::new(channel),
            });
        }

        // Spawn daemon
        Self::spawn_daemon()?;

        // Wait for socket
        Self::wait_for_socket(&socket_path, 10).await?;

        drop(lock);

        let channel = Self::create_channel(socket_path).await?;
        Ok(Self {
            inner: NeuromanceClient::new(channel),
        })
    }

    /// Starts a bidirectional chat session.
    pub async fn chat(
        &mut self,
        conversation_id: Option<String>,
        content: String,
    ) -> Result<ChatSession> {
        let (tx, rx) = tokio::sync::mpsc::channel(16);

        // Send the initial message
        let first = proto::ChatClientMessage {
            message: Some(proto::chat_client_message::Message::SendMessage(
                proto::SendMessageRequest {
                    conversation_id: conversation_id.unwrap_or_default(),
                    content,
                },
            )),
        };
        tx.send(first)
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send initial message"))?;

        let in_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let response = self
            .inner
            .chat(in_stream)
            .await
            .map_err(|e| anyhow::anyhow!("Chat RPC failed: {e}"))?;

        Ok(ChatSession {
            tx,
            rx: response.into_inner(),
        })
    }

    /// Creates a new conversation.
    pub async fn new_conversation(
        &mut self,
        model: Option<String>,
        system_message: Option<String>,
    ) -> Result<proto::NewConversationResponse> {
        self.inner
            .new_conversation(proto::NewConversationRequest {
                model,
                system_message,
            })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Lists messages from a conversation.
    pub async fn list_messages(
        &mut self,
        conversation_id: Option<String>,
        limit: Option<usize>,
    ) -> Result<proto::ListMessagesResponse> {
        self.inner
            .list_messages(proto::ListMessagesRequest {
                conversation_id: conversation_id.unwrap_or_default(),
                limit: limit.map(|l| l as u64),
            })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Lists all conversations.
    pub async fn list_conversations(
        &mut self,
        limit: Option<usize>,
    ) -> Result<proto::ListConversationsResponse> {
        self.inner
            .list_conversations(proto::ListConversationsRequest {
                limit: limit.map(|l| l as u64),
            })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Sets a bookmark.
    pub async fn set_bookmark(
        &mut self,
        conversation_id: String,
        name: String,
    ) -> Result<proto::SetBookmarkResponse> {
        self.inner
            .set_bookmark(proto::SetBookmarkRequest {
                conversation_id,
                name,
            })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Removes a bookmark.
    pub async fn remove_bookmark(&mut self, name: String) -> Result<proto::RemoveBookmarkResponse> {
        self.inner
            .remove_bookmark(proto::RemoveBookmarkRequest { name })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Deletes a conversation.
    pub async fn delete_conversation(
        &mut self,
        conversation_id: String,
    ) -> Result<proto::DeleteConversationResponse> {
        self.inner
            .delete_conversation(proto::DeleteConversationRequest { conversation_id })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Switches a conversation's model.
    pub async fn switch_model(
        &mut self,
        conversation_id: Option<String>,
        model_nickname: String,
    ) -> Result<proto::SwitchModelResponse> {
        self.inner
            .switch_model(proto::SwitchModelRequest {
                conversation_id: conversation_id.unwrap_or_default(),
                model_nickname,
            })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Lists available models.
    pub async fn list_models(&mut self) -> Result<proto::ListModelsResponse> {
        self.inner
            .list_models(proto::ListModelsRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Gets basic daemon status.
    pub async fn get_status(&mut self) -> Result<proto::GetStatusResponse> {
        self.inner
            .get_status(proto::GetStatusRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Gets detailed status.
    pub async fn get_detailed_status(&mut self) -> Result<proto::GetDetailedStatusResponse> {
        self.inner
            .get_detailed_status(proto::GetDetailedStatusRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Health check.
    pub async fn health_check(&mut self) -> Result<proto::HealthCheckResponse> {
        let client_version = env!("CARGO_PKG_VERSION").to_string();
        self.inner
            .health_check(proto::HealthCheckRequest { client_version })
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Requests daemon shutdown.
    pub async fn shutdown(&mut self) -> Result<proto::ShutdownResponse> {
        self.inner
            .shutdown(proto::ShutdownRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    // --- Internal helpers (unchanged) ---

    fn data_dir() -> Result<PathBuf> {
        dirs::data_local_dir()
            .context("Failed to determine data directory")
            .map(|dir| dir.join("neuromance"))
    }

    fn socket_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("neuromance.sock"))
    }

    fn pid_file() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("neuromance.pid"))
    }

    fn lock_file() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("neuromance.lock"))
    }

    fn read_pid(pid_file: &Path) -> Result<Option<u32>> {
        if !pid_file.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(pid_file).context("Failed to read PID file")?;
        Ok(content.trim().parse::<u32>().ok())
    }

    fn acquire_spawn_lock(lock_file: &Path) -> Result<File> {
        if let Some(parent) = lock_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_file)
            .context("Failed to open lock file")?;

        file.try_lock_exclusive()
            .context("Failed to acquire spawn lock")?;

        Ok(file)
    }

    fn spawn_daemon() -> Result<()> {
        let stderr_stdio = Self::open_daemon_log().map_or_else(|_| Stdio::null(), Stdio::from);

        Command::new("neuromance-daemon")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr_stdio)
            .spawn()
            .context("Failed to spawn daemon. Is neuromance-daemon in PATH?")?;

        Ok(())
    }

    fn open_daemon_log() -> Result<File> {
        let log_path = Self::data_dir()?.join("daemon.log");

        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .context("Failed to open daemon log file")
    }
}
