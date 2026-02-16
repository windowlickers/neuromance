//! gRPC server for handling client connections.
//!
//! Implements the `Neuromance` gRPC service over a Unix domain socket using
//! tonic. Replaces the previous line-delimited JSON protocol.

use std::os::unix::fs::PermissionsExt;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use neuromance_common::protocol::DaemonResponse;
use neuromance_proto::proto;
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::Instant;
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::{Request, Response, Status, Streaming};
use tracing::{error, info, instrument, warn};

use neuromance_daemon::process::is_process_running;

use crate::config::DaemonConfig;
use crate::conversation_manager::ConversationManager;
use crate::error::{DaemonError, Result};
use crate::storage::Storage;

/// Converts a `DaemonError` into a gRPC `Status`.
fn daemon_error_to_status(err: &DaemonError) -> Status {
    match err {
        DaemonError::ConversationNotFound(_)
        | DaemonError::ModelNotFound(_)
        | DaemonError::BookmarkNotFound(_) => Status::not_found(err.to_string()),
        DaemonError::BookmarkExists(_) => Status::already_exists(err.to_string()),
        DaemonError::NoActiveConversation => Status::failed_precondition(err.to_string()),
        DaemonError::InvalidConversationId(_) => Status::invalid_argument(err.to_string()),
        DaemonError::Storage(_) => Status::unavailable(err.to_string()),
        _ => Status::internal(err.to_string()),
    }
}

/// Maps a `DaemonError` to a proto `ErrorCode`.
const fn daemon_error_to_proto_code(err: &DaemonError) -> proto::ErrorCode {
    match err {
        DaemonError::ConversationNotFound(_) => proto::ErrorCode::ConversationNotFound,
        DaemonError::ModelNotFound(_) => proto::ErrorCode::ModelNotFound,
        DaemonError::BookmarkNotFound(_) => proto::ErrorCode::BookmarkNotFound,
        DaemonError::BookmarkExists(_) => proto::ErrorCode::BookmarkExists,
        DaemonError::NoActiveConversation => proto::ErrorCode::NoActiveConversation,
        DaemonError::InvalidConversationId(_) => proto::ErrorCode::InvalidConversationId,
        DaemonError::Core(_) | DaemonError::Client(_) | DaemonError::Tool(_) => {
            proto::ErrorCode::LlmError
        }
        DaemonError::Config(_) | DaemonError::Toml(_) => proto::ErrorCode::ConfigError,
        DaemonError::Storage(_) => proto::ErrorCode::StorageError,
        _ => proto::ErrorCode::Internal,
    }
}

/// Checks version compatibility between daemon and client.
///
/// Returns (compatible, warning). Compatible means same major.minor.
fn check_version_compatibility(
    daemon_version: &str,
    client_version: &str,
) -> (bool, Option<String>) {
    let daemon_parts: Vec<&str> = daemon_version.split('.').collect();
    let client_parts: Vec<&str> = client_version.split('.').collect();

    if daemon_parts.len() < 2 || client_parts.len() < 2 {
        return (false, Some("Invalid version format".to_string()));
    }

    let daemon_mm = format!("{}.{}", daemon_parts[0], daemon_parts[1]);
    let client_mm = format!("{}.{}", client_parts[0], client_parts[1]);

    if daemon_mm == client_mm {
        (true, None)
    } else {
        (
            false,
            Some(format!(
                "Version mismatch: daemon={daemon_version}, \
                 client={client_version}. \
                 Please upgrade to matching versions."
            )),
        )
    }
}

/// Daemon server state.
pub struct Server {
    /// Conversation manager
    manager: Arc<ConversationManager>,

    /// Storage backend
    storage: Arc<Storage>,

    /// Daemon configuration
    config: Arc<DaemonConfig>,

    /// Last activity timestamp for inactivity shutdown
    last_activity: Arc<RwLock<Instant>>,

    /// Server start time for uptime calculation
    start_time: Instant,

    /// Shutdown signal broadcaster
    shutdown_tx: broadcast::Sender<()>,
}

impl Server {
    /// Creates a new server.
    pub fn new(
        manager: Arc<ConversationManager>,
        storage: Arc<Storage>,
        config: Arc<DaemonConfig>,
        shutdown_tx: broadcast::Sender<()>,
    ) -> Self {
        Self {
            manager,
            storage,
            config,
            last_activity: Arc::new(RwLock::new(Instant::now())),
            start_time: Instant::now(),
            shutdown_tx,
        }
    }

    /// Runs the gRPC server over a Unix domain socket.
    ///
    /// # Errors
    ///
    /// Returns an error if socket binding or server startup fails.
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let socket_path = self.storage.socket_path();

        // Check for an existing daemon
        if socket_path.exists() {
            if let Some(existing_pid) = self.storage.read_pid()
                && is_process_running(existing_pid)
            {
                return Err(DaemonError::Other(format!(
                    "Another daemon is running (pid {existing_pid})"
                )));
            }
            std::fs::remove_file(socket_path)?;
        }

        // Write PID file
        let pid = std::process::id();
        self.storage.write_pid(pid)?;
        info!(pid = %pid, "Daemon started");

        // Bind Unix socket
        let listener = tokio::net::UnixListener::bind(socket_path)?;
        std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))?;
        info!(
            socket_path = %socket_path.display(),
            "Daemon listening"
        );

        let uds_stream = UnixListenerStream::new(listener);

        // Spawn inactivity checker
        let server_clone = Arc::clone(&self);
        tokio::spawn(async move {
            server_clone.check_inactivity().await;
        });

        // Subscribe to shutdown signal
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        // Build tonic server
        let grpc_service = GrpcService {
            inner: Arc::clone(&self),
        };
        let svc = neuromance_proto::NeuromanceServer::new(grpc_service);

        tonic::transport::Server::builder()
            .add_service(svc)
            .serve_with_incoming_shutdown(uds_stream, async move {
                let _ = shutdown_rx.recv().await;
                info!("Shutdown signal received, stopping server");
            })
            .await
            .map_err(|e| DaemonError::Other(format!("gRPC server error: {e}")))?;

        // Clean up socket file
        if socket_path.exists() {
            if let Err(e) = std::fs::remove_file(socket_path) {
                warn!("Failed to remove socket file: {e}");
            } else {
                info!("Cleaned up socket file");
            }
        }

        // Clean up PID file
        if let Err(e) = self.storage.remove_pid() {
            warn!("Failed to remove PID file: {e}");
        } else {
            info!("Cleaned up PID file");
        }

        Ok(())
    }

    /// Updates the last activity timestamp.
    async fn touch_activity(&self) {
        *self.last_activity.write().await = Instant::now();
    }

    /// Periodically checks for inactivity and shuts down.
    async fn check_inactivity(self: Arc<Self>) {
        let timeout = Duration::from_secs(self.config.settings.inactivity_timeout);
        let check_interval = Duration::from_secs(60);

        loop {
            tokio::time::sleep(check_interval).await;

            let last = *self.last_activity.read().await;
            if last.elapsed() > timeout {
                info!(
                    inactive_seconds = last.elapsed().as_secs(),
                    timeout_seconds = timeout.as_secs(),
                    "Shutting down due to inactivity"
                );
                let _ = self.shutdown_tx.send(());
                break;
            }
        }
    }
}

/// Wrapper that implements the tonic `Neuromance` service trait.
struct GrpcService {
    inner: Arc<Server>,
}

type ChatStream =
    Pin<Box<dyn futures::Stream<Item = std::result::Result<proto::ChatEvent, Status>> + Send>>;

#[allow(clippy::too_many_lines)]
#[tonic::async_trait]
impl neuromance_proto::Neuromance for GrpcService {
    type ChatStream = ChatStream;

    async fn chat(
        &self,
        request: Request<Streaming<proto::ChatClientMessage>>,
    ) -> std::result::Result<Response<Self::ChatStream>, Status> {
        self.inner.touch_activity().await;

        let mut in_stream = request.into_inner();

        // First message must be SendMessageRequest
        let first = in_stream
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("Empty chat stream"))?
            .map_err(|e| Status::internal(format!("Stream read error: {e}")))?;

        let (conversation_id, content) = match first.message {
            Some(proto::chat_client_message::Message::SendMessage(req)) => {
                let conv_id = if req.conversation_id.is_empty() {
                    None
                } else {
                    Some(req.conversation_id)
                };
                (conv_id, req.content)
            }
            _ => {
                return Err(Status::invalid_argument(
                    "First message must be SendMessageRequest",
                ));
            }
        };

        let (event_tx, event_rx) = mpsc::channel(64);
        let (response_tx, mut response_rx) = mpsc::channel(64);

        let manager = Arc::clone(&self.inner.manager);
        let server = Arc::clone(&self.inner);

        // Spawn message processing task
        let handle = tokio::spawn(async move {
            if let Err(e) = manager
                .send_message(conversation_id, content, response_tx.clone())
                .await
            {
                let _ = response_tx.send(e.into()).await;
            }
        });

        // Bridge task: DaemonResponse → ChatEvent,
        // reads tool approvals from client input stream
        tokio::spawn(async move {
            while let Some(response) = response_rx.recv().await {
                server.touch_activity().await;

                let event = match response {
                    DaemonResponse::StreamChunk {
                        conversation_id: cid,
                        content: c,
                    } => proto::ChatEvent {
                        conversation_id: cid,
                        event: Some(proto::chat_event::Event::StreamChunk(proto::StreamChunk {
                            content: c,
                        })),
                    },
                    DaemonResponse::ToolApprovalRequest {
                        conversation_id: cid,
                        tool_call,
                    } => {
                        let event = proto::ChatEvent {
                            conversation_id: cid.clone(),
                            event: Some(proto::chat_event::Event::ToolApprovalRequest(
                                proto::ToolApprovalRequestProto {
                                    tool_call: Some(proto::ToolCallProto::from(&tool_call)),
                                },
                            )),
                        };

                        if event_tx.send(Ok(event)).await.is_err() {
                            break;
                        }

                        // Read tool approval from client
                        let approval = read_tool_approval(&mut in_stream).await;

                        if let Err(e) = server.manager.approve_tool(&cid, &tool_call.id, approval) {
                            let err = make_chat_error(&e);
                            let _ = event_tx.send(Ok(err)).await;
                        }

                        continue;
                    }
                    DaemonResponse::ToolResult {
                        conversation_id: cid,
                        tool_name,
                        result: res,
                        success,
                    } => proto::ChatEvent {
                        conversation_id: cid,
                        event: Some(proto::chat_event::Event::ToolResult(
                            proto::ToolResultProto {
                                tool_name,
                                result: res,
                                success,
                            },
                        )),
                    },
                    DaemonResponse::Usage {
                        conversation_id: cid,
                        usage,
                    } => proto::ChatEvent {
                        conversation_id: cid,
                        event: Some(proto::chat_event::Event::Usage(
                            proto::UsageProto::from(&usage),
                        )),
                    },
                    DaemonResponse::MessageCompleted {
                        conversation_id: cid,
                        message: msg,
                    } => proto::ChatEvent {
                        conversation_id: cid,
                        event: Some(proto::chat_event::Event::MessageCompleted(
                            proto::MessageCompleted {
                                message: Some(proto::MessageProto::from(msg.as_ref())),
                            },
                        )),
                    },
                    DaemonResponse::Error { code, message: msg } => proto::ChatEvent {
                        conversation_id: String::new(),
                        event: Some(proto::chat_event::Event::Error(proto::ChatError {
                            code: proto::ErrorCode::from(code).into(),
                            message: msg,
                        })),
                    },
                    _ => continue,
                };

                if event_tx.send(Ok(event)).await.is_err() {
                    break;
                }
            }

            // Detect task panics
            if let Err(join_err) = handle.await {
                error!(
                    error = %join_err,
                    "Message processing task panicked"
                );
                let err = proto::ChatEvent {
                    conversation_id: String::new(),
                    event: Some(proto::chat_event::Event::Error(proto::ChatError {
                        code: proto::ErrorCode::Internal.into(),
                        message: "Message processing failed unexpectedly".to_string(),
                    })),
                };
                let _ = event_tx.send(Ok(err)).await;
            }
        });

        let out_stream = ReceiverStream::new(event_rx);
        Ok(Response::new(Box::pin(out_stream)))
    }

    async fn new_conversation(
        &self,
        request: Request<proto::NewConversationRequest>,
    ) -> std::result::Result<Response<proto::NewConversationResponse>, Status> {
        self.inner.touch_activity().await;
        let req = request.into_inner();

        let summary = self
            .inner
            .manager
            .create_conversation(req.model, req.system_message)
            .await
            .map_err(|e| daemon_error_to_status(&e))?;

        Ok(Response::new(proto::NewConversationResponse {
            conversation: Some(proto::ConversationSummaryProto::from(&summary)),
        }))
    }

    async fn list_messages(
        &self,
        request: Request<proto::ListMessagesRequest>,
    ) -> std::result::Result<Response<proto::ListMessagesResponse>, Status> {
        self.inner.touch_activity().await;
        let req = request.into_inner();

        let conv_id = if req.conversation_id.is_empty() {
            None
        } else {
            Some(req.conversation_id)
        };
        #[allow(clippy::cast_possible_truncation)]
        let limit = req.limit.map(|l| l as usize);

        let (messages, total_count, cid) = self
            .inner
            .manager
            .get_messages(conv_id, limit)
            .await
            .map_err(|e| daemon_error_to_status(&e))?;

        let proto_messages: Vec<proto::MessageProto> =
            messages.iter().map(proto::MessageProto::from).collect();

        Ok(Response::new(proto::ListMessagesResponse {
            conversation_id: cid,
            messages: proto_messages,
            total_count: total_count as u64,
        }))
    }

    async fn list_conversations(
        &self,
        request: Request<proto::ListConversationsRequest>,
    ) -> std::result::Result<Response<proto::ListConversationsResponse>, Status> {
        self.inner.touch_activity().await;
        let req = request.into_inner();
        #[allow(clippy::cast_possible_truncation)]
        let limit = req.limit.map(|l| l as usize);

        let conversations = self
            .inner
            .manager
            .list_conversations(limit)
            .await
            .map_err(|e| daemon_error_to_status(&e))?;

        let proto_convs: Vec<proto::ConversationSummaryProto> = conversations
            .iter()
            .map(proto::ConversationSummaryProto::from)
            .collect();

        Ok(Response::new(proto::ListConversationsResponse {
            conversations: proto_convs,
        }))
    }

    async fn set_bookmark(
        &self,
        request: Request<proto::SetBookmarkRequest>,
    ) -> std::result::Result<Response<proto::SetBookmarkResponse>, Status> {
        self.inner.touch_activity().await;
        let req = request.into_inner();
        let conv_id = req.conversation_id.clone();
        let bm_name = req.name.clone();

        self.inner
            .storage
            .run(move |s| {
                let id = s.resolve_conversation_id(&conv_id)?;
                s.set_bookmark(&bm_name, &id)?;
                Ok(())
            })
            .await
            .map_err(|e| daemon_error_to_status(&e))?;

        Ok(Response::new(proto::SetBookmarkResponse {
            message: format!(
                "Set bookmark '{}' for conversation {}",
                req.name, req.conversation_id
            ),
        }))
    }

    async fn remove_bookmark(
        &self,
        request: Request<proto::RemoveBookmarkRequest>,
    ) -> std::result::Result<Response<proto::RemoveBookmarkResponse>, Status> {
        self.inner.touch_activity().await;
        let req = request.into_inner();
        let bm_name = req.name.clone();

        self.inner
            .storage
            .run(move |s| s.remove_bookmark(&bm_name))
            .await
            .map_err(|e| daemon_error_to_status(&e))?;

        Ok(Response::new(proto::RemoveBookmarkResponse {
            message: format!("Removed bookmark '{}'", req.name),
        }))
    }

    #[instrument(skip(self))]
    async fn delete_conversation(
        &self,
        request: Request<proto::DeleteConversationRequest>,
    ) -> std::result::Result<Response<proto::DeleteConversationResponse>, Status> {
        self.inner.touch_activity().await;
        let req = request.into_inner();

        let (summary, removed_bookmarks) = self
            .inner
            .manager
            .delete_conversation(&req.conversation_id)
            .await
            .map_err(|e| daemon_error_to_status(&e))?;

        let title_part = summary
            .title
            .as_deref()
            .map_or(String::new(), |t| format!(" '{t}'"));

        let bookmark_part = if removed_bookmarks.is_empty() {
            String::new()
        } else {
            format!(" (removed bookmarks: {})", removed_bookmarks.join(", "))
        };

        Ok(Response::new(proto::DeleteConversationResponse {
            message: format!(
                "Deleted conversation {}{title_part}\
                 {bookmark_part}",
                summary.short_id
            ),
        }))
    }

    async fn switch_model(
        &self,
        request: Request<proto::SwitchModelRequest>,
    ) -> std::result::Result<Response<proto::SwitchModelResponse>, Status> {
        self.inner.touch_activity().await;
        let req = request.into_inner();

        let conv_id = if req.conversation_id.is_empty() {
            None
        } else {
            Some(req.conversation_id)
        };

        let summary = self
            .inner
            .manager
            .switch_model(conv_id, req.model_nickname)
            .await
            .map_err(|e| daemon_error_to_status(&e))?;

        Ok(Response::new(proto::SwitchModelResponse {
            conversation: Some(proto::ConversationSummaryProto::from(&summary)),
        }))
    }

    async fn list_models(
        &self,
        _request: Request<proto::ListModelsRequest>,
    ) -> std::result::Result<Response<proto::ListModelsResponse>, Status> {
        self.inner.touch_activity().await;

        let models: Vec<proto::ModelProfileProto> = self
            .inner
            .config
            .models
            .iter()
            .map(proto::ModelProfileProto::from)
            .collect();

        Ok(Response::new(proto::ListModelsResponse {
            models,
            active: self.inner.config.active_model.clone(),
        }))
    }

    async fn get_status(
        &self,
        _request: Request<proto::GetStatusRequest>,
    ) -> std::result::Result<Response<proto::GetStatusResponse>, Status> {
        self.inner.touch_activity().await;

        let uptime = self.inner.start_time.elapsed().as_secs();
        let active = self.inner.manager.conversation_count().await.unwrap_or(0);

        Ok(Response::new(proto::GetStatusResponse {
            uptime_seconds: uptime,
            active_conversations: active as u64,
        }))
    }

    async fn get_detailed_status(
        &self,
        _request: Request<proto::GetDetailedStatusRequest>,
    ) -> std::result::Result<Response<proto::GetDetailedStatusResponse>, Status> {
        self.inner.touch_activity().await;

        let uptime = self.inner.start_time.elapsed().as_secs();
        let active = self.inner.manager.conversation_count().await.unwrap_or(0);

        let loaded = self
            .inner
            .storage
            .run(|s| {
                let Some(id) = s.get_active_conversation()? else {
                    return Ok(None);
                };
                let conv = s.load_conversation(&id)?;
                let bm = s.get_conversation_bookmarks(&id).unwrap_or_default();
                Ok(Some((id, conv, bm)))
            })
            .await;

        let current = match loaded {
            Ok(Some((id, conv, bm))) => {
                let model = self.inner.manager.get_conversation_model(&id);
                let summary = neuromance_common::protocol::ConversationSummary::from_conversation(
                    &conv, model, bm,
                );
                Some(proto::ConversationSummaryProto::from(&summary))
            }
            Ok(None) => None,
            Err(e) => {
                warn!(
                    error = %e,
                    "Failed to load active conversation"
                );
                None
            }
        };

        Ok(Response::new(proto::GetDetailedStatusResponse {
            uptime_seconds: uptime,
            active_conversations: active as u64,
            current_conversation: current,
        }))
    }

    async fn health_check(
        &self,
        request: Request<proto::HealthCheckRequest>,
    ) -> std::result::Result<Response<proto::HealthCheckResponse>, Status> {
        self.inner.touch_activity().await;
        let req = request.into_inner();

        let daemon_version = env!("CARGO_PKG_VERSION").to_string();
        let uptime = self.inner.start_time.elapsed().as_secs();
        let (compatible, warning) =
            check_version_compatibility(&daemon_version, &req.client_version);

        Ok(Response::new(proto::HealthCheckResponse {
            daemon_version,
            compatible,
            warning,
            uptime_seconds: uptime,
        }))
    }

    async fn shutdown(
        &self,
        _request: Request<proto::ShutdownRequest>,
    ) -> std::result::Result<Response<proto::ShutdownResponse>, Status> {
        info!("Shutdown requested by client");
        let _ = self.inner.shutdown_tx.send(());

        Ok(Response::new(proto::ShutdownResponse {
            message: "Shutdown initiated".to_string(),
        }))
    }
}

/// Reads a tool approval response from the client stream.
async fn read_tool_approval(
    stream: &mut Streaming<proto::ChatClientMessage>,
) -> neuromance_common::ToolApproval {
    match stream.next().await {
        Some(Ok(msg)) => match msg.message {
            Some(proto::chat_client_message::Message::ToolApproval(ta)) => ta.approval.map_or_else(
                || neuromance_common::ToolApproval::Denied("No approval decision".to_string()),
                neuromance_common::ToolApproval::from,
            ),
            _ => neuromance_common::ToolApproval::Denied("Expected tool approval".to_string()),
        },
        _ => neuromance_common::ToolApproval::Denied("Client disconnected".to_string()),
    }
}

/// Creates a `ChatError` event from a `DaemonError`.
fn make_chat_error(err: &DaemonError) -> proto::ChatEvent {
    proto::ChatEvent {
        conversation_id: String::new(),
        event: Some(proto::chat_event::Event::Error(proto::ChatError {
            code: daemon_error_to_proto_code(err).into(),
            message: err.to_string(),
        })),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_version_compatibility_same_version() {
        let (compatible, warning) = check_version_compatibility("0.0.6", "0.0.6");
        assert!(compatible);
        assert!(warning.is_none());
    }

    #[test]
    fn test_version_compatibility_same_major_minor() {
        let (compatible, warning) = check_version_compatibility("0.0.6", "0.0.7");
        assert!(compatible);
        assert!(warning.is_none());
    }

    #[test]
    fn test_version_compatibility_different_minor() {
        let (compatible, warning) = check_version_compatibility("0.1.0", "0.0.6");
        assert!(!compatible);
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("Version mismatch"));
    }

    #[test]
    fn test_version_compatibility_different_major() {
        let (compatible, warning) = check_version_compatibility("1.0.0", "0.0.6");
        assert!(!compatible);
        assert!(warning.is_some());
    }

    #[test]
    fn test_version_compatibility_invalid_format() {
        let (compatible, warning) = check_version_compatibility("invalid", "0.0.6");
        assert!(!compatible);
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("Invalid version format"));
    }

    #[test]
    fn test_version_compatibility_short_version() {
        let (compatible, warning) = check_version_compatibility("0", "0.0.6");
        assert!(!compatible);
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("Invalid version format"));
    }
}
