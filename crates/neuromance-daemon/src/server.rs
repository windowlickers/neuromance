//! Unix socket server for handling client connections.
//!
//! Listens on a Unix domain socket and handles line-delimited JSON requests
//! from the `nm` CLI client.

use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use neuromance_common::protocol::{DaemonRequest, DaemonResponse, ErrorCode};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::Instant;
use tracing::{debug, error, info, instrument, warn};

use crate::config::DaemonConfig;
use crate::conversation_manager::ConversationManager;
use crate::error::{DaemonError, Result};
use crate::storage::Storage;

/// Maximum size of a single request line (1 MB).
const MAX_REQUEST_SIZE: usize = 1024 * 1024;

/// Idle timeout for a client connection with no complete request (5 minutes).
const CONNECTION_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// Checks if a process with the given PID is running.
///
/// Uses `kill -0` which works on both Linux and macOS.
fn is_process_running(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
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

    /// Runs the server.
    ///
    /// Binds to the Unix socket and accepts connections until shutdown is requested.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Socket binding fails
    /// - Connection handling fails
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let socket_path = self.storage.socket_path();

        // Check for an existing daemon before removing the socket
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
        let listener = UnixListener::bind(socket_path)?;
        std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))?;
        info!(socket_path = %socket_path.display(), "Daemon listening");

        // Subscribe to shutdown signal
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        // Spawn inactivity checker with shutdown awareness
        let server_clone = Arc::clone(&self);
        tokio::spawn(async move {
            server_clone.check_inactivity().await;
        });

        // Accept connections until shutdown
        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, _addr)) => {
                            let server = Arc::clone(&self);
                            debug!("Client connected");
                            tokio::spawn(async move {
                                if let Err(e) = server.handle_connection(stream).await {
                                    error!(error = %e, "Connection handling error");
                                }
                            });
                        }
                        Err(e) => {
                            error!(error = %e, "Accept error");
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received, stopping accept loop");
                    break;
                }
            }
        }

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

    /// Handles a single client connection.
    async fn handle_connection(self: Arc<Self>, stream: UnixStream) -> Result<()> {
        // Update last activity
        *self.last_activity.write().await = Instant::now();

        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);
        let mut line = String::new();

        loop {
            // Read line-delimited JSON (size-limited, with idle timeout)
            let read_result = tokio::time::timeout(
                CONNECTION_IDLE_TIMEOUT,
                Self::read_line_limited(&mut reader, &mut line),
            )
            .await;

            match read_result {
                Err(_) => {
                    warn!(
                        "Client connection timed out after {}s of inactivity",
                        CONNECTION_IDLE_TIMEOUT.as_secs()
                    );
                    break;
                }
                Ok(Err(e)) => {
                    error!(error = %e, "Read error");
                    break;
                }
                Ok(Ok(0)) => {
                    debug!("Client disconnected");
                    break;
                }
                Ok(Ok(_)) => {
                    // Any data on the socket is evidence of activity
                    *self.last_activity.write().await = Instant::now();

                    // Parse request
                    match serde_json::from_str::<DaemonRequest>(&line) {
                        Ok(DaemonRequest::SendMessage {
                            conversation_id,
                            content,
                        }) => {
                            debug!("Received SendMessage request");
                            if let Err(e) = self
                                .handle_streaming_session(
                                    conversation_id,
                                    content,
                                    &mut reader,
                                    &mut writer,
                                )
                                .await
                            {
                                error!(error = %e, "Streaming session error");
                                let response: DaemonResponse = e.into();
                                let _ = Self::write_response(&mut writer, &response).await;
                            }
                        }
                        Ok(request) => {
                            debug!(request = ?request, "Received request");

                            // Handle request
                            match self.handle_request(request).await {
                                Ok(responses) => {
                                    for response in responses {
                                        if let Err(e) =
                                            Self::write_response(&mut writer, &response).await
                                        {
                                            error!(
                                                error = %e,
                                                "Failed to write response"
                                            );
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!(error = %e, "Request handling error");
                                    let response: DaemonResponse = e.into();
                                    let _ = Self::write_response(&mut writer, &response).await;
                                }
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "Invalid request JSON");
                            let response = DaemonResponse::Error {
                                code: ErrorCode::InvalidRequest,
                                message: format!("Invalid JSON: {e}"),
                            };
                            let _ = Self::write_response(&mut writer, &response).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Dispatches a daemon request to the appropriate handler.
    #[instrument(skip_all)]
    async fn handle_request(&self, request: DaemonRequest) -> Result<Vec<DaemonResponse>> {
        match request {
            DaemonRequest::NewConversation {
                model,
                system_message,
            } => self.handle_new_conversation(model, system_message).await,
            DaemonRequest::ListMessages {
                conversation_id,
                limit,
            } => self.handle_list_messages(conversation_id, limit).await,
            DaemonRequest::ListConversations { limit } => {
                self.handle_list_conversations(limit).await
            }
            DaemonRequest::SetBookmark {
                conversation_id,
                name,
            } => self.handle_set_bookmark(&conversation_id, &name).await,
            DaemonRequest::RemoveBookmark { name } => {
                self.handle_remove_bookmark(&name).await
            }
            DaemonRequest::DeleteConversation { conversation_id } => {
                self.handle_delete_conversation(&conversation_id).await
            }
            DaemonRequest::SwitchModel {
                conversation_id,
                model_nickname,
            } => {
                self.handle_switch_model(conversation_id, model_nickname)
                    .await
            }
            DaemonRequest::ListModels => Ok(self.handle_list_models()),
            DaemonRequest::ToolApproval {
                conversation_id,
                tool_call_id,
                approval,
            } => Ok(self.handle_tool_approval(&conversation_id, &tool_call_id, approval)?),
            DaemonRequest::Status => Ok(self.handle_status()),
            DaemonRequest::DetailedStatus => self.handle_detailed_status().await,
            DaemonRequest::Health { client_version } => Ok(self.handle_health(&client_version)),
            DaemonRequest::Shutdown => Ok(self.handle_shutdown()),
            _ => {
                warn!("Unknown request variant received");
                Ok(vec![DaemonResponse::Error {
                    code: ErrorCode::InvalidRequest,
                    message: "Unknown request variant".to_string(),
                }])
            }
        }
    }

    /// Streams responses directly to the client as they arrive.
    ///
    /// Handles tool approval requests inline by reading the client's
    /// response from the socket during the streaming session.
    async fn handle_streaming_session(
        &self,
        conversation_id: Option<String>,
        content: String,
        reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
        writer: &mut tokio::net::unix::OwnedWriteHalf,
    ) -> Result<()> {
        let (tx, mut rx) = mpsc::unbounded_channel();

        let manager = Arc::clone(&self.manager);
        tokio::spawn(async move {
            if let Err(e) = manager
                .send_message(conversation_id, content, tx.clone())
                .await
            {
                let _ = tx.send(e.into());
            }
        });

        while let Some(response) = rx.recv().await {
            let needs_approval = matches!(&response, DaemonResponse::ToolApprovalRequest { .. });
            Self::write_response(writer, &response).await?;

            if needs_approval {
                self.read_tool_approval(reader, writer).await?;
            }
        }

        Ok(())
    }

    /// Reads a tool approval request from the client during a streaming session.
    async fn read_tool_approval(
        &self,
        reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
        writer: &mut tokio::net::unix::OwnedWriteHalf,
    ) -> Result<()> {
        let mut line = String::new();
        let bytes_read = Self::read_line_limited(reader, &mut line).await?;

        if bytes_read == 0 {
            return Err(DaemonError::Other(
                "Client disconnected during tool approval".to_string(),
            ));
        }

        match serde_json::from_str::<DaemonRequest>(&line) {
            Ok(DaemonRequest::ToolApproval {
                conversation_id,
                tool_call_id,
                approval,
            }) => {
                let responses =
                    self.handle_tool_approval(&conversation_id, &tool_call_id, approval)?;
                for response in responses {
                    Self::write_response(writer, &response).await?;
                }
            }
            Ok(other) => {
                warn!(request = ?other, "Unexpected request during tool approval");
                let response = DaemonResponse::Error {
                    code: ErrorCode::InvalidRequest,
                    message: "Expected ToolApproval request".to_string(),
                };
                Self::write_response(writer, &response).await?;
            }
            Err(e) => {
                warn!(error = %e, "Invalid JSON during tool approval");
                let response = DaemonResponse::Error {
                    code: ErrorCode::InvalidRequest,
                    message: format!("Invalid JSON: {e}"),
                };
                Self::write_response(writer, &response).await?;
            }
        }

        Ok(())
    }

    async fn handle_new_conversation(
        &self,
        model: Option<String>,
        system_message: Option<String>,
    ) -> Result<Vec<DaemonResponse>> {
        let summary = self
            .manager
            .create_conversation(model, system_message)
            .await?;
        Ok(vec![DaemonResponse::ConversationCreated {
            conversation: summary,
        }])
    }

    async fn handle_list_messages(
        &self,
        conversation_id: Option<String>,
        limit: Option<usize>,
    ) -> Result<Vec<DaemonResponse>> {
        let (messages, total_count, conv_id) =
            self.manager.get_messages(conversation_id, limit).await?;
        Ok(vec![DaemonResponse::Messages {
            conversation_id: conv_id,
            messages,
            total_count,
        }])
    }

    async fn handle_list_conversations(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<DaemonResponse>> {
        let conversations =
            self.manager.list_conversations(limit).await?;
        Ok(vec![DaemonResponse::Conversations { conversations }])
    }

    async fn handle_set_bookmark(
        &self,
        conversation_id: &str,
        name: &str,
    ) -> Result<Vec<DaemonResponse>> {
        let conv_id = conversation_id.to_string();
        let bm_name = name.to_string();
        self.storage
            .run(move |s| {
                let id = s.resolve_conversation_id(&conv_id)?;
                s.set_bookmark(&bm_name, &id)?;
                Ok(())
            })
            .await?;
        Ok(vec![DaemonResponse::Success {
            message: format!(
                "Set bookmark '{name}' for conversation {conversation_id}"
            ),
        }])
    }

    async fn handle_remove_bookmark(
        &self,
        name: &str,
    ) -> Result<Vec<DaemonResponse>> {
        let bm_name = name.to_string();
        self.storage
            .run(move |s| s.remove_bookmark(&bm_name))
            .await?;
        Ok(vec![DaemonResponse::Success {
            message: format!("Removed bookmark '{name}'"),
        }])
    }

    async fn handle_delete_conversation(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<DaemonResponse>> {
        let (summary, removed_bookmarks) =
            self.manager.delete_conversation(conversation_id).await?;

        let title_part = summary
            .title
            .as_deref()
            .map_or(String::new(), |t| format!(" '{t}'"));

        let bookmark_part = if removed_bookmarks.is_empty() {
            String::new()
        } else {
            format!(
                " (removed bookmarks: {})",
                removed_bookmarks.join(", ")
            )
        };

        Ok(vec![DaemonResponse::Success {
            message: format!(
                "Deleted conversation {}{title_part}{bookmark_part}",
                summary.short_id
            ),
        }])
    }

    async fn handle_switch_model(
        &self,
        conversation_id: Option<String>,
        model_nickname: String,
    ) -> Result<Vec<DaemonResponse>> {
        let summary = self
            .manager
            .switch_model(conversation_id, model_nickname)
            .await?;
        Ok(vec![DaemonResponse::ModelSwitched {
            conversation: summary,
        }])
    }

    fn handle_list_models(&self) -> Vec<DaemonResponse> {
        let models = self.config.models.clone();
        let active = self.config.active_model.clone();
        vec![DaemonResponse::Models { models, active }]
    }

    fn handle_tool_approval(
        &self,
        conversation_id: &str,
        tool_call_id: &str,
        approval: neuromance_common::ToolApproval,
    ) -> Result<Vec<DaemonResponse>> {
        self.manager
            .approve_tool(conversation_id, tool_call_id, approval)?;
        Ok(vec![DaemonResponse::Success {
            message: "Tool approval processed".to_string(),
        }])
    }

    fn handle_status(&self) -> Vec<DaemonResponse> {
        let uptime_seconds = self.start_time.elapsed().as_secs();
        let active_conversations = self.manager.active_conversation_count();
        vec![DaemonResponse::Status {
            uptime_seconds,
            active_conversations,
            current_conversation: None,
        }]
    }

    async fn handle_detailed_status(&self) -> Result<Vec<DaemonResponse>> {
        let uptime_seconds = self.start_time.elapsed().as_secs();
        let active_conversations = self.manager.active_conversation_count();

        let loaded = self
            .storage
            .run(|s| {
                let active_id = s.get_active_conversation()?;
                let Some(id) = active_id else {
                    return Ok(None);
                };
                let conv = s.load_conversation(&id)?;
                let bookmarks = s
                    .get_conversation_bookmarks(&id)
                    .unwrap_or_default();
                Ok(Some((id, conv, bookmarks)))
            })
            .await;

        let current_conversation = match loaded {
            Ok(Some((id, conv, bookmarks))) => {
                let model = self.manager.get_conversation_model(&id);
                Some(
                    neuromance_common::protocol::ConversationSummary::from_conversation(
                        &conv, model, bookmarks,
                    ),
                )
            }
            Ok(None) => None,
            Err(e) => {
                warn!(error = %e, "Failed to load active conversation for status");
                None
            }
        };

        Ok(vec![DaemonResponse::Status {
            uptime_seconds,
            active_conversations,
            current_conversation,
        }])
    }

    fn handle_health(&self, client_version: &str) -> Vec<DaemonResponse> {
        let daemon_version = env!("CARGO_PKG_VERSION").to_string();
        let uptime_seconds = self.start_time.elapsed().as_secs();
        let (compatible, warning) =
            Self::check_version_compatibility(&daemon_version, client_version);
        vec![DaemonResponse::Health {
            daemon_version,
            compatible,
            warning,
            uptime_seconds,
        }]
    }

    fn handle_shutdown(&self) -> Vec<DaemonResponse> {
        info!("Shutdown requested by client");
        let _ = self.shutdown_tx.send(());
        vec![DaemonResponse::Success {
            message: "Shutdown initiated".to_string(),
        }]
    }

    /// Writes a response to the client.
    async fn write_response(
        writer: &mut tokio::net::unix::OwnedWriteHalf,
        response: &DaemonResponse,
    ) -> Result<()> {
        let json = serde_json::to_string(response)?;
        writer.write_all(json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
        Ok(())
    }

    /// Reads a newline-delimited line with a size limit.
    ///
    /// Returns the number of bytes read (0 = EOF). Errors if the
    /// line exceeds `MAX_REQUEST_SIZE` before a newline is found.
    async fn read_line_limited(
        reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
        buf: &mut String,
    ) -> Result<usize> {
        buf.clear();
        let mut total = 0;

        loop {
            let available = reader.fill_buf().await?;
            if available.is_empty() {
                return Ok(total);
            }

            let newline_pos = available.iter().position(|&b| b == b'\n');
            let n = newline_pos.map_or(available.len(), |p| p + 1);

            total += n;
            if total > MAX_REQUEST_SIZE {
                reader.consume(n);
                return Err(DaemonError::Other(format!(
                    "Request exceeds {MAX_REQUEST_SIZE} byte limit"
                )));
            }

            let chunk = std::str::from_utf8(&available[..n])
                .map_err(|_| DaemonError::Other("Invalid UTF-8 in request".to_string()))?;
            buf.push_str(chunk);
            reader.consume(n);

            if newline_pos.is_some() {
                return Ok(total);
            }
        }
    }

    /// Checks version compatibility between daemon and client.
    ///
    /// Returns (compatible, warning). Compatible means same major.minor version.
    fn check_version_compatibility(
        daemon_version: &str,
        client_version: &str,
    ) -> (bool, Option<String>) {
        let daemon_parts: Vec<&str> = daemon_version.split('.').collect();
        let client_parts: Vec<&str> = client_version.split('.').collect();

        if daemon_parts.len() < 2 || client_parts.len() < 2 {
            return (false, Some("Invalid version format".to_string()));
        }

        let daemon_major_minor = format!("{}.{}", daemon_parts[0], daemon_parts[1]);
        let client_major_minor = format!("{}.{}", client_parts[0], client_parts[1]);

        if daemon_major_minor == client_major_minor {
            (true, None)
        } else {
            (
                false,
                Some(format!(
                    "Version mismatch: daemon={daemon_version}, client={client_version}. \
                     Please upgrade to matching versions."
                )),
            )
        }
    }

    /// Periodically checks for inactivity and shuts down if timeout exceeded.
    async fn check_inactivity(self: Arc<Self>) {
        let timeout = Duration::from_secs(self.config.settings.inactivity_timeout);
        let check_interval = Duration::from_secs(60);

        loop {
            tokio::time::sleep(check_interval).await;

            let last_activity = *self.last_activity.read().await;
            if last_activity.elapsed() > timeout {
                let inactive_secs = last_activity.elapsed().as_secs();
                info!(
                    inactive_seconds = inactive_secs,
                    timeout_seconds = timeout.as_secs(),
                    "Shutting down due to inactivity"
                );
                let _ = self.shutdown_tx.send(());
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_version_compatibility_same_version() {
        let (compatible, warning) = Server::check_version_compatibility("0.0.6", "0.0.6");
        assert!(compatible);
        assert!(warning.is_none());
    }

    #[test]
    fn test_version_compatibility_same_major_minor() {
        let (compatible, warning) = Server::check_version_compatibility("0.0.6", "0.0.7");
        assert!(compatible);
        assert!(warning.is_none());
    }

    #[test]
    fn test_version_compatibility_different_minor() {
        let (compatible, warning) = Server::check_version_compatibility("0.1.0", "0.0.6");
        assert!(!compatible);
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("Version mismatch"));
    }

    #[test]
    fn test_version_compatibility_different_major() {
        let (compatible, warning) = Server::check_version_compatibility("1.0.0", "0.0.6");
        assert!(!compatible);
        assert!(warning.is_some());
    }

    #[test]
    fn test_version_compatibility_invalid_format() {
        let (compatible, warning) = Server::check_version_compatibility("invalid", "0.0.6");
        assert!(!compatible);
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("Invalid version format"));
    }

    #[test]
    fn test_version_compatibility_short_version() {
        let (compatible, warning) = Server::check_version_compatibility("0", "0.0.6");
        assert!(!compatible);
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("Invalid version format"));
    }
}
