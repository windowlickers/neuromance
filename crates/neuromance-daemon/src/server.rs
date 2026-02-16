//! Unix socket server for handling client connections.
//!
//! Listens on a Unix domain socket and handles line-delimited JSON requests
//! from the `nm` CLI client.

use std::sync::Arc;
use std::time::Duration;

use neuromance_common::protocol::{DaemonRequest, DaemonResponse};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{RwLock, mpsc, broadcast};
use tokio::time::Instant;
use tracing::{debug, error, info, warn, instrument};

use crate::config::DaemonConfig;
use crate::conversation_manager::ConversationManager;
use crate::error::Result;
use crate::storage::Storage;

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

        // Write PID file
        let pid = std::process::id();
        self.storage.write_pid(pid)?;
        info!(pid = %pid, "Daemon started");

        // Remove existing socket file
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        // Bind Unix socket
        let listener = UnixListener::bind(socket_path)?;
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
            line.clear();

            // Read line-delimited JSON
            match reader.read_line(&mut line).await {
                Ok(0) => {
                    // Connection closed
                    debug!("Client disconnected");
                    break;
                }
                Ok(_) => {
                    // Parse request
                    match serde_json::from_str::<DaemonRequest>(&line) {
                        Ok(request) => {
                            debug!(request = ?request, "Received request");

                            // Handle request
                            match self.handle_request(request).await {
                                Ok(responses) => {
                                    // Write responses (may be multiple for streaming)
                                    for response in responses {
                                        if let Err(e) =
                                            Self::write_response(&mut writer, &response).await
                                        {
                                            error!(error = %e, "Failed to write response");
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!(error = %e, "Request handling error");
                                    let response = DaemonResponse::Error {
                                        message: e.to_string(),
                                    };
                                    let _ = Self::write_response(&mut writer, &response).await;
                                }
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "Invalid request JSON");
                            let response = DaemonResponse::Error {
                                message: format!("Invalid JSON: {e}"),
                            };
                            let _ = Self::write_response(&mut writer, &response).await;
                        }
                    }
                }
                Err(e) => {
                    error!(error = %e, "Read error");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handles a daemon request.
    ///
    /// Returns a vector of responses (for streaming, multiple responses may be sent).
    #[instrument(skip(self), fields(request_type = ?request))]
    async fn handle_request(&self, request: DaemonRequest) -> Result<Vec<DaemonResponse>> {
        match request {
            DaemonRequest::SendMessage {
                conversation_id,
                content,
            } => {
                // Create channel for streaming responses
                let (tx, mut rx) = mpsc::unbounded_channel();

                // Spawn task to handle sending
                let manager = Arc::clone(&self.manager);
                tokio::spawn(async move {
                    if let Err(e) = manager
                        .send_message(conversation_id, content, tx.clone())
                        .await
                    {
                        let _ = tx.send(DaemonResponse::Error {
                            message: e.to_string(),
                        });
                    }
                });

                // Collect responses
                let mut responses = Vec::new();
                while let Some(response) = rx.recv().await {
                    responses.push(response);
                }

                Ok(responses)
            }

            DaemonRequest::NewConversation {
                model,
                system_message,
            } => {
                let summary = self
                    .manager
                    .create_conversation(model, system_message)
                    .await?;
                Ok(vec![DaemonResponse::ConversationCreated {
                    conversation: summary,
                }])
            }

            DaemonRequest::ListMessages {
                conversation_id,
                limit,
            } => {
                let (messages, total_count, conv_id) =
                    self.manager.get_messages(conversation_id, limit)?;

                Ok(vec![DaemonResponse::Messages {
                    conversation_id: conv_id,
                    messages,
                    total_count,
                }])
            }

            DaemonRequest::ListConversations { limit } => {
                let conversations = self.manager.list_conversations(limit)?;
                Ok(vec![DaemonResponse::Conversations { conversations }])
            }

            DaemonRequest::SetBookmark {
                conversation_id,
                name,
            } => {
                let id = self.storage.resolve_conversation_id(&conversation_id)?;
                self.storage.set_bookmark(&name, &id)?;

                Ok(vec![DaemonResponse::Success {
                    message: format!("Set bookmark '{name}' for conversation {conversation_id}"),
                }])
            }

            DaemonRequest::RemoveBookmark { name } => {
                self.storage.remove_bookmark(&name)?;

                Ok(vec![DaemonResponse::Success {
                    message: format!("Removed bookmark '{name}'"),
                }])
            }

            DaemonRequest::SwitchModel {
                conversation_id,
                model_nickname,
            } => {
                let summary = self
                    .manager
                    .switch_model(conversation_id, model_nickname)
                    .await?;
                Ok(vec![DaemonResponse::ConversationCreated {
                    conversation: summary,
                }])
            }

            DaemonRequest::ListModels => {
                let models = self.config.models.clone();
                let active = self.config.active_model.clone();

                Ok(vec![DaemonResponse::Models { models, active }])
            }

            DaemonRequest::ToolApproval {
                conversation_id,
                tool_call_id,
                approval,
            } => {
                self.manager
                    .approve_tool(&conversation_id, &tool_call_id, approval)?;

                Ok(vec![DaemonResponse::Success {
                    message: "Tool approval processed".to_string(),
                }])
            }

            DaemonRequest::Status => {
                let uptime_seconds = self.start_time.elapsed().as_secs();
                let active_conversations = self.manager.clients.len();

                Ok(vec![DaemonResponse::Status {
                    uptime_seconds,
                    active_conversations,
                    current_conversation: None,
                }])
            }

            DaemonRequest::DetailedStatus => {
                let uptime_seconds = self.start_time.elapsed().as_secs();
                let active_conversations = self.manager.clients.len();

                // Get current conversation details
                let current_conversation = self
                    .storage
                    .get_active_conversation()
                    .ok()
                    .flatten()
                    .and_then(|id| {
                        // Load conversation
                        let conv = self.storage.load_conversation(&id).ok()?;

                        // Get model for this conversation
                        let model = self.manager.get_conversation_model(&id);

                        // Get bookmarks for this conversation
                        let bookmarks = self
                            .storage
                            .get_conversation_bookmarks(&id)
                            .unwrap_or_default();

                        Some(neuromance_common::protocol::ConversationSummary::from_conversation(
                            &conv, model, bookmarks,
                        ))
                    });

                Ok(vec![DaemonResponse::Status {
                    uptime_seconds,
                    active_conversations,
                    current_conversation,
                }])
            }

            DaemonRequest::Health { client_version } => {
                let daemon_version = env!("CARGO_PKG_VERSION").to_string();
                let uptime_seconds = self.start_time.elapsed().as_secs();

                // Check version compatibility (same major.minor)
                let (compatible, warning) = Self::check_version_compatibility(
                    &daemon_version,
                    &client_version,
                );

                Ok(vec![DaemonResponse::Health {
                    daemon_version,
                    compatible,
                    warning,
                    uptime_seconds,
                }])
            }

            DaemonRequest::Shutdown => {
                info!("Shutdown requested by client");
                let _ = self.shutdown_tx.send(());
                Ok(vec![DaemonResponse::Success {
                    message: "Shutdown initiated".to_string(),
                }])
            }

            _ => {
                // Handle any future variants
                Ok(vec![DaemonResponse::Error {
                    message: "Unknown request variant".to_string(),
                }])
            }
        }
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

    /// Checks version compatibility between daemon and client.
    ///
    /// Returns (compatible, optional_warning). Compatible means same major.minor version.
    fn check_version_compatibility(daemon_version: &str, client_version: &str) -> (bool, Option<String>) {
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
