//! Unix socket server for handling client connections.
//!
//! Listens on a Unix domain socket and handles line-delimited JSON requests
//! from the `nm` CLI client.

use std::sync::Arc;
use std::time::Duration;

use log::{debug, error, info, warn};
use neuromance_common::protocol::{DaemonRequest, DaemonResponse};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{RwLock, mpsc};
use tokio::time::Instant;

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
}

impl Server {
    /// Creates a new server.
    pub fn new(
        manager: Arc<ConversationManager>,
        storage: Arc<Storage>,
        config: Arc<DaemonConfig>,
    ) -> Self {
        Self {
            manager,
            storage,
            config,
            last_activity: Arc::new(RwLock::new(Instant::now())),
            start_time: Instant::now(),
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

        // Remove existing socket file
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        // Bind Unix socket
        let listener = UnixListener::bind(socket_path)?;
        info!("Daemon listening on {}", socket_path.display());

        // Spawn inactivity checker
        let server_clone = Arc::clone(&self);
        tokio::spawn(async move {
            server_clone.check_inactivity().await;
        });

        // Accept connections
        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let server = Arc::clone(&self);
                    tokio::spawn(async move {
                        if let Err(e) = server.handle_connection(stream).await {
                            error!("Connection handling error: {e}");
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {e}");
                }
            }
        }
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
                            debug!("Received request: {request:?}");

                            // Handle request
                            match self.handle_request(request).await {
                                Ok(responses) => {
                                    // Write responses (may be multiple for streaming)
                                    for response in responses {
                                        if let Err(e) =
                                            Self::write_response(&mut writer, &response).await
                                        {
                                            error!("Failed to write response: {e}");
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Request handling error: {e}");
                                    let response = DaemonResponse::Error {
                                        message: e.to_string(),
                                    };
                                    let _ = Self::write_response(&mut writer, &response).await;
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Invalid request JSON: {e}");
                            let response = DaemonResponse::Error {
                                message: format!("Invalid JSON: {e}"),
                            };
                            let _ = Self::write_response(&mut writer, &response).await;
                        }
                    }
                }
                Err(e) => {
                    error!("Read error: {e}");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handles a daemon request.
    ///
    /// Returns a vector of responses (for streaming, multiple responses may be sent).
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
                    self.manager.get_messages(conversation_id, limit).await?;

                Ok(vec![DaemonResponse::Messages {
                    conversation_id: conv_id,
                    messages,
                    total_count,
                }])
            }

            DaemonRequest::ListConversations { limit } => {
                let conversations = self.manager.list_conversations(limit).await?;
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
                conversation_id: _,
                model_nickname: _,
            } => {
                // TODO: Implement model switching
                Ok(vec![DaemonResponse::Error {
                    message: "Model switching not yet implemented".to_string(),
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
                    .approve_tool(conversation_id, tool_call_id, approval)?;

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
                }])
            }

            DaemonRequest::Shutdown => {
                info!("Shutdown requested by client");
                std::process::exit(0);
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

    /// Periodically checks for inactivity and shuts down if timeout exceeded.
    async fn check_inactivity(self: Arc<Self>) {
        let timeout = Duration::from_secs(self.config.settings.inactivity_timeout);
        let check_interval = Duration::from_secs(60);

        loop {
            tokio::time::sleep(check_interval).await;

            let last_activity = *self.last_activity.read().await;
            if last_activity.elapsed() > timeout {
                info!(
                    "Shutting down due to inactivity ({}s)",
                    last_activity.elapsed().as_secs()
                );
                std::process::exit(0);
            }
        }
    }
}
