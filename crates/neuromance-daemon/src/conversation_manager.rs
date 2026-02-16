//! Conversation and Core instance management.
//!
//! Manages active Core instances for each conversation, creates new conversations,
//! handles message sending with tool execution, and coordinates tool approval.

use std::collections::HashSet;
use std::env;
use std::sync::Arc;

use dashmap::DashMap;
use neuromance::Core;
use neuromance_client::{AnthropicClient, LLMClient, OpenAIClient, ResponsesClient};
use neuromance_common::protocol::{ConversationSummary, DaemonResponse};
use neuromance_common::{Config, Conversation, Message, MessageRole, ToolApproval};
use neuromance_tools::ToolImplementation;
use neuromance_tools::mcp::McpManager;
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::config::DaemonConfig;
use crate::error::{DaemonError, Result};
use crate::storage::Storage;

/// Core wrapper that abstracts over different client types.
///
/// This allows storing client configurations for different providers.
#[derive(Clone)]
pub enum ClientType {
    Anthropic(AnthropicClient),
    OpenAI(OpenAIClient),
    Responses(ResponsesClient),
}

/// Configuration for the chat loop execution.
struct ChatLoopConfig {
    conversation_id: String,
    response_tx: mpsc::Sender<DaemonResponse>,
    pending_approvals: Arc<DashMap<(String, String), oneshot::Sender<ToolApproval>>>,
    auto_approve: bool,
    max_turns: Option<u32>,
    thinking_budget: u32,
    tools: Vec<Arc<dyn ToolImplementation>>,
}

/// Manages conversations and their associated Core instances.
pub struct ConversationManager {
    /// Storage backend
    storage: Arc<Storage>,

    /// Daemon configuration (models, settings)
    config: Arc<DaemonConfig>,

    /// Active clients (conversation ID -> Client)
    ///
    /// Uses `DashMap` for concurrent access without blocking
    clients: DashMap<Uuid, ClientType>,

    /// Per-conversation locks to serialize `send_message` operations
    conversation_locks: DashMap<Uuid, Arc<Mutex<()>>>,

    /// Pending tool approvals (conversation ID, tool call ID) -> response channel
    pending_approvals: Arc<DashMap<(String, String), oneshot::Sender<ToolApproval>>>,

    /// Model per conversation (conversation ID -> model nickname)
    conversation_models: DashMap<Uuid, String>,

    /// MCP server manager (None if no MCP servers configured)
    mcp_manager: Option<Arc<McpManager>>,

    /// Built-in tools to register on each Core instance
    builtin_tools: Vec<Arc<dyn ToolImplementation>>,
}

impl ConversationManager {
    /// Creates a new conversation manager.
    pub fn new(
        storage: Arc<Storage>,
        config: Arc<DaemonConfig>,
        mcp_manager: Option<Arc<McpManager>>,
        builtin_tools: Vec<Arc<dyn ToolImplementation>>,
    ) -> Self {
        Self {
            storage,
            config,
            clients: DashMap::new(),
            conversation_locks: DashMap::new(),
            pending_approvals: Arc::new(DashMap::new()),
            conversation_models: DashMap::new(),
            mcp_manager,
            builtin_tools,
        }
    }

    /// Creates a new conversation.
    ///
    /// # Arguments
    ///
    /// * `model` - Optional model nickname (uses active model if None)
    /// * `system_message` - Optional system message to initialize with
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model is not found
    /// - Storage operations fail
    #[instrument(skip(self, system_message), fields(model = ?model, has_system = system_message.is_some()))]
    pub async fn create_conversation(
        &self,
        model: Option<String>,
        system_message: Option<String>,
    ) -> Result<ConversationSummary> {
        let model_nickname = model
            .as_deref()
            .unwrap_or(&self.config.active_model)
            .to_string();

        // Validate model exists
        self.config.get_model(&model_nickname)?;

        let mut conversation = Conversation::new();

        // Add system message if provided
        if let Some(system_msg) = system_message {
            let msg = conversation.system_message(system_msg);
            conversation
                .add_message(msg)
                .map_err(|e| DaemonError::Other(e.to_string()))?;
        }

        // Persist model in conversation metadata
        conversation.metadata.insert(
            "model".to_string(),
            serde_json::Value::String(model_nickname.clone()),
        );

        // Save conversation
        let conv_for_save = conversation.clone();
        self.storage
            .run(move |s| {
                s.save_conversation(&conv_for_save)?;
                s.set_active_conversation(&conv_for_save.id)
            })
            .await?;

        // Track model for this conversation (in-memory cache)
        self.conversation_models
            .insert(conversation.id, model_nickname.clone());

        info!(
            conversation_id = %conversation.id,
            model = %model_nickname,
            "Created new conversation"
        );

        let conv_id = conversation.id;
        let bookmarks = self
            .storage
            .run(move |s| s.get_conversation_bookmarks(&conv_id))
            .await?;
        Ok(ConversationSummary::from_conversation(
            &conversation,
            model_nickname,
            bookmarks,
        ))
    }

    /// Sends a message to a conversation and executes the tool loop.
    ///
    /// Streams responses via the response channel.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The conversation ID (or None for active conversation)
    /// * `content` - The message content
    /// * `response_tx` - Channel to send `DaemonResponse` events
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Conversation not found
    /// - Core creation fails
    /// - Message sending fails
    #[instrument(skip(self, content, response_tx), fields(conversation_id = ?conversation_id, message_length = content.len()))]
    pub async fn send_message(
        &self,
        conversation_id: Option<String>,
        content: String,
        response_tx: mpsc::Sender<DaemonResponse>,
    ) -> Result<()> {
        let id = self.resolve_or_active(conversation_id)?;

        debug!(conversation_id = %id, "Resolved conversation ID");

        // Acquire per-conversation lock to prevent concurrent modifications
        let lock = self
            .conversation_locks
            .entry(id)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _guard = lock.lock().await;

        // Load conversation
        let mut conversation = self.storage.run(move |s| s.load_conversation(&id)).await?;
        debug!(
            conversation_id = %id,
            message_count = conversation.messages.len(),
            "Loaded conversation"
        );

        // Restore model from metadata if not cached (e.g. after restart)
        self.populate_model_from_metadata(&conversation);

        // Add user message
        let user_msg = conversation.user_message(&content);
        conversation
            .add_message(user_msg)
            .map_err(|e| DaemonError::Other(e.to_string()))?;

        // Get or create client
        let client_enum = self.get_or_create_client(&id).await?;

        // Prepare messages for Core
        let messages: Vec<Message> = conversation.messages.to_vec();

        // Collect tools: built-in + MCP
        let mut tools: Vec<Arc<dyn ToolImplementation>> = self.builtin_tools.clone();

        if let Some(ref mcp) = self.mcp_manager {
            match mcp.get_all_tools().await {
                Ok(mcp_tools) => tools.extend(mcp_tools),
                Err(e) => {
                    warn!("Failed to load MCP tools: {e}");
                }
            }
        }

        let conv_id_str = id.to_string();
        let config = ChatLoopConfig {
            conversation_id: conv_id_str.clone(),
            response_tx: response_tx.clone(),
            pending_approvals: Arc::clone(&self.pending_approvals),
            auto_approve: self.config.settings.auto_approve_tools,
            max_turns: Some(self.config.settings.max_turns),
            thinking_budget: self.config.settings.thinking_budget,
            tools,
        };

        let updated_messages = match client_enum {
            ClientType::Anthropic(client) => {
                Self::execute_chat_loop(client, messages, config).await
            }
            ClientType::OpenAI(client) => Self::execute_chat_loop(client, messages, config).await,
            ClientType::Responses(client) => {
                Self::execute_chat_loop(client, messages, config).await
            }
        }
        .map_err(|e| DaemonError::Core(e.to_string()))?;

        debug!(
            conversation_id = %id,
            message_count = updated_messages.len(),
            "Chat loop completed"
        );

        // Update conversation with new messages
        let existing_ids: HashSet<Uuid> = conversation.messages.iter().map(|m| m.id).collect();
        for msg in &updated_messages {
            if !existing_ids.contains(&msg.id) {
                conversation
                    .add_message(msg.clone())
                    .map_err(|e| DaemonError::Other(e.to_string()))?;
            }
        }

        // Save updated conversation
        let conv_for_save = conversation.clone();
        self.storage
            .run(move |s| s.save_conversation(&conv_for_save))
            .await?;
        debug!(conversation_id = %id, "Saved conversation");

        // Send completion notification
        if let Some(last_msg) = updated_messages.last()
            && last_msg.role == MessageRole::Assistant
        {
            let _ = response_tx
                .send(DaemonResponse::MessageCompleted {
                    conversation_id: conv_id_str,
                    message: Box::new(last_msg.clone()),
                })
                .await;
        }

        Ok(())
    }

    /// Executes the chat loop with event callbacks.
    async fn execute_chat_loop<C>(
        client: C,
        messages: Vec<Message>,
        config: ChatLoopConfig,
    ) -> anyhow::Result<Vec<Message>>
    where
        C: LLMClient + Send + Sync,
    {
        let conv_id = config.conversation_id.clone();
        let tx = config.response_tx.clone();

        let mut core = Core::new(client)
            .with_streaming()
            .with_thinking_budget(config.thinking_budget);

        core.auto_approve_tools = config.auto_approve;
        core.max_turns = config.max_turns;

        for tool in config.tools {
            core.tool_executor.add_tool_arc(tool);
        }

        core = core.with_event_callback(move |event| {
            let tx = tx.clone();
            let conv_id = conv_id.clone();
            async move {
                match event {
                    neuromance::CoreEvent::Streaming(content) => {
                        let _ = tx
                            .send(DaemonResponse::StreamChunk {
                                conversation_id: conv_id,
                                content,
                            })
                            .await;
                    }
                    neuromance::CoreEvent::ToolResult {
                        name,
                        result,
                        success,
                    } => {
                        let _ = tx
                            .send(DaemonResponse::ToolResult {
                                conversation_id: conv_id,
                                tool_name: name,
                                result,
                                success,
                            })
                            .await;
                    }
                    neuromance::CoreEvent::Usage(usage) => {
                        let _ = tx
                            .send(DaemonResponse::Usage {
                                conversation_id: conv_id,
                                usage,
                            })
                            .await;
                    }
                }
            }
        });

        let conv_id_for_approval = config.conversation_id;
        let response_tx = config.response_tx;
        let pending_approvals = config.pending_approvals;
        core = core.with_tool_approval_callback(move |tool_call| {
            let conv_id = conv_id_for_approval.clone();
            let tool_call = tool_call.clone();
            let response_tx = response_tx.clone();
            let pending_approvals = Arc::clone(&pending_approvals);

            async move {
                let (tx, rx) = oneshot::channel();

                let key = (conv_id.clone(), tool_call.id.clone());
                pending_approvals.insert(key, tx);

                let _ = response_tx
                    .send(DaemonResponse::ToolApprovalRequest {
                        conversation_id: conv_id,
                        tool_call,
                    })
                    .await;

                rx.await.unwrap_or_else(|_| {
                    warn!("Tool approval channel closed unexpectedly");
                    ToolApproval::Denied("Approval channel closed".to_string())
                })
            }
        });

        core.chat_with_tool_loop(messages).await
    }

    /// Responds to a pending tool approval request.
    ///
    /// # Errors
    ///
    /// Returns an error if the approval request is not found.
    #[instrument(skip(self), fields(conversation_id = %conversation_id, tool_call_id = %tool_call_id, approval = ?approval))]
    pub fn approve_tool(
        &self,
        conversation_id: &str,
        tool_call_id: &str,
        approval: ToolApproval,
    ) -> Result<()> {
        let key = (conversation_id.to_string(), tool_call_id.to_string());

        if let Some((_, tx)) = self.pending_approvals.remove(&key) {
            let _ = tx.send(approval);
            debug!("Tool approval sent");
            Ok(())
        } else {
            warn!("No pending approval found");
            Err(DaemonError::Other(format!(
                "No pending approval for conversation {conversation_id}, tool call {tool_call_id}"
            )))
        }
    }

    /// Gets or creates a client for a conversation.
    #[instrument(skip(self), fields(conversation_id = %conversation_id))]
    async fn get_or_create_client(&self, conversation_id: &Uuid) -> Result<ClientType> {
        // Check if client already exists
        if let Some(entry) = self.clients.get(conversation_id) {
            debug!("Using existing client");
            return Ok(entry.value().clone());
        }

        // Determine model for this conversation
        let model_nickname = self
            .conversation_models
            .get(conversation_id)
            .map_or_else(|| self.config.active_model.clone(), |e| e.value().clone());

        // Get model profile
        let model_profile = self.config.get_model(&model_nickname)?;

        debug!(
            model = %model_nickname,
            provider = %model_profile.provider,
            "Creating new client"
        );

        // Get API key from environment
        let api_key = env::var(&model_profile.api_key_env).map_err(|_| {
            DaemonError::Config(format!(
                "Environment variable {} not set for model {}",
                model_profile.api_key_env, model_nickname
            ))
        })?;

        // Create config
        let mut config =
            Config::new(&model_profile.provider, &model_profile.model).with_api_key(api_key);

        // Set custom base URL if specified
        if let Some(ref base_url) = model_profile.base_url {
            config.base_url = Some(base_url.clone());
        }

        // Create client based on provider
        let client = match model_profile.provider.as_str() {
            "anthropic" => {
                let client =
                    AnthropicClient::new(config).map_err(|e| DaemonError::Client(e.to_string()))?;
                ClientType::Anthropic(client)
            }
            "openai" => {
                let client =
                    OpenAIClient::new(config).map_err(|e| DaemonError::Client(e.to_string()))?;
                ClientType::OpenAI(client)
            }
            "responses" => {
                let client =
                    ResponsesClient::new(config).map_err(|e| DaemonError::Client(e.to_string()))?;
                ClientType::Responses(client)
            }
            _ => {
                return Err(DaemonError::Config(format!(
                    "Unsupported provider: {}",
                    model_profile.provider
                )));
            }
        };

        // Insert and return cloned client
        self.clients.insert(*conversation_id, client.clone());
        info!(
            conversation_id = %conversation_id,
            model = %model_nickname,
            provider = %model_profile.provider,
            "Client created and cached"
        );
        Ok(client)
    }

    /// Deletes a conversation and cleans up all associated state.
    ///
    /// Removes bookmarks, clears active conversation if it matches,
    /// deletes the conversation file, and cleans up in-memory state.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The conversation ID cannot be resolved
    /// - The conversation cannot be loaded
    /// - Storage operations fail
    #[instrument(skip(self), fields(conversation_id = %conversation_id))]
    pub async fn delete_conversation(
        &self,
        conversation_id: &str,
    ) -> Result<(ConversationSummary, Vec<String>)> {
        let conv_id_str = conversation_id.to_string();
        let (id, conversation, bookmarks, removed_bookmarks) = self
            .storage
            .run(move |s| {
                let id = s.resolve_conversation_id(&conv_id_str)?;
                let conversation = s.load_conversation(&id)?;
                let bookmarks = s.get_conversation_bookmarks(&id)?;
                let removed = s.remove_bookmarks_for_conversation(&id)?;

                if let Ok(Some(active_id)) = s.get_active_conversation()
                    && active_id == id
                {
                    s.clear_active_conversation()?;
                }

                s.delete_conversation(&id)?;
                Ok((id, conversation, bookmarks, removed))
            })
            .await?;

        self.populate_model_from_metadata(&conversation);
        let model = self.get_conversation_model(&id);
        let summary = ConversationSummary::from_conversation(&conversation, model, bookmarks);

        // Clean up in-memory state
        self.clients.remove(&id);
        self.conversation_locks.remove(&id);
        self.conversation_models.remove(&id);

        info!(
            conversation_id = %id,
            removed_bookmarks = ?removed_bookmarks,
            "Deleted conversation"
        );

        Ok((summary, removed_bookmarks))
    }

    /// Lists conversations.
    ///
    /// # Errors
    ///
    /// Returns an error if storage operations fail.
    pub async fn list_conversations(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<ConversationSummary>> {
        let loaded = self
            .storage
            .run(move |s| {
                let ids = s.list_conversations()?;

                let mut loaded: Vec<(Uuid, Conversation, Vec<String>)> = ids
                    .into_iter()
                    .filter_map(|id| match s.load_conversation(&id) {
                        Ok(conv) => {
                            let bm = s.get_conversation_bookmarks(&id).ok().unwrap_or_default();
                            Some((id, conv, bm))
                        }
                        Err(e) => {
                            warn!(
                                conversation_id = %id,
                                error = %e,
                                "Skipping unloadable conversation"
                            );
                            None
                        }
                    })
                    .collect();

                loaded.sort_by(|a, b| b.1.updated_at.cmp(&a.1.updated_at));

                if let Some(limit) = limit {
                    loaded.truncate(limit);
                }

                Ok(loaded)
            })
            .await?;

        let mut summaries = Vec::with_capacity(loaded.len());
        for (id, conv, bookmarks) in &loaded {
            self.populate_model_from_metadata(conv);
            let model = self.get_conversation_model(id);
            summaries.push(ConversationSummary::from_conversation(
                conv,
                model,
                bookmarks.clone(),
            ));
        }

        Ok(summaries)
    }

    /// Gets messages from a conversation.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversation is not found.
    pub async fn get_messages(
        &self,
        conversation_id: Option<String>,
        limit: Option<usize>,
    ) -> Result<(Vec<Message>, usize, String)> {
        let id = self.resolve_or_active(conversation_id)?;

        let conversation = self.storage.run(move |s| s.load_conversation(&id)).await?;
        let total_count = conversation.messages.len();

        let mut messages = conversation.messages.to_vec();

        // Apply limit (most recent first)
        if let Some(limit) = limit {
            messages = messages.into_iter().rev().take(limit).rev().collect();
        }

        Ok((messages, total_count, id.to_string()))
    }

    /// Resolves a conversation ID from an optional string, falling back to the active conversation.
    fn resolve_or_active(&self, id: Option<String>) -> Result<Uuid> {
        if let Some(id_str) = id {
            self.storage.resolve_conversation_id(&id_str)
        } else {
            self.storage
                .get_active_conversation()?
                .ok_or(DaemonError::NoActiveConversation)
        }
    }

    /// Returns the number of persisted conversations.
    pub async fn conversation_count(&self) -> Result<usize> {
        let count = self
            .storage
            .run(|s| Ok(s.list_conversations()?.len()))
            .await?;
        Ok(count)
    }

    /// Gets the model nickname for a conversation.
    ///
    /// Checks in-memory cache first, then falls back to conversation
    /// metadata on disk (handles daemon restart). Returns the active
    /// model if no specific model is set.
    #[must_use]
    pub fn get_conversation_model(&self, conversation_id: &Uuid) -> String {
        // Check in-memory cache
        if let Some(model) = self.conversation_models.get(conversation_id) {
            return model.value().clone();
        }

        // Fall back to conversation metadata (handles daemon restart)
        if let Ok(conv) = self.storage.load_conversation(conversation_id)
            && let Some(model) = conv.metadata.get("model").and_then(|v| v.as_str())
        {
            let model = model.to_string();
            self.conversation_models
                .insert(*conversation_id, model.clone());
            return model;
        }

        self.config.active_model.clone()
    }

    /// Populates the in-memory model cache from conversation metadata.
    fn populate_model_from_metadata(&self, conversation: &Conversation) {
        if !self.conversation_models.contains_key(&conversation.id)
            && let Some(model) = conversation.metadata.get("model").and_then(|v| v.as_str())
        {
            self.conversation_models
                .insert(conversation.id, model.to_string());
        }
    }

    /// Switches the model for a conversation.
    ///
    /// Preserves conversation history but changes which model will be used
    /// for future messages. The client cache is invalidated to force creation
    /// of a new client with the updated model on the next message.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The conversation ID (or None for active conversation)
    /// * `model_nickname` - The new model nickname to use
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Conversation not found
    /// - Model not found in configuration
    /// - Storage operations fail
    #[instrument(skip(self), fields(conversation_id = ?conversation_id, model_nickname = %model_nickname))]
    pub async fn switch_model(
        &self,
        conversation_id: Option<String>,
        model_nickname: String,
    ) -> Result<ConversationSummary> {
        let id = self.resolve_or_active(conversation_id)?;

        // Validate model exists
        self.config.get_model(&model_nickname)?;

        // Update conversation model mapping (in-memory cache)
        self.conversation_models.insert(id, model_nickname.clone());

        // Invalidate cached client to force recreation with new model
        self.clients.remove(&id);

        // Persist model in conversation metadata
        let nickname_for_save = model_nickname.clone();
        let conversation = self
            .storage
            .run(move |s| {
                let mut conv = s.load_conversation(&id)?;
                conv.metadata.insert(
                    "model".to_string(),
                    serde_json::Value::String(nickname_for_save),
                );
                conv.touch();
                s.save_conversation(&conv)?;
                let bookmarks = s.get_conversation_bookmarks(&id)?;
                Ok((conv, bookmarks))
            })
            .await?;

        info!(
            conversation_id = %id,
            model = %model_nickname,
            "Switched conversation model"
        );
        Ok(ConversationSummary::from_conversation(
            &conversation.0,
            model_nickname,
            conversation.1,
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::config::DaemonConfig;
    use tempfile::TempDir;

    /// Creates a test conversation manager with temporary storage.
    fn test_manager() -> (ConversationManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(Storage::new_test(temp_dir.path()));

        let config_toml = r#"
active_model = "test-model"

[[models]]
nickname = "test-model"
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
api_key_env = "ANTHROPIC_API_KEY"

[[models]]
nickname = "other-model"
provider = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
        "#;

        let config: DaemonConfig = toml::from_str(config_toml).unwrap();
        let manager = ConversationManager::new(storage, Arc::new(config), None, Vec::new());

        (manager, temp_dir)
    }

    #[tokio::test]
    async fn test_delete_conversation() {
        let (manager, _temp) = test_manager();

        let summary = manager.create_conversation(None, None).await.unwrap();
        let conv_id = summary.id;

        // Set a bookmark
        let uuid = Uuid::parse_str(&conv_id).unwrap();
        manager.storage.set_bookmark("test-bm", &uuid).unwrap();

        // Delete the conversation
        let (deleted, removed_bookmarks) = manager.delete_conversation(&conv_id).await.unwrap();
        assert_eq!(deleted.id, conv_id);
        assert_eq!(removed_bookmarks, vec!["test-bm".to_string()]);

        // Verify conversation is gone from storage
        assert!(manager.storage.load_conversation(&uuid).is_err());

        // Verify active conversation is cleared
        assert!(manager.storage.get_active_conversation().unwrap().is_none());

        // Verify in-memory state is cleaned up
        assert!(!manager.clients.contains_key(&uuid));
        assert!(!manager.conversation_locks.contains_key(&uuid));
        assert!(!manager.conversation_models.contains_key(&uuid));
    }

    #[tokio::test]
    async fn test_delete_conversation_not_found() {
        let (manager, _temp) = test_manager();
        let result = manager.delete_conversation("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_preserves_other_active() {
        let (manager, _temp) = test_manager();

        // Create two conversations
        let first = manager.create_conversation(None, None).await.unwrap();
        let second = manager.create_conversation(None, None).await.unwrap();

        // Active conversation is now `second`
        let second_uuid = Uuid::parse_str(&second.id).unwrap();
        assert_eq!(
            manager.storage.get_active_conversation().unwrap(),
            Some(second_uuid)
        );

        // Delete the first — should not clear active
        manager.delete_conversation(&first.id).await.unwrap();
        assert_eq!(
            manager.storage.get_active_conversation().unwrap(),
            Some(second_uuid)
        );
    }

    #[tokio::test]
    async fn test_switch_model() {
        let (manager, _temp) = test_manager();

        // Create a conversation
        let summary = manager
            .create_conversation(None, Some("test system".to_string()))
            .await
            .unwrap();
        assert_eq!(summary.model, "test-model");

        let conv_id = summary.id.clone();

        // Switch model
        let updated = manager
            .switch_model(Some(conv_id.clone()), "other-model".to_string())
            .await
            .unwrap();
        assert_eq!(updated.model, "other-model");
        assert_eq!(updated.id, conv_id);

        // Verify model mapping was updated
        let uuid = uuid::Uuid::parse_str(&conv_id).unwrap();
        assert_eq!(
            manager.conversation_models.get(&uuid).unwrap().value(),
            "other-model"
        );

        // Verify client cache was cleared
        assert!(!manager.clients.contains_key(&uuid));
    }

    #[tokio::test]
    async fn test_switch_model_invalid_model() {
        let (manager, _temp) = test_manager();

        // Create a conversation
        let summary = manager.create_conversation(None, None).await.unwrap();

        // Try to switch to non-existent model
        let result = manager
            .switch_model(Some(summary.id.clone()), "nonexistent".to_string())
            .await;

        assert!(matches!(&result, Err(DaemonError::ModelNotFound(name)) if name == "nonexistent"));
    }

    #[tokio::test]
    async fn test_switch_model_no_active() {
        let (manager, _temp) = test_manager();

        // Try to switch without active conversation
        let result = manager.switch_model(None, "other-model".to_string()).await;

        assert!(result.is_err());
        assert!(matches!(result, Err(DaemonError::NoActiveConversation)));
    }
}
