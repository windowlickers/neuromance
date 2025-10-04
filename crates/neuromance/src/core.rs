use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::Utc;
use futures::StreamExt;
use log::{debug, info};

use neuromance_client::{ClientError, LLMClient};
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatRequest, ChatResponse, ToolChoice};
use neuromance_common::tools::{ToolApproval, ToolCall};
use neuromance_tools::ToolExecutor;

use crate::error::CoreError;

/// Type alias for tool approval callback functions
pub type ToolApprovalCallback = Box<dyn Fn(&ToolCall) -> ToolApproval + Send + Sync>;

/// Type alias for streaming content callback functions
pub type StreamingCallback = Box<dyn Fn(&str) + Send + Sync>;

pub struct Core<C: LLMClient> {
    pub client: C,
    // REVIEW could just work off common::client::Config
    // pub config: CoreConfig,
    pub max_turns: Option<u32>,
    /// Execute all tools regardless of their auto_approve value
    pub auto_approve_tools: bool,
    pub tool_choice: ToolChoice,
    /// Enable streaming mode for chat responses
    pub streaming: bool,

    pub tool_executor: ToolExecutor,
    pub tool_approval_callback: Option<ToolApprovalCallback>,
    /// Optional callback for streaming content chunks
    pub streaming_callback: Option<StreamingCallback>,
}

impl<C: LLMClient> Core<C> {
    pub fn new(client: C) -> Self {
        Self {
            client,
            max_turns: None,
            auto_approve_tools: false,
            tool_choice: ToolChoice::Auto,
            streaming: false,
            tool_executor: ToolExecutor::new(),
            tool_approval_callback: None,
            streaming_callback: None,
        }
    }

    pub fn with_tool_approval_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&ToolCall) -> ToolApproval + Send + Sync + 'static,
    {
        self.tool_approval_callback = Some(Box::new(callback));
        self
    }

    pub fn with_streaming<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.streaming = true;
        self.streaming_callback = Some(Box::new(callback));
        self
    }

    /// Send a chat request with retry logic for transient failures
    async fn chat_with_retry(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut last_error = None;

        let config = self.client.config();

        for attempt in 0..=config.retry_config.max_retries {
            match self.client.chat(request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    // Check if this is a retryable error
                    let is_retryable = e
                        .downcast_ref::<ClientError>()
                        .map(|client_err| client_err.is_retryable())
                        .unwrap_or(false);

                    if attempt < config.retry_config.max_retries && is_retryable {
                        debug!(
                            "Request failed (attempt {}), retrying in {:?}: {}",
                            attempt + 1,
                            config.retry_config.initial_delay,
                            e
                        );
                        last_error = Some(e);
                        tokio::time::sleep(config.retry_config.initial_delay).await;
                        continue;
                    }
                    // Non-retryable error or max attempts reached
                    last_error = Some(e);
                    break;
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Send a streaming chat request and accumulate the response
    async fn chat_stream_accumulated(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut stream = self.client.chat_stream(request).await?;

        let mut accumulated_content = String::new();
        let mut response_metadata = None;
        let mut role = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut finish_reason = None;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;

            // Accumulate content and invoke callback if present
            if let Some(ref content) = chunk.delta_content {
                accumulated_content.push_str(content);
                if let Some(ref callback) = self.streaming_callback {
                    callback(content);
                }
            }

            // Capture role from first chunk
            if role.is_none() {
                role = chunk.delta_role;
            }

            // Merge tool call deltas
            if let Some(ref delta_tool_calls) = chunk.delta_tool_calls {
                debug!("Received {} tool call delta(s)", delta_tool_calls.len());
                tool_calls = ToolCall::merge_deltas(tool_calls, delta_tool_calls);
            }

            // Capture finish reason
            if chunk.finish_reason.is_some() {
                finish_reason = chunk.finish_reason;
            }

            // Store metadata from last chunk
            response_metadata = Some(chunk);
        }

        // Get the conversation_id from the first message in the request
        let conversation_id = request
            .messages
            .first()
            .ok_or_else(|| anyhow::anyhow!("Request must contain at least one message"))?
            .conversation_id;

        // Construct the final response
        let last_chunk =
            response_metadata.ok_or_else(|| anyhow::anyhow!("Stream ended without any chunks"))?;

        let message = Message {
            id: uuid::Uuid::new_v4(),
            conversation_id,
            role: role.unwrap_or(MessageRole::Assistant),
            content: accumulated_content,
            tool_calls: tool_calls.into_iter().collect(),
            tool_call_id: None,
            name: None,
            timestamp: Utc::now(),
            metadata: last_chunk.metadata,
        };

        Ok(ChatResponse {
            message,
            model: last_chunk.model,
            usage: last_chunk.usage,
            finish_reason,
            created_at: last_chunk.created_at,
            response_id: last_chunk.response_id,
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Internal method to handle the chat loop with tool execution
    pub async fn chat_with_tool_loop(&self, mut messages: Vec<Message>) -> Result<Vec<Message>> {
        let mut turn_count = 0;
        let mut pending_tool_calls: HashSet<String> = HashSet::new();
        let start_time = Instant::now();
        let mut messages_arc: Arc<[Message]> = messages.clone().into();

        loop {
            // Create chat request
            let request = ChatRequest::from((self.client.config(), messages_arc.clone()))
                .with_tools(self.tool_executor.get_all_tools())
                .with_tool_choice(self.tool_choice.clone());

            info!(
                "Executing chat turn ({}/{})",
                turn_count + 1,
                self.max_turns
                    .map_or("unlimited".to_string(), |max| max.to_string()),
            );
            debug!(
                "Chat request:\n {}",
                serde_json::to_string_pretty(&request)?
            );

            // Send to LLM with retry logic or streaming
            let response = if self.streaming {
                self.chat_stream_accumulated(&request).await?
            } else {
                self.chat_with_retry(&request).await?
            };

            debug!("Received response from LLM");
            debug!(
                "Assistant Response:\n {}",
                serde_json::to_string_pretty(&response)?
            );

            // Extract data we need before consuming the message
            let conversation_id = response.message.conversation_id;
            let tool_calls = response.message.tool_calls.clone();
            let tool_calls_count = tool_calls.len();

            // Add assistant message to history
            messages.push(response.message);

            // NOTE: this is an exit condition
            // Check if tools were called,
            if tool_calls.is_empty() {
                let duration = start_time.elapsed();
                debug!(
                    "No tool calls in response, chat completed in {} turns ({:.2?})",
                    turn_count + 1,
                    duration
                );
                // Exit condition: No tools called, return all messages
                return Ok(messages);
            }

            // Execute tool calls
            for tool_call in &tool_calls {
                let tool_name = &tool_call.function.name;
                let call_id = &tool_call.id;

                // Track pending tool call
                pending_tool_calls.insert(tool_call.id.clone());

                debug!("Tool Name: {} (id: {})", tool_name, call_id);
                debug!("Tool Arguments: {:?}", tool_call.function.arguments);

                // Check if tool is auto-approved
                let is_auto_approved =
                    self.auto_approve_tools || self.tool_executor.is_tool_auto_approved(tool_name);
                debug!("Tool auto-approved: {}", is_auto_approved);

                let approval = if is_auto_approved {
                    ToolApproval::Approved
                } else if let Some(ref callback) = self.tool_approval_callback {
                    callback(tool_call)
                } else {
                    // No callback provided and not auto-approved, deny by default
                    ToolApproval::Denied("No approval mechanism configured".to_string())
                };

                debug!("Tool Approval Status: {:?}", approval);

                match approval {
                    ToolApproval::Approved => {
                        debug!("Executing tool: {}", tool_name);
                        // Execute the tool
                        match self.tool_executor.execute_tool(tool_call).await {
                            Ok(result) => {
                                debug!("Tool {} executed successfully", tool_name);
                                debug!("Tool result: {}", result);
                                // Add tool result as a message
                                let tool_message = Message::tool(
                                    conversation_id,
                                    result,
                                    tool_call.id.clone(),
                                    tool_call.function.name.clone(),
                                )?;
                                messages.push(tool_message);
                                pending_tool_calls.remove(&tool_call.id);
                            }
                            Err(e) => {
                                debug!("Tool {} execution failed: {}", tool_name, e);
                                // Add error as tool message
                                let error_message = Message::tool(
                                    conversation_id,
                                    format!("Tool execution failed: {}", e),
                                    tool_call.id.clone(),
                                    tool_call.function.name.clone(),
                                )?;
                                messages.push(error_message);
                                pending_tool_calls.remove(&tool_call.id);
                            }
                        }
                    }
                    ToolApproval::Denied(reason) => {
                        debug!("Tool {} denied: {}", tool_name, reason);
                        // Tool not approved
                        let denial_message = Message::tool(
                            conversation_id,
                            format!("Tool execution denied: {}", reason),
                            tool_call.id.clone(),
                            tool_call.function.name.clone(),
                        )?;
                        messages.push(denial_message);
                        pending_tool_calls.remove(&tool_call.id);
                    }
                    ToolApproval::Quit => {
                        debug!("User quit during tool approval");
                        // User requested to quit
                        return Err(CoreError::Other(anyhow::anyhow!(
                            "User quit during tool approval"
                        ))
                        .into());
                    }
                }
            }

            debug!(
                "Completed processing {} tool calls, continuing conversation",
                tool_calls_count
            );

            // Update Arc with new messages after tool execution
            messages_arc = messages.clone().into();

            // Sanity check: ensure all tool calls were handled
            if !pending_tool_calls.is_empty() {
                debug!(
                    "Warning: {} tool calls still pending",
                    pending_tool_calls.len()
                );
            }

            // Increment turn count after processing tool calls
            turn_count += 1;

            // Check if we've exceeded max turns
            if let Some(max) = self.max_turns
                && turn_count >= max
            {
                return Err(CoreError::MaxTurnsExceeded(format!(
                    "Exceeded maximum turns: {} (configured max: {})",
                    turn_count, max
                ))
                .into());
            }
        }
    }
}
