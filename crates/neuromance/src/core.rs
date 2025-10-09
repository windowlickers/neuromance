use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::Utc;
use futures::{FutureExt, StreamExt};
use log::{debug, info, warn};

use neuromance_client::{ClientError, LLMClient};
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatRequest, ChatResponse, ToolChoice};
use neuromance_common::tools::{ToolApproval, ToolCall};
use neuromance_tools::ToolExecutor;

use crate::error::CoreError;
use crate::events::{CoreEvent, EventCallback, ToolApprovalCallback};

/// Core orchestration layer for LLM conversations with tool execution
///
/// Core manages the conversation loop, including streaming, tool execution,
/// and event emission. It uses an event-driven architecture where a single
/// event callback receives all events (streaming, tool results, usage, etc.).
pub struct Core<C: LLMClient> {
    pub client: C,
    /// Enable streaming mode for chat responses
    pub streaming: bool,
    /// Total number of tool calls the LLM can make before returning to the user.
    pub max_turns: Option<u32>,
    /// Execute all tools regardless of their auto_approve value
    pub auto_approve_tools: bool,
    /// how the model selects which tool to call, if any.
    pub tool_choice: ToolChoice,
    /// Holds tools in ToolRegistry and executes tools
    pub tool_executor: ToolExecutor,
    /// Optional bi-directional callback for Tool approval
    pub tool_approval_callback: Option<ToolApprovalCallback>,
    /// Optional event callback for all Core events
    pub event_callback: Option<EventCallback>,
}

impl<C: LLMClient> Core<C> {
    pub fn new(client: C) -> Self {
        Self {
            client,
            streaming: false,
            max_turns: None,
            auto_approve_tools: false,
            tool_choice: ToolChoice::Auto,
            tool_executor: ToolExecutor::new(),
            tool_approval_callback: None,
            event_callback: None,
        }
    }

    /// Set callback for tool approval decisions
    /// Use when a tool is not auto-approved and `auto_approve_tools` is false.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use neuromance::{Core, ToolApproval};
    /// # use neuromance_client::openai::OpenAIClient;
    /// # let client: OpenAIClient = unimplemented!();
    /// let core = Core::new(client)
    ///     .with_tool_approval_callback(|tool_call| {
    ///         // Clone to move into async block (avoids lifetime issues)
    ///         let tool_call = tool_call.clone();
    ///         async move {
    ///             // Could prompt user, check policy, etc.
    ///             println!("Tool requested: {}", tool_call.function.name);
    ///             ToolApproval::Approved
    ///         }
    ///     });
    /// ```
    pub fn with_tool_approval_callback<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(&ToolCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolApproval> + Send + 'static,
    {
        self.tool_approval_callback =
            Some(Box::new(move |tool_call| Box::pin(callback(tool_call))));
        self
    }

    /// Callback for all CoreEvents
    /// This callback receives notifications about streaming content, tool execution
    /// results, and token usage.
    ///
    /// # Events
    /// - [`CoreEvent::Streaming`] - Content chunks as they arrive from the LLM
    /// - [`CoreEvent::ToolResult`] - Results from tool execution (success or failure)
    /// - [`CoreEvent::Usage`] - Token usage information from LLM responses
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use neuromance::{Core, CoreEvent};
    /// # use neuromance_client::openai::OpenAIClient;
    /// # let client: OpenAIClient = unimplemented!();
    /// let core = Core::new(client)
    ///     .with_event_callback(|event| async move {
    ///         match event {
    ///             CoreEvent::Streaming(chunk) => print!("{}", chunk),
    ///             CoreEvent::ToolResult { name, result, success } => {
    ///                 println!("Tool '{}' completed: {}", name, result);
    ///             }
    ///             CoreEvent::Usage(usage) => {
    ///                 println!("Tokens used: {}", usage.total_tokens);
    ///             }
    ///         }
    ///     });
    /// ```
    pub fn with_event_callback<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(CoreEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.event_callback = Some(Box::new(move |event| Box::pin(callback(event))));
        self
    }

    /// Enable streaming mode
    pub fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    /// Emit an event, catching any panics from the callback
    async fn emit_event(&self, event: CoreEvent) {
        if let Some(ref callback) = self.event_callback {
            // Use catch_unwind to prevent callback panics from propagating
            match std::panic::AssertUnwindSafe(callback(event.clone()))
                .catch_unwind()
                .await
            {
                Ok(()) => {}
                Err(e) => {
                    // Log the panic but continue execution
                    let panic_msg = if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };
                    warn!(
                        "Event callback panicked while processing {:?}: {}",
                        event, panic_msg
                    );
                }
            }
        }
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

        // Pre-allocate capacity for typical streaming responses
        // Average LLM response is ~200-500 chars, allocate for 1KB to reduce reallocations
        let mut accumulated_content = String::with_capacity(1024);
        let mut response_metadata = None;
        let mut role = None;
        // Most responses have 0-3 tool calls, pre-allocate for 4 to avoid most reallocations
        let mut tool_calls: Vec<ToolCall> = Vec::with_capacity(4);
        let mut finish_reason = None;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;

            // Accumulate content and emit event if callback present
            if let Some(ref content) = chunk.delta_content {
                accumulated_content.push_str(content);
                self.emit_event(CoreEvent::Streaming(content.clone())).await;
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

            // Emit usage event if callback present
            if let Some(ref usage) = response.usage {
                self.emit_event(CoreEvent::Usage(usage.clone())).await;
            }

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
                    callback(tool_call).await
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

                                // Emit tool result event
                                self.emit_event(CoreEvent::ToolResult {
                                    name: tool_name.clone(),
                                    result: result.clone(),
                                    success: true,
                                })
                                .await;

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
                                let error_msg = format!("Tool execution failed: {}", e);

                                // Emit tool result event
                                self.emit_event(CoreEvent::ToolResult {
                                    name: tool_name.clone(),
                                    result: error_msg.clone(),
                                    success: false,
                                })
                                .await;

                                // Add error as tool message
                                let error_message = Message::tool(
                                    conversation_id,
                                    error_msg,
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

#[cfg(test)]
mod tests {
    use super::*;
    use neuromance_client::openai::OpenAIClient;
    use neuromance_common::client::Config;

    /// Test that event callbacks handle panics gracefully
    #[tokio::test]
    async fn test_event_callback_panic_handling() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = OpenAIClient::new(config).expect("Failed to create client");

        let counter = Arc::new(tokio::sync::Mutex::new(0));
        let counter_clone = Arc::clone(&counter);

        let core = Core::new(client).with_event_callback(move |event| {
            let counter = Arc::clone(&counter_clone);
            async move {
                let mut count = counter.lock().await;
                *count += 1;
                if *count == 2 {
                    panic!("Intentional panic in event callback");
                }
                drop(event);
            }
        });

        core.emit_event(CoreEvent::Streaming("test1".to_string()))
            .await;
        core.emit_event(CoreEvent::Streaming("test2".to_string()))
            .await;
        core.emit_event(CoreEvent::Streaming("test3".to_string()))
            .await;

        let final_count = *counter.lock().await;
        assert_eq!(final_count, 3);
    }

    /// Test that tool approval callback is registered and can deny tools
    #[tokio::test]
    async fn test_tool_approval_callback() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = OpenAIClient::new(config).expect("Failed to create client");

        let core = Core::new(client).with_tool_approval_callback(|tool_call| {
            let tool_name = tool_call.function.name.clone();
            async move {
                if tool_name == "dangerous" {
                    ToolApproval::Denied("Not allowed".to_string())
                } else {
                    ToolApproval::Approved
                }
            }
        });

        assert!(core.tool_approval_callback.is_some());
    }

    /// Test Core without callbacks
    #[tokio::test]
    async fn test_core_without_callbacks() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = OpenAIClient::new(config).expect("Failed to create client");
        let core = Core::new(client);

        assert!(core.event_callback.is_none());
        assert!(core.tool_approval_callback.is_none());

        // Should not panic
        core.emit_event(CoreEvent::Streaming("test".to_string()))
            .await;
    }

    /// Test multiple event types
    #[tokio::test]
    async fn test_multiple_event_types() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = OpenAIClient::new(config).expect("Failed to create client");

        let events = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        let core = Core::new(client).with_event_callback(move |event| {
            let events = Arc::clone(&events_clone);
            async move {
                let event_type = match event {
                    CoreEvent::Streaming(_) => "streaming",
                    CoreEvent::ToolResult { .. } => "tool",
                    CoreEvent::Usage(_) => "usage",
                };
                events.lock().await.push(event_type);
            }
        });

        core.emit_event(CoreEvent::Streaming("chunk".to_string()))
            .await;
        core.emit_event(CoreEvent::ToolResult {
            name: "test".to_string(),
            result: "ok".to_string(),
            success: true,
        })
        .await;

        let captured = events.lock().await;
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], "streaming");
        assert_eq!(captured[1], "tool");
    }
}
