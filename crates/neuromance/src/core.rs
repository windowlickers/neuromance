use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use async_stream::try_stream;
use chrono::Utc;
use futures::{Stream, StreamExt};
use log::{debug, info};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use neuromance_client::LLMClient;
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatRequest, ChatResponse, ToolChoice};
use neuromance_common::features::ThinkingMode;
use neuromance_common::tools::{ToolApproval, ToolCall};
use neuromance_tools::ToolExecutor;

use crate::error::CoreError;
use crate::events::{CoreEvent, ToolApprovalCallback, TurnCallback};
use crate::stats::RunStats;

/// Core orchestration layer for LLM conversations with tool execution.
///
/// [`Core::run`] returns a [`Stream`] of [`CoreEvent`]s. The stream borrows
/// `&mut Core` for its lifetime and terminates with [`CoreEvent::Completed`].
pub struct Core<C: LLMClient> {
    pub client: C,
    /// Enable streaming mode for chat responses.
    pub streaming: bool,
    /// Total number of tool calls the LLM can make before returning to the user.
    pub max_turns: Option<u32>,
    /// Execute all tools regardless of their `auto_approve` value.
    pub auto_approve_tools: bool,
    /// How the model selects which tool to call, if any.
    pub tool_choice: ToolChoice,
    /// Holds tools in `ToolRegistry` and executes tools.
    pub tool_executor: ToolExecutor,
    /// Optional stored callback for tool approval. When set, Core answers
    /// approvals internally and never yields [`CoreEvent::ApprovalRequest`].
    pub tool_approval_callback: Option<ToolApprovalCallback>,
    /// Optional turn callback for transforming messages between turns (e.g., compaction).
    pub turn_callback: Option<TurnCallback>,
    /// Thinking/reasoning mode configuration.
    pub thinking: ThinkingMode,
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
            turn_callback: None,
            thinking: ThinkingMode::Default,
        }
    }

    /// Set callback for tool approval decisions.
    ///
    /// Escape hatch for consumers who prefer stored callbacks over reacting to
    /// [`CoreEvent::ApprovalRequest`] in the stream. When set, Core answers
    /// approvals internally and never yields [`CoreEvent::ApprovalRequest`].
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use neuromance::{Core, ToolApproval};
    /// # use neuromance_client::chat_completions::ChatCompletionsClient;
    /// # let client: ChatCompletionsClient = unimplemented!();
    /// let core = Core::new(client)
    ///     .with_tool_approval_callback(|tool_call| {
    ///         let tool_call = tool_call.clone();
    ///         async move {
    ///             println!("Tool requested: {}", tool_call.function.name);
    ///             ToolApproval::Approved
    ///         }
    ///     });
    /// ```
    #[must_use]
    pub fn with_tool_approval_callback<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(&ToolCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolApproval> + Send + 'static,
    {
        self.tool_approval_callback =
            Some(Box::new(move |tool_call| Box::pin(callback(tool_call))));
        self
    }

    /// Set callback for transforming messages between turns.
    #[must_use]
    pub fn with_turn_callback<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(Vec<Message>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<Message>>> + Send + 'static,
    {
        self.turn_callback = Some(Box::new(move |messages| Box::pin(callback(messages))));
        self
    }

    /// Enable streaming mode.
    #[must_use]
    pub const fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    /// Set extended thinking budget (Anthropic Claude models).
    #[must_use]
    pub const fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking = ThinkingMode::Extended {
            budget_tokens: budget,
        };
        self
    }

    /// Enable interleaved thinking between tool calls with the given budget.
    #[must_use]
    pub const fn with_interleaved_thinking(mut self, budget: u32) -> Self {
        self.thinking = ThinkingMode::Interleaved {
            budget_tokens: budget,
        };
        self
    }

    /// Set the thinking mode directly.
    #[must_use]
    pub const fn with_thinking_mode(mut self, mode: ThinkingMode) -> Self {
        self.thinking = mode;
        self
    }

    /// Send a chat request with retry logic for transient failures.
    async fn chat_with_retry(&self, request: &ChatRequest) -> Result<ChatResponse, CoreError> {
        let mut last_error = None;
        let config = self.client.config();

        for attempt in 0..=config.retry_config.max_retries {
            match self.client.chat(request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if attempt < config.retry_config.max_retries && e.is_retryable() {
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
                    last_error = Some(e);
                    break;
                }
            }
        }

        Err(last_error.map_or_else(
            || CoreError::NoResponse("No response received after retries".to_string()),
            CoreError::Client,
        ))
    }

    /// Run the conversation loop as a stream of [`CoreEvent`]s.
    ///
    /// The stream borrows `&mut self` and ends with [`CoreEvent::Completed`]
    /// carrying the final message history. Errors surface inline as `Err`
    /// items and terminate the stream.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use neuromance::{Core, CoreEvent, Message};
    /// # use neuromance_client::chat_completions::ChatCompletionsClient;
    /// # use futures::StreamExt;
    /// # use tokio_util::sync::CancellationToken;
    /// # async fn example(mut core: Core<ChatCompletionsClient>, messages: Vec<Message>)
    /// #     -> Result<Vec<Message>, neuromance::CoreError> {
    /// let cancel = CancellationToken::new();
    /// let mut stream = Box::pin(core.run(messages, cancel));
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         CoreEvent::Delta(chunk)      => print!("{chunk}"),
    ///         CoreEvent::Completed(msgs)   => return Ok(msgs),
    ///         _ => {}
    ///     }
    /// }
    /// # unreachable!()
    /// # }
    /// ```
    #[allow(clippy::too_many_lines)]
    pub fn run(
        &mut self,
        messages: Vec<Message>,
        cancel: CancellationToken,
    ) -> impl Stream<Item = Result<CoreEvent, CoreError>> + Send + '_ {
        try_stream! {
            let mut messages = messages;
            let mut turn_count: u32 = 0;
            let start_time = Instant::now();
            let mut messages_arc: Arc<[Message]> = messages.clone().into();

            loop {
                if cancel.is_cancelled() {
                    Err(CoreError::Cancelled("loop start".to_string()))?;
                }

                let mut request = ChatRequest::from((self.client.config(), messages_arc.clone()))
                    .with_tools(self.tool_executor.get_all_tools())
                    .with_tool_choice(self.tool_choice.clone());
                request = request.with_thinking_mode(self.thinking);

                info!(
                    "Executing chat turn ({}/{})",
                    turn_count + 1,
                    self.max_turns
                        .map_or_else(|| "unlimited".to_string(), |max| max.to_string()),
                );
                if log::log_enabled!(log::Level::Debug) {
                    let pretty = serde_json::to_string_pretty(&request)?;
                    debug!("Chat request:\n {pretty}");
                }

                let response = if self.streaming {
                    let mut inner = self.client.chat_stream(&request).await?;
                    let mut accumulated_content = String::with_capacity(1024);
                    let mut response_metadata = None;
                    let mut role = None;
                    let mut tool_calls: Vec<ToolCall> = Vec::with_capacity(4);
                    let mut finish_reason = None;
                    let mut accumulated_usage: Option<neuromance_common::Usage> = None;

                    loop {
                        let next_chunk: Result<_, CoreError> = tokio::select! {
                            biased;
                            () = cancel.cancelled() => Err(CoreError::Cancelled("stream chunk".to_string())),
                            next = inner.next() => Ok(next),
                        };
                        let Some(chunk_result) = next_chunk? else { break };
                        let chunk = chunk_result?;

                        if let Some(ref content) = chunk.delta_content {
                            accumulated_content.push_str(content);
                            yield CoreEvent::Delta(content.clone());
                        }

                        if role.is_none() {
                            role = chunk.delta_role;
                        }

                        if let Some(ref delta_tool_calls) = chunk.delta_tool_calls {
                            debug!("Received {} tool call delta(s)", delta_tool_calls.len());
                            tool_calls = ToolCall::merge_deltas(tool_calls, delta_tool_calls);
                        }

                        if chunk.finish_reason.is_some() {
                            finish_reason = chunk.finish_reason;
                        }

                        if let Some(ref chunk_usage) = chunk.usage {
                            accumulated_usage = Some(match accumulated_usage {
                                None => chunk_usage.clone(),
                                Some(mut acc) => {
                                    acc.prompt_tokens =
                                        acc.prompt_tokens.max(chunk_usage.prompt_tokens);
                                    acc.completion_tokens =
                                        acc.completion_tokens.max(chunk_usage.completion_tokens);
                                    acc.total_tokens = acc.prompt_tokens + acc.completion_tokens;
                                    if acc.input_tokens_details.is_none() {
                                        acc.input_tokens_details
                                            .clone_from(&chunk_usage.input_tokens_details);
                                    }
                                    if acc.output_tokens_details.is_none() {
                                        acc.output_tokens_details
                                            .clone_from(&chunk_usage.output_tokens_details);
                                    }
                                    acc
                                }
                            });
                        }

                        response_metadata = Some(chunk);
                    }

                    let conversation_id = request
                        .messages
                        .first()
                        .ok_or_else(|| {
                            CoreError::NoResponse(
                                "Request must contain at least one message".to_string(),
                            )
                        })?
                        .conversation_id;

                    let last_chunk = response_metadata.ok_or_else(|| {
                        CoreError::NoResponse("Stream ended without any chunks".to_string())
                    })?;

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
                        reasoning: None,
                    };

                    ChatResponse {
                        message,
                        model: last_chunk.model,
                        usage: accumulated_usage,
                        finish_reason,
                        created_at: last_chunk.created_at,
                        response_id: last_chunk.response_id,
                        metadata: std::collections::HashMap::new(),
                    }
                } else {
                    let outcome: Result<ChatResponse, CoreError> = tokio::select! {
                        biased;
                        () = cancel.cancelled() => Err(CoreError::Cancelled("chat_with_retry".to_string())),
                        res = self.chat_with_retry(&request) => res,
                    };
                    outcome?
                };

                debug!("Received response from LLM");
                if log::log_enabled!(log::Level::Debug) {
                    let pretty = serde_json::to_string_pretty(&response)?;
                    debug!("Assistant Response:\n {pretty}");
                }

                if let Some(ref usage) = response.usage {
                    yield CoreEvent::Usage(usage.clone());
                }

                let conversation_id = response.message.conversation_id;
                let tool_calls = response.message.tool_calls.clone();
                let tool_calls_count = tool_calls.len();
                messages.push(response.message);

                if tool_calls.is_empty() {
                    let duration = start_time.elapsed();
                    debug!(
                        "No tool calls in response, chat completed in {} turns ({:.2?})",
                        turn_count + 1,
                        duration
                    );
                    yield CoreEvent::Completed(messages);
                    return;
                }

                for tool_call in &tool_calls {
                    let tool_name = &tool_call.function.name;
                    let call_id = &tool_call.id;
                    debug!("Tool Name: {tool_name} (id: {call_id})");
                    debug!("Tool Arguments: {:?}", tool_call.function.arguments);

                    let is_auto_approved = self.auto_approve_tools
                        || self.tool_executor.is_tool_auto_approved(tool_name);
                    debug!("Tool auto-approved: {is_auto_approved}");

                    let approval = if is_auto_approved {
                        ToolApproval::Approved
                    } else if let Some(ref callback) = self.tool_approval_callback {
                        let outcome: Result<ToolApproval, CoreError> = tokio::select! {
                            biased;
                            () = cancel.cancelled() => Err(CoreError::Cancelled("approval callback".to_string())),
                            a = callback(tool_call) => Ok(a),
                        };
                        outcome?
                    } else {
                        let (tx, rx) = oneshot::channel();
                        yield CoreEvent::ApprovalRequest {
                            tool_call: tool_call.clone(),
                            responder: tx,
                        };
                        let outcome: Result<ToolApproval, CoreError> = tokio::select! {
                            biased;
                            () = cancel.cancelled() => Err(CoreError::Cancelled("approval responder".to_string())),
                            res = rx => Ok(res.unwrap_or_else(|_| {
                                ToolApproval::Denied("Approval responder dropped".to_string())
                            })),
                        };
                        outcome?
                    };

                    debug!("Tool Approval Status: {approval:?}");

                    match approval {
                        ToolApproval::Approved => {
                            debug!("Executing tool: {tool_name}");
                            let exec_outcome: Result<Result<String, anyhow::Error>, CoreError> = tokio::select! {
                                biased;
                                () = cancel.cancelled() => Err(CoreError::Cancelled("tool execution".to_string())),
                                r = self.tool_executor.execute_tool(tool_call) => Ok(r),
                            };
                            match exec_outcome? {
                                Ok(result) => {
                                    debug!("Tool {tool_name} executed successfully");
                                    yield CoreEvent::ToolResult {
                                        name: tool_name.clone(),
                                        result: result.clone(),
                                        success: true,
                                    };
                                    let tool_message = Message::tool(
                                        conversation_id,
                                        result,
                                        tool_call.id.clone(),
                                        tool_call.function.name.clone(),
                                    )
                                    .map_err(|e| CoreError::ToolError(e.to_string()))?;
                                    messages.push(tool_message);
                                }
                                Err(e) => {
                                    debug!("Tool {tool_name} execution failed: {e}");
                                    let error_msg = format!("Tool execution failed: {e}");
                                    yield CoreEvent::ToolResult {
                                        name: tool_name.clone(),
                                        result: error_msg.clone(),
                                        success: false,
                                    };
                                    let error_message = Message::tool(
                                        conversation_id,
                                        error_msg,
                                        tool_call.id.clone(),
                                        tool_call.function.name.clone(),
                                    )
                                    .map_err(|e| CoreError::ToolError(e.to_string()))?;
                                    messages.push(error_message);
                                }
                            }
                        }
                        ToolApproval::Denied(reason) => {
                            debug!("Tool {tool_name} denied: {reason}");
                            let denial_message = Message::tool(
                                conversation_id,
                                format!("Tool execution denied: {reason}"),
                                tool_call.id.clone(),
                                tool_call.function.name.clone(),
                            )
                            .map_err(|e| CoreError::ToolError(e.to_string()))?;
                            messages.push(denial_message);
                        }
                        ToolApproval::Quit => {
                            debug!("User quit during tool approval");
                            Err(CoreError::UserQuit(
                                "User quit during tool approval".to_string(),
                            ))?;
                        }
                    }
                }

                debug!("Completed processing {tool_calls_count} tool calls, continuing");

                if let Some(ref callback) = self.turn_callback {
                    let outcome: Result<Result<Vec<Message>, anyhow::Error>, CoreError> = tokio::select! {
                        biased;
                        () = cancel.cancelled() => Err(CoreError::Cancelled("turn callback".to_string())),
                        r = callback(messages) => Ok(r),
                    };
                    messages = outcome?.map_err(|e| CoreError::TurnCallback(e.into()))?;
                }

                messages_arc = messages.clone().into();
                turn_count += 1;

                if let Some(max) = self.max_turns
                    && turn_count >= max
                {
                    Err(CoreError::MaxTurnsExceeded(format!(
                        "Exceeded maximum turns: {turn_count} (configured max: {max})"
                    )))?;
                }
            }
        }
    }

    /// Convenience wrapper around [`Core::run`] that drains the stream and
    /// returns the final message history along with [`RunStats`] aggregated
    /// over the run.
    ///
    /// When no [`Core::with_tool_approval_callback`] is set and a non-auto-approved
    /// tool is requested, the yielded [`CoreEvent::ApprovalRequest`] is answered
    /// with `Denied("No approval mechanism configured")`.
    ///
    /// # Errors
    ///
    /// Returns any error produced by [`Core::run`]; also returns
    /// [`CoreError::NoResponse`] if the stream ends without [`CoreEvent::Completed`].
    pub async fn chat_with_tool_loop(
        &mut self,
        messages: Vec<Message>,
        cancel: CancellationToken,
    ) -> Result<(Vec<Message>, RunStats), CoreError> {
        let mut stats = RunStats::default();
        let mut stream = Box::pin(self.run(messages, cancel));
        while let Some(event) = stream.next().await {
            let event = event?;
            stats.observe(&event);
            match event {
                CoreEvent::Completed(msgs) => return Ok((msgs, stats)),
                CoreEvent::ApprovalRequest { responder, .. } => {
                    let _ = responder.send(ToolApproval::Denied(
                        "No approval mechanism configured".into(),
                    ));
                }
                CoreEvent::Delta(_) | CoreEvent::ToolResult { .. } | CoreEvent::Usage(_) => {}
            }
        }
        Err(CoreError::NoResponse(
            "Stream ended without Completed event".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_client::chat_completions::ChatCompletionsClient;
    use neuromance_common::client::Config;

    /// `with_tool_approval_callback` stores the callback.
    #[tokio::test]
    async fn test_tool_approval_callback() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

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

    /// Core without any callbacks builds cleanly.
    #[tokio::test]
    async fn test_core_without_callbacks() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");
        let core = Core::new(client);

        assert!(core.tool_approval_callback.is_none());
        assert!(core.turn_callback.is_none());
    }

    /// `with_turn_callback` stores the callback.
    #[tokio::test]
    async fn test_core_with_turn_callback() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

        let core = Core::new(client).with_turn_callback(|messages| async move { Ok(messages) });

        assert!(core.turn_callback.is_some());
    }

    /// Turn callback can transform messages.
    #[tokio::test]
    async fn test_turn_callback_transforms_messages() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

        let core = Core::new(client).with_turn_callback(|mut messages| async move {
            let conv_id = messages
                .first()
                .map_or_else(uuid::Uuid::new_v4, |m| m.conversation_id);
            messages.push(Message::system(conv_id, "[compacted]"));
            Ok(messages)
        });

        let conv_id = uuid::Uuid::new_v4();
        let input = vec![Message::user(conv_id, "hello")];
        let output = (core.turn_callback.unwrap())(input).await.unwrap();

        assert_eq!(output.len(), 2);
        assert_eq!(output[1].content, "[compacted]");
    }

    /// Pre-cancelling the token surfaces `CoreError::Cancelled` immediately:
    /// the loop-start check fires before any client request is built.
    #[tokio::test]
    async fn test_run_yields_cancelled_when_pre_cancelled() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");
        let mut core = Core::new(client);

        let cancel = CancellationToken::new();
        cancel.cancel();

        let messages = vec![Message::user(uuid::Uuid::new_v4(), "hello")];
        let mut stream = Box::pin(core.run(messages, cancel));

        let event = stream.next().await.expect("stream should yield an event");
        assert!(
            matches!(event, Err(CoreError::Cancelled(_))),
            "unexpected: {event:?}"
        );
    }

    /// `chat_with_tool_loop` propagates `CoreError::Cancelled` from `run`.
    #[tokio::test]
    async fn test_chat_with_tool_loop_propagates_cancelled() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");
        let mut core = Core::new(client);

        let cancel = CancellationToken::new();
        cancel.cancel();

        let messages = vec![Message::user(uuid::Uuid::new_v4(), "hello")];
        let result = core.chat_with_tool_loop(messages, cancel).await;
        assert!(
            matches!(result, Err(CoreError::Cancelled(_))),
            "unexpected: {result:?}"
        );
    }

    /// Errors from turn callback propagate as `anyhow::Error`.
    #[tokio::test]
    async fn test_turn_callback_error_propagation() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

        let core = Core::new(client).with_turn_callback(|_messages| async move {
            Err(anyhow::anyhow!("compaction failed"))
        });

        let input = vec![Message::user(uuid::Uuid::new_v4(), "hello")];
        let result = (core.turn_callback.unwrap())(input).await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "compaction failed");
    }
}
