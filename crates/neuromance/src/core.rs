use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_stream::try_stream;
use chrono::Utc;
use futures::{Stream, StreamExt};
use metrics::{counter, histogram};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, info_span, trace};

/// How often to emit an info-level "still streaming" progress log while a
/// single turn is in flight. Keeps long completions visible without flooding.
const STREAM_PROGRESS_INTERVAL: Duration = Duration::from_secs(30);

use neuromance_client::LLMClient;
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatRequest, ChatResponse, ToolChoice};
use neuromance_common::features::ThinkingMode;
use neuromance_common::tools::{ToolApproval, ToolCall};
use neuromance_tools::{ToolExecutor, ToolExecutorError};

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
    /// Optional sink that persists conversation history as the run progresses.
    /// Write failures are logged and never abort the run.
    #[cfg(feature = "db")]
    pub persistence: Option<Arc<dyn neuromance_db::ConversationSink>>,
    /// Conversation that spawned this run (e.g. a delegating parent agent),
    /// recorded against the persisted conversation so a delegation tree can be
    /// reconstructed. `None` for a root conversation.
    #[cfg(feature = "db")]
    pub parent_conversation_id: Option<uuid::Uuid>,
    /// Runtime task this run belongs to, persisted alongside
    /// [`Self::parent_conversation_id`].
    #[cfg(feature = "db")]
    pub parent_task_id: Option<uuid::Uuid>,
    /// Assistant message in the parent conversation that emitted the tool call
    /// spawning this run, recorded so a delegation can be traced to the exact
    /// message. `None` for a root conversation.
    #[cfg(feature = "db")]
    pub parent_message_id: Option<uuid::Uuid>,
    /// Id of the specific tool call within [`Self::parent_message_id`] that
    /// spawned this run.
    #[cfg(feature = "db")]
    pub parent_tool_call_id: Option<String>,
    /// Optional context manager for automatic compaction between turns.
    #[cfg(feature = "context")]
    context_manager: Option<crate::context_management::ContextManager<C>>,
}

impl<C: LLMClient> Core<C> {
    #[must_use]
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
            #[cfg(feature = "db")]
            persistence: None,
            #[cfg(feature = "db")]
            parent_conversation_id: None,
            #[cfg(feature = "db")]
            parent_task_id: None,
            #[cfg(feature = "db")]
            parent_message_id: None,
            #[cfg(feature = "db")]
            parent_tool_call_id: None,
            #[cfg(feature = "context")]
            context_manager: None,
        }
    }

    /// Set a sink that durably records conversation history.
    ///
    /// Messages are persisted incrementally during [`Core::run`]: the seed
    /// snapshot at run start, the assistant message after each turn, and tool
    /// results after each turn's tool calls complete — so a crashed run still
    /// has its prefix recorded. Persistence is best-effort: write failures are
    /// logged and retried at the next persist point, never aborting the run.
    #[cfg(feature = "db")]
    #[must_use]
    pub fn with_persistence(mut self, sink: Arc<dyn neuromance_db::ConversationSink>) -> Self {
        self.persistence = Some(sink);
        self
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

    /// Enable automatic context compaction between conversation turns.
    ///
    /// When configured, compaction runs inside the conversation loop, keeping
    /// the conversation within the configured token budget.
    ///
    /// Requires the `context` feature and `C: Clone` (all built-in clients
    /// implement `Clone`). For non-`Clone` clients such as `Box<dyn LLMClient>`,
    /// use [`with_context_management_client`](Self::with_context_management_client).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use neuromance::Core;
    /// # use neuromance::context_management::ContextConfig;
    /// # use neuromance_client::ChatCompletionsClient;
    /// # let client: ChatCompletionsClient = unimplemented!();
    /// let config = ContextConfig::new(128_000);
    /// let core = Core::new(client)
    ///     .with_context_management(config);
    /// ```
    #[cfg(feature = "context")]
    #[must_use]
    pub fn with_context_management(self, config: crate::context_management::ContextConfig) -> Self
    where
        C: Clone,
    {
        let compaction_client = self.client.clone();
        self.with_context_management_client(compaction_client, config)
    }

    /// Enable automatic context compaction using a separate client for
    /// summarization calls.
    ///
    /// Like [`with_context_management`](Self::with_context_management), but
    /// without a `Clone` bound on `C` — the caller supplies the client used
    /// for compaction summaries (typically built from the same config as the
    /// main client).
    #[cfg(feature = "context")]
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn with_context_management_client(
        mut self,
        compaction_client: C,
        config: crate::context_management::ContextConfig,
    ) -> Self {
        self.context_manager = Some(crate::context_management::ContextManager::new(
            compaction_client,
            &config,
        ));
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
            // `prompt_tokens` from each turn is the whole context sent that turn,
            // so the turn-over-turn delta shows how fast the conversation grows.
            let mut prev_prompt_tokens: u32 = 0;
            let start_time = Instant::now();
            let mut messages_arc: Arc<[Message]> = messages.clone().into();

            // Persist the seed snapshot (system + user messages) up front so
            // the input is durable even if the first LLM call fails.
            #[cfg(feature = "db")]
            let mut persisted_ids = std::collections::HashSet::new();
            #[cfg(feature = "db")]
            if let Some(ref sink) = self.persistence {
                persist_new_messages(sink.as_ref(), &messages, &mut persisted_ids).await;
                // Link this conversation to its spawning parent, once the seed
                // append has created the row. Best-effort, like message writes.
                if let (Some(parent), Some(first)) =
                    (self.parent_conversation_id, messages.first())
                    && let Err(e) = sink
                        .set_conversation_parent(
                            first.conversation_id,
                            parent,
                            self.parent_task_id,
                            self.parent_message_id,
                            self.parent_tool_call_id.as_deref(),
                        )
                        .await
                {
                    tracing::warn!(
                        conversation_id = %first.conversation_id,
                        parent_conversation_id = %parent,
                        error = %e,
                        "recording conversation parent failed; continuing without it"
                    );
                }
            }

            loop {
                if cancel.is_cancelled() {
                    Err(CoreError::Cancelled("loop start".to_string()))?;
                }

                let mut request = ChatRequest::from((self.client.config(), messages_arc.clone()))
                    .with_tools(self.tool_executor.get_all_tools())
                    .with_tool_choice(self.tool_choice.clone());
                request = request.with_thinking_mode(self.thinking);

                let turn_number = turn_count + 1;
                let max_turns_label = self
                    .max_turns
                    .map_or_else(|| "unlimited".to_string(), |max| max.to_string());
                let turn_start = Instant::now();
                let turn_span = info_span!(
                    "chat_turn",
                    turn = turn_number,
                    max_turns = %max_turns_label,
                    model = tracing::field::Empty,
                    finish_reason = tracing::field::Empty,
                );
                let _turn_enter = turn_span.enter();
                info!(turn = turn_number, max_turns = %max_turns_label, "executing chat turn");
                debug!(
                    model = request.model.as_deref().unwrap_or("default"),
                    messages = request.messages.len(),
                    tools = request.tools.as_ref().map_or(0, Vec::len),
                    "chat request",
                );
                if tracing::enabled!(target: "neuromance::wire", tracing::Level::TRACE) {
                    let body = serde_json::to_string(&request)?;
                    trace!(target: "neuromance::wire", %body, "chat request body");
                }

                let response = if self.streaming {
                    let mut inner = self.client.chat_stream(&request).await?;
                    let mut accumulated_content = String::with_capacity(1024);
                    let mut response_metadata = None;
                    let mut role = None;
                    let mut tool_calls: Vec<ToolCall> = Vec::with_capacity(4);
                    let mut finish_reason = None;
                    let mut accumulated_usage: Option<neuromance_common::Usage> = None;
                    let mut first_chunk_at: Option<Instant> = None;
                    let mut last_progress_log = turn_start;
                    let mut tool_call_deltas_seen: u32 = 0;
                    let mut reasoning_bytes: usize = 0;
                    let mut chunks_seen: u32 = 0;

                    loop {
                        let next_chunk: Result<_, CoreError> = tokio::select! {
                            biased;
                            () = cancel.cancelled() => Err(CoreError::Cancelled("stream chunk".to_string())),
                            next = inner.next() => Ok(next),
                        };
                        let Some(chunk_result) = next_chunk? else { break };
                        let chunk = chunk_result?;
                        chunks_seen = chunks_seen.saturating_add(1);

                        if first_chunk_at.is_none() {
                            let now = Instant::now();
                            first_chunk_at = Some(now);
                            let latency = now.duration_since(turn_start);
                            let latency_ms = u64::try_from(latency.as_millis()).unwrap_or(u64::MAX);
                            info!(
                                turn = turn_number,
                                latency_ms,
                                "first stream chunk received",
                            );
                            histogram!("neuromance_first_chunk_latency_seconds")
                                .record(latency.as_secs_f64());
                        }

                        if let Some(ref content) = chunk.delta_content {
                            accumulated_content.push_str(content);
                            yield CoreEvent::Delta(content.clone());
                        }

                        if let Some(ref reasoning) = chunk.delta_reasoning_content {
                            reasoning_bytes = reasoning_bytes.saturating_add(reasoning.len());
                        }

                        if role.is_none() {
                            role = chunk.delta_role;
                        }

                        if let Some(ref delta_tool_calls) = chunk.delta_tool_calls {
                            let delta_count = delta_tool_calls.len();
                            tool_call_deltas_seen = tool_call_deltas_seen
                                .saturating_add(u32::try_from(delta_count).unwrap_or(u32::MAX));
                            debug!(deltas = delta_count, "received tool call deltas");
                            tool_calls = ToolCall::merge_deltas(tool_calls, delta_tool_calls);
                        }

                        if last_progress_log.elapsed() >= STREAM_PROGRESS_INTERVAL {
                            let elapsed_ms =
                                u64::try_from(turn_start.elapsed().as_millis()).unwrap_or(u64::MAX);
                            let content_bytes = accumulated_content.len();
                            info!(
                                turn = turn_number,
                                elapsed_ms,
                                chunks_seen,
                                content_bytes,
                                reasoning_bytes,
                                tool_call_deltas_seen,
                                "stream progress",
                            );
                            last_progress_log = Instant::now();
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
                        model: None,
                        provider: None,
                        usage: None,
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

                if tracing::enabled!(target: "neuromance::wire", tracing::Level::TRACE) {
                    let body = serde_json::to_string(&response)?;
                    trace!(target: "neuromance::wire", %body, "assistant response body");
                }

                if let Some(ref usage) = response.usage {
                    yield CoreEvent::Usage(usage.clone());
                }

                let turn_duration = turn_start.elapsed();
                let turn_duration_ms =
                    u64::try_from(turn_duration.as_millis()).unwrap_or(u64::MAX);
                let conversation_id = response.message.conversation_id;
                let tool_calls = response.message.tool_calls.clone();
                let tool_calls_count = tool_calls.len();
                let finish_label = response
                    .finish_reason
                    .as_ref()
                    .map_or_else(|| "unknown".to_string(), ToString::to_string);
                let (prompt_tokens, completion_tokens) = response
                    .usage
                    .as_ref()
                    .map_or((0_u32, 0_u32), |u| (u.prompt_tokens, u.completion_tokens));
                let cached_tokens = response
                    .usage
                    .as_ref()
                    .and_then(|u| u.input_tokens_details.as_ref())
                    .map_or(0_u32, |d| d.cached_tokens);
                // Growth in the input context since the previous turn — the cost of
                // the last assistant message, tool results, and any new input. Goes
                // negative when a turn shrinks the context (e.g. after compaction).
                let prompt_delta = i64::from(prompt_tokens) - i64::from(prev_prompt_tokens);
                prev_prompt_tokens = prompt_tokens;
                // Current footprint of the stored conversation: the full input
                // context plus the assistant reply that joins it. Not a sum across
                // turns — `prompt_tokens` already accumulates the history.
                let conv_total_tokens = prompt_tokens.saturating_add(completion_tokens);
                // Provider-reported size of the context after this turn:
                // prompt covers everything sent, completion the new assistant
                // output. Tool results appended below aren't counted — the
                // compaction threshold ratio absorbs that lag.
                #[cfg(feature = "context")]
                let reported_tokens: Option<usize> = response
                    .usage
                    .as_ref()
                    .map(|u| u.prompt_tokens as usize + u.completion_tokens as usize);
                turn_span.record("model", tracing::field::display(&response.model));
                turn_span.record("finish_reason", tracing::field::display(&finish_label));
                info!(
                    turn = turn_number,
                    duration_ms = turn_duration_ms,
                    finish = %finish_label,
                    prompt_tokens,
                    prompt_delta,
                    completion_tokens,
                    cached_tokens,
                    conv_total_tokens,
                    tool_calls = tool_calls_count,
                    "chat turn completed",
                );
                let model_label = response.model.clone();
                histogram!(
                    "neuromance_turn_duration_seconds",
                    "model" => model_label.clone(),
                )
                .record(turn_duration.as_secs_f64());
                counter!(
                    "neuromance_chat_turns_total",
                    "model" => model_label.clone(),
                    "finish_reason" => finish_label.clone(),
                )
                .increment(1);
                counter!(
                    "neuromance_tokens_total",
                    "kind" => "prompt",
                    "model" => model_label.clone(),
                )
                .increment(u64::from(prompt_tokens));
                counter!(
                    "neuromance_tokens_total",
                    "kind" => "completion",
                    "model" => model_label,
                )
                .increment(u64::from(completion_tokens));
                let mut assistant_message = response.message;
                assistant_message.model = Some(response.model);
                assistant_message.provider = Some(self.client.config().provider.clone());
                assistant_message.usage = response.usage;
                // Pins any subagent spawned by this turn's tool calls to the exact
                // message that launched it (see the per-tool-call delegation scope).
                let assistant_message_id = assistant_message.id;
                messages.push(assistant_message);

                // Persist the assistant message before tool execution so a
                // crashed run still has its prefix recorded.
                #[cfg(feature = "db")]
                if let Some(ref sink) = self.persistence {
                    persist_new_messages(sink.as_ref(), &messages, &mut persisted_ids).await;
                }

                if tool_calls.is_empty() {
                    // Compact before yielding so the stored history handed
                    // back to the caller is already within budget.
                    #[cfg(feature = "context")]
                    if let Some(ref ctx) = self.context_manager {
                        let (compacted, result) = ctx.maybe_compact(messages, reported_tokens).await;
                        messages = compacted;
                        if let Some(r) = result {
                            if r.was_compacted {
                                info!(
                                    original_tokens = r.original_tokens,
                                    compacted_tokens = r.compacted_tokens,
                                    messages_summarized = r.messages_summarized,
                                    "context compacted",
                                );
                                counter!("neuromance_compactions_total").increment(1);
                            }
                            yield CoreEvent::Compaction {
                                original_tokens: r.original_tokens,
                                compacted_tokens: r.compacted_tokens,
                                messages_summarized: r.messages_summarized,
                                was_compacted: r.was_compacted,
                            };
                        }
                    }

                    let total_ms =
                        u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);
                    info!(
                        turns = turn_number,
                        duration_ms = total_ms,
                        "chat completed",
                    );
                    yield CoreEvent::Completed(messages);
                    return;
                }

                for tool_call in &tool_calls {
                    let tool_name = &tool_call.function.name;
                    let call_id = &tool_call.id;
                    let tool_span = info_span!(
                        "tool_call",
                        tool = %tool_name,
                        call_id = %call_id,
                    );
                    let _tool_enter = tool_span.enter();
                    info!(tool = %tool_name, call_id = %call_id, "tool call requested");
                    debug!(arguments = ?tool_call.function.arguments, "tool arguments");

                    let is_auto_approved = self.auto_approve_tools
                        || self.tool_executor.is_tool_auto_approved(tool_name);
                    debug!(auto_approved = is_auto_approved, "tool approval policy");

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

                    debug!(approval = ?approval, "tool approval decided");

                    match approval {
                        ToolApproval::Approved => {
                            info!(tool = %tool_name, "executing tool");
                            let tool_start = Instant::now();
                            // Carry the launch site down to any subagent this tool
                            // spawns: the assistant message and the specific tool
                            // call. Conversation/task ids are preserved from the
                            // enclosing scope.
                            let mut child_ctx = neuromance_common::delegation::current();
                            child_ctx.parent_message_id = Some(assistant_message_id);
                            child_ctx.parent_tool_call_id = Some(tool_call.id.clone());
                            let exec_outcome: Result<Result<String, ToolExecutorError>, CoreError> = tokio::select! {
                                biased;
                                () = cancel.cancelled() => Err(CoreError::Cancelled("tool execution".to_string())),
                                r = neuromance_common::delegation::scope(
                                    child_ctx,
                                    self.tool_executor.execute_tool(tool_call),
                                ) => Ok(r),
                            };
                            let tool_elapsed = tool_start.elapsed();
                            let tool_duration_ms =
                                u64::try_from(tool_elapsed.as_millis()).unwrap_or(u64::MAX);
                            histogram!(
                                "neuromance_tool_duration_seconds",
                                "tool" => tool_name.clone(),
                            )
                            .record(tool_elapsed.as_secs_f64());
                            match exec_outcome? {
                                Ok(result) => {
                                    let bytes = result.len();
                                    info!(
                                        tool = %tool_name,
                                        duration_ms = tool_duration_ms,
                                        bytes,
                                        "tool call succeeded",
                                    );
                                    counter!(
                                        "neuromance_tool_calls_total",
                                        "tool" => tool_name.clone(),
                                        "outcome" => "success",
                                    )
                                    .increment(1);
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
                                    info!(
                                        tool = %tool_name,
                                        duration_ms = tool_duration_ms,
                                        error = %e,
                                        "tool call failed",
                                    );
                                    counter!(
                                        "neuromance_tool_calls_total",
                                        "tool" => tool_name.clone(),
                                        "outcome" => "failure",
                                    )
                                    .increment(1);
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
                            info!(tool = %tool_name, reason = %reason, "tool call denied");
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
                            debug!("user quit during tool approval");
                            Err(CoreError::UserQuit(
                                "User quit during tool approval".to_string(),
                            ))?;
                        }
                    }
                }

                debug!(tool_calls = tool_calls_count, "completed tool calls; continuing");

                // Persist this turn's tool results (and retry any backlog from
                // earlier failed writes) before the turn callback can rewrite
                // the history.
                #[cfg(feature = "db")]
                if let Some(ref sink) = self.persistence {
                    persist_new_messages(sink.as_ref(), &messages, &mut persisted_ids).await;
                }

                // Run context compaction (if configured) before the turn callback.
                #[cfg(feature = "context")]
                if let Some(ref ctx) = self.context_manager {
                    let (compacted, result) = ctx.maybe_compact(messages, reported_tokens).await;
                    messages = compacted;
                    if let Some(r) = result {
                        if r.was_compacted {
                            info!(
                                original_tokens = r.original_tokens,
                                compacted_tokens = r.compacted_tokens,
                                messages_summarized = r.messages_summarized,
                                "context compacted",
                            );
                            counter!("neuromance_compactions_total").increment(1);
                        }
                        yield CoreEvent::Compaction {
                            original_tokens: r.original_tokens,
                            compacted_tokens: r.compacted_tokens,
                            messages_summarized: r.messages_summarized,
                            was_compacted: r.was_compacted,
                        };
                    }
                }


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
                CoreEvent::Delta(_)
                | CoreEvent::ToolResult { .. }
                | CoreEvent::Usage(_)
                | CoreEvent::Compaction { .. } => {}
            }
        }
        Err(CoreError::NoResponse(
            "Stream ended without Completed event".to_string(),
        ))
    }
}

/// Persists messages not yet recorded this run, tolerating failures.
///
/// This is the single place implementing the log-and-continue policy: on
/// success the message ids are marked persisted; on failure they are left
/// unmarked so the next persist point retries the backlog (the sink's
/// per-id idempotency makes the retry safe).
#[cfg(feature = "db")]
async fn persist_new_messages(
    sink: &dyn neuromance_db::ConversationSink,
    messages: &[Message],
    persisted: &mut std::collections::HashSet<uuid::Uuid>,
) {
    let pending: Vec<Message> = messages
        .iter()
        .filter(|m| !persisted.contains(&m.id))
        .cloned()
        .collect();
    let Some(first) = pending.first() else { return };
    match sink.append_messages(first.conversation_id, &pending).await {
        Ok(inserted) => {
            persisted.extend(pending.iter().map(|m| m.id));
            counter!("neuromance_db_messages_persisted_total").increment(inserted);
        }
        Err(e) => {
            tracing::warn!(
                conversation_id = %first.conversation_id,
                error = %e,
                pending = pending.len(),
                "conversation persistence failed; continuing without it"
            );
            counter!("neuromance_db_persist_failures_total").increment(1);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_client::chat_completions::ChatCompletionsClient;
    use neuromance_common::client::Config;

    /// Returns a plain assistant reply (no tool calls) with a huge reported
    /// prompt size, so compaction must trigger before `Completed`.
    #[cfg(feature = "context")]
    struct HugeUsageClient {
        config: Config,
    }

    #[cfg(feature = "context")]
    #[async_trait::async_trait]
    impl LLMClient for HugeUsageClient {
        fn config(&self) -> &Config {
            &self.config
        }

        async fn chat(
            &self,
            request: &ChatRequest,
        ) -> Result<ChatResponse, neuromance_client::ClientError> {
            let conv_id = request
                .messages
                .first()
                .map_or_else(uuid::Uuid::new_v4, |m| m.conversation_id);
            Ok(ChatResponse {
                message: Message::assistant(conv_id, "summary or reply"),
                model: "mock-model".to_string(),
                usage: Some(neuromance_common::client::Usage {
                    prompt_tokens: 200_000,
                    completion_tokens: 10,
                    total_tokens: 200_010,
                    cost: None,
                    input_tokens_details: None,
                    output_tokens_details: None,
                }),
                finish_reason: None,
                created_at: chrono::Utc::now(),
                response_id: None,
                metadata: std::collections::HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<
                                neuromance_common::client::ChatChunk,
                                neuromance_client::ClientError,
                            >,
                        > + Send,
                >,
            >,
            neuromance_client::ClientError,
        > {
            Ok(Box::pin(futures::stream::pending()))
        }

        fn supports_tools(&self) -> bool {
            false
        }

        fn supports_streaming(&self) -> bool {
            false
        }
    }

    /// Compaction runs on the no-tool-call completion path, so the final
    /// history handed back to callers is already within budget.
    #[cfg(feature = "context")]
    #[tokio::test]
    async fn test_run_emits_compaction_event_on_completion_path() {
        use crate::context_management::ContextConfig;

        let mock_config = Config::new("mock", "mock-model");
        let client = HugeUsageClient {
            config: mock_config.clone(),
        };
        let compaction_client = HugeUsageClient {
            config: mock_config,
        };

        let context_config = ContextConfig::new(1000).with_preserve_recent_turns(1);
        let mut core =
            Core::new(client).with_context_management_client(compaction_client, context_config);

        let conv_id = uuid::Uuid::new_v4();
        let messages: Vec<Message> = (0..4)
            .flat_map(|i| {
                vec![
                    Message::user(conv_id, format!("question {i}")),
                    Message::assistant(conv_id, format!("answer {i}")),
                ]
            })
            .collect();

        let cancel = CancellationToken::new();
        let mut stream = Box::pin(core.run(messages, cancel));

        let mut saw_compaction = false;
        let mut completed: Option<Vec<Message>> = None;
        while let Some(event) = stream.next().await {
            match event.expect("run should not error") {
                CoreEvent::Compaction { was_compacted, .. } => {
                    assert!(
                        completed.is_none(),
                        "compaction must precede the completed event"
                    );
                    saw_compaction |= was_compacted;
                }
                CoreEvent::Completed(msgs) => completed = Some(msgs),
                _ => {}
            }
        }

        assert!(saw_compaction, "expected a was_compacted=true event");
        let completed = completed.expect("stream must complete");
        assert!(
            completed.len() < 9,
            "history should shrink below input+reply length, got {}",
            completed.len()
        );
        assert!(
            completed
                .iter()
                .any(|m| m.content.contains("summarized")
                    || m.content.contains("summary or reply")),
            "compacted history should contain the summary message"
        );
    }

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

    /// A mock sink that records every batch it receives and can be told to fail.
    #[cfg(feature = "db")]
    mod persistence {
        use std::collections::HashSet;
        use std::sync::Mutex;

        use neuromance_db::{ConversationSink, DbError};
        use uuid::Uuid;

        use super::*;

        #[derive(Default)]
        struct MockSink {
            fail: bool,
            batches: Mutex<Vec<Vec<Uuid>>>,
        }

        #[async_trait::async_trait]
        impl ConversationSink for MockSink {
            async fn append_messages(
                &self,
                conversation_id: Uuid,
                messages: &[Message],
            ) -> Result<u64, DbError> {
                if self.fail {
                    return Err(DbError::UnknownRole {
                        value: "mock failure".to_string(),
                        message_id: conversation_id,
                    });
                }
                let ids: Vec<Uuid> = messages.iter().map(|m| m.id).collect();
                let count = ids.len() as u64;
                self.batches.lock().unwrap().push(ids);
                Ok(count)
            }
        }

        /// `with_persistence` stores the sink that `Core` later drives: writing
        /// through the stored handle reaches our `MockSink` and records the batch.
        #[tokio::test]
        async fn test_with_persistence_drives_the_stored_sink() {
            let config = Config::new("test", "test-model").with_api_key("test-key");
            let client = ChatCompletionsClient::new(config).expect("Failed to create client");
            let sink = Arc::new(MockSink::default());
            let core = Core::new(client).with_persistence(sink.clone());

            let conv_id = uuid::Uuid::new_v4();
            let message = Message::user(conv_id, "hello");
            let stored = core.persistence.as_ref().expect("sink should be stored");
            let count = stored
                .append_messages(conv_id, std::slice::from_ref(&message))
                .await
                .unwrap();

            assert_eq!(count, 1);
            assert_eq!(*sink.batches.lock().unwrap(), vec![vec![message.id]]);
        }

        /// Already-persisted ids are filtered out of subsequent batches.
        #[tokio::test]
        async fn test_persist_new_messages_skips_persisted_ids() {
            let sink = MockSink::default();
            let conv_id = uuid::Uuid::new_v4();
            let first = Message::user(conv_id, "one");
            let mut persisted = HashSet::new();

            persist_new_messages(&sink, std::slice::from_ref(&first), &mut persisted).await;
            assert!(persisted.contains(&first.id));

            let second = Message::assistant(conv_id, "two");
            persist_new_messages(&sink, &[first.clone(), second.clone()], &mut persisted).await;

            let batches = sink.batches.lock().unwrap().clone();
            assert_eq!(batches, vec![vec![first.id], vec![second.id]]);
        }

        /// Nothing pending means the sink is never called.
        #[tokio::test]
        async fn test_persist_new_messages_noop_when_all_persisted() {
            let sink = MockSink::default();
            let message = Message::user(uuid::Uuid::new_v4(), "one");
            let mut persisted = HashSet::from([message.id]);

            persist_new_messages(&sink, &[message], &mut persisted).await;
            assert!(sink.batches.lock().unwrap().is_empty());
        }

        /// A failed write leaves ids unmarked so the next call retries them.
        #[tokio::test]
        async fn test_persist_failure_leaves_ids_unpersisted() {
            let sink = MockSink {
                fail: true,
                ..MockSink::default()
            };
            let message = Message::user(uuid::Uuid::new_v4(), "one");
            let mut persisted = HashSet::new();

            persist_new_messages(&sink, std::slice::from_ref(&message), &mut persisted).await;
            assert!(
                !persisted.contains(&message.id),
                "failed writes must be retried at the next persist point"
            );
        }
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
