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
use neuromance_common::chat::{Conversation, Message, MessageRole};
use neuromance_common::client::{ChatRequest, ChatResponse, ToolChoice, Usage};
use neuromance_common::context::{ContextLedger, EditSource};
use neuromance_common::features::ThinkingMode;
use neuromance_common::hook::{CompactionStats, Hook, HookContext};
use neuromance_common::tools::{ToolApproval, ToolCall};
use neuromance_tools::{ToolExecutor, ToolExecutorError};

use crate::error::CoreError;
use crate::events::CoreEvent;
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
    /// Lifecycle hooks dispatched at each stage of the run, in registration order.
    pub hooks: Vec<Arc<dyn Hook>>,
    /// Thinking/reasoning mode configuration.
    pub thinking: ThinkingMode,
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
            hooks: Vec::new(),
            thinking: ThinkingMode::Default,
        }
    }

    /// Register a lifecycle [`Hook`].
    ///
    /// Hooks run in registration order at each stage of [`Core::run`]: they can
    /// inject context, decide tool approval, and rewrite or compact history.
    #[must_use]
    pub fn with_hook(mut self, hook: Arc<dyn Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Register a lifecycle [`Hook`] in place.
    pub fn add_hook(&mut self, hook: Arc<dyn Hook>) {
        self.hooks.push(hook);
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

    /// Run every hook's `on_conversation_start`, recording each hook's injected
    /// messages in the ledger attributed to that hook.
    async fn hooks_conversation_start(
        &self,
        ctx: &HookContext,
        ledger: &mut ContextLedger,
        cancel: &CancellationToken,
    ) -> Result<(), CoreError> {
        for hook in &self.hooks {
            let outcome = run_hook(
                cancel,
                hook.name(),
                hook.on_conversation_start(ctx, ledger.messages()),
            )
            .await?;
            ledger.append(EditSource::hook(hook.name()), outcome.messages);
        }
        Ok(())
    }

    /// Run every hook's `on_messages` observer for the current history.
    async fn hooks_messages(
        &self,
        ctx: &HookContext,
        messages: &[Message],
        cancel: &CancellationToken,
    ) -> Result<(), CoreError> {
        for hook in &self.hooks {
            run_hook(cancel, hook.name(), hook.on_messages(ctx, messages)).await?;
        }
        Ok(())
    }

    /// Run every hook's `on_usage` observer.
    async fn hooks_usage(
        &self,
        ctx: &HookContext,
        usage: &Usage,
        cancel: &CancellationToken,
    ) -> Result<(), CoreError> {
        for hook in &self.hooks {
            run_hook(cancel, hook.name(), hook.on_usage(ctx, usage)).await?;
        }
        Ok(())
    }

    /// Poll hooks for a tool-approval decision; the first non-abstaining hook wins.
    async fn hooks_review_tool(
        &self,
        ctx: &HookContext,
        call: &ToolCall,
        cancel: &CancellationToken,
    ) -> Result<Option<ToolApproval>, CoreError> {
        for hook in &self.hooks {
            if let Some(decision) =
                run_hook(cancel, hook.name(), hook.review_tool(ctx, call)).await?
            {
                return Ok(Some(decision));
            }
        }
        Ok(None)
    }

    /// Run every hook's `after_tool`, collecting each hook's injected messages
    /// tagged with its source so the caller records them with provenance.
    ///
    /// `after_tool` hooks key off the tool result, not the message history, so
    /// they need not observe one another's injections; gathering here and
    /// applying at the call site is equivalent.
    async fn hooks_after_tool(
        &self,
        ctx: &HookContext,
        call: &ToolCall,
        result: &str,
        success: bool,
        cancel: &CancellationToken,
    ) -> Result<Vec<(EditSource, Vec<Message>)>, CoreError> {
        let mut injected = Vec::new();
        for hook in &self.hooks {
            let outcome = run_hook(
                cancel,
                hook.name(),
                hook.after_tool(ctx, call, result, success),
            )
            .await?;
            if !outcome.messages.is_empty() {
                injected.push((EditSource::hook(hook.name()), outcome.messages));
            }
        }
        Ok(injected)
    }

    /// Fold the history through every hook's `on_turn_end`, recording a ledger
    /// `Replace` for any hook that actually rewrote the history and collecting
    /// any compaction statistics for the caller to emit.
    async fn hooks_turn_end(
        &self,
        ctx: &HookContext,
        ledger: &mut ContextLedger,
        cancel: &CancellationToken,
    ) -> Result<Vec<CompactionStats>, CoreError> {
        let mut stats = Vec::new();
        for hook in &self.hooks {
            let before: Vec<uuid::Uuid> = ledger.messages().iter().map(|m| m.id).collect();
            let end = run_hook(
                cancel,
                hook.name(),
                hook.on_turn_end(ctx, ledger.messages().to_vec()),
            )
            .await?;
            let changed = end.messages.iter().map(|m| m.id).ne(before.iter().copied());
            if changed {
                let details = end
                    .compaction
                    .map(|s| serde_json::json!({ "was_compacted": s.was_compacted }));
                ledger.replace(EditSource::hook(hook.name()), end.messages, details);
            }
            if let Some(s) = end.compaction {
                stats.push(s);
            }
        }
        Ok(stats)
    }

    /// Run every hook's `on_completion` observer.
    async fn hooks_completion(
        &self,
        ctx: &HookContext,
        messages: &[Message],
        cancel: &CancellationToken,
    ) -> Result<(), CoreError> {
        for hook in &self.hooks {
            run_hook(cancel, hook.name(), hook.on_completion(ctx, messages)).await?;
        }
        Ok(())
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
            let mut turn_count: u32 = 0;
            // `prompt_tokens` from each turn is the whole context sent that turn,
            // so the turn-over-turn delta shows how fast the conversation grows.
            let mut prev_prompt_tokens: u32 = 0;
            let start_time = Instant::now();

            // The conversation id is stable across the run; derive it from the
            // seed so hooks have it before the first response.
            let conversation_id = messages
                .first()
                .map_or_else(uuid::Uuid::new_v4, |m| m.conversation_id);
            let start_ctx = HookContext::new(conversation_id, 0);

            // Every edit to the history funnels through the ledger, which records
            // its provenance. The seed is the first recorded edit.
            let mut seed_conversation = Conversation::new();
            seed_conversation.id = conversation_id;
            let mut ledger = ContextLedger::new(seed_conversation);
            ledger.append(EditSource::core(), messages);

            // Conversation-start hooks inject always-on context before the
            // first LLM call (e.g. rule files that always apply).
            self.hooks_conversation_start(&start_ctx, &mut ledger, &cancel)
                .await?;

            // Hooks observe the seed snapshot before the first LLM call (e.g.
            // persistence records it so the input is durable even if the call
            // fails, and links the conversation to its spawning parent).
            self.hooks_messages(&start_ctx, ledger.messages(), &cancel).await?;

            loop {
                if cancel.is_cancelled() {
                    Err(CoreError::Cancelled("loop start".to_string()))?;
                }

                let turn_ctx = HookContext::new(conversation_id, turn_count);

                let mut request = ChatRequest::from((self.client.config(), ledger.snapshot()))
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
                    self.hooks_usage(&turn_ctx, usage, &cancel).await?;
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
                ledger.append(EditSource::model(), [assistant_message]);

                // Hooks observe the assistant message before tool execution
                // (e.g. persistence records it so a crashed run keeps its prefix).
                self.hooks_messages(&turn_ctx, ledger.messages(), &cancel).await?;

                if tool_calls.is_empty() {
                    // End-of-turn hooks run on the final turn too; they may
                    // compact or rewrite the history handed back to the caller.
                    let turn_stats =
                        self.hooks_turn_end(&turn_ctx, &mut ledger, &cancel).await?;
                    for s in turn_stats {
                        if s.was_compacted {
                            counter!("neuromance_compactions_total").increment(1);
                        }
                        yield CoreEvent::Compaction {
                            original_tokens: s.original_tokens,
                            compacted_tokens: s.compacted_tokens,
                            messages_summarized: s.messages_summarized,
                            was_compacted: s.was_compacted,
                        };
                    }

                    self.hooks_completion(&turn_ctx, ledger.messages(), &cancel).await?;

                    let total_ms =
                        u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);
                    info!(
                        turns = turn_number,
                        duration_ms = total_ms,
                        "chat completed",
                    );
                    yield CoreEvent::Completed(ledger.into_messages());
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
                    } else if let Some(decision) =
                        self.hooks_review_tool(&turn_ctx, tool_call, &cancel).await?
                    {
                        decision
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

                    // Captured from an executed tool so `after_tool` hooks can
                    // inject follow-on context once the result message is in place.
                    let mut tool_outcome: Option<(String, bool)> = None;

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
                                        result.clone(),
                                        tool_call.id.clone(),
                                        tool_call.function.name.clone(),
                                    )
                                    .map_err(|e| CoreError::ToolError(e.to_string()))?;
                                    ledger.append(EditSource::tool(), [tool_message]);
                                    tool_outcome = Some((result, true));
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
                                        error_msg.clone(),
                                        tool_call.id.clone(),
                                        tool_call.function.name.clone(),
                                    )
                                    .map_err(|e| CoreError::ToolError(e.to_string()))?;
                                    ledger.append(EditSource::tool(), [error_message]);
                                    tool_outcome = Some((error_msg, false));
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
                            ledger.append(EditSource::core(), [denial_message]);
                        }
                        ToolApproval::Quit => {
                            debug!("user quit during tool approval");
                            Err(CoreError::UserQuit(
                                "User quit during tool approval".to_string(),
                            ))?;
                        }
                    }

                    // After-tool hooks inject follow-on context (e.g. a rule
                    // file keyed to the touched path) right after the result.
                    if let Some((result, success)) = tool_outcome {
                        let injected = self
                            .hooks_after_tool(&turn_ctx, tool_call, &result, success, &cancel)
                            .await?;
                        for (source, msgs) in injected {
                            ledger.append(source, msgs);
                        }
                    }
                }

                debug!(tool_calls = tool_calls_count, "completed tool calls; continuing");

                // Hooks observe this turn's tool results (e.g. persistence
                // records them, retrying any backlog from earlier failed writes).
                self.hooks_messages(&turn_ctx, ledger.messages(), &cancel).await?;

                // End-of-turn hooks may rewrite or compact the history before
                // the next turn; emit any compaction they report.
                let turn_stats =
                    self.hooks_turn_end(&turn_ctx, &mut ledger, &cancel).await?;
                for s in turn_stats {
                    if s.was_compacted {
                        counter!("neuromance_compactions_total").increment(1);
                    }
                    yield CoreEvent::Compaction {
                        original_tokens: s.original_tokens,
                        compacted_tokens: s.compacted_tokens,
                        messages_summarized: s.messages_summarized,
                        was_compacted: s.was_compacted,
                    };
                }

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

/// Await a hook future under cancellation, mapping its error to
/// [`CoreError::Hook`] with the hook's name for context.
async fn run_hook<T>(
    cancel: &CancellationToken,
    name: &str,
    fut: impl Future<Output = anyhow::Result<T>>,
) -> Result<T, CoreError> {
    tokio::select! {
        biased;
        () = cancel.cancelled() => Err(CoreError::Cancelled(format!("hook {name}"))),
        r = fut => r.map_err(|source| CoreError::Hook {
            hook: name.to_string(),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use async_trait::async_trait;
    use neuromance_client::chat_completions::ChatCompletionsClient;
    use neuromance_common::client::Config;
    use neuromance_common::context::Operation;
    use neuromance_common::hook::{HookOutcome, TurnEnd};

    /// Hook that appends a marker system message in `on_turn_end`.
    struct AppendTurnEndHook(&'static str);

    #[async_trait]
    impl Hook for AppendTurnEndHook {
        async fn on_turn_end(
            &self,
            ctx: &HookContext,
            mut messages: Vec<Message>,
        ) -> anyhow::Result<TurnEnd> {
            messages.push(Message::system(ctx.conversation_id, self.0));
            Ok(TurnEnd::unchanged(messages))
        }
    }

    /// Hook whose `on_turn_end` always fails.
    struct FailingTurnEndHook;

    #[async_trait]
    impl Hook for FailingTurnEndHook {
        async fn on_turn_end(
            &self,
            _ctx: &HookContext,
            _messages: Vec<Message>,
        ) -> anyhow::Result<TurnEnd> {
            Err(anyhow::anyhow!("boom"))
        }
    }

    /// Hook that decides every tool call with a fixed approval.
    struct FixedReviewHook(ToolApproval);

    #[async_trait]
    impl Hook for FixedReviewHook {
        async fn review_tool(
            &self,
            _ctx: &HookContext,
            _call: &ToolCall,
        ) -> anyhow::Result<Option<ToolApproval>> {
            Ok(Some(self.0.clone()))
        }
    }

    /// Hook that injects a marker message in `on_conversation_start`.
    struct StartInjectHook(&'static str);

    #[async_trait]
    impl Hook for StartInjectHook {
        fn name(&self) -> &'static str {
            "start-inject"
        }

        async fn on_conversation_start(
            &self,
            ctx: &HookContext,
            _messages: &[Message],
        ) -> anyhow::Result<HookOutcome> {
            Ok(HookOutcome::inject(vec![Message::system(
                ctx.conversation_id,
                self.0,
            )]))
        }
    }

    /// Mirrors the real `SkillsHook`: a hook named `"skills"` that appends a
    /// `System` menu message at conversation start.
    struct SkillsMenuHook;

    #[async_trait]
    impl Hook for SkillsMenuHook {
        fn name(&self) -> &'static str {
            "skills"
        }

        async fn on_conversation_start(
            &self,
            ctx: &HookContext,
            _messages: &[Message],
        ) -> anyhow::Result<HookOutcome> {
            Ok(HookOutcome::inject(vec![Message::system(
                ctx.conversation_id,
                "<skills-menu>",
            )]))
        }
    }

    /// Hook that injects a marker message in `after_tool`.
    struct AfterToolInjectHook(&'static str);

    #[async_trait]
    impl Hook for AfterToolInjectHook {
        async fn after_tool(
            &self,
            ctx: &HookContext,
            _call: &ToolCall,
            _result: &str,
            _success: bool,
        ) -> anyhow::Result<HookOutcome> {
            Ok(HookOutcome::inject(vec![Message::system(
                ctx.conversation_id,
                self.0,
            )]))
        }
    }

    /// Hook that reports compaction stats from `on_turn_end`.
    struct CompactingHook;

    #[async_trait]
    impl Hook for CompactingHook {
        async fn on_turn_end(
            &self,
            _ctx: &HookContext,
            messages: Vec<Message>,
        ) -> anyhow::Result<TurnEnd> {
            Ok(TurnEnd::compacted(
                messages,
                CompactionStats::new(100, 50, 2, true),
            ))
        }
    }

    /// Returns a plain assistant reply with no tool calls, so a run reaches the
    /// no-tool completion path in a single turn.
    struct HugeUsageClient {
        config: Config,
    }

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

    /// A hook reporting compaction surfaces as a `CoreEvent::Compaction` on the
    /// no-tool completion path, before `Completed`.
    #[tokio::test]
    async fn test_run_emits_compaction_event_on_completion_path() {
        let mock_config = Config::new("mock", "mock-model");
        let mut core = Core::new(HugeUsageClient {
            config: mock_config,
        })
        .with_hook(Arc::new(CompactingHook));

        let conv_id = uuid::Uuid::new_v4();
        let messages = vec![
            Message::system(conv_id, "sys"),
            Message::user(conv_id, "hello"),
        ];

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
        assert!(completed.is_some(), "stream must complete");
    }

    /// `on_turn_end` hooks can transform the history.
    #[tokio::test]
    async fn test_on_turn_end_transforms_messages() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

        let core = Core::new(client).with_hook(Arc::new(AppendTurnEndHook("[compacted]")));

        let conv_id = uuid::Uuid::new_v4();
        let ctx = HookContext::new(conv_id, 0);
        let mut ledger = seeded_ledger(conv_id, vec![Message::user(conv_id, "hello")]);
        let stats = core
            .hooks_turn_end(&ctx, &mut ledger, &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(ledger.messages().len(), 2);
        assert_eq!(ledger.messages()[1].content, "[compacted]");
        assert!(stats.is_empty());
    }

    fn test_core() -> Core<ChatCompletionsClient> {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        Core::new(ChatCompletionsClient::new(config).expect("client"))
    }

    /// Builds a ledger whose conversation carries `conv_id`, seeded with `msgs`.
    fn seeded_ledger(conv_id: uuid::Uuid, msgs: Vec<Message>) -> ContextLedger {
        let mut conversation = Conversation::new();
        conversation.id = conv_id;
        let mut ledger = ContextLedger::new(conversation);
        ledger.append(EditSource::core(), msgs);
        ledger
    }

    /// `on_conversation_start` hooks append their injected messages.
    #[tokio::test]
    async fn test_on_conversation_start_injects() {
        let core = test_core().with_hook(Arc::new(StartInjectHook("[rules]")));
        let conv_id = uuid::Uuid::new_v4();
        let ctx = HookContext::new(conv_id, 0);
        let mut ledger = seeded_ledger(conv_id, vec![Message::user(conv_id, "hi")]);

        core.hooks_conversation_start(&ctx, &mut ledger, &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(ledger.messages().len(), 2);
        assert_eq!(ledger.messages()[1].content, "[rules]");
        // The injected message is attributed to the hook that produced it.
        let injected = ledger
            .metadata()
            .records_from(&EditSource::hook("start-inject"));
        assert_eq!(injected.len(), 1);
        assert_eq!(injected[0].roles, vec![MessageRole::System]);
    }

    /// Regression: the skills menu append is recorded in the ledger attributed
    /// to the `"skills"` hook, rather than slipping into the history untracked.
    #[tokio::test]
    async fn test_skills_menu_append_is_tracked_with_provenance() {
        let core = test_core().with_hook(Arc::new(SkillsMenuHook));
        let conv_id = uuid::Uuid::new_v4();
        let ctx = HookContext::new(conv_id, 0);
        let mut ledger = seeded_ledger(conv_id, vec![Message::user(conv_id, "hi")]);

        core.hooks_conversation_start(&ctx, &mut ledger, &CancellationToken::new())
            .await
            .unwrap();

        let from_skills = ledger.metadata().records_from(&EditSource::hook("skills"));
        assert_eq!(from_skills.len(), 1);
        assert_eq!(from_skills[0].operation, Operation::Append);
        assert_eq!(from_skills[0].roles, vec![MessageRole::System]);
        assert_eq!(ledger.messages().last().unwrap().content, "<skills-menu>");
    }

    /// The first hook to return a decision wins the approval.
    #[tokio::test]
    async fn test_review_tool_first_decision_wins() {
        let core = test_core()
            .with_hook(Arc::new(FixedReviewHook(ToolApproval::Denied("no".into()))))
            .with_hook(Arc::new(FixedReviewHook(ToolApproval::Approved)));
        let ctx = HookContext::new(uuid::Uuid::new_v4(), 0);
        let call = ToolCall::new("t", "");

        let decision = core
            .hooks_review_tool(&ctx, &call, &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(decision, Some(ToolApproval::Denied("no".into())));
    }

    /// With no deciding hook, approval abstains (caller falls back).
    #[tokio::test]
    async fn test_review_tool_all_abstain_returns_none() {
        let core = test_core().with_hook(Arc::new(AppendTurnEndHook("[x]")));
        let ctx = HookContext::new(uuid::Uuid::new_v4(), 0);
        let call = ToolCall::new("t", "");

        let decision = core
            .hooks_review_tool(&ctx, &call, &CancellationToken::new())
            .await
            .unwrap();

        assert!(decision.is_none());
    }

    /// `after_tool` collects injected messages across hooks in order.
    #[tokio::test]
    async fn test_after_tool_collects_injections() {
        let core = test_core()
            .with_hook(Arc::new(AfterToolInjectHook("[a]")))
            .with_hook(Arc::new(AfterToolInjectHook("[b]")));
        let ctx = HookContext::new(uuid::Uuid::new_v4(), 0);
        let call = ToolCall::new("read", "");

        let injected = core
            .hooks_after_tool(&ctx, &call, "result", true, &CancellationToken::new())
            .await
            .unwrap();

        let contents: Vec<&str> = injected
            .iter()
            .flat_map(|(_, msgs)| msgs.iter())
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(contents, vec!["[a]", "[b]"]);
    }

    /// Multiple `on_turn_end` hooks run in registration order.
    #[tokio::test]
    async fn test_multiple_hooks_run_in_registration_order() {
        let core = test_core()
            .with_hook(Arc::new(AppendTurnEndHook("[first]")))
            .with_hook(Arc::new(AppendTurnEndHook("[second]")));
        let conv_id = uuid::Uuid::new_v4();
        let ctx = HookContext::new(conv_id, 0);
        let mut ledger = seeded_ledger(conv_id, vec![]);

        core.hooks_turn_end(&ctx, &mut ledger, &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(ledger.messages()[0].content, "[first]");
        assert_eq!(ledger.messages()[1].content, "[second]");
    }

    /// `on_turn_end` surfaces the compaction stats a hook reports.
    #[tokio::test]
    async fn test_on_turn_end_collects_compaction_stats() {
        let core = test_core().with_hook(Arc::new(CompactingHook));
        let ctx = HookContext::new(uuid::Uuid::new_v4(), 0);
        let mut ledger = seeded_ledger(ctx.conversation_id, vec![]);

        let stats = core
            .hooks_turn_end(&ctx, &mut ledger, &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(stats.len(), 1);
        assert!(stats[0].was_compacted);
        assert_eq!(stats[0].original_tokens, 100);
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

    /// A failing hook surfaces as `CoreError::Hook` naming the hook.
    #[tokio::test]
    async fn test_hook_error_maps_to_core_error_hook() {
        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

        let core = Core::new(client).with_hook(Arc::new(FailingTurnEndHook));

        let ctx = HookContext::new(uuid::Uuid::new_v4(), 0);
        let mut ledger = seeded_ledger(
            ctx.conversation_id,
            vec![Message::user(ctx.conversation_id, "hello")],
        );
        let result = core
            .hooks_turn_end(&ctx, &mut ledger, &CancellationToken::new())
            .await;

        let err = result.unwrap_err();
        assert!(matches!(err, CoreError::Hook { .. }));
        assert!(err.to_string().contains("boom"));
    }
}
