//! Compaction module for context window management.
//!
//! This module provides functionality to compact long conversations into shorter
//! summaries while preserving important context. It uses a one-shot LLM call
//! to intelligently summarize conversation history.
//!
//! ## Example
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use neuromance_context::{Compactor, CompactionConfig, TokenCounter, ModelConfig};
//! use neuromance_client::ChatCompletionsClient;
//! use neuromance_common::Conversation;
//! use neuromance_common::client::Config;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let client = ChatCompletionsClient::new(
//!     Config::new("openai", "gpt-4o").with_api_key("sk-..."),
//! )?;
//!
//! // Without a tokenizer, token counts are heuristic estimates; add one
//! // via `with_token_counter` for accurate local counting.
//! let token_counter = TokenCounter::new(ModelConfig::gpt_oss_20b()).await?;
//!
//! let compactor = Compactor::new(client)
//!     .with_token_counter(Arc::new(token_counter))
//!     .with_config(CompactionConfig::new(4000).with_preserve_recent_turns(3));
//!
//! let conversation = Conversation::new();
//! let compacted = compactor.compact(&conversation).await?;
//! # Ok(())
//! # }
//! ```

use neuromance_client::LLMClient;
use neuromance_common::chat::{Conversation, Message, MessageRole};
use neuromance_common::client::ChatRequest;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

use crate::TokenCounter;
use crate::error::TokenCounterError;

/// Strategy for compacting conversations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStrategy {
    /// Single LLM call to summarize the middle section of the conversation.
    /// Most efficient for typical conversations.
    #[default]
    OneShot,

    /// Hierarchical summarization for very long conversations.
    /// Splits into chunks, summarizes each, then summarizes the summaries.
    Hierarchical,

    /// Simple truncation of older messages without summarization.
    /// Fastest but loses context.
    Truncate,
}

/// Configuration for conversation compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Target token budget after compaction.
    /// The compactor will attempt to reduce the conversation to this size.
    pub target_tokens: usize,

    /// Whether to always preserve the system prompt unchanged.
    /// Highly recommended to keep this true.
    pub preserve_system_prompt: bool,

    /// Number of recent message turns (user+assistant pairs) to preserve verbatim.
    /// These messages won't be summarized.
    pub preserve_recent_turns: usize,

    /// The compaction strategy to use.
    pub strategy: CompactionStrategy,

    /// Custom summarization prompt. If None, uses the default prompt.
    pub custom_prompt: Option<String>,

    /// Minimum tokens before compaction is triggered.
    /// Prevents unnecessary compaction of short conversations.
    pub compaction_threshold: Option<usize>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            target_tokens: 4000,
            preserve_system_prompt: true,
            preserve_recent_turns: 2,
            strategy: CompactionStrategy::OneShot,
            custom_prompt: None,
            compaction_threshold: None,
        }
    }
}

impl CompactionConfig {
    /// Creates a new compaction config with the specified target token budget.
    #[must_use]
    pub fn new(target_tokens: usize) -> Self {
        Self {
            target_tokens,
            ..Default::default()
        }
    }

    /// Sets whether to preserve the system prompt.
    #[must_use]
    pub fn with_preserve_system_prompt(mut self, preserve: bool) -> Self {
        self.preserve_system_prompt = preserve;
        self
    }

    /// Sets the number of recent turns to preserve.
    #[must_use]
    pub fn with_preserve_recent_turns(mut self, turns: usize) -> Self {
        self.preserve_recent_turns = turns;
        self
    }

    /// Sets the compaction strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: CompactionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets a custom summarization prompt.
    #[must_use]
    pub fn with_custom_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.custom_prompt = Some(prompt.into());
        self
    }

    /// Sets the minimum token threshold before compaction is triggered.
    #[must_use]
    pub fn with_compaction_threshold(mut self, threshold: usize) -> Self {
        self.compaction_threshold = Some(threshold);
        self
    }

    /// Effective trigger threshold: `compaction_threshold` if set, else `target_tokens`.
    #[must_use]
    pub fn threshold(&self) -> usize {
        self.compaction_threshold.unwrap_or(self.target_tokens)
    }
}

/// Result of a compaction operation.
///
/// When the compactor has no tokenizer configured, `original_tokens` and
/// `compacted_tokens` are heuristic estimates (~4 chars/token). The exception
/// is `original_tokens` from
/// [`Compactor::compact_with_reported_tokens`], which is the provider-reported
/// count passed by the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    /// The compacted conversation.
    pub conversation: Conversation,

    /// Original token count before compaction.
    pub original_tokens: usize,

    /// Token count after compaction.
    pub compacted_tokens: usize,

    /// Number of messages that were summarized.
    pub messages_summarized: usize,

    /// The generated summary (if any).
    pub summary: Option<String>,

    /// Whether compaction was actually performed.
    pub was_compacted: bool,
}

/// Compactor for reducing conversation size while preserving context.
///
/// Uses an LLM to intelligently summarize conversation history, keeping
/// important context while reducing token count.
pub struct Compactor<C: LLMClient> {
    client: C,
    token_counter: Option<Arc<TokenCounter>>,
    config: CompactionConfig,
}

impl<C: LLMClient> std::fmt::Debug for Compactor<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compactor")
            .field("client", &"<LLMClient>")
            .field("token_counter", &self.token_counter)
            .field("config", &self.config)
            .finish()
    }
}

impl<C: LLMClient> Compactor<C> {
    /// Creates a new compactor with no local tokenizer.
    ///
    /// Token counts in results are heuristic estimates (~4 chars/token) unless
    /// reported counts are supplied via
    /// [`compact_with_reported_tokens`](Self::compact_with_reported_tokens).
    #[must_use]
    pub fn new(client: C) -> Self {
        Self {
            client,
            token_counter: None,
            config: CompactionConfig::default(),
        }
    }

    /// Sets a token counter for accurate local counting.
    #[must_use]
    pub fn with_token_counter(mut self, token_counter: Arc<TokenCounter>) -> Self {
        self.token_counter = Some(token_counter);
        self
    }

    /// Sets the compaction configuration.
    #[must_use]
    pub fn with_config(mut self, config: CompactionConfig) -> Self {
        self.config = config;
        self
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &CompactionConfig {
        &self.config
    }

    /// Returns the token counter, if one is configured.
    pub fn token_counter(&self) -> Option<&TokenCounter> {
        self.token_counter.as_deref()
    }

    /// Counts conversation tokens with the configured tokenizer, falling back
    /// to a heuristic estimate (~4 chars/token plus per-message overhead).
    fn count_or_estimate(&self, conversation: &Conversation) -> Result<usize, TokenCounterError> {
        match &self.token_counter {
            Some(counter) => counter.count_conversation_tokens(conversation),
            None => Ok(estimate_conversation_tokens(conversation)),
        }
    }

    /// Compacts a conversation according to the configured strategy.
    ///
    /// Returns a `CompactionResult` containing the compacted conversation
    /// and metadata about the compaction process.
    pub async fn compact(
        &self,
        conversation: &Conversation,
    ) -> Result<CompactionResult, TokenCounterError> {
        let original_tokens = self.count_or_estimate(conversation)?;
        self.compact_inner(conversation, original_tokens).await
    }

    /// Compacts a conversation using a provider-reported pre-compaction token
    /// count (e.g. from `Usage`), skipping local counting for the trigger.
    ///
    /// `original_tokens` in the result is `reported_tokens`; `compacted_tokens`
    /// is counted (or estimated) locally.
    pub async fn compact_with_reported_tokens(
        &self,
        conversation: &Conversation,
        reported_tokens: usize,
    ) -> Result<CompactionResult, TokenCounterError> {
        self.compact_inner(conversation, reported_tokens).await
    }

    /// Threshold check and strategy dispatch shared by [`compact`](Self::compact)
    /// and [`compact_with_reported_tokens`](Self::compact_with_reported_tokens).
    async fn compact_inner(
        &self,
        conversation: &Conversation,
        original_tokens: usize,
    ) -> Result<CompactionResult, TokenCounterError> {
        info!(
            "Compaction requested: {} tokens, target: {} tokens",
            original_tokens, self.config.target_tokens
        );

        if original_tokens <= self.config.threshold() {
            debug!("Conversation below threshold, skipping compaction");
            return Ok(CompactionResult {
                conversation: conversation.clone(),
                original_tokens,
                compacted_tokens: original_tokens,
                messages_summarized: 0,
                summary: None,
                was_compacted: false,
            });
        }

        match self.config.strategy {
            CompactionStrategy::OneShot => {
                self.compact_one_shot(conversation, original_tokens).await
            }
            CompactionStrategy::Hierarchical => {
                self.compact_hierarchical(conversation, original_tokens)
                    .await
            }
            CompactionStrategy::Truncate => {
                self.compact_truncate(conversation, original_tokens).await
            }
        }
    }

    /// Performs one-shot compaction using a single LLM call.
    async fn compact_one_shot(
        &self,
        conversation: &Conversation,
        original_tokens: usize,
    ) -> Result<CompactionResult, TokenCounterError> {
        let messages = conversation.get_messages();

        if messages.is_empty() {
            return Ok(CompactionResult {
                conversation: conversation.clone(),
                original_tokens,
                compacted_tokens: original_tokens,
                messages_summarized: 0,
                summary: None,
                was_compacted: false,
            });
        }

        // Split conversation into sections
        let (system_msg, middle_msgs, recent_msgs) = self.split_conversation(messages);

        debug!(
            "Split conversation: system={}, middle={}, recent={}",
            system_msg.is_some(),
            middle_msgs.len(),
            recent_msgs.len()
        );

        // If there's nothing to summarize in the middle, return as-is
        if middle_msgs.is_empty() {
            return Ok(CompactionResult {
                conversation: conversation.clone(),
                original_tokens,
                compacted_tokens: original_tokens,
                messages_summarized: 0,
                summary: None,
                was_compacted: false,
            });
        }

        // Generate summary of middle section
        let summary = self.generate_summary(&middle_msgs).await?;

        // Reconstruct conversation
        let compacted_conversation =
            self.reconstruct_conversation(conversation, system_msg, &summary, &recent_msgs)?;

        let compacted_tokens = self.count_or_estimate(&compacted_conversation)?;

        info!(
            "Compaction complete: {} -> {} tokens ({} messages summarized)",
            original_tokens,
            compacted_tokens,
            middle_msgs.len()
        );

        Ok(CompactionResult {
            conversation: compacted_conversation,
            original_tokens,
            compacted_tokens,
            messages_summarized: middle_msgs.len(),
            summary: Some(summary),
            was_compacted: true,
        })
    }

    /// Performs hierarchical compaction for very long conversations.
    async fn compact_hierarchical(
        &self,
        conversation: &Conversation,
        original_tokens: usize,
    ) -> Result<CompactionResult, TokenCounterError> {
        let messages = conversation.get_messages();

        if messages.is_empty() {
            return Ok(CompactionResult {
                conversation: conversation.clone(),
                original_tokens,
                compacted_tokens: original_tokens,
                messages_summarized: 0,
                summary: None,
                was_compacted: false,
            });
        }

        let (system_msg, middle_msgs, recent_msgs) = self.split_conversation(messages);

        if middle_msgs.is_empty() {
            return Ok(CompactionResult {
                conversation: conversation.clone(),
                original_tokens,
                compacted_tokens: original_tokens,
                messages_summarized: 0,
                summary: None,
                was_compacted: false,
            });
        }

        // Split middle into chunks and summarize each
        let chunk_size = 10; // messages per chunk
        let chunks: Vec<_> = middle_msgs.chunks(chunk_size).collect();

        let mut chunk_summaries = Vec::new();
        for chunk in &chunks {
            let chunk_summary = self.generate_summary(chunk).await?;
            chunk_summaries.push(chunk_summary);
        }

        // If we have multiple summaries, summarize the summaries
        let final_summary = if chunk_summaries.len() > 1 {
            self.summarize_summaries(&chunk_summaries).await?
        } else {
            chunk_summaries.into_iter().next().ok_or_else(|| {
                TokenCounterError::Compaction(
                    "no chunk summaries were produced for a non-empty message history".to_string(),
                )
            })?
        };

        let compacted_conversation =
            self.reconstruct_conversation(conversation, system_msg, &final_summary, &recent_msgs)?;

        let compacted_tokens = self.count_or_estimate(&compacted_conversation)?;

        Ok(CompactionResult {
            conversation: compacted_conversation,
            original_tokens,
            compacted_tokens,
            messages_summarized: middle_msgs.len(),
            summary: Some(final_summary),
            was_compacted: true,
        })
    }

    /// Performs simple truncation without LLM summarization.
    async fn compact_truncate(
        &self,
        conversation: &Conversation,
        original_tokens: usize,
    ) -> Result<CompactionResult, TokenCounterError> {
        let messages = conversation.get_messages();

        if messages.is_empty() {
            return Ok(CompactionResult {
                conversation: conversation.clone(),
                original_tokens,
                compacted_tokens: original_tokens,
                messages_summarized: 0,
                summary: None,
                was_compacted: false,
            });
        }

        let (system_msg, middle_msgs, recent_msgs) = self.split_conversation(messages);

        // For truncation, we just keep system + recent, with a note about truncation
        let truncation_note =
            "[Previous conversation history truncated to fit context window]".to_string();

        let compacted_conversation = self.reconstruct_conversation(
            conversation,
            system_msg,
            &truncation_note,
            &recent_msgs,
        )?;

        let compacted_tokens = self.count_or_estimate(&compacted_conversation)?;

        Ok(CompactionResult {
            conversation: compacted_conversation,
            original_tokens,
            compacted_tokens,
            messages_summarized: middle_msgs.len(),
            summary: Some(truncation_note),
            was_compacted: true,
        })
    }

    /// Splits a conversation into system message, middle (to summarize), and recent (to preserve).
    fn split_conversation<'a>(
        &self,
        messages: &'a [Message],
    ) -> (Option<&'a Message>, Vec<&'a Message>, Vec<&'a Message>) {
        if messages.is_empty() {
            return (None, vec![], vec![]);
        }

        let mut system_msg = None;
        let mut start_idx = 0;

        // Extract system message if present and configured to preserve
        if self.config.preserve_system_prompt && messages[0].role == MessageRole::System {
            system_msg = Some(&messages[0]);
            start_idx = 1;
        }

        let remaining = &messages[start_idx..];

        // Calculate how many messages to preserve at the end
        // A "turn" is typically user + assistant, so we preserve 2 * preserve_recent_turns messages
        let preserve_count = self.config.preserve_recent_turns * 2;
        let preserve_count = preserve_count.min(remaining.len());

        let split_point = remaining.len().saturating_sub(preserve_count);

        let middle: Vec<&Message> = remaining[..split_point].iter().collect();
        let recent: Vec<&Message> = remaining[split_point..].iter().collect();

        (system_msg, middle, recent)
    }

    /// Generates a summary of the given messages using the LLM.
    async fn generate_summary(&self, messages: &[&Message]) -> Result<String, TokenCounterError> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        let formatted_messages = self.format_messages_for_summary(messages);
        let prompt = self.build_summary_prompt(&formatted_messages);

        debug!("Generating summary for {} messages", messages.len());

        let summary_message = Message::new(uuid::Uuid::new_v4(), MessageRole::User, prompt);

        let summary_request = ChatRequest::new(vec![summary_message])
            .with_model(self.client.config().model.clone())
            .with_temperature(0.3);

        let response = self.client.chat(&summary_request).await.map_err(|e| {
            TokenCounterError::CompactionLlm {
                context: "Failed to generate summary".to_string(),
                source: Box::new(e),
            }
        })?;

        Ok(response.message.content.clone())
    }

    /// Summarizes multiple chunk summaries into a final summary.
    async fn summarize_summaries(&self, summaries: &[String]) -> Result<String, TokenCounterError> {
        let combined = summaries.join("\n\n---\n\n");
        let prompt = format!(
            "Combine and consolidate the following conversation \
             summaries into a single, coherent summary. Remove \
             redundancy and preserve the most important \
             information:\n\n{}\n\nConsolidated Summary:",
            combined
        );

        let consolidation_message = Message::new(uuid::Uuid::new_v4(), MessageRole::User, prompt);

        let request = ChatRequest::new(vec![consolidation_message])
            .with_model(self.client.config().model.clone())
            .with_temperature(0.3);

        let response =
            self.client
                .chat(&request)
                .await
                .map_err(|e| TokenCounterError::CompactionLlm {
                    context: "Failed to consolidate summaries".to_string(),
                    source: Box::new(e),
                })?;

        Ok(response.message.content.clone())
    }

    /// Formats messages for inclusion in the summary prompt.
    fn format_messages_for_summary(&self, messages: &[&Message]) -> String {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "System",
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant",
                    MessageRole::Tool => "Tool",
                    _ => "Unknown",
                };

                let mut formatted = format!("[{}]: {}", role, msg.content);

                // Include tool call info if present
                if !msg.tool_calls.is_empty() {
                    let tool_names: Vec<_> =
                        msg.tool_calls.iter().map(|tc| &tc.function.name).collect();
                    formatted.push_str(&format!("\n  (Called tools: {:?})", tool_names));
                }

                // Include tool response info
                if let Some(ref tool_call_id) = msg.tool_call_id {
                    formatted.push_str(&format!("\n  (Response to tool call: {})", tool_call_id));
                }

                formatted
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Builds the prompt for summarization.
    fn build_summary_prompt(&self, formatted_messages: &str) -> String {
        if let Some(ref custom) = self.config.custom_prompt {
            return format!("{}\n\nConversation:\n{}", custom, formatted_messages);
        }

        format!(
            r#"You are summarizing a conversation history for an AI assistant. Create a concise but comprehensive summary that preserves critical context.

PRESERVE:
- All tool call results that affect current state or provide important information
- Key decisions, agreements, and requirements established
- User preferences, constraints, and explicit instructions
- Error states or issues that were encountered and their resolutions
- Any data, values, or facts that may be referenced later

OMIT:
- Pleasantries, greetings, and acknowledgments
- Redundant confirmations and repetitive exchanges
- Failed tool attempts that were successfully retried
- Verbose explanations when the outcome is what matters

FORMAT:
Write a structured summary with clear sections. Use bullet points for clarity.
Keep the summary under 500 words unless critical context requires more.

Conversation to summarize:
{}

Summary:"#,
            formatted_messages
        )
    }

    /// Reconstructs the conversation with the summary inserted.
    fn reconstruct_conversation(
        &self,
        original: &Conversation,
        system_msg: Option<&Message>,
        summary: &str,
        recent_msgs: &[&Message],
    ) -> Result<Conversation, TokenCounterError> {
        let mut new_conversation = Conversation::new();
        new_conversation.id = original.id;
        new_conversation.title = original.title.clone();
        new_conversation.description = original.description.clone();
        new_conversation.metadata = original.metadata.clone();

        if let Some(system) = system_msg {
            new_conversation.add_message(system.clone())?;
        }

        if !summary.is_empty() {
            let summary_msg = Message::new(
                original.id,
                MessageRole::System,
                format!(
                    "[Conversation Summary - Previous messages \
                     have been summarized]\n\n{}",
                    summary
                ),
            );
            new_conversation.add_message(summary_msg)?;
        }

        for msg in recent_msgs {
            new_conversation.add_message((*msg).clone())?;
        }

        Ok(new_conversation)
    }

    /// Checks if a conversation needs compaction based on the current configuration.
    ///
    /// Counts with the configured tokenizer, or a heuristic estimate when none
    /// is set. Prefer [`needs_compaction_reported`](Self::needs_compaction_reported)
    /// when a provider-reported count is available.
    pub fn needs_compaction(&self, conversation: &Conversation) -> Result<bool, TokenCounterError> {
        let tokens = self.count_or_estimate(conversation)?;
        Ok(tokens > self.config.threshold())
    }

    /// Checks if compaction is needed given an externally known token count
    /// (e.g. provider-reported `Usage`).
    #[must_use]
    pub fn needs_compaction_reported(&self, reported_tokens: usize) -> bool {
        reported_tokens > self.config.threshold()
    }

    /// Compacts a `Vec<Message>` directly, bridging to the `Conversation`-based API.
    ///
    /// This is designed to be used with `Core::with_turn_callback` for transparent
    /// context compaction during the tool loop. It handles conversation ID remapping
    /// so that messages from any conversation can be compacted.
    ///
    /// Returns the original messages unchanged if compaction is not needed.
    pub async fn compact_messages(
        &self,
        messages: Vec<Message>,
    ) -> Result<Vec<Message>, TokenCounterError> {
        if messages.is_empty() {
            return Ok(messages);
        }

        // Build a temporary Conversation, remapping conversation_ids to satisfy add_message validation
        let mut conversation = Conversation::new();
        let conv_id = conversation.id;
        let remapped: Vec<Message> = messages
            .iter()
            .map(|msg| {
                let mut m = msg.clone();
                m.conversation_id = conv_id;
                m
            })
            .collect();
        conversation.messages = Arc::new(remapped);

        if !self.needs_compaction(&conversation)? {
            return Ok(messages);
        }

        let result = self.compact(&conversation).await?;

        if result.was_compacted {
            // Restore original conversation_id from input messages
            let original_id = messages
                .first()
                .map(|m| m.conversation_id)
                .unwrap_or(conv_id);
            Ok(result
                .conversation
                .get_messages()
                .iter()
                .map(|msg| {
                    let mut m = msg.clone();
                    m.conversation_id = original_id;
                    m
                })
                .collect())
        } else {
            Ok(messages)
        }
    }

    /// Returns the current token count for a conversation.
    ///
    /// Counts with the configured tokenizer, or a heuristic estimate
    /// (~4 chars/token) when none is set.
    pub fn count_tokens(&self, conversation: &Conversation) -> Result<usize, TokenCounterError> {
        self.count_or_estimate(conversation)
    }
}

/// Heuristic token estimate for a conversation: ~4 characters per token over
/// message content and tool-call arguments, plus a fixed per-message overhead
/// for role markers.
fn estimate_conversation_tokens(conversation: &Conversation) -> usize {
    const CHARS_PER_TOKEN: usize = 4;
    const PER_MESSAGE_OVERHEAD: usize = 4;

    conversation
        .get_messages()
        .iter()
        .map(|msg| {
            let tool_call_chars: usize = msg
                .tool_calls
                .iter()
                .map(|tc| tc.function.name.len() + tc.function.arguments.len())
                .sum();
            (msg.content.len() + tool_call_chars) / CHARS_PER_TOKEN + PER_MESSAGE_OVERHEAD
        })
        .sum()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures::Stream;

    use neuromance_client::ClientError;
    use neuromance_common::client::{ChatChunk, ChatResponse, Config};

    use super::*;

    /// `LLMClient` stub that returns a canned summary and counts invocations,
    /// so tests can assert whether the compaction LLM call happened.
    struct MockSummaryClient {
        config: Config,
        summary: String,
        calls: AtomicUsize,
    }

    impl MockSummaryClient {
        fn new(summary: &str) -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                summary: summary.to_string(),
                calls: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LLMClient for &MockSummaryClient {
        fn config(&self) -> &Config {
            &self.config
        }

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let conv_id = request
                .messages
                .first()
                .map_or_else(uuid::Uuid::new_v4, |m| m.conversation_id);
            Ok(ChatResponse {
                message: Message::assistant(conv_id, &self.summary),
                model: "mock-model".to_string(),
                usage: None,
                finish_reason: None,
                created_at: chrono::Utc::now(),
                response_id: None,
                metadata: std::collections::HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
        {
            Ok(Box::pin(futures::stream::pending()))
        }

        fn supports_tools(&self) -> bool {
            false
        }

        fn supports_streaming(&self) -> bool {
            false
        }
    }

    fn conversation_with_messages(contents: &[(MessageRole, &str)]) -> Conversation {
        let mut conversation = Conversation::new();
        for (role, content) in contents {
            let msg = Message::new(conversation.id, *role, (*content).to_string());
            conversation.add_message(msg).expect("add message");
        }
        conversation
    }

    #[test]
    fn test_threshold_prefers_explicit_over_target() {
        let config = CompactionConfig::new(4000);
        assert_eq!(config.threshold(), 4000);

        let config = CompactionConfig::new(4000).with_compaction_threshold(6000);
        assert_eq!(config.threshold(), 6000);
    }

    #[test]
    fn test_needs_compaction_reported_uses_threshold() {
        let client = MockSummaryClient::new("summary");
        let compactor = Compactor::new(&client).with_config(CompactionConfig::new(4000));

        assert!(!compactor.needs_compaction_reported(4000));
        assert!(compactor.needs_compaction_reported(4001));
    }

    #[test]
    fn test_heuristic_estimate_scales_with_content() {
        let client = MockSummaryClient::new("summary");
        let compactor = Compactor::new(&client);

        let short = conversation_with_messages(&[(MessageRole::User, "hi")]);
        let long = conversation_with_messages(&[(
            MessageRole::User,
            "a much longer message with substantially more content to count",
        )]);

        let short_tokens = compactor.count_tokens(&short).expect("estimate");
        let long_tokens = compactor.count_tokens(&long).expect("estimate");
        assert!(
            long_tokens > short_tokens,
            "expected longer content to estimate more tokens: {long_tokens} <= {short_tokens}"
        );
    }

    #[tokio::test]
    async fn test_compact_with_reported_tokens_below_threshold_skips_llm() {
        let client = MockSummaryClient::new("summary");
        let compactor = Compactor::new(&client).with_config(CompactionConfig::new(4000));

        let conversation = conversation_with_messages(&[
            (MessageRole::User, "hello"),
            (MessageRole::Assistant, "world"),
        ]);

        let result = compactor
            .compact_with_reported_tokens(&conversation, 100)
            .await
            .expect("compact");

        assert!(!result.was_compacted);
        assert_eq!(result.original_tokens, 100);
        assert_eq!(
            client.call_count(),
            0,
            "LLM must not be called below threshold"
        );
    }

    #[tokio::test]
    async fn test_compact_with_reported_tokens_summarizes_middle() {
        let client = MockSummaryClient::new("the summary of earlier messages");
        let compactor = Compactor::new(&client)
            .with_config(CompactionConfig::new(1000).with_preserve_recent_turns(1));

        let conversation = conversation_with_messages(&[
            (MessageRole::System, "be helpful"),
            (MessageRole::User, "old question one"),
            (MessageRole::Assistant, "old answer one"),
            (MessageRole::User, "old question two"),
            (MessageRole::Assistant, "old answer two"),
            (MessageRole::User, "recent question"),
            (MessageRole::Assistant, "recent answer"),
        ]);

        let result = compactor
            .compact_with_reported_tokens(&conversation, 50_000)
            .await
            .expect("compact");

        assert!(result.was_compacted);
        assert_eq!(result.original_tokens, 50_000);
        assert_eq!(result.messages_summarized, 4);
        assert!(client.call_count() >= 1, "LLM summary call expected");

        let messages = result.conversation.get_messages();
        // system prompt + summary message + 2 preserved recent messages
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].content, "be helpful");
        assert!(
            messages[1]
                .content
                .contains("the summary of earlier messages")
        );
        assert_eq!(messages[2].content, "recent question");
        assert_eq!(messages[3].content, "recent answer");
    }

    #[test]
    fn test_compaction_strategy_serde_is_snake_case() {
        let json = serde_json::to_string(&CompactionStrategy::OneShot).expect("serialize");
        assert_eq!(json, "\"one_shot\"");
        let parsed: CompactionStrategy =
            serde_json::from_str("\"hierarchical\"").expect("deserialize");
        assert_eq!(parsed, CompactionStrategy::Hierarchical);
        assert!(serde_json::from_str::<CompactionStrategy>("\"OneShot\"").is_err());
    }

    #[test]
    fn test_compaction_config_builder() {
        let config = CompactionConfig::new(8000)
            .with_preserve_system_prompt(true)
            .with_preserve_recent_turns(3)
            .with_strategy(CompactionStrategy::Hierarchical)
            .with_compaction_threshold(10000);

        assert_eq!(config.target_tokens, 8000);
        assert!(config.preserve_system_prompt);
        assert_eq!(config.preserve_recent_turns, 3);
        assert_eq!(config.strategy, CompactionStrategy::Hierarchical);
        assert_eq!(config.compaction_threshold, Some(10000));
    }

    #[test]
    fn test_compaction_config_default() {
        let config = CompactionConfig::default();

        assert_eq!(config.target_tokens, 4000);
        assert!(config.preserve_system_prompt);
        assert_eq!(config.preserve_recent_turns, 2);
        assert_eq!(config.strategy, CompactionStrategy::OneShot);
        assert!(config.custom_prompt.is_none());
    }

    #[tokio::test]
    async fn test_compact_messages_empty() {
        use neuromance_client::ChatCompletionsClient;
        use neuromance_common::client::Config;

        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

        // Create a minimal tokenizer for testing
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "hello": 1, "world": 2 },
                "unk_token": "[UNK]"
            },
            "pre_tokenizer": { "type": "Whitespace" }
        });
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json.to_string()).expect("tokenizer");
        let counter = crate::TokenCounter::from_tokenizer(tokenizer);

        let compactor = Compactor::new(client).with_token_counter(Arc::new(counter));
        let result = compactor
            .compact_messages(vec![])
            .await
            .expect("should succeed");
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_compact_messages_below_threshold() {
        use neuromance_client::ChatCompletionsClient;
        use neuromance_common::client::Config;

        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "hello": 1, "world": 2 },
                "unk_token": "[UNK]"
            },
            "pre_tokenizer": { "type": "Whitespace" }
        });
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json.to_string()).expect("tokenizer");
        let counter = crate::TokenCounter::from_tokenizer(tokenizer);

        // Default target is 4000 tokens, so a tiny message should be below threshold
        let compactor = Compactor::new(client).with_token_counter(Arc::new(counter));
        let conv_id = uuid::Uuid::new_v4();
        let messages = vec![
            Message::user(conv_id, "hello"),
            Message::assistant(conv_id, "world"),
        ];

        let result = compactor
            .compact_messages(messages.clone())
            .await
            .expect("should succeed");

        // Messages below threshold should be returned unchanged
        assert_eq!(result.len(), messages.len());
        assert_eq!(result[0].content, "hello");
        assert_eq!(result[1].content, "world");
        // conversation_ids should be preserved
        assert_eq!(result[0].conversation_id, conv_id);
        assert_eq!(result[1].conversation_id, conv_id);
    }

    #[test]
    fn test_split_conversation_clamps_preserve_to_available() {
        use neuromance_client::ChatCompletionsClient;
        use neuromance_common::client::Config;

        let config = Config::new("test", "test-model").with_api_key("test-key");
        let client = ChatCompletionsClient::new(config).expect("Failed to create client");

        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "hello": 1, "world": 2 },
                "unk_token": "[UNK]"
            },
            "pre_tokenizer": { "type": "Whitespace" }
        });
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json.to_string()).expect("tokenizer");
        let counter = crate::TokenCounter::from_tokenizer(tokenizer);

        // preserve_recent_turns=5 wants to keep 10 messages, but only 2 exist,
        // so the preserve count clamps and nothing is left in the middle.
        let compactor = Compactor::new(client)
            .with_token_counter(Arc::new(counter))
            .with_config(CompactionConfig::new(4000).with_preserve_recent_turns(5));

        let conv_id = uuid::Uuid::new_v4();
        let messages = vec![
            Message::user(conv_id, "hello"),
            Message::assistant(conv_id, "world"),
        ];

        let (system, middle, recent) = compactor.split_conversation(&messages);
        assert!(system.is_none());
        assert!(middle.is_empty());
        assert_eq!(recent.len(), 2);
    }
}
