//! Compaction module for context window management.
//!
//! This module provides functionality to compact long conversations into shorter
//! summaries while preserving important context. It uses a one-shot LLM call
//! to intelligently summarize conversation history.
//!
//! ## Example
//!
//! ```rust,ignore
//! use neuromance_context::{Compactor, CompactionConfig, CompactionStrategy, TokenCounter, ModelConfig};
//! use neuromance_client::OpenAIClient;
//! use neuromance_common::Conversation;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let client = OpenAIClient::new(config)?;
//! let token_counter = TokenCounter::new(ModelConfig::gpt_oss_20b()).await?;
//!
//! let compactor = Compactor::new(client, token_counter)
//!     .with_config(CompactionConfig {
//!         target_tokens: 4000,
//!         preserve_system_prompt: true,
//!         preserve_recent_turns: 3,
//!         strategy: CompactionStrategy::OneShot,
//!     });
//!
//! let compacted = compactor.compact(&conversation).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use tracing::{debug, info};
use neuromance_client::LLMClient;
use neuromance_common::chat::{Conversation, Message, MessageRole};
use neuromance_common::client::ChatRequest;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::TokenCounter;

/// Strategy for compacting conversations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
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
    pub fn new(target_tokens: usize) -> Self {
        Self {
            target_tokens,
            ..Default::default()
        }
    }

    /// Sets whether to preserve the system prompt.
    pub fn with_preserve_system_prompt(mut self, preserve: bool) -> Self {
        self.preserve_system_prompt = preserve;
        self
    }

    /// Sets the number of recent turns to preserve.
    pub fn with_preserve_recent_turns(mut self, turns: usize) -> Self {
        self.preserve_recent_turns = turns;
        self
    }

    /// Sets the compaction strategy.
    pub fn with_strategy(mut self, strategy: CompactionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets a custom summarization prompt.
    pub fn with_custom_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.custom_prompt = Some(prompt.into());
        self
    }

    /// Sets the minimum token threshold before compaction is triggered.
    pub fn with_compaction_threshold(mut self, threshold: usize) -> Self {
        self.compaction_threshold = Some(threshold);
        self
    }
}

/// Result of a compaction operation.
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
    token_counter: Arc<TokenCounter>,
    config: CompactionConfig,
}

impl<C: LLMClient> Compactor<C> {
    /// Creates a new compactor with the given LLM client and token counter.
    pub fn new(client: C, token_counter: TokenCounter) -> Self {
        Self {
            client,
            token_counter: Arc::new(token_counter),
            config: CompactionConfig::default(),
        }
    }

    /// Creates a new compactor with a shared token counter.
    pub fn with_shared_counter(client: C, token_counter: Arc<TokenCounter>) -> Self {
        Self {
            client,
            token_counter,
            config: CompactionConfig::default(),
        }
    }

    /// Sets the compaction configuration.
    pub fn with_config(mut self, config: CompactionConfig) -> Self {
        self.config = config;
        self
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &CompactionConfig {
        &self.config
    }

    /// Returns a reference to the token counter.
    pub fn token_counter(&self) -> &TokenCounter {
        &self.token_counter
    }

    /// Compacts a conversation according to the configured strategy.
    ///
    /// Returns a `CompactionResult` containing the compacted conversation
    /// and metadata about the compaction process.
    pub async fn compact(&self, conversation: &Conversation) -> Result<CompactionResult> {
        let original_tokens = self
            .token_counter
            .count_conversation_tokens(conversation)
            .context("Failed to count original conversation tokens")?;

        info!(
            "Compaction requested: {} tokens, target: {} tokens",
            original_tokens, self.config.target_tokens
        );

        // Check if compaction is needed
        let threshold = self
            .config
            .compaction_threshold
            .unwrap_or(self.config.target_tokens);

        if original_tokens <= threshold {
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
    ) -> Result<CompactionResult> {
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

        let compacted_tokens = self
            .token_counter
            .count_conversation_tokens(&compacted_conversation)
            .context("Failed to count compacted conversation tokens")?;

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
    ) -> Result<CompactionResult> {
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
            chunk_summaries.into_iter().next().unwrap_or_default()
        };

        let compacted_conversation =
            self.reconstruct_conversation(conversation, system_msg, &final_summary, &recent_msgs)?;

        let compacted_tokens = self
            .token_counter
            .count_conversation_tokens(&compacted_conversation)
            .context("Failed to count compacted conversation tokens")?;

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
    ) -> Result<CompactionResult> {
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

        let (system_msg, _middle_msgs, recent_msgs) = self.split_conversation(messages);

        // For truncation, we just keep system + recent, with a note about truncation
        let truncation_note =
            "[Previous conversation history truncated to fit context window]".to_string();

        let compacted_conversation = self.reconstruct_conversation(
            conversation,
            system_msg,
            &truncation_note,
            &recent_msgs,
        )?;

        let compacted_tokens = self
            .token_counter
            .count_conversation_tokens(&compacted_conversation)
            .context("Failed to count compacted conversation tokens")?;

        Ok(CompactionResult {
            conversation: compacted_conversation,
            original_tokens,
            compacted_tokens,
            messages_summarized: _middle_msgs.len(),
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
    async fn generate_summary(&self, messages: &[&Message]) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        let formatted_messages = self.format_messages_for_summary(messages);
        let prompt = self.build_summary_prompt(&formatted_messages);

        debug!("Generating summary for {} messages", messages.len());

        // Create a user message for the summary request
        let summary_message = Message::new(uuid::Uuid::new_v4(), MessageRole::User, prompt);

        let summary_request = ChatRequest::new(vec![summary_message])
            .with_model(self.client.config().model.clone())
            .with_temperature(0.3); // Lower temperature for more consistent summaries

        let response = self
            .client
            .chat(&summary_request)
            .await
            .context("Failed to generate summary")?;

        Ok(response.message.content.clone())
    }

    /// Summarizes multiple chunk summaries into a final summary.
    async fn summarize_summaries(&self, summaries: &[String]) -> Result<String> {
        let combined = summaries.join("\n\n---\n\n");
        let prompt = format!(
            "Combine and consolidate the following conversation summaries into a single, \
             coherent summary. Remove redundancy and preserve the most important information:\n\n\
             {}\n\n\
             Consolidated Summary:",
            combined
        );

        // Create a user message for the consolidation request
        let consolidation_message = Message::new(uuid::Uuid::new_v4(), MessageRole::User, prompt);

        let request = ChatRequest::new(vec![consolidation_message])
            .with_model(self.client.config().model.clone())
            .with_temperature(0.3);

        let response = self
            .client
            .chat(&request)
            .await
            .context("Failed to consolidate summaries")?;

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
    ) -> Result<Conversation> {
        let mut new_conversation = Conversation::new();
        new_conversation.id = original.id;
        new_conversation.title = original.title.clone();
        new_conversation.description = original.description.clone();
        new_conversation.metadata = original.metadata.clone();

        // Add system message if present
        if let Some(system) = system_msg {
            new_conversation.add_message(system.clone())?;
        }

        // Add summary as a system message (context injection)
        if !summary.is_empty() {
            let summary_msg = Message::new(
                original.id,
                MessageRole::System,
                format!(
                    "[Conversation Summary - Previous messages have been summarized]\n\n{}",
                    summary
                ),
            );
            new_conversation.add_message(summary_msg)?;
        }

        // Add recent messages
        for msg in recent_msgs {
            new_conversation.add_message((*msg).clone())?;
        }

        Ok(new_conversation)
    }

    /// Checks if a conversation needs compaction based on the current configuration.
    pub fn needs_compaction(&self, conversation: &Conversation) -> Result<bool> {
        let tokens = self
            .token_counter
            .count_conversation_tokens(conversation)
            .context("Failed to count conversation tokens")?;

        let threshold = self
            .config
            .compaction_threshold
            .unwrap_or(self.config.target_tokens);

        Ok(tokens > threshold)
    }

    /// Returns the current token count for a conversation.
    pub fn count_tokens(&self, conversation: &Conversation) -> Result<usize> {
        self.token_counter
            .count_conversation_tokens(conversation)
            .context("Failed to count conversation tokens")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
