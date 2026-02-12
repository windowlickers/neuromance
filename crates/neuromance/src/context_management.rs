//! First-class context management for Core.
//!
//! Provides automatic context compaction as built-in infrastructure in the conversation loop.
//! Users configure compaction through [`ContextConfig`] and wire it into [`Core`](crate::core::Core)
//! via [`Core::with_context_management()`](crate::core::Core::with_context_management).
//!
//! Compaction runs automatically before each turn callback, keeping the conversation
//! within the configured token budget.

use std::sync::Arc;

use tracing::warn;

use neuromance_client::LLMClient;
use neuromance_common::chat::{Conversation, Message};
use neuromance_context::compaction::{CompactionConfig, CompactionResult, CompactionStrategy};
use neuromance_context::{Compactor, TokenCounter};

/// High-level configuration for automatic context management.
///
/// Wraps the lower-level [`CompactionConfig`] with user-friendly ratio-based settings.
/// Sensible defaults are derived from `context_window_size`.
///
/// # Example
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use neuromance_context::TokenCounter;
/// # let token_counter: Arc<TokenCounter> = unimplemented!();
/// use neuromance::context_management::ContextConfig;
///
/// let config = ContextConfig::new(128_000, token_counter)
///     .with_compaction_threshold_ratio(0.85)
///     .with_target_ratio(0.50)
///     .with_preserve_recent_turns(4);
/// ```
pub struct ContextConfig {
    /// Total context window size in tokens (e.g., `128_000`).
    pub context_window_size: usize,
    /// Shared token counter for measuring conversation size.
    pub token_counter: Arc<TokenCounter>,
    /// Ratio of context window at which compaction triggers (default: 0.80).
    pub compaction_threshold_ratio: f64,
    /// Target ratio of context window after compaction (default: 0.50).
    pub target_ratio: f64,
    /// Number of recent message turns to preserve verbatim (default: 3).
    pub preserve_recent_turns: usize,
    /// Compaction strategy (default: `OneShot`).
    pub strategy: CompactionStrategy,
    /// Optional custom summarization prompt.
    pub custom_prompt: Option<String>,
}

impl ContextConfig {
    /// Create a new context config with sensible defaults.
    ///
    /// - `compaction_threshold_ratio`: 0.80
    /// - `target_ratio`: 0.50
    /// - `preserve_recent_turns`: 3
    /// - `strategy`: `OneShot`
    #[must_use]
    pub const fn new(context_window_size: usize, token_counter: Arc<TokenCounter>) -> Self {
        Self {
            context_window_size,
            token_counter,
            compaction_threshold_ratio: 0.80,
            target_ratio: 0.50,
            preserve_recent_turns: 3,
            strategy: CompactionStrategy::OneShot,
            custom_prompt: None,
        }
    }

    /// Set the ratio of context window at which compaction triggers.
    #[must_use]
    pub const fn with_compaction_threshold_ratio(mut self, ratio: f64) -> Self {
        self.compaction_threshold_ratio = ratio;
        self
    }

    /// Set the target ratio of context window after compaction.
    #[must_use]
    pub const fn with_target_ratio(mut self, ratio: f64) -> Self {
        self.target_ratio = ratio;
        self
    }

    /// Set the number of recent turns to preserve verbatim.
    #[must_use]
    pub const fn with_preserve_recent_turns(mut self, turns: usize) -> Self {
        self.preserve_recent_turns = turns;
        self
    }

    /// Set the compaction strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: CompactionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set a custom summarization prompt.
    #[must_use]
    pub fn with_custom_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.custom_prompt = Some(prompt.into());
        self
    }

    /// Convert ratios into absolute token counts and build a [`CompactionConfig`].
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn to_compaction_config(&self) -> CompactionConfig {
        let threshold =
            (self.context_window_size as f64 * self.compaction_threshold_ratio) as usize;
        let target = (self.context_window_size as f64 * self.target_ratio) as usize;

        let mut config = CompactionConfig::new(target)
            .with_compaction_threshold(threshold)
            .with_preserve_recent_turns(self.preserve_recent_turns)
            .with_strategy(self.strategy);

        if let Some(ref prompt) = self.custom_prompt {
            config = config.with_custom_prompt(prompt.clone());
        }

        config
    }
}

/// Internal context manager that owns a [`Compactor`] and handles conversation remapping.
///
/// This is `pub(crate)` — users interact with [`ContextConfig`] and
/// [`Core::with_context_management()`](crate::core::Core::with_context_management).
pub(crate) struct ContextManager<C: LLMClient> {
    compactor: Compactor<C>,
}

impl<C: LLMClient> ContextManager<C> {
    /// Create a new context manager from a client and user-facing config.
    pub(crate) fn new(client: C, config: &ContextConfig) -> Self {
        let compaction_config = config.to_compaction_config();
        let compactor = Compactor::with_shared_counter(client, Arc::clone(&config.token_counter))
            .with_config(compaction_config);
        Self { compactor }
    }

    /// Check if compaction is needed and perform it if so.
    ///
    /// Returns the (possibly compacted) messages and an optional [`CompactionResult`]
    /// if compaction was attempted. On error, logs a warning and returns originals unchanged.
    pub(crate) async fn maybe_compact(
        &self,
        messages: Vec<Message>,
    ) -> (Vec<Message>, Option<CompactionResult>) {
        if messages.is_empty() {
            return (messages, None);
        }

        // Build a temporary Conversation from the messages, remapping conversation_ids
        let mut temp_conversation = Conversation::new();
        let original_conversation_id = messages.first().map(|m| m.conversation_id);

        for msg in &messages {
            let mut remapped = msg.clone();
            remapped.conversation_id = temp_conversation.id;
            if let Err(e) = temp_conversation.add_message(remapped) {
                warn!("Failed to build temp conversation for compaction: {e}");
                return (messages, None);
            }
        }

        // Check if compaction is needed
        match self.compactor.needs_compaction(&temp_conversation) {
            Ok(false) => return (messages, None),
            Ok(true) => {}
            Err(e) => {
                warn!("Failed to check compaction threshold: {e}");
                return (messages, None);
            }
        }

        // Perform compaction
        match self.compactor.compact(&temp_conversation).await {
            Ok(result) => {
                // Restore original conversation_id on compacted messages
                let compacted_messages: Vec<Message> = result
                    .conversation
                    .messages
                    .iter()
                    .map(|msg| {
                        let mut restored = msg.clone();
                        if let Some(orig_id) = original_conversation_id {
                            restored.conversation_id = orig_id;
                        }
                        restored
                    })
                    .collect();

                (compacted_messages, Some(result))
            }
            Err(e) => {
                warn!(
                    "Context compaction failed (non-fatal, continuing with original messages): {e}"
                );
                (messages, None)
            }
        }
    }
}
