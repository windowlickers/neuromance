//! First-class context management for Core.
//!
//! Provides automatic context compaction as built-in infrastructure in the conversation loop.
//! Users configure compaction through [`ContextConfig`] and wire it into [`Core`](crate::core::Core)
//! via [`Core::with_context_management()`](crate::core::Core::with_context_management) or
//! [`Core::with_context_management_client()`](crate::core::Core::with_context_management_client).
//!
//! Compaction runs automatically inside the conversation loop, keeping the
//! conversation within the configured token budget. Conversation size is
//! measured according to [`TokenSource`]: by default the provider-reported
//! `Usage` from the latest response, or a local tokenizer when one is
//! configured.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::warn;

use neuromance_client::LLMClient;
use neuromance_common::chat::{Conversation, Message};
use neuromance_context::compaction::{CompactionConfig, CompactionResult, CompactionStrategy};
use neuromance_context::{Compactor, TokenCounter};

/// How conversation size is measured for compaction triggering.
#[derive(Clone, Default)]
pub enum TokenSource {
    /// Use the provider-reported `Usage` from the latest response.
    ///
    /// Requires no tokenizer; compaction never triggers on turns where the
    /// provider omits usage data.
    #[default]
    Reported,
    /// Count locally with a `HuggingFace` tokenizer.
    Tokenizer(Arc<TokenCounter>),
}

/// High-level configuration for automatic context management.
///
/// Wraps the lower-level [`CompactionConfig`] with user-friendly ratio-based settings.
/// Sensible defaults are derived from `context_window_size`.
///
/// # Example
///
/// ```rust
/// use neuromance::context_management::ContextConfig;
///
/// let config = ContextConfig::new(128_000)
///     .with_compaction_threshold_ratio(0.85)
///     .with_target_ratio(0.50)
///     .with_preserve_recent_turns(4);
/// ```
pub struct ContextConfig {
    /// Total context window size in tokens (e.g., `128_000`).
    pub context_window_size: usize,
    /// How conversation size is measured (default: provider-reported `Usage`).
    pub token_source: TokenSource,
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
    /// Create a new context config with sensible defaults, measuring
    /// conversation size from provider-reported `Usage`.
    ///
    /// - `token_source`: `Reported`
    /// - `compaction_threshold_ratio`: 0.80
    /// - `target_ratio`: 0.50
    /// - `preserve_recent_turns`: 3
    /// - `strategy`: `OneShot`
    #[must_use]
    pub const fn new(context_window_size: usize) -> Self {
        Self {
            context_window_size,
            token_source: TokenSource::Reported,
            compaction_threshold_ratio: 0.80,
            target_ratio: 0.50,
            preserve_recent_turns: 3,
            strategy: CompactionStrategy::OneShot,
            custom_prompt: None,
        }
    }

    /// Measure conversation size with a local tokenizer instead of
    /// provider-reported `Usage`.
    #[must_use]
    pub fn with_tokenizer(mut self, token_counter: Arc<TokenCounter>) -> Self {
        self.token_source = TokenSource::Tokenizer(token_counter);
        self
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
    use_reported: bool,
    missing_usage_warned: AtomicBool,
}

impl<C: LLMClient> ContextManager<C> {
    /// Create a new context manager from a client and user-facing config.
    pub(crate) fn new(client: C, config: &ContextConfig) -> Self {
        let compaction_config = config.to_compaction_config();
        let mut compactor = Compactor::new(client).with_config(compaction_config);
        let use_reported = match &config.token_source {
            TokenSource::Reported => true,
            TokenSource::Tokenizer(counter) => {
                compactor = compactor.with_token_counter(Arc::clone(counter));
                false
            }
        };
        Self {
            compactor,
            use_reported,
            missing_usage_warned: AtomicBool::new(false),
        }
    }

    /// Check if compaction is needed and perform it if so.
    ///
    /// `reported_tokens` is the conversation size from the latest response's
    /// `Usage` (prompt + completion tokens), consulted when the manager is
    /// configured with [`TokenSource::Reported`].
    ///
    /// Returns the (possibly compacted) messages and an optional [`CompactionResult`]
    /// if compaction was attempted. On error, logs a warning and returns originals unchanged.
    pub(crate) async fn maybe_compact(
        &self,
        messages: Vec<Message>,
        reported_tokens: Option<usize>,
    ) -> (Vec<Message>, Option<CompactionResult>) {
        if messages.is_empty() {
            return (messages, None);
        }

        let reported = if self.use_reported {
            let Some(tokens) = reported_tokens else {
                if !self.missing_usage_warned.swap(true, Ordering::Relaxed) {
                    warn!(
                        "context management is configured with TokenSource::Reported but \
                         the provider returned no usage data; compaction will never trigger"
                    );
                }
                return (messages, None);
            };
            if !self.compactor.needs_compaction_reported(tokens) {
                return (messages, None);
            }
            Some(tokens)
        } else {
            None
        };

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

        // In tokenizer mode the trigger check needs the rebuilt conversation.
        if reported.is_none() {
            match self.compactor.needs_compaction(&temp_conversation) {
                Ok(false) => return (messages, None),
                Ok(true) => {}
                Err(e) => {
                    warn!("Failed to check compaction threshold: {e}");
                    return (messages, None);
                }
            }
        }

        let outcome = match reported {
            Some(tokens) => {
                self.compactor
                    .compact_with_reported_tokens(&temp_conversation, tokens)
                    .await
            }
            None => self.compactor.compact(&temp_conversation).await,
        };

        match outcome {
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::pin::Pin;
    use std::sync::atomic::AtomicUsize;

    use async_trait::async_trait;
    use futures::Stream;
    use uuid::Uuid;

    use neuromance_client::ClientError;
    use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config};

    use super::*;

    /// `LLMClient` stub that returns a canned summary and counts invocations.
    struct MockSummaryClient {
        config: Config,
        summary: String,
        calls: Arc<AtomicUsize>,
    }

    impl MockSummaryClient {
        fn new(summary: &str) -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                summary: summary.to_string(),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn call_counter(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.calls)
        }
    }

    #[async_trait]
    impl LLMClient for MockSummaryClient {
        fn config(&self) -> &Config {
            &self.config
        }

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let conv_id = request
                .messages
                .first()
                .map_or_else(Uuid::new_v4, |m| m.conversation_id);
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

    fn history(conv_id: Uuid, pairs: usize) -> Vec<Message> {
        (0..pairs)
            .flat_map(|i| {
                vec![
                    Message::user(conv_id, format!("question {i}")),
                    Message::assistant(conv_id, format!("answer {i}")),
                ]
            })
            .collect()
    }

    fn reported_manager(window: usize) -> (ContextManager<MockSummaryClient>, Arc<AtomicUsize>) {
        let client = MockSummaryClient::new("summary of earlier turns");
        let calls = client.call_counter();
        let config = ContextConfig::new(window).with_preserve_recent_turns(1);
        (ContextManager::new(client, &config), calls)
    }

    #[tokio::test]
    async fn test_maybe_compact_reported_none_is_noop() {
        let (manager, calls) = reported_manager(1000);
        let conv_id = Uuid::new_v4();
        let messages = history(conv_id, 4);

        let (out, result) = manager.maybe_compact(messages.clone(), None).await;

        assert_eq!(out.len(), messages.len());
        assert!(result.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_maybe_compact_reported_below_threshold_is_noop() {
        // window 1000, threshold ratio 0.8 -> trigger above 800 tokens
        let (manager, calls) = reported_manager(1000);
        let conv_id = Uuid::new_v4();
        let messages = history(conv_id, 4);

        let (out, result) = manager.maybe_compact(messages.clone(), Some(800)).await;

        assert_eq!(out.len(), messages.len());
        assert!(result.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_maybe_compact_reported_triggers_and_restores_conversation_id() {
        let (manager, calls) = reported_manager(1000);
        let conv_id = Uuid::new_v4();
        let messages = history(conv_id, 4);

        let (out, result) = manager.maybe_compact(messages, Some(5000)).await;

        let result = result.expect("compaction attempted");
        assert!(result.was_compacted);
        assert_eq!(result.original_tokens, 5000);
        assert!(
            calls.load(Ordering::SeqCst) >= 1,
            "summary LLM call expected"
        );

        // summary message + 2 preserved recent messages
        assert_eq!(out.len(), 3);
        assert!(out[0].content.contains("summary of earlier turns"));
        assert!(
            out.iter().all(|m| m.conversation_id == conv_id),
            "conversation_id must be restored on compacted messages"
        );
    }
}
