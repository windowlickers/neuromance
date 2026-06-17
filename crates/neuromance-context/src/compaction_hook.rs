//! Automatic context compaction as a conversation [`Hook`].
//!
//! [`CompactionHook`] wraps a [`Compactor`] and plugs into the conversation
//! loop: it records the provider-reported token count on each turn
//! ([`Hook::on_usage`]) and compacts the history when it grows past the
//! configured threshold ([`Hook::on_turn_end`]), reporting [`CompactionStats`]
//! the core surfaces as a compaction event.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tracing::{info, warn};

use neuromance_client::LLMClient;
use neuromance_common::chat::{Conversation, Message};
use neuromance_common::client::Usage;
use neuromance_common::hook::{CompactionStats, Hook, HookContext, TurnEnd};

use crate::Compactor;
use crate::TokenCounter;
use crate::compaction::CompactionStrategy;

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
    Tokenizer(std::sync::Arc<TokenCounter>),
}

/// High-level configuration for automatic context management.
///
/// Wraps the lower-level [`CompactionConfig`](crate::compaction::CompactionConfig)
/// with user-friendly ratio-based settings. Sensible defaults are derived from
/// `context_window_size`.
///
/// # Example
///
/// ```rust
/// use neuromance_context::ContextConfig;
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
    pub fn with_tokenizer(mut self, token_counter: std::sync::Arc<TokenCounter>) -> Self {
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

    /// Convert ratios into absolute token counts and build a `CompactionConfig`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn to_compaction_config(&self) -> crate::compaction::CompactionConfig {
        let threshold =
            (self.context_window_size as f64 * self.compaction_threshold_ratio) as usize;
        let target = (self.context_window_size as f64 * self.target_ratio) as usize;

        let mut config = crate::compaction::CompactionConfig::new(target)
            .with_compaction_threshold(threshold)
            .with_preserve_recent_turns(self.preserve_recent_turns)
            .with_strategy(self.strategy);

        if let Some(ref prompt) = self.custom_prompt {
            config = config.with_custom_prompt(prompt.clone());
        }

        config
    }
}

/// Conversation hook that compacts history when it grows past a token budget.
///
/// Generic over the client used for summarization; registered as
/// `Arc<dyn Hook>` so the client type is erased from the core.
pub struct CompactionHook<C: LLMClient> {
    compactor: Compactor<C>,
    use_reported: bool,
    /// Latest provider-reported token count, captured in `on_usage`.
    reported_tokens: Mutex<Option<usize>>,
    missing_usage_warned: AtomicBool,
}

impl<C: LLMClient> CompactionHook<C> {
    /// Create a compaction hook from a summarization `client` and config.
    #[must_use]
    pub fn new(client: C, config: &ContextConfig) -> Self {
        let compaction_config = config.to_compaction_config();
        let mut compactor = Compactor::new(client).with_config(compaction_config);
        let use_reported = match &config.token_source {
            TokenSource::Reported => true,
            TokenSource::Tokenizer(counter) => {
                compactor = compactor.with_token_counter(std::sync::Arc::clone(counter));
                false
            }
        };
        Self {
            compactor,
            use_reported,
            reported_tokens: Mutex::new(None),
            missing_usage_warned: AtomicBool::new(false),
        }
    }

    /// Compact `messages` if needed, returning the (possibly compacted) history
    /// and statistics when compaction was attempted.
    async fn maybe_compact(
        &self,
        messages: Vec<Message>,
    ) -> (Vec<Message>, Option<CompactionStats>) {
        if messages.is_empty() {
            return (messages, None);
        }

        let reported = if self.use_reported {
            let tokens = {
                *self
                    .reported_tokens
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
            };
            let Some(tokens) = tokens else {
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

        // Build a temporary Conversation from the messages, remapping conversation_ids.
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

                if result.was_compacted {
                    info!(
                        original_tokens = result.original_tokens,
                        compacted_tokens = result.compacted_tokens,
                        messages_summarized = result.messages_summarized,
                        "context compacted",
                    );
                }

                let stats = CompactionStats::new(
                    result.original_tokens,
                    result.compacted_tokens,
                    result.messages_summarized,
                    result.was_compacted,
                );
                (compacted_messages, Some(stats))
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

#[async_trait]
impl<C: LLMClient> Hook for CompactionHook<C> {
    fn name(&self) -> &'static str {
        "compaction"
    }

    async fn on_usage(&self, _ctx: &HookContext, usage: &Usage) -> anyhow::Result<()> {
        if self.use_reported {
            let tokens = usage.prompt_tokens as usize + usage.completion_tokens as usize;
            *self
                .reported_tokens
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(tokens);
        }
        Ok(())
    }

    async fn on_turn_end(
        &self,
        _ctx: &HookContext,
        messages: Vec<Message>,
    ) -> anyhow::Result<TurnEnd> {
        let (messages, stats) = self.maybe_compact(messages).await;
        Ok(match stats {
            Some(stats) => TurnEnd::compacted(messages, stats),
            None => TurnEnd::unchanged(messages),
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::pin::Pin;
    use std::sync::Arc;
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

    fn reported_hook(window: usize) -> (CompactionHook<MockSummaryClient>, Arc<AtomicUsize>) {
        let client = MockSummaryClient::new("summary of earlier turns");
        let calls = client.call_counter();
        let config = ContextConfig::new(window).with_preserve_recent_turns(1);
        (CompactionHook::new(client, &config), calls)
    }

    fn usage_with(prompt: u32) -> Usage {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        }
    }

    #[tokio::test]
    async fn test_no_usage_reported_is_noop() {
        let (hook, calls) = reported_hook(1000);
        let conv_id = Uuid::new_v4();
        let ctx = HookContext::new(conv_id, 0);
        let messages = history(conv_id, 4);

        // No on_usage call -> no reported tokens -> never compacts.
        let end = hook.on_turn_end(&ctx, messages.clone()).await.unwrap();

        assert_eq!(end.messages.len(), messages.len());
        assert!(end.compaction.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_below_threshold_is_noop() {
        // window 1000, threshold ratio 0.8 -> trigger above 800 tokens.
        let (hook, calls) = reported_hook(1000);
        let conv_id = Uuid::new_v4();
        let ctx = HookContext::new(conv_id, 0);
        let messages = history(conv_id, 4);

        hook.on_usage(&ctx, &usage_with(800)).await.unwrap();
        let end = hook.on_turn_end(&ctx, messages.clone()).await.unwrap();

        assert_eq!(end.messages.len(), messages.len());
        assert!(end.compaction.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_triggers_and_restores_conversation_id() {
        let (hook, calls) = reported_hook(1000);
        let conv_id = Uuid::new_v4();
        let ctx = HookContext::new(conv_id, 0);
        let messages = history(conv_id, 4);

        hook.on_usage(&ctx, &usage_with(5000)).await.unwrap();
        let end = hook.on_turn_end(&ctx, messages).await.unwrap();

        let stats = end.compaction.expect("compaction attempted");
        assert!(stats.was_compacted);
        assert_eq!(stats.original_tokens, 5000);
        assert!(
            calls.load(Ordering::SeqCst) >= 1,
            "summary LLM call expected"
        );

        // summary message + 2 preserved recent messages
        assert_eq!(end.messages.len(), 3);
        assert!(end.messages[0].content.contains("summary of earlier turns"));
        assert!(
            end.messages.iter().all(|m| m.conversation_id == conv_id),
            "conversation_id must be restored on compacted messages"
        );
    }
}
