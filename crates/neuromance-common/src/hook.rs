//! Generic lifecycle hooks for the conversation loop.
//!
//! A [`Hook`] is the single extension point the orchestration core dispatches to
//! at each stage of a run: conversation start, usage reporting, tool approval,
//! after each tool result, and turn end. Every method has a no-op default, so an
//! implementation overrides only the stages it cares about.
//!
//! Hooks compose: the core holds an ordered list and invokes each in turn.
//! Message-injecting methods return a [`HookOutcome`]; the approval method
//! returns `Option<ToolApproval>` (`None` abstains, deferring to the next hook);
//! [`Hook::on_turn_end`] returns a [`TurnEnd`] that may rewrite history and
//! optionally report [`CompactionStats`] for the core to surface.

use std::future::Future;

use async_trait::async_trait;
use uuid::Uuid;

use crate::chat::Message;
use crate::client::Usage;
use crate::tools::{ToolApproval, ToolCall};

/// Per-turn context handed to every [`Hook`] method.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HookContext {
    /// The conversation the run belongs to.
    pub conversation_id: Uuid,
    /// Zero-based index of the turn about to execute (or just executed).
    pub turn: u32,
}

impl HookContext {
    /// Create a context for `conversation_id` at `turn`.
    #[must_use]
    pub const fn new(conversation_id: Uuid, turn: u32) -> Self {
        Self {
            conversation_id,
            turn,
        }
    }
}

/// Append-only messages a hook injects into the conversation.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HookOutcome {
    /// Messages to append to the conversation history.
    pub messages: Vec<Message>,
}

impl HookOutcome {
    /// An outcome that injects nothing.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// An outcome that injects `messages`.
    #[must_use]
    pub const fn inject(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Whether the outcome injects no messages.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// Statistics from a compaction performed in [`Hook::on_turn_end`].
///
/// The core surfaces these to consumers (e.g. as a compaction event).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompactionStats {
    /// Token count before compaction.
    pub original_tokens: usize,
    /// Token count after compaction.
    pub compacted_tokens: usize,
    /// Number of messages that were summarized.
    pub messages_summarized: usize,
    /// Whether compaction was actually performed.
    pub was_compacted: bool,
}

impl CompactionStats {
    /// Create compaction statistics.
    #[must_use]
    pub const fn new(
        original_tokens: usize,
        compacted_tokens: usize,
        messages_summarized: usize,
        was_compacted: bool,
    ) -> Self {
        Self {
            original_tokens,
            compacted_tokens,
            messages_summarized,
            was_compacted,
        }
    }
}

/// The result of [`Hook::on_turn_end`]: the (possibly rewritten) history plus
/// optional compaction statistics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TurnEnd {
    /// The message history after the hook ran.
    pub messages: Vec<Message>,
    /// Compaction statistics, if this hook compacted the history.
    pub compaction: Option<CompactionStats>,
}

impl TurnEnd {
    /// Return `messages` unchanged, with no compaction reported.
    #[must_use]
    pub const fn unchanged(messages: Vec<Message>) -> Self {
        Self {
            messages,
            compaction: None,
        }
    }

    /// Return the compacted `messages` along with their `stats`.
    #[must_use]
    pub const fn compacted(messages: Vec<Message>, stats: CompactionStats) -> Self {
        Self {
            messages,
            compaction: Some(stats),
        }
    }
}

/// A lifecycle hook the conversation loop dispatches to.
///
/// All methods default to a no-op; implement only the stages you need. Hooks
/// are held as `Arc<dyn Hook>` and run in registration order.
#[async_trait]
pub trait Hook: Send + Sync {
    /// A short name for this hook, used in logs and error context.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Runs once before the first LLM call, with the seed history.
    ///
    /// Use to inject always-on context (e.g. rule files that always apply).
    async fn on_conversation_start(
        &self,
        _ctx: &HookContext,
        _messages: &[Message],
    ) -> anyhow::Result<HookOutcome> {
        Ok(HookOutcome::none())
    }

    /// Runs whenever the history advances (seed, assistant message, tool
    /// results). Use to durably record messages.
    async fn on_messages(&self, _ctx: &HookContext, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    /// Runs after the provider reports token usage for a turn.
    async fn on_usage(&self, _ctx: &HookContext, _usage: &Usage) -> anyhow::Result<()> {
        Ok(())
    }

    /// Decides a non-auto-approved tool call.
    ///
    /// `None` abstains, deferring to the next hook and ultimately to the core's
    /// default approval mechanism. The first hook to return `Some` wins.
    async fn review_tool(
        &self,
        _ctx: &HookContext,
        _call: &ToolCall,
    ) -> anyhow::Result<Option<ToolApproval>> {
        Ok(None)
    }

    /// Runs after a tool produces a result. Use to inject follow-on context
    /// (e.g. a rule file keyed to the touched path).
    async fn after_tool(
        &self,
        _ctx: &HookContext,
        _call: &ToolCall,
        _result: &str,
        _success: bool,
    ) -> anyhow::Result<HookOutcome> {
        Ok(HookOutcome::none())
    }

    /// Runs at the end of every turn. May rewrite history and report
    /// compaction statistics.
    async fn on_turn_end(
        &self,
        _ctx: &HookContext,
        messages: Vec<Message>,
    ) -> anyhow::Result<TurnEnd> {
        Ok(TurnEnd::unchanged(messages))
    }

    /// Runs once when the conversation finishes.
    async fn on_completion(&self, _ctx: &HookContext, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Adapts an approval closure into a [`Hook`] that only decides tool calls.
///
/// The closure mirrors the previous stored approval callback: it receives each
/// non-auto-approved [`ToolCall`] and returns a [`ToolApproval`].
pub struct FnReviewHook<F> {
    review: F,
}

impl<F> std::fmt::Debug for FnReviewHook<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnReviewHook").finish_non_exhaustive()
    }
}

impl<F, Fut> FnReviewHook<F>
where
    F: Fn(&ToolCall) -> Fut + Send + Sync,
    Fut: Future<Output = ToolApproval> + Send,
{
    /// Wrap `review` as an approval-only hook.
    pub const fn new(review: F) -> Self {
        Self { review }
    }
}

#[async_trait]
impl<F, Fut> Hook for FnReviewHook<F>
where
    F: Fn(&ToolCall) -> Fut + Send + Sync,
    Fut: Future<Output = ToolApproval> + Send,
{
    async fn review_tool(
        &self,
        _ctx: &HookContext,
        call: &ToolCall,
    ) -> anyhow::Result<Option<ToolApproval>> {
        Ok(Some((self.review)(call).await))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    struct Noop;
    #[async_trait]
    impl Hook for Noop {}

    #[tokio::test]
    async fn test_default_methods_are_noop() {
        let hook = Noop;
        let ctx = HookContext::new(Uuid::new_v4(), 0);
        let call = ToolCall::new("t", "");

        assert!(
            hook.on_conversation_start(&ctx, &[])
                .await
                .unwrap()
                .is_empty()
        );
        assert!(hook.review_tool(&ctx, &call).await.unwrap().is_none());
        assert!(
            hook.after_tool(&ctx, &call, "result", true)
                .await
                .unwrap()
                .is_empty()
        );

        let msgs = vec![Message::user(ctx.conversation_id, "hi")];
        let end = hook.on_turn_end(&ctx, msgs).await.unwrap();
        assert_eq!(end.messages.len(), 1);
        assert_eq!(end.messages[0].content, "hi");
        assert!(end.compaction.is_none());
    }

    #[tokio::test]
    async fn test_fn_review_hook_decides() {
        let hook = FnReviewHook::new(|_call: &ToolCall| async { ToolApproval::Approved });
        let ctx = HookContext::new(Uuid::new_v4(), 0);
        let call = ToolCall::new("t", "");
        assert_eq!(
            hook.review_tool(&ctx, &call).await.unwrap(),
            Some(ToolApproval::Approved)
        );
    }

    #[test]
    fn test_hook_outcome_inject_carries_messages() {
        let id = Uuid::new_v4();
        let outcome = HookOutcome::inject(vec![Message::system(id, "x")]);
        assert!(!outcome.is_empty());
        assert_eq!(outcome.messages.len(), 1);
    }
}
