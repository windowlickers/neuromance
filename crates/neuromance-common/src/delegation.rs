//! Out-of-band propagation of delegation lineage to spawned subagents.
//!
//! `ToolImplementation::execute` carries no calling context, so when a parent
//! agent delegates to a subagent the parent's conversation and task ids must
//! travel out-of-band. A [`tokio::task_local!`] carries [`DelegationContext`]:
//! the subagent fan-out paths poll children on the parent's task (via
//! `futures::join_all`, not `tokio::spawn`), so a value set with [`scope`] is
//! visible to every child run on that task.

use std::future::Future;
use std::path::PathBuf;

use uuid::Uuid;

/// Lineage of the enclosing agent run, propagated to any subagent it spawns.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct DelegationContext {
    /// Conversation of the enclosing agent, or `None` at a root scope (e.g. one
    /// seeded by the runtime solely to carry [`DelegationContext::task_id`]).
    pub conversation_id: Option<Uuid>,
    /// Runtime task this delegation tree belongs to, when known.
    pub task_id: Option<Uuid>,
    /// Assistant message that emitted the tool call spawning the subagent, when
    /// the enclosing run is delegating from a specific tool call.
    pub parent_message_id: Option<Uuid>,
    /// Id of the specific tool call within [`Self::parent_message_id`] that
    /// spawned the subagent.
    pub parent_tool_call_id: Option<String>,
    /// Absolute per-conversation workspace directory for tool execution,
    /// when the runtime has one configured. Subagents inherit the parent's —
    /// same conversation tree, same workspace.
    pub workspace_dir: Option<PathBuf>,
}

tokio::task_local! {
    static TASK_CTX: DelegationContext;
}

/// The delegation context in effect, or the default when none is set.
#[must_use]
pub fn current() -> DelegationContext {
    TASK_CTX.try_with(Clone::clone).unwrap_or_default()
}

/// Run `fut` with `ctx` as the ambient delegation context for its task.
pub async fn scope<F>(ctx: DelegationContext, fut: F) -> F::Output
where
    F: Future,
{
    TASK_CTX.scope(ctx, fut).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_is_default_outside_any_scope() {
        assert_eq!(current(), DelegationContext::default());
    }

    #[tokio::test]
    async fn scope_makes_context_visible_to_current() {
        let ctx = DelegationContext {
            conversation_id: Some(Uuid::new_v4()),
            task_id: Some(Uuid::new_v4()),
            parent_message_id: Some(Uuid::new_v4()),
            parent_tool_call_id: Some("call_abc".to_string()),
            workspace_dir: Some(std::path::PathBuf::from("/workspace/test")),
        };
        let observed = scope(ctx.clone(), async { current() }).await;
        assert_eq!(observed, ctx);
    }
}
