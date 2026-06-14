//! Out-of-band propagation of delegation lineage to spawned subagents.
//!
//! `ToolImplementation::execute` carries no calling context, so when a parent
//! agent delegates to a subagent the parent's conversation and task ids must
//! travel out-of-band. Two cooperating relays carry [`DelegationContext`]:
//!
//! - A [`tokio::task_local!`] carries it within a single task. The subagent
//!   fan-out paths poll children on the parent's task (via `futures::join_all`,
//!   not `tokio::spawn`), so a value set with [`scope`] is visible to every
//!   child run on that task.
//! - A thread-local mirrors it across a `spawn_blocking`/`block_on` boundary,
//!   which a task-local does not cross. The Python REPL bridge runs subagents
//!   from inside `spawn_blocking`; it captures [`current`] before the hop and
//!   re-establishes it with [`with_thread_local`] on the blocking thread.
//!
//! [`current`] reads the task-local first and falls back to the thread-local,
//! so both relays feed the same lookup.

use std::cell::Cell;
use std::future::Future;

use uuid::Uuid;

/// Lineage of the enclosing agent run, propagated to any subagent it spawns.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DelegationContext {
    /// Conversation of the enclosing agent, or `None` at a root scope (e.g. one
    /// seeded by the runtime solely to carry [`DelegationContext::task_id`]).
    pub conversation_id: Option<Uuid>,
    /// Runtime task this delegation tree belongs to, when known.
    pub task_id: Option<Uuid>,
}

tokio::task_local! {
    static TASK_CTX: DelegationContext;
}

thread_local! {
    static THREAD_CTX: Cell<Option<DelegationContext>> = const { Cell::new(None) };
}

/// The delegation context in effect, or the default when none is set.
///
/// Reads the task-local first (set by [`scope`]); on a blocking thread where the
/// task-local is absent, falls back to the thread-local relay set by
/// [`with_thread_local`].
#[must_use]
pub fn current() -> DelegationContext {
    if let Ok(ctx) = TASK_CTX.try_with(|ctx| *ctx) {
        return ctx;
    }
    THREAD_CTX.with(Cell::get).unwrap_or_default()
}

/// Run `fut` with `ctx` as the ambient delegation context for its task.
pub async fn scope<F>(ctx: DelegationContext, fut: F) -> F::Output
where
    F: Future,
{
    TASK_CTX.scope(ctx, fut).await
}

/// Run `f` with `ctx` mirrored into the thread-local relay, restoring the
/// previous value when it returns (or unwinds).
///
/// Bridges the `spawn_blocking`/`block_on` boundary that a task-local cannot
/// cross: capture [`current`] before `spawn_blocking`, then wrap the blocking
/// body in this so a subagent run driven via `block_on` on that thread observes
/// the parent context.
pub fn with_thread_local<R>(ctx: DelegationContext, f: impl FnOnce() -> R) -> R {
    // Restores the prior value on drop so an unwinding panic can't leak `ctx`
    // onto a pooled blocking thread reused by a later, unrelated task.
    struct Restore(Option<DelegationContext>);
    impl Drop for Restore {
        fn drop(&mut self) {
            THREAD_CTX.with(|c| c.set(self.0));
        }
    }

    let _restore = Restore(THREAD_CTX.with(|c| c.replace(Some(ctx))));
    f()
}

#[cfg(test)]
#[allow(clippy::panic)]
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
        };
        let observed = scope(ctx, async { current() }).await;
        assert_eq!(observed, ctx);
    }

    #[test]
    fn thread_local_relay_feeds_current_and_restores() {
        let ctx = DelegationContext {
            conversation_id: Some(Uuid::new_v4()),
            task_id: None,
        };
        let observed = with_thread_local(ctx, current);
        assert_eq!(observed, ctx);
        // The relay is cleared once the scope returns.
        assert_eq!(current(), DelegationContext::default());
    }

    #[test]
    fn thread_local_relay_restores_previous_on_unwind() {
        let outer = DelegationContext {
            conversation_id: Some(Uuid::new_v4()),
            task_id: None,
        };
        let inner = DelegationContext {
            conversation_id: Some(Uuid::new_v4()),
            task_id: None,
        };
        with_thread_local(outer, || {
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_thread_local(inner, || panic!("boom"));
            }));
            assert!(caught.is_err());
            // The inner scope unwound; the outer context must still be in effect.
            assert_eq!(current(), outer);
        });
    }
}
