//! Conversation persistence as a [`Hook`].
//!
//! [`PersistenceHook`] records conversation history through a
//! [`ConversationSink`] as a run progresses: the seed snapshot at conversation
//! start and every subsequent assistant/tool message. Mid-run writes are
//! best-effort and idempotent per [`Message::id`] — a failed write leaves its
//! ids unmarked so the next persist point retries the backlog. At completion the
//! write is fail-closed: any backlog is flushed and a remaining failure aborts
//! the run, so a task reports success only when its history is durable.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use uuid::Uuid;

use neuromance_common::chat::Message;
use neuromance_common::delegation;
use neuromance_common::hook::{Hook, HookContext, HookOutcome};

use crate::sink::ConversationSink;

/// Parent lineage linking a spawned child conversation to the run that
/// delegated it, captured from the enclosing delegation scope.
#[derive(Debug, Clone, Default)]
#[allow(clippy::struct_field_names)] // mirrors the delegation context's *_id fields
struct ParentLineage {
    conversation_id: Option<Uuid>,
    task_id: Option<Uuid>,
    message_id: Option<Uuid>,
    tool_call_id: Option<String>,
}

/// Hook that durably records conversation history via a [`ConversationSink`].
pub struct PersistenceHook {
    sink: Arc<dyn ConversationSink>,
    /// Message ids already persisted this run; failed writes stay unmarked so
    /// the next persist point retries them.
    persisted: Mutex<HashSet<Uuid>>,
    lineage: ParentLineage,
}

impl std::fmt::Debug for PersistenceHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistenceHook")
            .field("persisted", &self.persisted)
            .field("lineage", &self.lineage)
            .finish_non_exhaustive()
    }
}

impl PersistenceHook {
    /// Build a persistence hook over `sink`.
    ///
    /// Captures the enclosing delegation context so that, when this run is a
    /// delegated child, its conversation is linked to the parent that spawned
    /// it. Construct the hook within the parent's delegation scope (as the
    /// subagent factory does) for the link to be recorded; a root run sees no
    /// parent and records none.
    #[must_use]
    pub fn new(sink: Arc<dyn ConversationSink>) -> Self {
        let ctx = delegation::current();
        Self {
            sink,
            persisted: Mutex::new(HashSet::new()),
            lineage: ParentLineage {
                conversation_id: ctx.conversation_id,
                task_id: ctx.task_id,
                message_id: ctx.parent_message_id,
                tool_call_id: ctx.parent_tool_call_id,
            },
        }
    }

    /// Persist messages not yet recorded this run.
    ///
    /// On success the ids are marked persisted; on failure they are left
    /// unmarked so the next persist point retries the backlog (the sink's
    /// per-id idempotency makes the retry safe), and the error is returned so
    /// the caller can decide whether to tolerate it (mid-run) or fail the run
    /// (at completion).
    async fn persist(&self, messages: &[Message]) -> anyhow::Result<()> {
        let pending: Vec<Message> = {
            let persisted = self
                .persisted
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            messages
                .iter()
                .filter(|m| !persisted.contains(&m.id))
                .cloned()
                .collect()
        };
        let Some(first) = pending.first() else {
            return Ok(());
        };
        let conversation_id = first.conversation_id;
        match self.sink.append_messages(conversation_id, &pending).await {
            Ok(inserted) => {
                {
                    let mut persisted = self
                        .persisted
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    persisted.extend(pending.iter().map(|m| m.id));
                }
                metrics::counter!("neuromance_db_messages_persisted_total").increment(inserted);
                Ok(())
            }
            Err(e) => {
                metrics::counter!("neuromance_db_persist_failures_total").increment(1);
                Err(anyhow::Error::new(e).context(format!(
                    "persist {} message(s) for conversation {conversation_id}",
                    pending.len(),
                )))
            }
        }
    }

    /// Persist messages, logging and swallowing any failure.
    ///
    /// Used at mid-run persist points where a transient failure should not abort
    /// the run: the unmarked ids are retried at the next persist point and the
    /// write is ultimately enforced by [`PersistenceHook::on_completion`].
    async fn persist_best_effort(&self, messages: &[Message]) {
        if let Err(e) = self.persist(messages).await {
            tracing::warn!(error = %e, "conversation persistence failed; will retry");
        }
    }

    /// Link the conversation to its spawning parent, if any.
    async fn link_parent(&self, messages: &[Message]) {
        if let (Some(parent), Some(first)) = (self.lineage.conversation_id, messages.first())
            && let Err(e) = self
                .sink
                .set_conversation_parent(
                    first.conversation_id,
                    parent,
                    self.lineage.task_id,
                    self.lineage.message_id,
                    self.lineage.tool_call_id.as_deref(),
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
}

#[async_trait]
impl Hook for PersistenceHook {
    fn name(&self) -> &'static str {
        "persistence"
    }

    async fn on_conversation_start(
        &self,
        _ctx: &HookContext,
        messages: &[Message],
    ) -> anyhow::Result<HookOutcome> {
        // Reset per-run dedup state: a shared serve agent reuses one hook across
        // every task, so without this the set would accumulate every message id
        // the process ever persists. Intra-run dedup is all that is needed since
        // the sink's append is idempotent per message id.
        self.persisted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        // Persist the seed snapshot up front so the input is durable even if
        // the first LLM call fails, then link to the spawning parent once the
        // append has created the conversation row. Best-effort: a failure here
        // is retried by the next persist point and enforced at completion.
        self.persist_best_effort(messages).await;
        self.link_parent(messages).await;
        Ok(HookOutcome::none())
    }

    async fn on_messages(&self, _ctx: &HookContext, messages: &[Message]) -> anyhow::Result<()> {
        // Best-effort mid-run: a transient write failure leaves ids unmarked so
        // the next persist point (or completion) retries the backlog, rather
        // than aborting a run that may still recover.
        self.persist_best_effort(messages).await;
        Ok(())
    }

    async fn on_completion(&self, _ctx: &HookContext, messages: &[Message]) -> anyhow::Result<()> {
        // Fail-closed barrier: flush any messages earlier persist points failed
        // to write before the run reports success. A continuation served from
        // another replica reads history from the store, so a `Succeeded` task
        // must imply its history is durable — otherwise the continuation would
        // silently resume from a truncated conversation.
        self.persist(messages).await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::sync::Mutex as StdMutex;

    use super::*;
    use crate::error::DbError;

    /// Sink that records appended ids and can be told to fail once.
    #[derive(Default)]
    struct RecordingSink {
        appended: StdMutex<Vec<Uuid>>,
        fail_next: StdMutex<bool>,
    }

    #[async_trait]
    impl ConversationSink for RecordingSink {
        async fn append_messages(
            &self,
            _conversation_id: Uuid,
            messages: &[Message],
        ) -> Result<u64, DbError> {
            let should_fail = {
                let mut guard = self.fail_next.lock().unwrap();
                std::mem::replace(&mut *guard, false)
            };
            if should_fail {
                return Err(DbError::UnknownStatus {
                    value: "forced".to_string(),
                    conversation_id: Uuid::nil(),
                });
            }
            {
                let mut appended = self.appended.lock().unwrap();
                for m in messages {
                    appended.push(m.id);
                }
            }
            Ok(messages.len() as u64)
        }
    }

    fn hook(sink: Arc<RecordingSink>) -> PersistenceHook {
        PersistenceHook::new(sink)
    }

    #[tokio::test]
    async fn test_persists_seed_then_dedups_on_messages() {
        let sink = Arc::new(RecordingSink::default());
        let hook = hook(Arc::clone(&sink));
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let seed = vec![Message::system(conv, "s"), Message::user(conv, "u")];

        hook.on_conversation_start(&ctx, &seed).await.unwrap();
        // A later assistant message extends the same slice; already-seen ids
        // must not be re-appended.
        let mut next = seed.clone();
        next.push(Message::assistant(conv, "a"));
        hook.on_messages(&ctx, &next).await.unwrap();

        assert_eq!(sink.appended.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_failed_write_is_retried_at_next_persist() {
        let sink = Arc::new(RecordingSink::default());
        *sink.fail_next.lock().unwrap() = true;
        let hook = hook(Arc::clone(&sink));
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let seed = vec![Message::system(conv, "s")];

        // First write fails -> nothing recorded, id stays unmarked.
        hook.on_messages(&ctx, &seed).await.unwrap();
        assert!(sink.appended.lock().unwrap().is_empty());

        // Next persist retries the backlog.
        hook.on_messages(&ctx, &seed).await.unwrap();
        assert_eq!(sink.appended.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_on_completion_propagates_persist_failure() {
        let sink = Arc::new(RecordingSink::default());
        *sink.fail_next.lock().unwrap() = true;
        let hook = hook(Arc::clone(&sink));
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let messages = vec![Message::system(conv, "s")];

        // A write failure at completion must surface, not be swallowed.
        assert!(hook.on_completion(&ctx, &messages).await.is_err());
        assert!(sink.appended.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_on_completion_flushes_backlog() {
        let sink = Arc::new(RecordingSink::default());
        *sink.fail_next.lock().unwrap() = true;
        let hook = hook(Arc::clone(&sink));
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let messages = vec![Message::system(conv, "s")];

        // Mid-run write fails and is swallowed, leaving the id unmarked.
        hook.on_messages(&ctx, &messages).await.unwrap();
        assert!(sink.appended.lock().unwrap().is_empty());

        // Completion flushes the backlog and succeeds.
        hook.on_completion(&ctx, &messages).await.unwrap();
        assert_eq!(sink.appended.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_dedup_state_resets_each_run() {
        // A shared serve agent reuses one hook across runs; starting a new
        // conversation must clear the dedup set so it cannot grow without bound.
        let sink = Arc::new(RecordingSink::default());
        let hook = hook(Arc::clone(&sink));
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let seed = vec![Message::system(conv, "s")];

        hook.on_conversation_start(&ctx, &seed).await.unwrap();
        assert_eq!(sink.appended.lock().unwrap().len(), 1);

        // Starting the next run clears prior ids, so the same slice is treated
        // as unseen and persisted again rather than retained forever.
        hook.on_conversation_start(&ctx, &seed).await.unwrap();
        assert_eq!(sink.appended.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_on_completion_noop_when_all_persisted() {
        let sink = Arc::new(RecordingSink::default());
        let hook = hook(Arc::clone(&sink));
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let messages = vec![Message::system(conv, "s")];

        hook.on_messages(&ctx, &messages).await.unwrap();
        assert_eq!(sink.appended.lock().unwrap().len(), 1);

        // Nothing pending: completion appends nothing and still succeeds.
        hook.on_completion(&ctx, &messages).await.unwrap();
        assert_eq!(sink.appended.lock().unwrap().len(), 1);
    }
}
