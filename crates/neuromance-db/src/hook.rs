//! Conversation persistence as a [`Hook`].
//!
//! [`PersistenceHook`] records conversation history through a
//! [`ConversationSink`] as a run progresses: the seed snapshot at conversation
//! start and every subsequent assistant/tool message. Writes are best-effort
//! and idempotent per [`Message::id`] — a failed write leaves its ids unmarked
//! so the next persist point retries the backlog.

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

    /// Persist messages not yet recorded this run, tolerating failures.
    ///
    /// On success the ids are marked persisted; on failure they are left
    /// unmarked so the next persist point retries the backlog (the sink's
    /// per-id idempotency makes the retry safe).
    async fn persist(&self, messages: &[Message]) {
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
            return;
        };
        match self
            .sink
            .append_messages(first.conversation_id, &pending)
            .await
        {
            Ok(inserted) => {
                {
                    let mut persisted = self
                        .persisted
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    persisted.extend(pending.iter().map(|m| m.id));
                }
                metrics::counter!("neuromance_db_messages_persisted_total").increment(inserted);
            }
            Err(e) => {
                tracing::warn!(
                    conversation_id = %first.conversation_id,
                    error = %e,
                    pending = pending.len(),
                    "conversation persistence failed; continuing without it"
                );
                metrics::counter!("neuromance_db_persist_failures_total").increment(1);
            }
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
        // Persist the seed snapshot up front so the input is durable even if
        // the first LLM call fails, then link to the spawning parent once the
        // append has created the conversation row.
        self.persist(messages).await;
        self.link_parent(messages).await;
        Ok(HookOutcome::none())
    }

    async fn on_messages(&self, _ctx: &HookContext, messages: &[Message]) -> anyhow::Result<()> {
        self.persist(messages).await;
        Ok(())
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
}
