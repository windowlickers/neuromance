//! The context edit ledger.
//!
//! [`ContextLedger`] pairs a [`Conversation`] with an append-only log of every
//! edit applied to it. It is the single funnel through which a conversation's
//! history is mutated: each edit goes through [`append`](ContextLedger::append)
//! or [`replace`](ContextLedger::replace), which record an [`EditRecord`] into
//! [`ContextMetadata`] capturing the [`Operation`], the [`EditSource`] that
//! produced it (provenance), the roles touched, and the affected message ids.
//!
//! The inner message vector is private and never handed out mutably, so it is
//! impossible to change the history without leaving a record.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::chat::{Conversation, Message, MessageRole};

/// The kinds of edits recorded in the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Operation {
    /// One or more messages appended to the history.
    Append,
    /// The history was rewritten wholesale (e.g. compaction, filtering).
    Replace,
}

/// Identifies what produced a context edit, for provenance in the ledger.
///
/// Hook-produced edits carry the hook's [`name`](crate::hook::Hook::name); edits
/// the orchestration core makes itself use [`core`](EditSource::core),
/// [`model`](EditSource::model), or [`tool`](EditSource::tool).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EditSource(String);

impl EditSource {
    /// Attributes an edit to a hook by its name.
    #[must_use]
    pub fn hook(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// An edit made directly by the orchestration core (seed, tool denials).
    #[must_use]
    pub fn core() -> Self {
        Self("core".to_string())
    }

    /// An edit carrying the model's assistant output.
    #[must_use]
    pub fn model() -> Self {
        Self("model".to_string())
    }

    /// An edit carrying a tool result or error.
    #[must_use]
    pub fn tool() -> Self {
        Self("tool".to_string())
    }

    /// The source label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EditSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A single recorded edit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditRecord {
    /// The kind of edit.
    pub operation: Operation,

    /// What produced the edit.
    pub source: EditSource,

    /// When the edit was recorded.
    pub timestamp: DateTime<Utc>,

    /// The distinct roles the edit touched, in order of first appearance.
    pub roles: Vec<MessageRole>,

    /// The ids of the messages the edit added or produced.
    pub message_ids: Vec<Uuid>,

    /// Structured detail about the edit (e.g. compaction stats, filter criteria).
    pub details: Option<Value>,
}

/// An append-only log of every edit applied to a conversation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextMetadata {
    edits: Vec<EditRecord>,
}

impl ContextMetadata {
    /// Creates an empty ledger log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one edit. `roles` and `message_ids` are derived from the messages
    /// the edit touched, so the log reflects exactly which roles changed.
    pub fn record_edit(
        &mut self,
        operation: Operation,
        source: EditSource,
        roles: Vec<MessageRole>,
        message_ids: Vec<Uuid>,
        details: Option<Value>,
    ) {
        self.edits.push(EditRecord {
            operation,
            source,
            timestamp: Utc::now(),
            roles,
            message_ids,
            details,
        });
    }

    /// Returns every recorded edit in order.
    #[must_use]
    pub fn edits(&self) -> &[EditRecord] {
        &self.edits
    }

    /// Returns the number of recorded edits.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.edits.len()
    }

    /// Returns whether the log is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Returns whether any edit recorded `operation`.
    #[must_use]
    pub fn has_operation(&self, operation: Operation) -> bool {
        self.edits.iter().any(|r| r.operation == operation)
    }

    /// Returns all edits matching `operation`.
    #[must_use]
    pub fn records_for(&self, operation: Operation) -> Vec<&EditRecord> {
        self.edits
            .iter()
            .filter(|r| r.operation == operation)
            .collect()
    }

    /// Returns all edits attributed to `source`.
    #[must_use]
    pub fn records_from(&self, source: &EditSource) -> Vec<&EditRecord> {
        self.edits.iter().filter(|r| &r.source == source).collect()
    }
}

/// A conversation paired with the ledger of every edit applied to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextLedger {
    conversation: Conversation,
    metadata: ContextMetadata,
}

impl ContextLedger {
    /// Wraps `conversation` with an empty ledger.
    #[must_use]
    pub fn new(conversation: Conversation) -> Self {
        Self {
            conversation,
            metadata: ContextMetadata::new(),
        }
    }

    /// Appends `msgs` to the history, recording one [`Operation::Append`]
    /// attributed to `source`.
    ///
    /// An empty iterator is a no-op and records nothing.
    pub fn append(&mut self, source: EditSource, msgs: impl IntoIterator<Item = Message>) {
        let added: Vec<Message> = msgs.into_iter().collect();
        if added.is_empty() {
            return;
        }
        let roles = distinct_roles(&added);
        let ids = added.iter().map(|m| m.id).collect();
        Arc::make_mut(&mut self.conversation.messages).extend(added);
        self.conversation.touch();
        self.metadata
            .record_edit(Operation::Append, source, roles, ids, None);
    }

    /// Replaces the entire history with `msgs`, recording one
    /// [`Operation::Replace`] attributed to `source`.
    ///
    /// `details` carries structured context for the edit (e.g. compaction
    /// statistics, the filter criteria that produced it).
    pub fn replace(&mut self, source: EditSource, msgs: Vec<Message>, details: Option<Value>) {
        let roles = distinct_roles(&msgs);
        let ids = msgs.iter().map(|m| m.id).collect();
        self.conversation.messages = Arc::new(msgs);
        self.conversation.touch();
        self.metadata
            .record_edit(Operation::Replace, source, roles, ids, details);
    }

    /// Returns the current message history.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        self.conversation.get_messages()
    }

    /// Returns an `Arc` snapshot of the current history for a chat request.
    #[must_use]
    pub fn snapshot(&self) -> Arc<[Message]> {
        self.conversation.get_messages().into()
    }

    /// Returns a reference to the underlying conversation.
    #[must_use]
    pub const fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// Returns the edit ledger.
    #[must_use]
    pub const fn metadata(&self) -> &ContextMetadata {
        &self.metadata
    }

    /// Consumes the ledger and returns the final message history.
    #[must_use]
    pub fn into_messages(self) -> Vec<Message> {
        self.conversation.get_messages().to_vec()
    }
}

/// Collects the distinct roles across `messages`, in order of first appearance.
fn distinct_roles(messages: &[Message]) -> Vec<MessageRole> {
    let mut roles = Vec::new();
    for message in messages {
        if !roles.contains(&message.role) {
            roles.push(message.role);
        }
    }
    roles
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    fn ledger() -> ContextLedger {
        ContextLedger::new(Conversation::new())
    }

    fn conv_id(ledger: &ContextLedger) -> Uuid {
        ledger.conversation().id
    }

    #[test]
    fn test_append_records_role_and_ids() {
        let mut ledger = ledger();
        let id = conv_id(&ledger);
        let msg = Message::system(id, "menu");
        let msg_id = msg.id;

        ledger.append(EditSource::hook("skills"), [msg]);

        assert_eq!(ledger.messages().len(), 1);
        let record = &ledger.metadata().edits()[0];
        assert_eq!(record.operation, Operation::Append);
        assert_eq!(record.source, EditSource::hook("skills"));
        assert_eq!(record.roles, vec![MessageRole::System]);
        assert_eq!(record.message_ids, vec![msg_id]);
    }

    #[test]
    fn test_append_collapses_distinct_roles() {
        let mut ledger = ledger();
        let id = conv_id(&ledger);

        ledger.append(
            EditSource::hook("rules"),
            [
                Message::system(id, "rule"),
                Message::user(id, "context"),
                Message::system(id, "another rule"),
            ],
        );

        let record = &ledger.metadata().edits()[0];
        assert_eq!(record.roles, vec![MessageRole::System, MessageRole::User]);
        assert_eq!(record.message_ids.len(), 3);
    }

    #[test]
    fn test_empty_append_records_nothing() {
        let mut ledger = ledger();
        ledger.append(EditSource::model(), std::iter::empty());

        assert!(ledger.messages().is_empty());
        assert!(ledger.metadata().is_empty());
    }

    #[test]
    fn test_replace_swaps_history_and_records_details() {
        let mut ledger = ledger();
        let id = conv_id(&ledger);
        ledger.append(EditSource::core(), [Message::user(id, "first")]);

        ledger.replace(
            EditSource::hook("compaction"),
            vec![Message::system(id, "summary")],
            Some(serde_json::json!({"was_compacted": true})),
        );

        assert_eq!(ledger.messages().len(), 1);
        assert_eq!(ledger.messages()[0].content, "summary");
        let record = ledger
            .metadata()
            .records_for(Operation::Replace)
            .pop()
            .expect("replace recorded");
        assert_eq!(record.source, EditSource::hook("compaction"));
        assert_eq!(
            record.details,
            Some(serde_json::json!({"was_compacted": true}))
        );
    }

    #[test]
    fn test_metadata_round_trips_through_json() {
        let mut ledger = ledger();
        let id = conv_id(&ledger);
        ledger.append(
            EditSource::hook("rules"),
            [Message::system(id, "a"), Message::user(id, "b")],
        );

        let json = serde_json::to_string(ledger.metadata()).expect("serialize");
        let restored: ContextMetadata = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.len(), 1);
        assert_eq!(
            restored.edits()[0].roles,
            vec![MessageRole::System, MessageRole::User]
        );
        assert_eq!(restored.edits()[0].source, EditSource::hook("rules"));
    }
}
