//! Transformation operations for context state transitions.
//!
//! This module provides the actual transformation logic that is applied
//! when contexts move between states. Transformations include filtering,
//! merging consecutive messages, deduplication, and more.
//!
//! ## Example
//!
//! ```rust,ignore
//! use neuromance_context::transforms::{FilterCriteria, TransformPipeline, TransformOperation};
//! use neuromance_common::chat::MessageRole;
//!
//! let pipeline = TransformPipeline::new()
//!     .add(TransformOperation::Deduplicate)
//!     .add(TransformOperation::MergeConsecutive);
//!
//! let transformed = pipeline.apply(conversation);
//! ```

use chrono::{DateTime, Utc};
use neuromance_common::chat::{Conversation, Message, MessageRole};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tracing::debug;

/// Criteria for filtering messages in a conversation.
///
/// Multiple criteria can be combined to create complex filters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilterCriteria {
    /// Filter by message roles (if specified)
    pub roles: Option<Vec<MessageRole>>,

    /// Filter messages after this timestamp
    pub after: Option<DateTime<Utc>>,

    /// Filter messages before this timestamp
    pub before: Option<DateTime<Utc>>,

    /// Maximum number of messages to keep (most recent)
    pub limit: Option<usize>,

    /// Skip the first N messages
    pub offset: Option<usize>,
}

impl FilterCriteria {
    /// Filters by specific message roles.
    pub fn with_roles(mut self, roles: Vec<MessageRole>) -> Self {
        self.roles = Some(roles);
        self
    }

    /// Filters messages after a specific timestamp.
    pub fn after(mut self, timestamp: DateTime<Utc>) -> Self {
        self.after = Some(timestamp);
        self
    }

    /// Filters messages before a specific timestamp.
    pub fn before(mut self, timestamp: DateTime<Utc>) -> Self {
        self.before = Some(timestamp);
        self
    }

    /// Limits the number of messages (keeps most recent).
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Skips the first N messages.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// Represents a transformation operation that can be applied to messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TransformOperation {
    /// Merge consecutive messages from the same role into a single message.
    /// Useful for reducing message count when the same role sends multiple messages.
    MergeConsecutive,

    /// Remove duplicate messages based on content hash.
    /// Keeps the first occurrence of each unique message.
    Deduplicate,

    /// Remove tool messages (both tool calls and tool responses).
    /// Useful when tool context is no longer needed.
    RemoveToolMessages,

    /// Keep only the most recent N messages.
    TruncateOld(usize),

    /// Remove empty or whitespace-only messages.
    RemoveEmpty,

    /// Custom transformation with a name (for extensibility).
    Custom(String),
}

/// A pipeline of transformation operations to apply in sequence.
#[derive(Debug, Clone, Default)]
pub struct TransformPipeline {
    operations: Vec<TransformOperation>,
}

impl TransformPipeline {
    /// Creates a new empty transform pipeline.
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Adds an operation to the pipeline.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, operation: TransformOperation) -> Self {
        self.operations.push(operation);
        self
    }

    /// Applies all operations in the pipeline to a conversation.
    pub fn apply(&self, mut conversation: Conversation) -> Conversation {
        for operation in &self.operations {
            conversation = apply_operation(conversation, operation);
        }
        conversation
    }

    /// Returns the number of operations in the pipeline.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns true if the pipeline has no operations.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

/// Applies a single transform operation to a conversation.
fn apply_operation(conversation: Conversation, operation: &TransformOperation) -> Conversation {
    match operation {
        TransformOperation::MergeConsecutive => merge_consecutive_messages(conversation),
        TransformOperation::Deduplicate => deduplicate_messages(conversation),
        TransformOperation::RemoveToolMessages => remove_tool_messages(conversation),
        TransformOperation::TruncateOld(keep) => truncate_old_messages(conversation, *keep),
        TransformOperation::RemoveEmpty => remove_empty_messages(conversation),
        TransformOperation::Custom(_) => conversation, // Custom ops are no-ops by default
    }
}

/// Merges consecutive messages from the same role.
fn merge_consecutive_messages(conversation: Conversation) -> Conversation {
    let messages = conversation.get_messages();
    if messages.len() < 2 {
        return conversation;
    }

    let mut merged_messages: Vec<Message> = Vec::new();

    for msg in messages.iter() {
        if let Some(last) = merged_messages.last_mut() {
            // Merge if same role and neither has tool calls
            if last.role == msg.role
                && last.tool_calls.is_empty()
                && msg.tool_calls.is_empty()
                && last.tool_call_id.is_none()
                && msg.tool_call_id.is_none()
            {
                // Merge content with newline separator
                last.content = format!("{}\n\n{}", last.content, msg.content);
                last.timestamp = msg.timestamp; // Use later timestamp
                debug!("Merged consecutive {:?} messages", msg.role);
                continue;
            }
        }
        merged_messages.push(msg.clone());
    }

    rebuild_conversation(conversation, merged_messages)
}

/// Removes duplicate messages based on content.
fn deduplicate_messages(conversation: Conversation) -> Conversation {
    let messages = conversation.get_messages();
    let mut seen_hashes: HashSet<u64> = HashSet::new();
    let mut unique_messages: Vec<Message> = Vec::new();

    for msg in messages.iter() {
        // Create a simple hash of role + content
        let hash = calculate_message_hash(msg);

        if seen_hashes.insert(hash) {
            unique_messages.push(msg.clone());
        } else {
            debug!("Removed duplicate message: {:?}", msg.role);
        }
    }

    rebuild_conversation(conversation, unique_messages)
}

/// Calculates a simple hash for a message based on role and content.
fn calculate_message_hash(msg: &Message) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    msg.role.hash(&mut hasher);
    msg.content.hash(&mut hasher);
    hasher.finish()
}

/// Removes all tool-related messages.
fn remove_tool_messages(conversation: Conversation) -> Conversation {
    let messages = conversation.get_messages();
    let filtered: Vec<Message> = messages
        .iter()
        .filter(|msg| {
            // Keep if not a tool message and has no tool calls
            msg.role != MessageRole::Tool && msg.tool_calls.is_empty()
        })
        .cloned()
        .collect();

    rebuild_conversation(conversation, filtered)
}

/// Keeps only the most recent N messages.
fn truncate_old_messages(conversation: Conversation, keep: usize) -> Conversation {
    let messages = conversation.get_messages();
    if messages.len() <= keep {
        return conversation;
    }

    let start = messages.len() - keep;
    let truncated: Vec<Message> = messages[start..].to_vec();

    debug!(
        "Truncated {} old messages, keeping {}",
        messages.len() - keep,
        keep
    );

    rebuild_conversation(conversation, truncated)
}

/// Removes empty or whitespace-only messages.
fn remove_empty_messages(conversation: Conversation) -> Conversation {
    let messages = conversation.get_messages();
    let filtered: Vec<Message> = messages
        .iter()
        .filter(|msg| !msg.content.trim().is_empty() || !msg.tool_calls.is_empty())
        .cloned()
        .collect();

    rebuild_conversation(conversation, filtered)
}

/// Helper to rebuild a conversation with new messages.
fn rebuild_conversation(original: Conversation, messages: Vec<Message>) -> Conversation {
    let mut new_conversation = Conversation::new();
    new_conversation.id = original.id;
    new_conversation.title = original.title;
    new_conversation.description = original.description;
    new_conversation.created_at = original.created_at;
    new_conversation.updated_at = Utc::now();
    new_conversation.metadata = original.metadata;
    new_conversation.status = original.status;
    new_conversation.messages = Arc::new(messages);
    new_conversation
}

/// Applies filter criteria to a conversation, returning a new filtered conversation.
pub fn apply_filter(conversation: Conversation, criteria: FilterCriteria) -> Conversation {
    let messages = conversation.get_messages();
    let mut filtered_messages = Vec::new();

    for message in messages {
        // Filter by role
        if let Some(ref roles) = criteria.roles
            && !roles.contains(&message.role)
        {
            continue;
        }

        // Filter by timestamp
        if let Some(after) = criteria.after
            && message.timestamp <= after
        {
            continue;
        }

        if let Some(before) = criteria.before
            && message.timestamp >= before
        {
            continue;
        }

        filtered_messages.push(message.clone());
    }

    // Apply offset
    if let Some(offset) = criteria.offset {
        filtered_messages = filtered_messages.into_iter().skip(offset).collect();
    }

    // Apply limit
    if let Some(limit) = criteria.limit {
        filtered_messages.truncate(limit);
    }

    rebuild_conversation(conversation, filtered_messages)
}

/// Applies default transformation operations to a conversation.
///
/// The default pipeline applies:
/// 1. Remove empty messages
/// 2. Deduplicate identical messages
///
/// For more control, use `TransformPipeline` directly.
pub fn apply_transform(conversation: Conversation) -> Conversation {
    TransformPipeline::new()
        .add(TransformOperation::RemoveEmpty)
        .add(TransformOperation::Deduplicate)
        .apply(conversation)
}

/// Applies a custom transform pipeline to a conversation.
pub fn apply_transform_pipeline(
    conversation: Conversation,
    pipeline: &TransformPipeline,
) -> Conversation {
    pipeline.apply(conversation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use neuromance_common::chat::{Conversation, Message};

    #[test]
    fn test_filter_criteria_fluent() {
        let criteria = FilterCriteria::default()
            .with_roles(vec![MessageRole::User, MessageRole::Assistant])
            .limit(10);

        assert!(criteria.roles.is_some());
        assert_eq!(criteria.limit, Some(10));
    }

    #[test]
    fn test_filter_by_role() {
        let mut conv = Conversation::new();
        let user_msg = Message::user(conv.id, "Hello");
        let system_msg = Message::system(conv.id, "System");
        let assistant_msg = Message::assistant(conv.id, "Hi");

        conv.add_message(user_msg).unwrap();
        conv.add_message(system_msg).unwrap();
        conv.add_message(assistant_msg).unwrap();

        let criteria =
            FilterCriteria::default().with_roles(vec![MessageRole::User, MessageRole::Assistant]);

        let filtered = apply_filter(conv, criteria);
        assert_eq!(filtered.get_messages().len(), 2);
    }

    #[test]
    fn test_filter_limit() {
        let mut conv = Conversation::new();
        for i in 0..10 {
            let msg = Message::user(conv.id, format!("Message {}", i));
            conv.add_message(msg).unwrap();
        }

        let criteria = FilterCriteria::default().limit(5);
        let filtered = apply_filter(conv, criteria);
        assert_eq!(filtered.get_messages().len(), 5);
    }

    #[test]
    fn test_filter_offset() {
        let mut conv = Conversation::new();
        for i in 0..10 {
            let msg = Message::user(conv.id, format!("Message {}", i));
            conv.add_message(msg).unwrap();
        }

        let criteria = FilterCriteria::default().offset(5);
        let filtered = apply_filter(conv, criteria);
        assert_eq!(filtered.get_messages().len(), 5);
        assert_eq!(filtered.get_messages()[0].content, "Message 5");
    }

    #[test]
    fn test_filter_offset_and_limit() {
        let mut conv = Conversation::new();
        for i in 0..10 {
            let msg = Message::user(conv.id, format!("Message {}", i));
            conv.add_message(msg).unwrap();
        }

        let criteria = FilterCriteria::default().offset(2).limit(3);
        let filtered = apply_filter(conv, criteria);
        assert_eq!(filtered.get_messages().len(), 3);
        assert_eq!(filtered.get_messages()[0].content, "Message 2");
        assert_eq!(filtered.get_messages()[2].content, "Message 4");
    }

    #[test]
    fn test_transform_pipeline() {
        let pipeline = TransformPipeline::new()
            .add(TransformOperation::RemoveEmpty)
            .add(TransformOperation::Deduplicate);

        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn test_merge_consecutive_messages() {
        let mut conv = Conversation::new();
        conv.add_message(Message::user(conv.id, "Hello")).unwrap();
        conv.add_message(Message::user(conv.id, "How are you?"))
            .unwrap();
        conv.add_message(Message::assistant(conv.id, "Hi!"))
            .unwrap();
        conv.add_message(Message::assistant(conv.id, "I'm good!"))
            .unwrap();

        let merged = merge_consecutive_messages(conv);
        let messages = merged.get_messages();

        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("Hello"));
        assert!(messages[0].content.contains("How are you?"));
        assert!(messages[1].content.contains("Hi!"));
        assert!(messages[1].content.contains("I'm good!"));
    }

    #[test]
    fn test_deduplicate_messages() {
        let mut conv = Conversation::new();
        conv.add_message(Message::user(conv.id, "Hello")).unwrap();
        conv.add_message(Message::assistant(conv.id, "Hi")).unwrap();
        conv.add_message(Message::user(conv.id, "Hello")).unwrap(); // Duplicate
        conv.add_message(Message::assistant(conv.id, "Hello again"))
            .unwrap();

        let deduped = deduplicate_messages(conv);
        assert_eq!(deduped.get_messages().len(), 3);
    }

    #[test]
    fn test_remove_tool_messages() {
        let mut conv = Conversation::new();
        conv.add_message(Message::user(conv.id, "Hello")).unwrap();
        conv.add_message(Message::assistant(conv.id, "Let me check"))
            .unwrap();
        conv.add_message(Message::new(
            conv.id,
            MessageRole::Tool,
            "Tool result".to_string(),
        ))
        .unwrap();
        conv.add_message(Message::assistant(conv.id, "Done"))
            .unwrap();

        let filtered = remove_tool_messages(conv);
        assert_eq!(filtered.get_messages().len(), 3);

        // Verify no tool messages remain
        for msg in filtered.get_messages().iter() {
            assert_ne!(msg.role, MessageRole::Tool);
        }
    }

    #[test]
    fn test_truncate_old_messages() {
        let mut conv = Conversation::new();
        for i in 0..10 {
            let msg = Message::user(conv.id, format!("Message {}", i));
            conv.add_message(msg).unwrap();
        }

        let truncated = truncate_old_messages(conv, 3);
        let messages = truncated.get_messages();

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "Message 7");
        assert_eq!(messages[1].content, "Message 8");
        assert_eq!(messages[2].content, "Message 9");
    }

    #[test]
    fn test_remove_empty_messages() {
        let mut conv = Conversation::new();
        conv.add_message(Message::user(conv.id, "Hello")).unwrap();
        conv.add_message(Message::assistant(conv.id, "")).unwrap();
        conv.add_message(Message::user(conv.id, "   ")).unwrap();
        conv.add_message(Message::assistant(conv.id, "Response"))
            .unwrap();

        let filtered = remove_empty_messages(conv);
        assert_eq!(filtered.get_messages().len(), 2);
    }

    #[test]
    fn test_apply_transform_default() {
        let mut conv = Conversation::new();
        conv.add_message(Message::user(conv.id, "Hello")).unwrap();
        conv.add_message(Message::assistant(conv.id, "")).unwrap(); // Empty - will be removed
        conv.add_message(Message::user(conv.id, "Hello")).unwrap(); // Duplicate - will be removed
        conv.add_message(Message::assistant(conv.id, "Hi")).unwrap();

        let transformed = apply_transform(conv);
        assert_eq!(transformed.get_messages().len(), 2);
    }

    #[test]
    fn test_merge_consecutive_empty_unchanged() {
        let conv = Conversation::new();
        let merged = merge_consecutive_messages(conv);
        assert!(merged.get_messages().is_empty());
    }

    #[test]
    fn test_merge_consecutive_single_message_unchanged() {
        let mut conv = Conversation::new();
        conv.add_message(Message::user(conv.id, "only one"))
            .unwrap();

        let merged = merge_consecutive_messages(conv);
        assert_eq!(merged.get_messages().len(), 1);
        assert_eq!(merged.get_messages()[0].content, "only one");
    }

    #[test]
    fn test_filter_removes_all_when_no_role_matches() {
        let mut conv = Conversation::new();
        conv.add_message(Message::user(conv.id, "hi")).unwrap();
        conv.add_message(Message::user(conv.id, "there")).unwrap();

        let criteria = FilterCriteria::default().with_roles(vec![MessageRole::Assistant]);
        let filtered = apply_filter(conv, criteria);
        assert!(filtered.get_messages().is_empty());
    }

    #[test]
    fn test_filter_by_after_timestamp() {
        let mut conv = Conversation::new();
        let mut old = Message::user(conv.id, "old");
        old.timestamp = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let mut new = Message::user(conv.id, "new");
        new.timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        conv.add_message(old).unwrap();
        conv.add_message(new).unwrap();

        let cutoff = Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap();
        let filtered = apply_filter(conv, FilterCriteria::default().after(cutoff));

        assert_eq!(filtered.get_messages().len(), 1);
        assert_eq!(filtered.get_messages()[0].content, "new");
    }

    #[test]
    fn test_filter_by_before_timestamp() {
        let mut conv = Conversation::new();
        let mut old = Message::user(conv.id, "old");
        old.timestamp = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let mut new = Message::user(conv.id, "new");
        new.timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        conv.add_message(old).unwrap();
        conv.add_message(new).unwrap();

        let cutoff = Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap();
        let filtered = apply_filter(conv, FilterCriteria::default().before(cutoff));

        assert_eq!(filtered.get_messages().len(), 1);
        assert_eq!(filtered.get_messages()[0].content, "old");
    }

    #[test]
    fn test_transform_pipeline_apply() {
        let mut conv = Conversation::new();
        conv.add_message(Message::user(conv.id, "Hello")).unwrap();
        conv.add_message(Message::user(conv.id, "World")).unwrap();
        conv.add_message(Message::assistant(conv.id, "Hi")).unwrap();
        conv.add_message(Message::assistant(conv.id, "There"))
            .unwrap();

        let pipeline = TransformPipeline::new().add(TransformOperation::MergeConsecutive);

        let transformed = pipeline.apply(conv);
        assert_eq!(transformed.get_messages().len(), 2);
    }
}
