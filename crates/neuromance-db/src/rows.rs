//! Pure mapping between database rows and `neuromance-common` types.
//!
//! Everything here is side-effect free so the round-trip logic can be
//! unit-tested without a database.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use neuromance_common::chat::{
    ConversationStatus, Message, MessageRole, ReasoningContent, TaskStatus,
};
use neuromance_common::client::Usage;
use neuromance_common::tools::ToolCall;
use serde::Serialize;
use serde_json::Value;
use smallvec::SmallVec;
use uuid::Uuid;

use crate::error::DbError;

/// Serializes an enum that maps to a single JSON string (e.g. [`MessageRole`],
/// [`ConversationStatus`]) into that string.
///
/// Going through serde keeps the database representation in lockstep with the
/// wire format, including any variants added to these `#[non_exhaustive]`
/// enums later.
fn enum_to_db_string<T: Serialize>(
    value: &T,
    table: &'static str,
    column: &'static str,
    id: Uuid,
) -> Result<String, DbError> {
    match serde_json::to_value(value) {
        Ok(Value::String(s)) => Ok(s),
        Ok(other) => Err(DbError::Encode {
            table,
            column,
            id,
            source: serde::ser::Error::custom(format!("expected a JSON string, got {other}")),
        }),
        Err(source) => Err(DbError::Encode {
            table,
            column,
            id,
            source,
        }),
    }
}

/// Maps a [`MessageRole`] to its stored `role` column string.
pub fn role_to_string(role: MessageRole, message_id: Uuid) -> Result<String, DbError> {
    enum_to_db_string(&role, "messages", "role", message_id)
}

/// Parses a stored `role` column string back into a [`MessageRole`].
pub fn role_from_str(value: &str, message_id: Uuid) -> Result<MessageRole, DbError> {
    match value {
        "system" => Ok(MessageRole::System),
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "tool" => Ok(MessageRole::Tool),
        _ => Err(DbError::UnknownRole {
            value: value.to_string(),
            message_id,
        }),
    }
}

/// Maps a [`ConversationStatus`] to its stored `status` column string.
pub fn status_to_string(
    status: &ConversationStatus,
    conversation_id: Uuid,
) -> Result<String, DbError> {
    enum_to_db_string(status, "conversations", "status", conversation_id)
}

/// Parses a stored `status` column string back into a [`ConversationStatus`].
pub fn status_from_str(value: &str, conversation_id: Uuid) -> Result<ConversationStatus, DbError> {
    match value {
        "active" => Ok(ConversationStatus::Active),
        "paused" => Ok(ConversationStatus::Paused),
        "archived" => Ok(ConversationStatus::Archived),
        "deleted" => Ok(ConversationStatus::Deleted),
        _ => Err(DbError::UnknownStatus {
            value: value.to_string(),
            conversation_id,
        }),
    }
}

/// Maps a [`TaskStatus`] to its stored `status` column string.
pub fn task_status_to_string(status: TaskStatus, task_id: Uuid) -> Result<String, DbError> {
    enum_to_db_string(&status, "tasks", "status", task_id)
}

/// Parses a stored task `status` column string back into a [`TaskStatus`].
pub fn task_status_from_str(value: &str, task_id: Uuid) -> Result<TaskStatus, DbError> {
    match value {
        "pending" => Ok(TaskStatus::Pending),
        "running" => Ok(TaskStatus::Running),
        "succeeded" => Ok(TaskStatus::Succeeded),
        "failed" => Ok(TaskStatus::Failed),
        "cancelled" => Ok(TaskStatus::Cancelled),
        _ => Err(DbError::UnknownTaskStatus {
            value: value.to_string(),
            task_id,
        }),
    }
}

/// The JSON/text column values derived from a [`Message`] for an INSERT.
pub struct MessageColumns {
    pub role: String,
    pub tool_calls: Value,
    pub reasoning: Option<Value>,
    pub metadata: Value,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub usage: Option<Value>,
}

/// Derives the encoded column values for inserting a [`Message`].
pub fn message_to_columns(message: &Message) -> Result<MessageColumns, DbError> {
    let encode = |column: &'static str, source: serde_json::Error| DbError::Encode {
        table: "messages",
        column,
        id: message.id,
        source,
    };
    Ok(MessageColumns {
        role: role_to_string(message.role, message.id)?,
        tool_calls: serde_json::to_value(&message.tool_calls)
            .map_err(|e| encode("tool_calls", e))?,
        reasoning: message
            .reasoning
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|e| encode("reasoning", e))?,
        metadata: serde_json::to_value(&message.metadata).map_err(|e| encode("metadata", e))?,
        model: message.model.clone(),
        provider: message.provider.clone(),
        usage: message
            .usage
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|e| encode("usage", e))?,
    })
}

/// A full row from the `messages` table.
pub struct MessageRow {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub role: String,
    pub content: String,
    pub tool_calls: Value,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
    pub reasoning: Option<Value>,
    pub metadata: Value,
    pub timestamp: DateTime<Utc>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub usage: Option<Value>,
}

impl MessageRow {
    /// Decodes this row into a [`Message`], parsing its JSON columns.
    pub fn into_message(self) -> Result<Message, DbError> {
        let id = self.id;
        let decode = move |column: &'static str, source: serde_json::Error| DbError::Decode {
            table: "messages",
            column,
            id,
            source,
        };
        Ok(Message {
            id,
            conversation_id: self.conversation_id,
            role: role_from_str(&self.role, id)?,
            content: self.content,
            metadata: serde_json::from_value::<HashMap<String, Value>>(self.metadata)
                .map_err(|e| decode("metadata", e))?,
            timestamp: self.timestamp,
            tool_calls: serde_json::from_value::<SmallVec<[ToolCall; 2]>>(self.tool_calls)
                .map_err(|e| decode("tool_calls", e))?,
            tool_call_id: self.tool_call_id,
            name: self.name,
            reasoning: self
                .reasoning
                .map(serde_json::from_value::<ReasoningContent>)
                .transpose()
                .map_err(|e| decode("reasoning", e))?,
            model: self.model,
            provider: self.provider,
            usage: self
                .usage
                .map(serde_json::from_value::<Usage>)
                .transpose()
                .map_err(|e| decode("usage", e))?,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use neuromance_common::client::InputTokensDetails;
    use proptest::prelude::*;

    use super::*;

    fn row_from(message: &Message, columns: MessageColumns) -> MessageRow {
        MessageRow {
            id: message.id,
            conversation_id: message.conversation_id,
            role: columns.role,
            content: message.content.clone(),
            tool_calls: columns.tool_calls,
            tool_call_id: message.tool_call_id.clone(),
            name: message.name.clone(),
            reasoning: columns.reasoning,
            metadata: columns.metadata,
            timestamp: message.timestamp,
            model: columns.model,
            provider: columns.provider,
            usage: columns.usage,
        }
    }

    fn assert_round_trip(message: &Message) {
        let columns = message_to_columns(message).unwrap();
        let restored = row_from(message, columns).into_message().unwrap();

        assert_eq!(restored.id, message.id);
        assert_eq!(restored.conversation_id, message.conversation_id);
        assert_eq!(restored.role, message.role);
        assert_eq!(restored.content, message.content);
        assert_eq!(restored.metadata, message.metadata);
        assert_eq!(restored.timestamp, message.timestamp);
        assert_eq!(restored.tool_calls, message.tool_calls);
        assert_eq!(restored.tool_call_id, message.tool_call_id);
        assert_eq!(restored.name, message.name);
        assert_eq!(restored.reasoning, message.reasoning);
        assert_eq!(restored.model, message.model);
        assert_eq!(restored.provider, message.provider);
        assert_eq!(restored.usage, message.usage);
    }

    #[test]
    fn test_plain_user_message_round_trips() {
        let message = Message::user(Uuid::new_v4(), "hello")
            .with_metadata("source".to_string(), serde_json::json!("test"));
        assert_round_trip(&message);
    }

    #[test]
    fn test_assistant_message_with_tool_calls_round_trips() {
        let message = Message::assistant(Uuid::new_v4(), "")
            .with_tool_calls(vec![
                ToolCall::new("get_weather", r#"{"city":"Berlin"}"#),
                ToolCall::new("get_time", "{}"),
            ])
            .unwrap();
        assert_round_trip(&message);
    }

    #[test]
    fn test_tool_result_message_round_trips() {
        let message = Message::tool(
            Uuid::new_v4(),
            "22C and sunny",
            "call_123".to_string(),
            "get_weather".to_string(),
        )
        .unwrap();
        assert_round_trip(&message);
    }

    #[test]
    fn test_reasoning_with_signature_round_trips() {
        let mut message = Message::assistant(Uuid::new_v4(), "answer");
        message.reasoning = Some(ReasoningContent::with_signature("thinking...", "sig-abc"));
        assert_round_trip(&message);
    }

    #[test]
    fn test_assistant_message_with_model_and_usage_round_trips() {
        let mut message = Message::assistant(Uuid::new_v4(), "the answer is 42");
        message.model = Some("claude-sonnet-4-5-20250929".to_string());
        message.provider = Some("anthropic".to_string());
        message.usage = Some(Usage {
            prompt_tokens: 120,
            completion_tokens: 34,
            total_tokens: 154,
            cost: Some(0.0012),
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: 80,
                cache_creation_tokens: 0,
            }),
            output_tokens_details: None,
        });
        assert_round_trip(&message);
    }

    #[test]
    fn test_model_and_usage_absent_encode_as_null() {
        let message = Message::user(Uuid::new_v4(), "hi");
        let columns = message_to_columns(&message).unwrap();
        assert_eq!(columns.model, None);
        assert_eq!(columns.provider, None);
        assert_eq!(columns.usage, None);
    }

    #[test]
    fn test_corrupt_usage_json_is_a_decode_error() {
        let mut message = Message::assistant(Uuid::new_v4(), "answer");
        message.usage = Some(Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        });
        let columns = message_to_columns(&message).unwrap();
        let mut row = row_from(&message, columns);
        row.usage = Some(serde_json::json!("not a usage object"));
        let err = row.into_message().unwrap_err();
        assert!(matches!(
            err,
            DbError::Decode {
                column: "usage",
                ..
            }
        ));
    }

    #[test]
    fn test_empty_tool_calls_encode_as_json_array() {
        let message = Message::user(Uuid::new_v4(), "hi");
        let columns = message_to_columns(&message).unwrap();
        assert_eq!(columns.tool_calls, serde_json::json!([]));
        assert_eq!(columns.reasoning, None);
    }

    #[test]
    fn test_all_roles_round_trip_through_strings() {
        let id = Uuid::new_v4();
        for role in [
            MessageRole::System,
            MessageRole::User,
            MessageRole::Assistant,
            MessageRole::Tool,
        ] {
            let s = role_to_string(role, id).unwrap();
            assert_eq!(role_from_str(&s, id).unwrap(), role);
        }
    }

    #[test]
    fn test_unknown_role_is_an_error() {
        let id = Uuid::new_v4();
        let err = role_from_str("operator", id).unwrap_err();
        assert!(matches!(
            err,
            DbError::UnknownRole { value, message_id } if value == "operator" && message_id == id
        ));
    }

    #[test]
    fn test_all_statuses_round_trip_through_strings() {
        let id = Uuid::new_v4();
        for status in [
            ConversationStatus::Active,
            ConversationStatus::Paused,
            ConversationStatus::Archived,
            ConversationStatus::Deleted,
        ] {
            let s = status_to_string(&status, id).unwrap();
            assert_eq!(status_from_str(&s, id).unwrap(), status);
        }
    }

    #[test]
    fn test_unknown_status_is_an_error() {
        let id = Uuid::new_v4();
        let err = status_from_str("tombstoned", id).unwrap_err();
        assert!(matches!(
            err,
            DbError::UnknownStatus { value, conversation_id } if value == "tombstoned" && conversation_id == id
        ));
    }

    #[test]
    fn test_all_task_statuses_round_trip_through_strings() {
        let id = Uuid::new_v4();
        for status in [
            TaskStatus::Pending,
            TaskStatus::Running,
            TaskStatus::Succeeded,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ] {
            let s = task_status_to_string(status, id).unwrap();
            assert_eq!(task_status_from_str(&s, id).unwrap(), status);
        }
    }

    #[test]
    fn test_unknown_task_status_is_an_error() {
        let id = Uuid::new_v4();
        let err = task_status_from_str("queued", id).unwrap_err();
        assert!(matches!(
            err,
            DbError::UnknownTaskStatus { value, task_id } if value == "queued" && task_id == id
        ));
    }

    #[test]
    fn test_corrupt_tool_calls_json_is_a_decode_error() {
        let message = Message::user(Uuid::new_v4(), "hi");
        let columns = message_to_columns(&message).unwrap();
        let mut row = row_from(&message, columns);
        row.tool_calls = serde_json::json!({"not": "an array"});
        let err = row.into_message().unwrap_err();
        assert!(matches!(
            err,
            DbError::Decode {
                column: "tool_calls",
                ..
            }
        ));
    }

    #[test]
    fn test_corrupt_metadata_json_is_a_decode_error() {
        let message = Message::user(Uuid::new_v4(), "hi");
        let columns = message_to_columns(&message).unwrap();
        let mut row = row_from(&message, columns);
        row.metadata = serde_json::json!("not a map");
        let err = row.into_message().unwrap_err();
        assert!(matches!(
            err,
            DbError::Decode {
                column: "metadata",
                ..
            }
        ));
    }

    #[test]
    fn test_corrupt_reasoning_json_is_a_decode_error() {
        let message = Message::user(Uuid::new_v4(), "hi");
        let columns = message_to_columns(&message).unwrap();
        let mut row = row_from(&message, columns);
        row.reasoning = Some(serde_json::json!("not a reasoning object"));
        let err = row.into_message().unwrap_err();
        assert!(matches!(
            err,
            DbError::Decode {
                column: "reasoning",
                ..
            }
        ));
    }

    /// Anchors the encoded role spellings to literal DB strings. Unlike
    /// [`test_all_roles_round_trip_through_strings`], which runs the live
    /// encoder on both sides, this catches a drift in the stored spelling.
    #[test]
    fn test_role_strings_match_stored_spelling() {
        let id = Uuid::new_v4();
        for (stored, role) in [
            ("system", MessageRole::System),
            ("user", MessageRole::User),
            ("assistant", MessageRole::Assistant),
            ("tool", MessageRole::Tool),
        ] {
            assert_eq!(role_from_str(stored, id).unwrap(), role);
            assert_eq!(role_to_string(role, id).unwrap(), stored);
        }
    }

    /// Anchors the encoded status spellings to literal DB strings, for the
    /// same reason as [`test_role_strings_match_stored_spelling`].
    #[test]
    fn test_status_strings_match_stored_spelling() {
        let id = Uuid::new_v4();
        for (stored, status) in [
            ("active", ConversationStatus::Active),
            ("paused", ConversationStatus::Paused),
            ("archived", ConversationStatus::Archived),
            ("deleted", ConversationStatus::Deleted),
        ] {
            assert_eq!(status_from_str(stored, id).unwrap(), status);
            assert_eq!(status_to_string(&status, id).unwrap(), stored);
        }
    }

    /// Anchors the encoded task-status spellings to literal DB strings, for the
    /// same reason as [`test_status_strings_match_stored_spelling`].
    #[test]
    fn test_task_status_strings_match_stored_spelling() {
        let id = Uuid::new_v4();
        for (stored, status) in [
            ("pending", TaskStatus::Pending),
            ("running", TaskStatus::Running),
            ("succeeded", TaskStatus::Succeeded),
            ("failed", TaskStatus::Failed),
            ("cancelled", TaskStatus::Cancelled),
        ] {
            assert_eq!(task_status_from_str(stored, id).unwrap(), status);
            assert_eq!(task_status_to_string(status, id).unwrap(), stored);
        }
    }

    proptest! {
        /// A message built from arbitrary JSON-backed fields survives the
        /// column encode → row decode round-trip unchanged.
        #[test]
        fn prop_message_round_trips(
            content in ".*",
            role_idx in 0usize..4,
            metadata in proptest::collection::hash_map(
                "[a-z]{1,8}",
                "[a-zA-Z0-9 ]{0,16}",
                0..4,
            ),
            reasoning in proptest::option::of((
                "[ -~]{0,32}",
                proptest::option::of("[a-z0-9]{1,16}"),
            )),
            tool_calls in proptest::collection::vec(
                ("[a-z_]{1,12}", "[ -~]{0,24}"),
                0..3,
            ),
        ) {
            let role = [
                MessageRole::System,
                MessageRole::User,
                MessageRole::Assistant,
                MessageRole::Tool,
            ][role_idx];

            let mut message = Message::new(Uuid::new_v4(), role, content);
            message.metadata = metadata
                .into_iter()
                .map(|(k, v)| (k, Value::String(v)))
                .collect();
            message.reasoning = reasoning.map(|(content, signature)| ReasoningContent {
                content,
                signature,
            });
            message.tool_calls = tool_calls
                .into_iter()
                .map(|(name, args)| ToolCall::new(name, args))
                .collect();

            assert_round_trip(&message);
        }
    }
}
