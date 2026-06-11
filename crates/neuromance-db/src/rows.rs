//! Pure mapping between database rows and `neuromance-common` types.
//!
//! Everything here is side-effect free so the round-trip logic can be
//! unit-tested without a database.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use neuromance_common::chat::{ConversationStatus, Message, MessageRole, ReasoningContent};
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
    column: &'static str,
    id: Uuid,
) -> Result<String, DbError> {
    match serde_json::to_value(value) {
        Ok(Value::String(s)) => Ok(s),
        Ok(other) => Err(DbError::Encode {
            column,
            id,
            source: serde::ser::Error::custom(format!("expected a JSON string, got {other}")),
        }),
        Err(source) => Err(DbError::Encode { column, id, source }),
    }
}

pub fn role_to_string(role: MessageRole, message_id: Uuid) -> Result<String, DbError> {
    enum_to_db_string(&role, "role", message_id)
}

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

pub fn status_to_string(
    status: &ConversationStatus,
    conversation_id: Uuid,
) -> Result<String, DbError> {
    enum_to_db_string(status, "status", conversation_id)
}

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

/// The JSON/text column values derived from a [`Message`] for an INSERT.
pub struct MessageColumns {
    pub role: String,
    pub tool_calls: Value,
    pub reasoning: Option<Value>,
    pub metadata: Value,
}

pub fn message_to_columns(message: &Message) -> Result<MessageColumns, DbError> {
    let encode = |column: &'static str, source: serde_json::Error| DbError::Encode {
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
}

impl MessageRow {
    pub fn into_message(self) -> Result<Message, DbError> {
        let id = self.id;
        let decode = move |column: &'static str, source: serde_json::Error| DbError::Decode {
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
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

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
}
