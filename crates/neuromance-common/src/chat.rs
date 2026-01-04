//! Conversation and message management for LLM interactions.
//!
//! This module provides the core abstractions for managing conversations and messages
//! when interacting with Large Language Models (LLMs). It handles message roles, tool
//! calling, conversation lifecycle, and metadata management.
//!
//! # Core Types
//!
//! - [`Message`]: Individual messages with role-based content (system, user, assistant, tool)
//! - [`Conversation`]: A thread of messages with lifecycle management and metadata
//! - [`MessageRole`]: Enum for message roles (system, user, assistant, tool)
//! - [`ConversationStatus`]: Enum for conversation lifecycle states
//!
//! # Example
//!
//! ```
//! use neuromance_common::chat::Conversation;
//! use neuromance_common::tools::ToolCall;
//!
//! let mut conv = Conversation::new();
//!
//! // Add messages
//! let system_msg = conv.system_message("You are a helpful assistant");
//! conv.add_message(system_msg).unwrap();
//!
//! let user_msg = conv.user_message("What's the weather in Tokyo?");
//! conv.add_message(user_msg).unwrap();
//!
//! // Assistant responds with a tool call
//! let tool_call = ToolCall::new("get_weather", [r#"{"location": "Tokyo"}"#]);
//! let assistant_msg = conv.assistant_message("Let me check that.")
//!     .with_tool_calls(vec![tool_call.clone()])
//!     .unwrap();
//! conv.add_message(assistant_msg).unwrap();
//!
//! // Tool result
//! let tool_msg = conv.tool_message(
//!     r#"{"temp": 18, "condition": "cloudy"}"#,
//!     tool_call.id,
//!     "get_weather".to_string()
//! ).unwrap();
//! conv.add_message(tool_msg).unwrap();
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use typed_builder::TypedBuilder;
use uuid::Uuid;

use crate::tools::ToolCall;

/// Represents the role of a message sender in a conversation.
///
/// Roles are serialized to lowercase strings matching the `OpenAI` API format.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MessageRole {
    /// System-level instructions or context for the LLM.
    #[serde(rename = "system")]
    System,

    /// Messages from the end user.
    #[serde(rename = "user")]
    User,

    /// Messages from the LLM assistant, optionally including tool calls.
    #[serde(rename = "assistant")]
    Assistant,

    /// Messages containing tool execution results with `tool_call_id` and `name` fields.
    #[serde(rename = "tool")]
    Tool,
}

/// A single message in a conversation.
///
/// Messages have a role (system, user, assistant, or tool), content, and optional metadata.
/// Tool calls are validated to only appear on assistant messages.
#[derive(Debug, Serialize, Deserialize, Clone, TypedBuilder)]
pub struct Message {
    /// Unique identifier for this message.
    #[builder(default = Uuid::new_v4())]
    pub id: Uuid,

    /// ID of the conversation this message belongs to.
    pub conversation_id: Uuid,

    /// The role of the message sender.
    pub role: MessageRole,

    /// The text content of the message.
    pub content: String,

    /// Application-specific metadata.
    #[builder(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// When this message was created.
    #[builder(default = Utc::now())]
    pub timestamp: DateTime<Utc>,

    /// Tool calls requested by this message (assistant messages only, uses `SmallVec` to avoid allocations for ≤2 calls).
    #[builder(default)]
    pub tool_calls: SmallVec<[ToolCall; 2]>,

    /// Tool call ID this message responds to (required for tool messages).
    #[builder(default)]
    pub tool_call_id: Option<String>,

    /// Function name (required for tool messages).
    #[builder(default)]
    pub name: Option<String>,

    /// Reasoning content from thinking models (optional, separate from main content).
    #[builder(default)]
    pub reasoning_content: Option<String>,

    /// Signature for reasoning content (required by some providers like Anthropic).
    #[builder(default)]
    pub reasoning_signature: Option<String>,
}

impl Message {
    /// Creates a new message with the specified role and content.
    pub fn new(conversation_id: Uuid, role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            conversation_id,
            role,
            content: content.into(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            tool_calls: SmallVec::new(),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            reasoning_signature: None,
        }
    }

    /// Creates a new system message.
    pub fn system(conversation_id: Uuid, content: impl Into<String>) -> Self {
        Self::new(conversation_id, MessageRole::System, content)
    }

    /// Creates a new user message.
    pub fn user(conversation_id: Uuid, content: impl Into<String>) -> Self {
        Self::new(conversation_id, MessageRole::User, content)
    }

    /// Creates a new assistant message.
    pub fn assistant(conversation_id: Uuid, content: impl Into<String>) -> Self {
        Self::new(conversation_id, MessageRole::Assistant, content)
    }

    /// Creates a new tool result message.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The conversation this message belongs to
    /// * `content` - The result/output from the tool execution
    /// * `tool_call_id` - The ID of the tool call this responds to
    /// * `function_name` - The name of the function that was called
    ///
    /// # Errors
    ///
    /// Returns an error if the `tool_call_id` is empty.
    pub fn tool(
        conversation_id: Uuid,
        content: impl Into<String>,
        tool_call_id: String,
        function_name: String,
    ) -> anyhow::Result<Self> {
        if tool_call_id.is_empty() {
            anyhow::bail!("Tool call ID cannot be empty");
        }
        if function_name.is_empty() {
            anyhow::bail!("Function name cannot be empty for tool messages");
        }
        let mut msg = Self::new(conversation_id, MessageRole::Tool, content);
        msg.tool_call_id = Some(tool_call_id);
        msg.name = Some(function_name);
        Ok(msg)
    }

    /// Adds a metadata key-value pair to this message.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Adds a metadata key-value pair to this message with automatic serialization.
    ///
    /// This is a convenience method that accepts any serializable type.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn with_metadata_typed<T: serde::Serialize>(
        mut self,
        key: impl Into<String>,
        value: T,
    ) -> anyhow::Result<Self> {
        let json_value = serde_json::to_value(value)?;
        self.metadata.insert(key.into(), json_value);
        Ok(self)
    }

    /// Sets the tool calls for this message.
    ///
    /// # Errors
    ///
    /// Returns an error if this message is not an assistant message.
    pub fn with_tool_calls(
        mut self,
        tool_calls: impl Into<SmallVec<[ToolCall; 2]>>,
    ) -> anyhow::Result<Self> {
        if self.role != MessageRole::Assistant {
            anyhow::bail!(
                "Tool calls can only be added to assistant messages, found {:?}",
                self.role
            );
        }
        self.tool_calls = tool_calls.into();
        Ok(self)
    }

    /// Adds a single tool call to this message.
    ///
    /// # Errors
    ///
    /// Returns an error if this message is not an assistant message.
    pub fn add_tool_call(&mut self, tool_call: ToolCall) -> anyhow::Result<()> {
        if self.role != MessageRole::Assistant {
            anyhow::bail!(
                "Tool calls can only be added to assistant messages, found {:?}",
                self.role
            );
        }
        self.tool_calls.push(tool_call);
        Ok(())
    }
}

/// The lifecycle status of a conversation.
///
/// Statuses serialize to lowercase strings: "active", "paused", "archived", "deleted".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum ConversationStatus {
    /// The conversation is currently active (default state).
    #[serde(rename = "active")]
    Active,

    /// The conversation is temporarily paused.
    #[serde(rename = "paused")]
    Paused,

    /// The conversation has been archived.
    #[serde(rename = "archived")]
    Archived,

    /// The conversation has been marked for deletion (soft delete).
    #[serde(rename = "deleted")]
    Deleted,
}

/// Represents a conversation thread containing multiple messages.
///
/// Manages conversation lifecycle, message ordering, and provides convenience methods
/// for creating properly-linked messages.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conversation {
    /// Unique identifier for this conversation.
    pub id: Uuid,

    /// Optional human-readable title.
    pub title: Option<String>,

    /// Optional longer description.
    pub description: Option<String>,

    /// When this conversation was created.
    pub created_at: DateTime<Utc>,

    /// When this conversation was last modified (updated on message add or status change).
    pub updated_at: DateTime<Utc>,

    /// Application-specific metadata.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Current status of the conversation (defaults to `Active`).
    pub status: ConversationStatus,

    /// Messages in this conversation (wrapped in `Arc` for efficient cloning).
    pub messages: Arc<Vec<Message>>,
}

impl Conversation {
    /// Creates a new active conversation with a generated ID.
    #[must_use]
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            title: None,
            description: None,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
            status: ConversationStatus::Active,
            messages: Arc::new(Vec::new()),
        }
    }

    /// Sets the title of this conversation.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the description of this conversation.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Changes the status of this conversation and updates the timestamp.
    pub fn set_status(&mut self, status: ConversationStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// Updates the `updated_at` timestamp to the current time.
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Adds a message to this conversation.
    ///
    /// # Errors
    ///
    /// Returns an error if the message's conversation ID doesn't match.
    pub fn add_message(&mut self, message: Message) -> anyhow::Result<()> {
        if message.conversation_id != self.id {
            anyhow::bail!(
                "Message conversation_id {} does not match conversation id {}",
                message.conversation_id,
                self.id
            );
        }
        Arc::make_mut(&mut self.messages).push(message);
        self.touch();
        Ok(())
    }

    /// Returns a reference to the messages in this conversation.
    #[must_use]
    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    /// Creates a new user message for this conversation.
    pub fn user_message(&self, content: impl Into<String>) -> Message {
        Message::user(self.id, content)
    }

    /// Creates a new assistant message for this conversation.
    pub fn assistant_message(&self, content: impl Into<String>) -> Message {
        Message::assistant(self.id, content)
    }

    /// Creates a new system message for this conversation.
    pub fn system_message(&self, content: impl Into<String>) -> Message {
        Message::system(self.id, content)
    }

    /// Creates a new tool result message for this conversation.
    ///
    /// # Errors
    ///
    /// Returns an error if tool message creation fails.
    pub fn tool_message(
        &self,
        content: impl Into<String>,
        tool_call_id: String,
        function_name: String,
    ) -> anyhow::Result<Message> {
        Message::tool(self.id, content, tool_call_id, function_name)
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_message_creation() {
        let conv_id = Uuid::new_v4();
        let msg = Message::user(conv_id, "Hello, world!");

        assert_eq!(msg.conversation_id, conv_id);
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "Hello, world!");
        assert!(msg.tool_calls.is_empty());
    }

    #[test]
    fn test_conversation_creation() {
        let conv = Conversation::new()
            .with_title("Test Conversation")
            .with_description("A test conversation");

        assert_eq!(conv.title, Some("Test Conversation".to_string()));
        assert_eq!(conv.description, Some("A test conversation".to_string()));
        assert_eq!(conv.status, ConversationStatus::Active);
    }

    #[test]
    fn test_tool_call_creation() {
        let tool_call = ToolCall::new("test_function", [r#"{"param": "value"}"#]);

        assert_eq!(tool_call.function.name, "test_function");
        assert_eq!(tool_call.function.arguments, vec![r#"{"param": "value"}"#]);
        assert_eq!(tool_call.call_type, "function");
        assert!(!tool_call.id.is_empty());
    }

    #[test]
    fn test_message_with_tool_calls() {
        let conv_id = Uuid::new_v4();
        let tool_call = ToolCall::new("get_weather", [r#"{"location": "New York"}"#]);
        let msg = Message::assistant(conv_id, "I'll check the weather for you.")
            .with_tool_calls(vec![tool_call])
            .expect("Failed to add tool calls");

        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            msg.tool_calls[0].function.arguments,
            vec![r#"{"location": "New York"}"#]
        );
    }

    #[test]
    fn test_message_tool_call_validation() {
        let conv_id = Uuid::new_v4();
        let tool_call = ToolCall::new("get_weather", [r#"{"location": "New York"}"#]);

        // Should fail on user message
        let user_msg = Message::user(conv_id, "What's the weather?");
        let result = user_msg.with_tool_calls(vec![tool_call.clone()]);
        assert!(result.is_err());

        // Should succeed on assistant message
        let assistant_msg = Message::assistant(conv_id, "Let me check.");
        let result = assistant_msg.with_tool_calls(vec![tool_call]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tool_message_validation() {
        let conv_id = Uuid::new_v4();

        // Should fail with empty tool_call_id
        let result = Message::tool(conv_id, "Result", String::new(), "test_func".to_string());
        assert!(result.is_err());

        // Should fail with empty function name
        let result = Message::tool(conv_id, "Result", "call_123".to_string(), String::new());
        assert!(result.is_err());

        // Should succeed with valid tool_call_id and function name
        let result = Message::tool(
            conv_id,
            "Result",
            "call_123".to_string(),
            "test_func".to_string(),
        );
        assert!(result.is_ok());
        let msg = result.unwrap();
        assert_eq!(msg.name, Some("test_func".to_string()));
    }

    #[test]
    fn test_conversation_add_message() {
        let mut conv = Conversation::new();
        let msg = Message::user(conv.id, "Hello");

        conv.add_message(msg).expect("Failed to add message");
        assert_eq!(conv.messages.len(), 1);
        assert_eq!(conv.messages[0].content, "Hello");
    }

    #[test]
    fn test_conversation_add_message_wrong_id() {
        let mut conv = Conversation::new();
        let other_id = Uuid::new_v4();
        let msg = Message::user(other_id, "Hello");

        let result = conv.add_message(msg);
        assert!(result.is_err());
    }

    #[test]
    fn test_message_with_metadata_typed() {
        let conv_id = Uuid::new_v4();
        let msg = Message::user(conv_id, "Hello")
            .with_metadata_typed("count", 42)
            .expect("Failed to add metadata");

        assert_eq!(msg.metadata.get("count"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn test_tool_call_with_multiple_args() {
        let tool_call = ToolCall::new(
            "complex_function",
            vec![
                "arg1".to_string(),
                "arg2".to_string(),
                r#"{"key": "value"}"#.to_string(),
            ],
        );

        assert_eq!(tool_call.function.name, "complex_function");
        assert_eq!(tool_call.function.arguments.len(), 3);
        assert_eq!(tool_call.function.arguments[0], "arg1");
        assert_eq!(tool_call.function.arguments[1], "arg2");
        assert_eq!(tool_call.function.arguments[2], r#"{"key": "value"}"#);
    }
}

#[cfg(test)]
mod proptests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn message_accepts_string_types(content in ".*") {
            let conv_id = Uuid::new_v4();

            // Test with &str
            let msg1 = Message::new(conv_id, MessageRole::User, content.as_str());
            assert_eq!(msg1.content, content);

            // Test with String
            let msg2 = Message::new(conv_id, MessageRole::User, content.clone());
            assert_eq!(msg2.content, content);

            // Test builder methods
            let msg3 = Message::user(conv_id, content.as_str());
            assert_eq!(msg3.role, MessageRole::User);
            assert_eq!(msg3.content, content);
        }

        #[test]
        fn message_serialization_roundtrip(
            content in ".*",
            role_idx in 0usize..4,
        ) {
            let conv_id = Uuid::new_v4();
            let role = match role_idx {
                0 => MessageRole::User,
                1 => MessageRole::Assistant,
                2 => MessageRole::System,
                _ => MessageRole::Tool,
            };

            let msg = Message::new(conv_id, role, content);
            let serialized = serde_json::to_string(&msg).expect("Failed to serialize");
            let deserialized: Message = serde_json::from_str(&serialized)
                .expect("Failed to deserialize");

            assert_eq!(msg.id, deserialized.id);
            assert_eq!(msg.conversation_id, deserialized.conversation_id);
            assert_eq!(msg.role, deserialized.role);
            assert_eq!(msg.content, deserialized.content);
        }

        #[test]
        fn conversation_builder_with_strings(
            title in ".*",
            description in ".*",
        ) {
            // Test with &str
            let conv1 = Conversation::new()
                .with_title(title.as_str())
                .with_description(description.as_str());
            assert_eq!(conv1.title, Some(title.clone()));
            assert_eq!(conv1.description, Some(description.clone()));

            // Test with String
            let conv2 = Conversation::new()
                .with_title(title.clone())
                .with_description(description.clone());
            assert_eq!(conv2.title, Some(title));
            assert_eq!(conv2.description, Some(description));
        }

        #[test]
        fn tool_call_accepts_various_argument_types(
            func_name in ".*",
            args in prop::collection::vec(".*", 0..10),
        ) {
            // Test with Vec<String>
            let tc1 = ToolCall::new(func_name.as_str(), args.clone());
            assert_eq!(tc1.function.name, func_name);
            assert_eq!(tc1.function.arguments, args);

            // Test with &[&str]
            let str_refs: Vec<&str> = args.iter().map(std::string::String::as_str).collect();
            let tc2 = ToolCall::new(func_name.as_str(), str_refs);
            assert_eq!(tc2.function.name, func_name);
            assert_eq!(tc2.function.arguments, args);
        }

        #[test]
        fn message_metadata_operations(
            key in ".*",
            value_num in 0i64..1_000_000,
        ) {
            let conv_id = Uuid::new_v4();
            let msg = Message::user(conv_id, "test")
                .with_metadata(key.as_str(), serde_json::json!(value_num));

            assert!(msg.metadata.contains_key(&key));
            assert_eq!(msg.metadata[&key], serde_json::json!(value_num));
        }

        #[test]
        fn conversation_status_transitions(
            status_idx in 0usize..4,
        ) {
            let status = match status_idx {
                1 => ConversationStatus::Archived,
                2 => ConversationStatus::Deleted,
                _ => ConversationStatus::Active,
            };

            let mut conv = Conversation::new();
            conv.set_status(status.clone());

            assert_eq!(conv.status, status);
        }

        #[test]
        fn message_clone_preserves_data(content in ".*") {
            let conv_id = Uuid::new_v4();
            let original = Message::user(conv_id, content.as_str());
            let cloned = original.clone();

            assert_eq!(original.id, cloned.id);
            assert_eq!(original.conversation_id, cloned.conversation_id);
            assert_eq!(original.role, cloned.role);
            assert_eq!(original.content, cloned.content);
            assert_eq!(original.timestamp, cloned.timestamp);
        }

        #[test]
        fn fuzz_message_deserialization(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            // Should not panic on arbitrary bytes
            let _ = serde_json::from_slice::<Message>(&data);
        }

        #[test]
        fn fuzz_message_json_with_invalid_roles(
            content in "[\\p{L}\\p{N}\\p{P}\\p{S} ]{0,100}",
            role_str in "[a-z]{1,20}",
        ) {
            let conv_id = Uuid::new_v4();
            let msg_id = Uuid::new_v4();
            // Escape content for JSON
            let escaped_content = content.replace('\\', "\\\\").replace('"', "\\\"");
            let json = format!(
                r#"{{"id":"{msg_id}","conversation_id":"{conv_id}","role":"{role_str}","content":"{escaped_content}","metadata":{{}},"timestamp":"2024-01-01T00:00:00Z","tool_calls":[],"tool_call_id":null,"name":null}}"#
            );
            // Should handle invalid roles gracefully (will fail deserialization for unknown roles)
            let _ = serde_json::from_str::<Message>(&json);
        }

        #[test]
        fn fuzz_message_with_extreme_lengths(
            content_len in 10000usize..20000,
        ) {
            let conv_id = Uuid::new_v4();
            // Generate large content string
            let content: String = "a".repeat(content_len);
            let msg = Message::user(conv_id, content);

            // Should serialize and deserialize large content
            let json = serde_json::to_string(&msg).unwrap();
            let deserialized: Message = serde_json::from_str(&json).unwrap();
            assert_eq!(msg.content, deserialized.content);
        }

        #[test]
        fn fuzz_tool_message_with_invalid_ids(
            content in ".*",
            tool_call_id in ".*",
            func_name in ".*",
        ) {
            let conv_id = Uuid::new_v4();
            let result = Message::tool(conv_id, content.clone(), tool_call_id.clone(), func_name.clone());

            // Empty IDs should fail, others should succeed
            if tool_call_id.is_empty() || func_name.is_empty() {
                assert!(result.is_err());
            } else {
                assert!(result.is_ok());
                let msg = result.unwrap();
                assert_eq!(msg.tool_call_id, Some(tool_call_id));
                assert_eq!(msg.name, Some(func_name));
                assert_eq!(msg.content, content);
            }
        }

        #[test]
        fn fuzz_message_with_special_characters(
            content in r#"[\x00-\x1F\x7F\n\r\t"'`{}\[\]]*"#,
        ) {
            let conv_id = Uuid::new_v4();
            let msg = Message::user(conv_id, content.clone());

            // Should handle special characters in serialization
            let json_result = serde_json::to_string(&msg);
            assert!(json_result.is_ok());

            if let Ok(json) = json_result {
                let parsed: Result<Message, _> = serde_json::from_str(&json);
                if let Ok(parsed_msg) = parsed {
                    assert_eq!(parsed_msg.content, content);
                }
            }
        }

        #[test]
        fn fuzz_conversation_serialization(
            title in prop::option::of(".*"),
            description in prop::option::of(".*"),
            num_messages in 0usize..20,
        ) {
            let mut conv = Conversation::new();
            if let Some(t) = title {
                conv = conv.with_title(t);
            }
            if let Some(d) = description {
                conv = conv.with_description(d);
            }

            // Add random messages
            for i in 0..num_messages {
                let msg = conv.user_message(format!("Message {i}"));
                let _ = conv.add_message(msg);
            }

            // Should serialize and deserialize
            let json = serde_json::to_string(&conv).unwrap();
            let parsed: Conversation = serde_json::from_str(&json).unwrap();

            assert_eq!(conv.id, parsed.id);
            assert_eq!(conv.title, parsed.title);
            assert_eq!(conv.description, parsed.description);
            assert_eq!(conv.messages.len(), parsed.messages.len());
        }
    }
}
