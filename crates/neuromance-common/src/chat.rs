//! Conversation and message management for LLM interactions.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use uuid::Uuid;

use crate::tools::ToolCall;

/// Represents the role of a message sender in a conversation.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum MessageRole {
    /// System-level instructions or context
    #[serde(rename = "system")]
    System,
    /// Messages from the end user
    #[serde(rename = "user")]
    User,
    /// Messages from the LLM assistant
    #[serde(rename = "assistant")]
    Assistant,
    /// Messages containing tool execution results
    #[serde(rename = "tool")]
    Tool,
}

/// A single message in a conversation.
///
/// Messages can contain text content, metadata, tool calls, and references to tool call results.
#[derive(Debug, Serialize, Deserialize, Clone, TypedBuilder)]
pub struct Message {
    /// Unique identifier for this message
    #[builder(default = Uuid::new_v4())]
    pub id: Uuid,
    /// ID of the conversation this message belongs to
    pub conversation_id: Uuid,
    /// The role of the message sender
    pub role: MessageRole,
    /// The text content of the message
    pub content: String,
    /// Additional metadata attached to this message
    #[builder(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// When this message was created
    #[builder(default = Utc::now())]
    pub timestamp: DateTime<Utc>,
    /// Tool calls requested by this message (for assistant messages)
    #[builder(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Reference to the tool call this message responds to (for tool messages)
    #[builder(default)]
    pub tool_call_id: Option<String>,
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
            tool_calls: Vec::new(),
            tool_call_id: None,
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
    /// # Errors
    ///
    /// Returns an error if the tool_call_id is empty.
    pub fn tool(
        conversation_id: Uuid,
        content: impl Into<String>,
        tool_call_id: String,
    ) -> anyhow::Result<Self> {
        if tool_call_id.is_empty() {
            anyhow::bail!("Tool call ID cannot be empty");
        }
        let mut msg = Self::new(conversation_id, MessageRole::Tool, content);
        msg.tool_call_id = Some(tool_call_id);
        Ok(msg)
    }

    /// Adds a metadata key-value pair to this message.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Adds a metadata key-value pair to this message with automatic serialization.
    ///
    /// This is a convenience method that accepts any serializable type.
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
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> anyhow::Result<Self> {
        if self.role != MessageRole::Assistant {
            anyhow::bail!(
                "Tool calls can only be added to assistant messages, found {:?}",
                self.role
            );
        }
        self.tool_calls = tool_calls;
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConversationStatus {
    /// The conversation is currently active
    #[serde(rename = "active")]
    Active,
    /// The conversation is temporarily paused
    #[serde(rename = "paused")]
    Paused,
    /// The conversation has been archived
    #[serde(rename = "archived")]
    Archived,
    /// The conversation has been marked for deletion
    #[serde(rename = "deleted")]
    Deleted,
}

/// Represents a conversation thread containing multiple messages.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conversation {
    /// Unique identifier for this conversation
    pub id: Uuid,
    /// Optional human-readable title
    pub title: Option<String>,
    /// Optional longer description
    pub description: Option<String>,
    /// When this conversation was created
    pub created_at: DateTime<Utc>,
    /// When this conversation was last modified
    pub updated_at: DateTime<Utc>,
    /// Additional metadata attached to this conversation
    pub metadata: HashMap<String, serde_json::Value>,
    /// Current status of the conversation
    pub status: ConversationStatus,
    /// Messages in this conversation
    pub messages: Vec<Message>,
}

impl Conversation {
    /// Creates a new active conversation with a generated ID.
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
            messages: Vec::new(),
        }
    }

    /// Sets the title of this conversation.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the description of this conversation.
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
    pub fn add_message(&mut self, message: Message) -> anyhow::Result<()> {
        if message.conversation_id != self.id {
            anyhow::bail!(
                "Message conversation_id {} does not match conversation id {}",
                message.conversation_id,
                self.id
            );
        }
        self.messages.push(message);
        self.touch();
        Ok(())
    }

    /// Returns a reference to the messages in this conversation.
    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
        let result = Message::tool(conv_id, "Result", String::new());
        assert!(result.is_err());

        // Should succeed with valid tool_call_id
        let result = Message::tool(conv_id, "Result", "call_123".to_string());
        assert!(result.is_ok());
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
