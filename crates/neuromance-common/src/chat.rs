//! Conversation and message management for LLM interactions.
//!
//! This module provides the core abstractions for managing conversations and messages
//! when interacting with Large Language Models (LLMs). It handles message roles, tool
//! calling, conversation lifecycle, and metadata management.
//!
//! # Overview
//!
//! The module centers around two main types:
//!
//! - [`Message`]: Individual messages with role-based content (system, user, assistant, tool)
//! - [`Conversation`]: A thread of messages with lifecycle management and metadata
//!
//! # Message Roles
//!
//! Messages support four distinct roles via [`MessageRole`]:
//!
//! - **System**: Instructions and context for the LLM (e.g., "You are a helpful assistant")
//! - **User**: Input from the end user
//! - **Assistant**: Responses from the LLM, including tool call requests
//! - **Tool**: Results from executing tools/functions
//!
//! # Tool Calling
//!
//! The module supports the full tool calling workflow:
//!
//! 1. Assistant messages can include tool calls via [`Message::with_tool_calls`]
//! 2. Tool messages provide execution results via [`Message::tool`]
//! 3. Tool calls are validated to ensure they only appear on assistant messages
//!
//! # Examples
//!
//! ## Basic Conversation
//!
//! ```
//! use neuromance_common::chat::{Conversation, Message, MessageRole};
//!
//! // Create a new conversation
//! let mut conversation = Conversation::new()
//!     .with_title("Customer Support")
//!     .with_description("Helping user with account issues");
//!
//! // Add a system message
//! let system_msg = conversation.system_message(
//!     "You are a helpful customer support agent. Be polite and concise."
//! );
//! conversation.add_message(system_msg).unwrap();
//!
//! // Add a user message
//! let user_msg = conversation.user_message("I forgot my password");
//! conversation.add_message(user_msg).unwrap();
//!
//! // The conversation now has 2 messages
//! assert_eq!(conversation.get_messages().len(), 2);
//! ```
//!
//! ## Tool Calling Workflow
//!
//! ```
//! use neuromance_common::chat::{Conversation, Message};
//! use neuromance_common::tools::ToolCall;
//!
//! let mut conversation = Conversation::new();
//!
//! // User asks a question that requires a tool
//! let user_msg = conversation.user_message("What's the weather in Tokyo?");
//! conversation.add_message(user_msg).unwrap();
//!
//! // Assistant responds with a tool call
//! let tool_call = ToolCall::new(
//!     "get_weather",
//!     [r#"{"location": "Tokyo", "unit": "celsius"}"#]
//! );
//! let assistant_msg = conversation
//!     .assistant_message("Let me check the weather for you.")
//!     .with_tool_calls(vec![tool_call.clone()])
//!     .unwrap();
//! conversation.add_message(assistant_msg).unwrap();
//!
//! // Tool executes and returns result
//! let tool_result = conversation.tool_message(
//!     r#"{"temperature": 18, "condition": "Partly cloudy"}"#,
//!     tool_call.id.clone(),
//!     "get_weather".to_string()
//! ).unwrap();
//! conversation.add_message(tool_result).unwrap();
//!
//! // Assistant uses the tool result to respond
//! let final_msg = conversation.assistant_message(
//!     "It's currently 18°C and partly cloudy in Tokyo."
//! );
//! conversation.add_message(final_msg).unwrap();
//! ```
//!
//! ## Message Metadata
//!
//! ```
//! use neuromance_common::chat::{Conversation, Message};
//! use serde_json::json;
//!
//! let conversation = Conversation::new();
//!
//! // Add metadata to messages for tracking or debugging
//! let msg = conversation
//!     .user_message("Show me the latest sales report")
//!     .with_metadata("department", json!("sales"))
//!     .with_metadata("priority", json!("high"));
//!
//! // Or use typed metadata
//! let msg = conversation
//!     .user_message("Another message")
//!     .with_metadata_typed("user_id", 12345)
//!     .unwrap();
//! ```
//!
//! ## Conversation Lifecycle
//!
//! ```
//! use neuromance_common::chat::{Conversation, ConversationStatus};
//!
//! let mut conversation = Conversation::new();
//! assert_eq!(conversation.status, ConversationStatus::Active);
//!
//! // Pause a conversation
//! conversation.set_status(ConversationStatus::Paused);
//!
//! // Archive when complete
//! conversation.set_status(ConversationStatus::Archived);
//!
//! // Mark for deletion
//! conversation.set_status(ConversationStatus::Deleted);
//! ```
//!
//! # Type Safety
//!
//! The module enforces several invariants at compile time and runtime:
//!
//! - Tool calls can only be added to assistant messages
//! - Tool messages must have a `tool_call_id` and `name` (function name)
//! - Messages must belong to the conversation they're added to
//!
//! # Performance
//!
//! - Uses `SmallVec` for tool calls to avoid heap allocations in the common case (≤2 tool calls)
//! - Messages and conversations use `Uuid` for efficient unique identification
//! - Timestamps use `chrono` for UTC time tracking

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use typed_builder::TypedBuilder;
use uuid::Uuid;

use crate::tools::ToolCall;

/// Represents the role of a message sender in a conversation.
///
/// Message roles determine how the LLM interprets and processes each message.
/// The role affects both the semantic meaning and the allowed message structure.
///
/// # Serialization
///
/// Roles are serialized to lowercase strings matching the OpenAI API format:
/// - `System` → "system"
/// - `User` → "user"
/// - `Assistant` → "assistant"
/// - `Tool` → "tool"
///
/// # Examples
///
/// ```
/// use neuromance_common::chat::MessageRole;
///
/// let role = MessageRole::User;
/// let json = serde_json::to_string(&role).unwrap();
/// assert_eq!(json, "\"user\"");
/// ```
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MessageRole {
    /// System-level instructions or context.
    ///
    /// System messages set the behavior, personality, or context for the LLM.
    /// They typically appear at the start of a conversation and are not visible
    /// to end users.
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::{Message, MessageRole};
    /// use uuid::Uuid;
    ///
    /// let conv_id = Uuid::new_v4();
    /// let msg = Message::system(
    ///     conv_id,
    ///     "You are a helpful assistant specialized in Rust programming."
    /// );
    /// assert_eq!(msg.role, MessageRole::System);
    /// ```
    #[serde(rename = "system")]
    System,

    /// Messages from the end user.
    ///
    /// User messages contain input, questions, or instructions from the human
    /// interacting with the LLM.
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::{Message, MessageRole};
    /// use uuid::Uuid;
    ///
    /// let conv_id = Uuid::new_v4();
    /// let msg = Message::user(conv_id, "How do I handle errors in Rust?");
    /// assert_eq!(msg.role, MessageRole::User);
    /// ```
    #[serde(rename = "user")]
    User,

    /// Messages from the LLM assistant.
    ///
    /// Assistant messages contain responses from the LLM, which may include:
    /// - Text responses to user queries
    /// - Tool/function call requests
    /// - Reasoning or explanations
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::{Message, MessageRole};
    /// use neuromance_common::tools::ToolCall;
    /// use uuid::Uuid;
    ///
    /// let conv_id = Uuid::new_v4();
    ///
    /// // Assistant message with text
    /// let msg = Message::assistant(conv_id, "I can help you with that!");
    /// assert_eq!(msg.role, MessageRole::Assistant);
    ///
    /// // Assistant message with tool call
    /// let tool_call = ToolCall::new("search", [r#"{"query": "Rust errors"}"#]);
    /// let msg = Message::assistant(conv_id, "Let me search for that.")
    ///     .with_tool_calls(vec![tool_call])
    ///     .unwrap();
    /// assert!(!msg.tool_calls.is_empty());
    /// ```
    #[serde(rename = "assistant")]
    Assistant,

    /// Messages containing tool execution results.
    ///
    /// Tool messages provide the results of executing a function/tool that was
    /// requested by an assistant message. They must include:
    /// - A `tool_call_id` linking to the original tool call
    /// - A `name` field with the function name
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::{Message, MessageRole};
    /// use uuid::Uuid;
    ///
    /// let conv_id = Uuid::new_v4();
    /// let msg = Message::tool(
    ///     conv_id,
    ///     r#"{"result": "Success"}"#,
    ///     "call_abc123".to_string(),
    ///     "search".to_string()
    /// ).unwrap();
    /// assert_eq!(msg.role, MessageRole::Tool);
    /// assert_eq!(msg.tool_call_id, Some("call_abc123".to_string()));
    /// assert_eq!(msg.name, Some("search".to_string()));
    /// ```
    #[serde(rename = "tool")]
    Tool,
}

/// A single message in a conversation.
///
/// Messages are the fundamental building blocks of LLM interactions. Each message has a
/// role that determines how it's interpreted, content that carries the actual information,
/// and optional metadata for tracking and customization.
///
/// # Structure
///
/// - **Identification**: Each message has a unique `id` and belongs to a `conversation_id`
/// - **Role & Content**: The `role` determines message type, and `content` holds the text
/// - **Tool Support**: Assistant messages can include `tool_calls`, and tool messages
///   must have `tool_call_id` and `name` set
/// - **Metadata**: Arbitrary JSON metadata can be attached for application-specific needs
/// - **Timestamp**: Automatically set to creation time, useful for ordering and debugging
///
/// # Creating Messages
///
/// Use the convenience constructors for common message types:
///
/// ```
/// use neuromance_common::chat::Message;
/// use uuid::Uuid;
///
/// let conv_id = Uuid::new_v4();
///
/// // System message
/// let system = Message::system(conv_id, "You are a helpful assistant");
///
/// // User message
/// let user = Message::user(conv_id, "What is Rust?");
///
/// // Assistant message
/// let assistant = Message::assistant(conv_id, "Rust is a systems programming language.");
///
/// // Tool message (with validation)
/// let tool = Message::tool(
///     conv_id,
///     r#"{"temperature": 72}"#,
///     "call_123".to_string(),
///     "get_weather".to_string()
/// ).expect("Valid tool message");
/// ```
///
/// # Builder Pattern
///
/// For more control, use the TypedBuilder pattern:
///
/// ```
/// use neuromance_common::chat::{Message, MessageRole};
/// use uuid::Uuid;
///
/// let msg = Message::builder()
///     .conversation_id(Uuid::new_v4())
///     .role(MessageRole::User)
///     .content("Hello!".to_string())
///     .build();
/// ```
///
/// # Tool Calling
///
/// Only assistant messages can have tool calls:
///
/// ```
/// use neuromance_common::chat::Message;
/// use neuromance_common::tools::ToolCall;
/// use uuid::Uuid;
///
/// let conv_id = Uuid::new_v4();
/// let tool_call = ToolCall::new("calculate", [r#"{"expr": "2+2"}"#]);
///
/// let msg = Message::assistant(conv_id, "Let me calculate that.")
///     .with_tool_calls(vec![tool_call])
///     .expect("Assistant can have tool calls");
/// ```
///
/// # Validation
///
/// The type system and runtime checks enforce:
/// - Tool calls only on assistant messages
/// - Tool messages must have non-empty `tool_call_id` and `name`
/// - Messages added to conversations must have matching `conversation_id`
#[derive(Debug, Serialize, Deserialize, Clone, TypedBuilder)]
pub struct Message {
    /// Unique identifier for this message.
    ///
    /// Automatically generated using UUIDv4. Used for deduplication,
    /// message updates, and referencing specific messages.
    #[builder(default = Uuid::new_v4())]
    pub id: Uuid,

    /// ID of the conversation this message belongs to.
    ///
    /// Messages must have the same `conversation_id` as the conversation
    /// they're added to, enforced by [`Conversation::add_message`].
    pub conversation_id: Uuid,

    /// The role of the message sender.
    ///
    /// Determines how the LLM interprets this message. See [`MessageRole`]
    /// for details on each role type.
    pub role: MessageRole,

    /// The text content of the message.
    ///
    /// - For system/user/assistant messages: The actual message text
    /// - For tool messages: The JSON result from tool execution
    /// - Can be empty for assistant messages that only contain tool calls
    pub content: String,

    /// Additional metadata attached to this message.
    ///
    /// Use this for application-specific data like:
    /// - User IDs or session information
    /// - Message priority or categorization
    /// - Debugging or analytics data
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::Message;
    /// use uuid::Uuid;
    /// use serde_json::json;
    ///
    /// let msg = Message::user(Uuid::new_v4(), "Hello")
    ///     .with_metadata("user_id", json!(12345))
    ///     .with_metadata("channel", json!("web"));
    /// ```
    #[builder(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// When this message was created.
    ///
    /// Automatically set to the current UTC time at creation. Useful for:
    /// - Ordering messages chronologically
    /// - Debugging conversation flow
    /// - Analytics and usage tracking
    #[builder(default = Utc::now())]
    pub timestamp: DateTime<Utc>,

    /// Tool calls requested by this message (for assistant messages).
    ///
    /// When an assistant message includes tool calls, the LLM is requesting
    /// to execute one or more functions. Each tool call should be executed
    /// and the result provided via a tool message.
    ///
    /// Uses `SmallVec` to avoid heap allocation for ≤2 tool calls (the common case).
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::Message;
    /// use neuromance_common::tools::ToolCall;
    /// use uuid::Uuid;
    ///
    /// let tool_call = ToolCall::new("get_weather", [r#"{"city": "Tokyo"}"#]);
    /// let msg = Message::assistant(Uuid::new_v4(), "Checking weather...")
    ///     .with_tool_calls(vec![tool_call])
    ///     .unwrap();
    /// ```
    #[builder(default)]
    pub tool_calls: SmallVec<[ToolCall; 2]>,

    /// Reference to the tool call this message responds to (for tool messages).
    ///
    /// Tool messages must set this to link the result back to the original
    /// tool call request. The ID should match the `id` field from the
    /// corresponding [`ToolCall`].
    ///
    /// Required for tool role messages.
    #[builder(default)]
    pub tool_call_id: Option<String>,

    /// Name of the function for tool messages (required for tool role).
    ///
    /// For tool messages, this must match the function name from the original
    /// tool call request.
    ///
    /// Required for tool role messages.
    #[builder(default)]
    pub name: Option<String>,
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
    /// Returns an error if the tool_call_id is empty.
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
    pub fn with_tool_calls(mut self, tool_calls: impl Into<SmallVec<[ToolCall; 2]>>) -> anyhow::Result<Self> {
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
/// Tracks the current state of a conversation through its lifecycle.
/// Applications can use these states to implement features like:
/// - Listing only active conversations
/// - Archiving completed conversations
/// - Soft-delete with recovery
///
/// # Serialization
///
/// Statuses serialize to lowercase strings: "active", "paused", "archived", "deleted"
///
/// # Example
///
/// ```
/// use neuromance_common::chat::{Conversation, ConversationStatus};
///
/// let mut conv = Conversation::new();
/// assert_eq!(conv.status, ConversationStatus::Active);
///
/// // Pause for later
/// conv.set_status(ConversationStatus::Paused);
///
/// // Archive when done
/// conv.set_status(ConversationStatus::Archived);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum ConversationStatus {
    /// The conversation is currently active.
    ///
    /// Default state for new conversations. Indicates the conversation
    /// is ongoing and can accept new messages.
    #[serde(rename = "active")]
    Active,

    /// The conversation is temporarily paused.
    ///
    /// Useful for conversations that are on hold but may resume later.
    /// Applications might hide paused conversations from the main list.
    #[serde(rename = "paused")]
    Paused,

    /// The conversation has been archived.
    ///
    /// For completed conversations that should be preserved but aren't
    /// actively used. Typically shown separately from active conversations.
    #[serde(rename = "archived")]
    Archived,

    /// The conversation has been marked for deletion.
    ///
    /// Soft-delete state allowing recovery before permanent removal.
    /// Applications can implement a "trash" feature using this state.
    #[serde(rename = "deleted")]
    Deleted,
}

/// Represents a conversation thread containing multiple messages.
///
/// A conversation is a container for a sequence of messages exchanged with an LLM.
/// It manages the conversation lifecycle, ensures message consistency, and provides
/// convenience methods for creating properly-linked messages.
///
/// # Structure
///
/// - **Identity**: Unique `id` for referencing and persistence
/// - **Metadata**: Optional `title`, `description`, and custom metadata
/// - **Lifecycle**: `status` tracks the conversation state, `created_at` and `updated_at` track timing
/// - **Messages**: Ordered vector of messages in the conversation
///
/// # Creating Conversations
///
/// ```
/// use neuromance_common::chat::Conversation;
///
/// let mut conversation = Conversation::new()
///     .with_title("Technical Support")
///     .with_description("User having login issues");
///
/// // Conversation starts in Active status
/// assert_eq!(conversation.status, neuromance_common::chat::ConversationStatus::Active);
/// ```
///
/// # Adding Messages
///
/// Use the convenience methods to create messages that are automatically linked
/// to the conversation:
///
/// ```
/// use neuromance_common::chat::Conversation;
///
/// let mut conv = Conversation::new();
///
/// // Create and add messages
/// let system_msg = conv.system_message("You are a helpful assistant");
/// conv.add_message(system_msg).unwrap();
///
/// let user_msg = conv.user_message("Hello!");
/// conv.add_message(user_msg).unwrap();
///
/// assert_eq!(conv.get_messages().len(), 2);
/// ```
///
/// # Timestamp Management
///
/// The conversation tracks creation and modification times:
///
/// - `created_at`: Set once when the conversation is created
/// - `updated_at`: Automatically updated when messages are added or status changes
///
/// ```
/// use neuromance_common::chat::{Conversation, ConversationStatus};
///
/// let mut conv = Conversation::new();
/// let created = conv.created_at;
///
/// // Adding a message updates the timestamp
/// conv.add_message(conv.user_message("Hello")).unwrap();
/// assert!(conv.updated_at > created);
///
/// // Changing status also updates the timestamp
/// conv.set_status(ConversationStatus::Archived);
/// ```
///
/// # Validation
///
/// Messages added to a conversation must have a matching `conversation_id`.
/// This ensures data consistency:
///
/// ```
/// use neuromance_common::chat::{Conversation, Message};
/// use uuid::Uuid;
///
/// let mut conv1 = Conversation::new();
/// let conv2 = Conversation::new();
///
/// let msg_for_conv2 = conv2.user_message("Hello");
///
/// // This will fail - message belongs to conv2, not conv1
/// assert!(conv1.add_message(msg_for_conv2).is_err());
/// ```
///
/// # Use Cases
///
/// - **Chat Applications**: Maintain conversation threads with users
/// - **Multi-turn Interactions**: Build complex interactions requiring context
/// - **Tool Calling**: Orchestrate tool execution across multiple turns
/// - **Conversation History**: Persist and replay interactions
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conversation {
    /// Unique identifier for this conversation.
    ///
    /// Automatically generated using UUIDv4. Use this for:
    /// - Persisting conversations to a database
    /// - Referencing conversations in logs or analytics
    /// - Ensuring messages belong to the correct conversation
    pub id: Uuid,

    /// Optional human-readable title.
    ///
    /// Useful for displaying conversations in a UI or for organization.
    /// Can be set manually or auto-generated from the first user message.
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::Conversation;
    ///
    /// let conv = Conversation::new()
    ///     .with_title("Troubleshooting Database Connection");
    /// assert_eq!(conv.title, Some("Troubleshooting Database Connection".to_string()));
    /// ```
    pub title: Option<String>,

    /// Optional longer description.
    ///
    /// Provides additional context about the conversation's purpose or content.
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::Conversation;
    ///
    /// let conv = Conversation::new()
    ///     .with_description("User reported connection timeout errors on production DB");
    /// ```
    pub description: Option<String>,

    /// When this conversation was created.
    ///
    /// Automatically set to the current UTC time. Never changes after creation.
    pub created_at: DateTime<Utc>,

    /// When this conversation was last modified.
    ///
    /// Automatically updated when:
    /// - Messages are added via [`add_message`](Self::add_message)
    /// - Status changes via [`set_status`](Self::set_status)
    /// - Manually via [`touch`](Self::touch)
    pub updated_at: DateTime<Utc>,

    /// Additional metadata attached to this conversation.
    ///
    /// Store application-specific data like:
    /// - User or session IDs
    /// - Tags or categories
    /// - Custom configuration
    /// - Analytics data
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::chat::Conversation;
    /// use serde_json::json;
    ///
    /// let mut conv = Conversation::new();
    /// conv.metadata.insert("user_id".to_string(), json!(12345));
    /// conv.metadata.insert("priority".to_string(), json!("high"));
    /// ```
    pub metadata: HashMap<String, serde_json::Value>,

    /// Current status of the conversation.
    ///
    /// See [`ConversationStatus`] for available states.
    /// Defaults to [`ConversationStatus::Active`].
    pub status: ConversationStatus,

    /// Messages in this conversation.
    ///
    /// Ordered chronologically (though not strictly enforced).
    /// Use [`add_message`](Self::add_message) to add messages with validation.
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
        let result = Message::tool(conv_id, "Result", "call_123".to_string(), "test_func".to_string());
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
            let str_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            let tc2 = ToolCall::new(func_name.as_str(), str_refs);
            assert_eq!(tc2.function.name, func_name);
            assert_eq!(tc2.function.arguments, args);
        }

        #[test]
        fn message_metadata_operations(
            key in ".*",
            value_num in 0i64..1000000,
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
                0 => ConversationStatus::Active,
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
                r#"{{"id":"{}","conversation_id":"{}","role":"{}","content":"{}","metadata":{{}},"timestamp":"2024-01-01T00:00:00Z","tool_calls":[],"tool_call_id":null,"name":null}}"#,
                msg_id, conv_id, role_str, escaped_content
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
            let msg = Message::user(conv_id, content.clone());

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
            if let Some(t) = title.clone() {
                conv = conv.with_title(t);
            }
            if let Some(d) = description.clone() {
                conv = conv.with_description(d);
            }

            // Add random messages
            for i in 0..num_messages {
                let msg = conv.user_message(format!("Message {}", i));
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
