//! # neuromance-common
//!
//! Common types and data structures for LLM conversation and tool management.
//!
//! This crate provides the foundational types for building LLM-powered applications:
//! - Conversation and message management
//! - Tool/function calling support
//! - Serializable data structures for persistence and API communication
//!
//! ## Example
//!
//! ```
//! use neuromance_common::{Conversation, Message, ToolCall, Tool, Function};
//! use uuid::Uuid;
//!
//! // Create a new conversation
//! let conv = Conversation::new()
//!     .with_title("Time Assistant")
//!     .with_description("Helping users get the current time");
//!
//! // Add a user message
//! let msg = Message::user(conv.id, "What time is it?");
//!
//! // Define a tool
//! let tool = Tool {
//!     r#type: "function".to_string(),
//!     function: Function {
//!         name: "get_current_time".to_string(),
//!         description: "Get the current date and time in UTC format. Takes no parameters.".to_string(),
//!         parameters: serde_json::json!({
//!             "type": "object",
//!             "properties": {},
//!             "required": [],
//!         }),
//!     },
//! };
//!
//! // Create a tool call
//! let tool_call = ToolCall::new("get_current_time", Vec::<String>::new());
//! ```

pub mod chat;
pub mod tools;

pub use chat::{Conversation, ConversationStatus, Message, MessageRole};
pub use tools::{Function, FunctionCall, Parameters, Property, Tool, ToolApproval, ToolCall};
