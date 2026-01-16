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
//! // Define a tool using the builder pattern
//! let tool = Tool::builder()
//!     .function(Function {
//!         name: "get_current_time".to_string(),
//!         description: "Get the current date and time in UTC format. Takes no parameters.".to_string(),
//!         parameters: serde_json::json!({
//!             "type": "object",
//!             "properties": {},
//!             "required": [],
//!         }),
//!     })
//!     .build();
//!
//! // Or using struct initialization (the r#type field defaults to "function")
//! let tool_alt = Tool {
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
//! // Create a tool call using the into() conversion
//! let tool_call = ToolCall::new("get_current_time", Vec::<String>::new());
//! ```

/// Chat conversation and message types.
///
/// Provides types for managing conversations, messages, and message roles.
pub mod chat;
/// Client configuration and request/response types.
///
/// Contains types for configuring LLM clients and making chat completion requests.
pub mod client;
/// Feature abstractions for cross-provider capabilities.
///
/// Provides types like `ThinkingMode` and `ReasoningLevel` that abstract
/// provider-specific features into a common interface.
pub mod features;
/// Tool calling and function execution types.
///
/// Provides types for defining and executing functions/tools that LLMs can call.
pub mod tools;

pub mod agents;

pub use agents::{AgentContext, AgentMemory, AgentMessage, AgentResponse, AgentState, AgentStats};
pub use chat::{Conversation, ConversationStatus, Message, MessageRole};
pub use client::{
    ChatRequest, ChatResponse, Config, FinishReason, ReasoningEffort, RetryConfig, ToolChoice,
    Usage,
};
pub use features::{ReasoningLevel, ThinkingMode};
pub use tools::{Function, FunctionCall, Parameters, Property, Tool, ToolApproval, ToolCall};
