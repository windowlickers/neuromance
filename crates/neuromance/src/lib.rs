//! # neuromance
//!
//! A Rust library for controlling and orchestrating LLM interactions.
//!
//! Neuromance provides high-level abstractions for building LLM-powered applications,
//! including conversation management, tool calling, and interaction orchestration.
//!
//! ## Quick Start
//!
//! ```rust
//! use neuromance::{Conversation, Message, ToolCall};
//!
//! // Create a conversation
//! let mut conversation = Conversation::new().with_title("My Chat");
//!
//! // Add messages
//! let user_msg = Message::user(conversation.id, "Hello!");
//! let assistant_msg = Message::assistant(conversation.id, "Hi there!");
//!
//! conversation.add_message(user_msg).expect("Failed to add message");
//! conversation.add_message(assistant_msg).expect("Failed to add message");
//!
//! // Create a tool call
//! let tool_call = ToolCall::new("get_current_time", Vec::<String>::new());
//! ```
//!
//! ## Features
//!
//! - **Conversation Management**: Track multi-turn conversations with metadata and status
//! - **Message Handling**: Support for system, user, assistant, and tool messages
//! - **Tool Calling**: Define and execute function calls from LLM responses
//! - **Serialization**: Full serde support for all types

pub mod core;
pub mod error;

pub use neuromance_client::*;
pub use neuromance_common::*;
pub use neuromance_tools::*;

pub use core::Core;
