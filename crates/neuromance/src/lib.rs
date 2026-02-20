//! # neuromance
//!
//! A Rust library for controlling and orchestrating LLM interactions.
//!
//! Neuromance provides high-level abstractions for building LLM-powered applications,
//! including conversation management, tool calling, and interaction orchestration.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use neuromance::{Conversation, Message, Core, CoreEvent, ToolApproval};
//! # use neuromance::OpenAIClient;
//! # let client: OpenAIClient = unimplemented!();
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
//! // Core uses event-driven architecture for streaming and monitoring
//! let core = Core::new(client)
//!     .with_streaming()
//!     .with_event_callback(|event| async move {
//!         match event {
//!             CoreEvent::Streaming(chunk) => print!("{}", chunk),
//!             CoreEvent::ToolResult { name, .. } => println!("Tool executed: {}", name),
//!             CoreEvent::Usage(usage) => println!("Tokens: {}", usage.total_tokens),
//!         }
//!     })
//!     .with_tool_approval_callback(|tool_call| {
//!         let tool_call = tool_call.clone();
//!         async move {
//!             println!("Approve {}?", tool_call.function.name);
//!             ToolApproval::Approved // Or prompt user
//!         }
//!     });
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
pub mod events;

pub use neuromance_client::*;
pub use neuromance_common::*;
pub use neuromance_tools::*;

pub use core::Core;
pub use events::{CoreEvent, EventCallback, ToolApprovalCallback, TurnCallback};
