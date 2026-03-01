use std::collections::HashMap;
use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::enums::FinishReason;
use super::usage::Usage;
use crate::chat::Message;

/// A response from a chat completion request.
///
/// Contains the generated message, usage statistics, and metadata about
/// how and why generation completed.
///
/// # Examples
///
/// ```no_run
/// # use neuromance_common::{ChatResponse, Message, MessageRole};
/// # use neuromance_common::client::FinishReason;
/// # use uuid::Uuid;
/// # use chrono::Utc;
/// # let message = Message::new(Uuid::new_v4(), MessageRole::Assistant, "Hello!");
/// # let response = ChatResponse {
/// #     message: message.clone(),
/// #     model: "gpt-4".to_string(),
/// #     usage: None,
/// #     finish_reason: Some(FinishReason::Stop),
/// #     created_at: Utc::now(),
/// #     response_id: Some("resp_123".to_string()),
/// #     metadata: std::collections::HashMap::new(),
/// # };
/// // Check why generation stopped
/// if response.finish_reason == Some(FinishReason::Length) {
///     println!("Response was truncated - consider increasing max_tokens");
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The generated message from the model.
    pub message: Message,
    /// The identifier of the model that generated this response.
    pub model: String,
    /// Token usage statistics for this request.
    pub usage: Option<Usage>,
    /// Reason why generation stopped.
    pub finish_reason: Option<FinishReason>,
    /// Timestamp when this response was created.
    pub created_at: DateTime<Utc>,
    /// Unique identifier for this response from the provider.
    pub response_id: Option<String>,
    /// Additional metadata about this response.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A chunk from a streaming chat completion.
///
/// Represents an incremental update to a chat response. Multiple chunks
/// are combined to form the complete response. Typically received from
/// streaming APIs where the response is delivered incrementally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    /// The model identifier that generated this chunk.
    pub model: String,
    /// Incremental content added in this chunk.
    pub delta_content: Option<String>,
    /// Incremental reasoning content from thinking models (o1, o3, etc.).
    pub delta_reasoning_content: Option<String>,
    /// The role of the message (only present in first chunk).
    pub delta_role: Option<crate::chat::MessageRole>,
    /// Tool calls being built incrementally.
    pub delta_tool_calls: Option<Vec<crate::tools::ToolCall>>,
    /// Reason why generation stopped (only present in final chunk).
    pub finish_reason: Option<FinishReason>,
    /// Token usage statistics (only present in final chunk for some providers).
    pub usage: Option<Usage>,
    /// Unique identifier for this response stream.
    pub response_id: Option<String>,
    /// Timestamp when this chunk was created.
    pub created_at: DateTime<Utc>,
    /// Additional metadata about this chunk.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl fmt::Display for ChatResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string(self) {
            Ok(json) => write!(f, "{json}"),
            Err(_) => write!(f, "Error serializing ChatResponse to JSON"),
        }
    }
}
