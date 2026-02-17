//! Internal event and domain types for daemon communication.
//!
//! Defines the `DaemonResponse` event enum streamed through mpsc channels,
//! plus domain value objects (`ConversationSummary`, `ModelProfile`, `ErrorCode`)
//! used throughout the daemon. The wire protocol lives in `neuromance-proto`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::chat::Message;
use crate::client::Usage;
use crate::tools::ToolCall;

/// Machine-readable error codes for programmatic error handling.
///
/// Allows clients to distinguish error types without parsing message strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorCode {
    /// Conversation not found
    ConversationNotFound,
    /// Model not found in configuration
    ModelNotFound,
    /// Bookmark not found
    BookmarkNotFound,
    /// Bookmark already exists
    BookmarkExists,
    /// No active conversation set
    NoActiveConversation,
    /// Invalid conversation ID format
    InvalidConversationId,
    /// LLM client or orchestration error
    LlmError,
    /// Configuration error
    ConfigError,
    /// Storage/persistence error
    StorageError,
    /// Invalid or malformed request
    InvalidRequest,
    /// Internal server error
    Internal,
}

/// Internal events sent from the conversation manager through mpsc channels.
///
/// The gRPC server bridge converts these into proto `ChatEvent` messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DaemonResponse {
    /// A chunk of streaming content from the assistant.
    StreamChunk {
        /// The conversation this chunk belongs to
        conversation_id: String,
        /// The incremental content
        content: String,
    },

    /// Result of a tool execution.
    ToolResult {
        /// The conversation this result belongs to
        conversation_id: String,
        /// The tool that was executed
        tool_name: String,
        /// The result output
        result: String,
        /// Whether execution succeeded
        success: bool,
    },

    /// Request for user approval of a tool call.
    ToolApprovalRequest {
        /// The conversation requiring approval
        conversation_id: String,
        /// The tool call awaiting approval
        tool_call: ToolCall,
    },

    /// Token usage statistics for a completed message.
    Usage {
        /// The conversation this usage belongs to
        conversation_id: String,
        /// Usage statistics
        usage: Usage,
    },

    /// Indicates a message exchange is complete.
    MessageCompleted {
        /// The conversation this message belongs to
        conversation_id: String,
        /// The completed assistant message
        message: Box<Message>,
    },

    /// An error occurred.
    Error {
        /// Machine-readable error code
        code: ErrorCode,
        /// Human-readable error message
        message: String,
    },
}

/// Summary information about a conversation.
///
/// Used in conversation listings and creation responses. Provides enough
/// information to identify and select conversations without loading full
/// message history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    /// Full conversation UUID
    pub id: String,

    /// Short ID (first 7 characters, git-style)
    pub short_id: String,

    /// Optional human-readable title
    pub title: Option<String>,

    /// Number of messages in the conversation
    pub message_count: usize,

    /// When the conversation was created
    pub created_at: DateTime<Utc>,

    /// When the conversation was last updated
    pub updated_at: DateTime<Utc>,

    /// Bookmark names for this conversation
    pub bookmarks: Vec<String>,

    /// Model nickname being used
    pub model: String,
}

impl ConversationSummary {
    /// Creates a conversation summary from a conversation and model.
    #[must_use]
    pub fn from_conversation(
        conversation: &crate::chat::Conversation,
        model: impl Into<String>,
        bookmarks: Vec<String>,
    ) -> Self {
        let id = conversation.id.to_string();
        let short_id = id.chars().take(7).collect();

        Self {
            id,
            short_id,
            title: conversation.title.clone(),
            message_count: conversation.messages.len(),
            created_at: conversation.created_at,
            updated_at: conversation.updated_at,
            bookmarks,
            model: model.into(),
        }
    }
}

/// Configuration for an LLM model.
///
/// Represents a named model profile from the daemon's configuration file.
/// Users can reference models by their nickname instead of full provider/model names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    /// User-friendly nickname for this model (e.g., "sonnet", "gpt4")
    pub nickname: String,

    /// Provider name (e.g., "anthropic", "openai")
    pub provider: String,

    /// Full model identifier (e.g., "claude-sonnet-4-5-20250929")
    pub model: String,

    /// Environment variable name for the API key (e.g., `ANTHROPIC_API_KEY`)
    pub api_key_env: String,

    /// Optional custom base URL (e.g., <https://openrouter.ai/api/v1>)
    #[serde(default)]
    pub base_url: Option<String>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::panic)]

    use super::*;

    #[test]
    fn test_daemon_response_serialization() {
        let response = DaemonResponse::StreamChunk {
            conversation_id: "abc123".to_string(),
            content: "Hello".to_string(),
        };

        let json = serde_json::to_string(&response).expect("Failed to serialize");
        let deserialized: DaemonResponse =
            serde_json::from_str(&json).expect("Failed to deserialize");

        match deserialized {
            DaemonResponse::StreamChunk {
                conversation_id,
                content,
            } => {
                assert_eq!(conversation_id, "abc123");
                assert_eq!(content, "Hello");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_conversation_summary_from_conversation() {
        let conv = crate::chat::Conversation::new().with_title("Test Conversation");

        let summary = ConversationSummary::from_conversation(
            &conv,
            "sonnet",
            vec!["bookmark1".to_string(), "bookmark2".to_string()],
        );

        assert_eq!(summary.id, conv.id.to_string());
        assert_eq!(summary.short_id.len(), 7);
        assert_eq!(summary.title, Some("Test Conversation".to_string()));
        assert_eq!(summary.message_count, 0);
        assert_eq!(summary.model, "sonnet");
        assert_eq!(summary.bookmarks.len(), 2);
    }

    #[test]
    fn test_model_profile_serialization() {
        let profile = ModelProfile {
            nickname: "sonnet".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            base_url: None,
        };

        let json = serde_json::to_value(&profile).expect("Failed to serialize");
        assert_eq!(json["nickname"], "sonnet");
        assert_eq!(json["provider"], "anthropic");
        assert_eq!(json["model"], "claude-sonnet-4-5-20250929");
        assert_eq!(json["api_key_env"], "ANTHROPIC_API_KEY");

        let deserialized: ModelProfile =
            serde_json::from_value(json).expect("Failed to deserialize");
        assert_eq!(profile.nickname, deserialized.nickname);
        assert_eq!(profile.provider, deserialized.provider);
    }

    #[test]
    fn test_tool_approval_request_response() {
        let tool_call = ToolCall::new("test_tool", "arg1");
        let conv_id = uuid::Uuid::new_v4().to_string();

        let response = DaemonResponse::ToolApprovalRequest {
            conversation_id: conv_id.clone(),
            tool_call,
        };

        let json = serde_json::to_string(&response).expect("Failed to serialize");
        let deserialized: DaemonResponse =
            serde_json::from_str(&json).expect("Failed to deserialize");

        match deserialized {
            DaemonResponse::ToolApprovalRequest {
                conversation_id,
                tool_call: tc,
            } => {
                assert_eq!(conversation_id, conv_id);
                assert_eq!(tc.function.name, "test_tool");
            }
            _ => panic!("Wrong variant"),
        }
    }
}
