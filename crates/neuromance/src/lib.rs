//! # neuromance
//!
//! A Rust library for controlling and orchestrating LLM interactions.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use neuromance::{ChatCompletionsClient, Config, Core, CoreEvent};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = Config::new("openai", "gpt-4").with_api_key("sk-...");
//! let client = ChatCompletionsClient::new(config)?;
//!
//! let core = Core::new(client)
//!     .with_streaming()
//!     .with_event_callback(|event| async move {
//!         if let CoreEvent::Streaming(chunk) = event {
//!             print!("{chunk}");
//!         }
//!     });
//! # Ok(())
//! # }
//! ```

pub mod core;
pub mod error;
pub mod events;

// --- Orchestration ---
pub use crate::core::Core;
pub use crate::error::CoreError;
pub use crate::events::CoreEvent;

// --- Clients ---
pub use neuromance_client::{
    AnthropicClient, ChatCompletionsClient, ClientError, LLMClient, ResponsesClient,
};

// --- Config, request, response ---
pub use neuromance_common::client::{
    CacheMetrics, ChatChunk, ChatRequest, ChatResponse, Config, FinishReason, InputTokensDetails,
    OutputTokensDetails, Provider, ProxyConfig, ReasoningEffort, RetryConfig, ToolChoice, Usage,
};

// --- Chat primitives ---
pub use neuromance_common::chat::{
    Conversation, ConversationStatus, Message, MessageRole, ReasoningContent,
};

// --- Cross-provider features ---
pub use neuromance_common::features::{ReasoningLevel, ThinkingMode};

// --- Embedding API clients and types ---
pub mod embedding {
    pub use neuromance_client::{
        EmbeddingClient, EmbeddingConfig, EmbeddingInput, EmbeddingRequest, EmbeddingResponse,
        OpenAIEmbedding,
    };
}

// --- Tools ---
pub use neuromance_common::tools::{
    Function, FunctionCall, ObjectSchema, Parameters, Property, Tool, ToolApproval, ToolCall,
};
pub use neuromance_tools::{ToolExecutor, ToolImplementation, ToolRegistry};

// --- Model Context Protocol integration ---
pub mod mcp {
    pub use neuromance_tools::mcp::{
        McpConfig, McpManager, McpServerConfig, McpSettings, McpTransportConfig, ServerStatus,
    };
}

// --- Tokenizer proxy for tools that need scoped credentials ---
pub mod proxy {
    pub use neuromance_tools::proxy::{ProxyAwareClient, ToolProxyConfig};
}
