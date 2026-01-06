//! # neuromance
//!
//! A Rust library for controlling and orchestrating LLM interactions.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use futures::StreamExt;
//! use neuromance::{Config, Core, CoreEvent, Message, build_client};
//! use tokio_util::sync::CancellationToken;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = Config::from_model("openai:gpt-4o")?.with_api_key("sk-...");
//! let client = build_client(config)?;
//! let mut core = Core::new(client).with_streaming();
//!
//! let messages: Vec<Message> = vec![/* ... */];
//! let cancel = CancellationToken::new();
//! let mut stream = Box::pin(core.run(messages, cancel));
//! while let Some(event) = stream.next().await {
//!     match event? {
//!         CoreEvent::Delta(chunk)    => print!("{chunk}"),
//!         CoreEvent::Completed(_)    => break,
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! - **Conversation Management**: Track multi-turn conversations with metadata and status
//! - **Message Handling**: Support for system, user, assistant, and tool messages
//! - **Tool Calling**: Define and execute function calls from LLM responses
//! - **Serialization**: Full serde support for all types
//!
//! ## Optional Features
//!
//! - **`context`**: Token counting and context management using Candle tokenizers.
//!   Enable with `neuromance = { features = ["context"] }`

pub mod core;
pub mod error;
pub mod events;
pub mod stats;

// --- Orchestration ---
pub use crate::core::Core;
pub use crate::error::CoreError;
pub use crate::events::CoreEvent;
pub use crate::stats::RunStats;

// --- Clients ---
pub use neuromance_client::{
    AnthropicClient, ChatCompletionsClient, ClientError, LLMClient, ResponsesClient, build_client,
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
// --- Postgres conversation persistence ---
#[cfg(feature = "db")]
pub mod db {
    pub use neuromance_db::{ConversationSink, ConversationSummary, DbError, PgConversationStore};
}

// --- Context management (optional) ---
#[cfg(feature = "context")]
pub use neuromance_context as context;
