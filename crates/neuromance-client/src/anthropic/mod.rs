//! Anthropic Messages API types and client implementation.
//!
//! This module provides types for the Anthropic Messages API
//! and a client implementation for Claude models.

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatRequest, Config, InputTokensDetails, Usage};
use neuromance_common::tools::{FunctionCall, Tool, ToolCall};

pub mod client;
pub use client::AnthropicClient;

/// The required API version header value for Anthropic API.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Beta header value for interleaved thinking feature (Claude 4+ models).
///
/// Enables Claude to think between tool calls, allowing more sophisticated
/// reasoning after receiving tool results.
pub const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// Default base URL for the Anthropic API.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

// ============================================================================
// Request Types
// ============================================================================

/// Cache control configuration for prompt caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Type of cache control, always "ephemeral".
    #[serde(rename = "type")]
    pub cache_type: String,
    /// Optional TTL for cache (e.g., "1h" for 1-hour cache).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl CacheControl {
    /// Creates ephemeral cache control (5-minute default TTL).
    #[must_use]
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            ttl: None,
        }
    }

    /// Creates cache control with 1-hour TTL.
    #[must_use]
    pub fn one_hour() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            ttl: Some("1h".to_string()),
        }
    }
}

/// Image source for multimodal content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64 {
        /// Media type (e.g., "image/jpeg", "image/png").
        media_type: String,
        /// Base64-encoded data.
        data: String,
    },
    /// URL-referenced image.
    Url {
        /// URL of the image.
        url: String,
    },
}

/// Document source for multimodal content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Base64-encoded document data.
    Base64 {
        /// Media type (must be "application/pdf").
        media_type: String,
        /// Base64-encoded data.
        data: String,
    },
    /// URL-referenced document.
    Url {
        /// URL of the document.
        url: String,
    },
}

/// Content block types that can appear in request messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestContentBlock {
    /// Text content block.
    Text {
        /// The text content.
        text: String,
        /// Optional cache control.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Image content block.
    Image {
        /// Image source data.
        source: ImageSource,
        /// Optional cache control.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Document content block (PDF).
    Document {
        /// Document source data.
        source: DocumentSource,
        /// Optional document title.
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Optional cache control.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Tool result content block.
    ToolResult {
        /// ID of the tool use this is responding to.
        tool_use_id: String,
        /// Content of the tool result (can be string or array).
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        /// Whether the tool execution resulted in an error.
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    /// Tool use content block (for assistant messages with tool calls).
    ToolUse {
        /// Unique ID for this tool use.
        id: String,
        /// Name of the tool to call.
        name: String,
        /// Input arguments as JSON.
        input: serde_json::Value,
    },
    /// Thinking content block (for extended thinking responses).
    /// Must be included in subsequent requests when thinking was enabled.
    Thinking {
        /// The thinking content.
        thinking: String,
        /// Signature for the thinking block (required by Anthropic).
        signature: String,
    },
}

/// Content for tool results - can be a string or array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Simple string content.
    Text(String),
    /// Array of content blocks.
    Blocks(Vec<ToolResultContentBlock>),
}

/// Content blocks that can appear in tool results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentBlock {
    /// Text content.
    Text {
        /// The text content.
        text: String,
    },
    /// Image content.
    Image {
        /// Image source data.
        source: ImageSource,
    },
}

/// An Anthropic message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    /// Role of the message author.
    pub role: AnthropicRole,
    /// Content of the message - either a string or array of content blocks.
    pub content: MessageContent,
}

/// Message content - can be a simple string or array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content (shorthand for single text block).
    Text(String),
    /// Array of content blocks.
    Blocks(Vec<RequestContentBlock>),
}

/// Role of message author in Anthropic API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    /// User message.
    User,
    /// Assistant message.
    Assistant,
}

impl From<MessageRole> for AnthropicRole {
    fn from(role: MessageRole) -> Self {
        match role {
            MessageRole::Assistant => Self::Assistant,
            // All other roles map to User in Anthropic's API
            _ => Self::User,
        }
    }
}

/// System prompt configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Simple text system prompt.
    Text(String),
    /// Array of system content blocks with cache control.
    Blocks(Vec<SystemContentBlock>),
}

/// Content blocks for system prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemContentBlock {
    /// Text content block.
    Text {
        /// The text content.
        text: String,
        /// Optional cache control.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Anthropic tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    /// Tool name (must match regex: ^[a-zA-Z0-9_-]{1,64}$).
    pub name: String,
    /// Description of what the tool does.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
    /// Optional cache control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl From<&Tool> for AnthropicTool {
    fn from(tool: &Tool) -> Self {
        Self {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            input_schema: tool.function.parameters.clone(),
            cache_control: None,
        }
    }
}

/// Tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    /// Let the model decide whether to use tools.
    Auto,
    /// Force the model to use at least one tool.
    Any {
        /// Disable parallel tool use.
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Force the model to use a specific tool.
    Tool {
        /// Name of the tool to use.
        name: String,
        /// Disable parallel tool use.
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Disable tool use entirely.
    None,
}

impl From<&neuromance_common::client::ToolChoice> for AnthropicToolChoice {
    fn from(choice: &neuromance_common::client::ToolChoice) -> Self {
        match choice {
            neuromance_common::client::ToolChoice::None => Self::None,
            neuromance_common::client::ToolChoice::Required => Self::Any {
                disable_parallel_tool_use: None,
            },
            neuromance_common::client::ToolChoice::Function { name } => Self::Tool {
                name: name.clone(),
                disable_parallel_tool_use: None,
            },
            // Auto and any future variants default to Auto
            _ => Self::Auto,
        }
    }
}

/// Extended thinking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Type, must be "enabled".
    #[serde(rename = "type")]
    pub config_type: String,
    /// Budget tokens for thinking (min 1024, must be < `max_tokens`).
    pub budget_tokens: u32,
}

impl ThinkingConfig {
    /// Creates a new thinking configuration with the given budget.
    #[must_use]
    pub fn new(budget_tokens: u32) -> Self {
        Self {
            config_type: "enabled".to_string(),
            budget_tokens,
        }
    }
}

/// Metadata for tracking requests.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RequestMetadata {
    /// End-user identifier for tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Request for the Anthropic Messages API.
#[derive(Debug, Clone, Serialize, TypedBuilder)]
pub struct CreateMessageRequest {
    /// Model identifier (e.g., "claude-sonnet-4-5-20250929").
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<AnthropicMessage>,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// System prompt.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Sampling temperature (0.0-1.0).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold (0.0-1.0).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences (max 8191 entries).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Enable streaming.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Request metadata.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RequestMetadata>,
    /// Available tools.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    /// Tool selection strategy.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    /// Extended thinking configuration.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

// ============================================================================
// Response Types
// ============================================================================

/// Citation reference in text responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Type of citation.
    #[serde(rename = "type")]
    pub citation_type: String,
    /// Citation text or reference.
    pub text: Option<String>,
}

/// Content block types that can appear in responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    /// Text content block.
    Text {
        /// The text content.
        text: String,
        /// Optional citations.
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Citation>>,
    },
    /// Tool use request from the model.
    ToolUse {
        /// Unique ID for this tool use.
        id: String,
        /// Name of the tool to call.
        name: String,
        /// Input arguments as JSON.
        input: serde_json::Value,
    },
    /// Thinking content block (extended thinking).
    Thinking {
        /// The thinking content.
        thinking: String,
        /// Cryptographic signature.
        signature: String,
    },
    /// Redacted thinking content block.
    RedactedThinking {
        /// Redacted data.
        data: String,
    },
}

/// Reason why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end of turn.
    EndTurn,
    /// Maximum tokens reached.
    MaxTokens,
    /// Stop sequence encountered.
    StopSequence,
    /// Model wants to use a tool.
    ToolUse,
    /// Model refused to respond.
    Refusal,
}

impl From<StopReason> for neuromance_common::client::FinishReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::MaxTokens => Self::Length,
            StopReason::ToolUse => Self::ToolCalls,
            StopReason::Refusal => Self::ContentFilter,
            // EndTurn and StopSequence both map to Stop
            StopReason::EndTurn | StopReason::StopSequence => Self::Stop,
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    /// Input tokens consumed.
    pub input_tokens: u32,
    /// Output tokens generated.
    pub output_tokens: u32,
    /// Tokens used to create cache entries.
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens read from cache.
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

impl From<AnthropicUsage> for Usage {
    fn from(usage: AnthropicUsage) -> Self {
        Self {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage.input_tokens + usage.output_tokens,
            cost: None,
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: usage.cache_read_input_tokens,
                cache_creation_tokens: usage.cache_creation_input_tokens,
            }),
            output_tokens_details: None,
        }
    }
}

/// Response from the Messages API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageResponse {
    /// Unique message ID (e.g., "`msg_01XFDUDYJgAACzvnptvVoYEL`").
    pub id: String,
    /// Object type, always "message".
    #[serde(rename = "type")]
    pub response_type: String,
    /// Role, always "assistant".
    pub role: String,
    /// Content blocks in the response.
    pub content: Vec<ResponseContentBlock>,
    /// Model that generated the response.
    pub model: String,
    /// Why generation stopped.
    pub stop_reason: Option<StopReason>,
    /// Stop sequence that caused generation to stop.
    pub stop_sequence: Option<String>,
    /// Token usage statistics.
    pub usage: AnthropicUsage,
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Content block start variants for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStart {
    /// Text block starting.
    Text {
        /// Initial text (usually empty).
        text: String,
    },
    /// Thinking block starting.
    Thinking {
        /// Initial thinking content.
        thinking: String,
    },
    /// Tool use block starting.
    ToolUse {
        /// Tool use ID.
        id: String,
        /// Tool name.
        name: String,
        /// Initial input (usually empty object, may be omitted).
        #[serde(default)]
        input: serde_json::Value,
    },
}

/// Delta variants for streaming content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    /// Text delta.
    TextDelta {
        /// Text to append.
        text: String,
    },
    /// JSON input delta for tool calls.
    InputJsonDelta {
        /// Partial JSON to accumulate.
        partial_json: String,
    },
    /// Thinking delta.
    ThinkingDelta {
        /// Thinking content to append.
        thinking: String,
    },
    /// Signature delta for thinking blocks.
    SignatureDelta {
        /// Signature to append.
        signature: String,
    },
}

/// Message delta data for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaData {
    /// Stop reason if generation ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// Stop sequence that caused generation to stop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// Usage delta for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageDelta {
    /// Output tokens generated so far.
    pub output_tokens: u32,
}

/// Stream event types from SSE.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Message has started.
    MessageStart {
        /// Initial message response.
        message: MessageResponse,
    },
    /// A content block has started.
    ContentBlockStart {
        /// Index of this content block.
        index: u32,
        /// The starting content block.
        content_block: ContentBlockStart,
    },
    /// Delta update for a content block.
    ContentBlockDelta {
        /// Index of the content block being updated.
        index: u32,
        /// The delta update.
        delta: Delta,
    },
    /// A content block has finished.
    ContentBlockStop {
        /// Index of the content block that finished.
        index: u32,
    },
    /// Message-level delta (stop reason, etc.).
    MessageDelta {
        /// Delta data.
        delta: MessageDeltaData,
        /// Usage update.
        usage: UsageDelta,
    },
    /// Message has finished.
    MessageStop,
    /// Keepalive ping.
    Ping,
    /// Error event.
    Error {
        /// Error details.
        error: ApiError,
    },
}

// ============================================================================
// Error Types
// ============================================================================

/// API error detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Error type (e.g., "`invalid_request_error`").
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message.
    pub message: String,
}

/// Full error response from the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Response type, always "error".
    #[serde(rename = "type")]
    pub response_type: String,
    /// Error details.
    pub error: ApiError,
    /// Optional request ID for debugging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

// ============================================================================
// Conversion Helpers
// ============================================================================

/// Convert our common Message to Anthropic format.
impl From<&Message> for AnthropicMessage {
    fn from(message: &Message) -> Self {
        let role = AnthropicRole::from(message.role);

        // Build content blocks based on message type
        let content = match message.role {
            MessageRole::Tool => {
                // Tool results go in a tool_result block
                let tool_use_id = message.tool_call_id.clone().unwrap_or_default();
                MessageContent::Blocks(vec![RequestContentBlock::ToolResult {
                    tool_use_id,
                    content: Some(ToolResultContent::Text(message.content.clone())),
                    is_error: None,
                }])
            }
            MessageRole::Assistant if !message.tool_calls.is_empty() => {
                // Assistant message with tool calls
                let mut blocks = Vec::new();

                // Add thinking block if present (must come first for Anthropic)
                if let (Some(thinking), Some(signature)) =
                    (&message.reasoning_content, &message.reasoning_signature)
                {
                    blocks.push(RequestContentBlock::Thinking {
                        thinking: thinking.clone(),
                        signature: signature.clone(),
                    });
                }

                // Add text content if present
                if !message.content.is_empty() {
                    blocks.push(RequestContentBlock::Text {
                        text: message.content.clone(),
                        cache_control: None,
                    });
                }

                // Add tool_use blocks for each tool call
                for tool_call in &message.tool_calls {
                    // Parse the arguments JSON string back to a Value
                    let input = tool_call
                        .function
                        .arguments
                        .first()
                        .and_then(|arg| serde_json::from_str(arg).ok())
                        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

                    blocks.push(RequestContentBlock::ToolUse {
                        id: tool_call.id.clone(),
                        name: tool_call.function.name.clone(),
                        input,
                    });
                }

                MessageContent::Blocks(blocks)
            }
            MessageRole::Assistant
                if message.reasoning_content.is_some() && message.reasoning_signature.is_some() =>
            {
                // Assistant message with thinking but no tool calls
                let mut blocks = Vec::new();

                if let (Some(thinking), Some(signature)) =
                    (&message.reasoning_content, &message.reasoning_signature)
                {
                    blocks.push(RequestContentBlock::Thinking {
                        thinking: thinking.clone(),
                        signature: signature.clone(),
                    });
                }

                if !message.content.is_empty() {
                    blocks.push(RequestContentBlock::Text {
                        text: message.content.clone(),
                        cache_control: None,
                    });
                }

                MessageContent::Blocks(blocks)
            }
            _ => {
                // Regular text content
                if message.content.is_empty() {
                    MessageContent::Text(String::new())
                } else {
                    MessageContent::Text(message.content.clone())
                }
            }
        };

        Self { role, content }
    }
}

/// Converts tools to Anthropic format with cache control on the last tool.
///
/// Anthropic's prompt caching caches everything up to and including the
/// content block marked with `cache_control`. By marking the last tool,
/// we ensure all tool definitions are cached together.
fn convert_tools_with_caching(tools: &[Tool]) -> Vec<AnthropicTool> {
    let mut anthropic_tools: Vec<AnthropicTool> = tools.iter().map(AnthropicTool::from).collect();

    // Apply cache control to the last tool for prompt caching
    if let Some(last) = anthropic_tools.last_mut() {
        last.cache_control = Some(CacheControl::ephemeral());
    }

    anthropic_tools
}

/// Conversion from `ChatRequest` to Anthropic request format.
impl From<(&ChatRequest, &Config)> for CreateMessageRequest {
    fn from((request, config): (&ChatRequest, &Config)) -> Self {
        // Collect all system messages into blocks
        let mut system_blocks: Vec<SystemContentBlock> = Vec::new();
        let mut anthropic_messages: Vec<AnthropicMessage> = Vec::new();

        for message in request.messages.iter() {
            match message.role {
                MessageRole::System => {
                    system_blocks.push(SystemContentBlock::Text {
                        text: message.content.clone(),
                        cache_control: None,
                    });
                }
                _ => {
                    anthropic_messages.push(AnthropicMessage::from(message));
                }
            }
        }

        // Apply cache control to the last system block for prompt caching
        if let Some(SystemContentBlock::Text { cache_control, .. }) = system_blocks.last_mut() {
            *cache_control = Some(CacheControl::ephemeral());
        }

        // Convert collected system blocks to SystemPrompt
        let system = if system_blocks.is_empty() {
            None
        } else {
            Some(SystemPrompt::Blocks(system_blocks))
        };

        // Convert tools if present, with cache control on the last tool
        let tools: Option<Vec<AnthropicTool>> = request
            .tools
            .as_ref()
            .map(|t| convert_tools_with_caching(t));

        // Convert tool choice if present
        let tool_choice: Option<AnthropicToolChoice> =
            request.tool_choice.as_ref().map(AnthropicToolChoice::from);

        // Create thinking config from ThinkingMode
        let thinking = request.thinking.budget().map(ThinkingConfig::new);

        // Determine max_tokens - Anthropic requires this, default to 4096
        // When thinking is enabled, max_tokens must be > budget_tokens
        let max_tokens = request.thinking.budget().map_or_else(
            || request.max_tokens.unwrap_or(4096),
            |budget| {
                request
                    .max_tokens
                    .unwrap_or(budget + 8192)
                    .max(budget + 1024)
            },
        );

        // When thinking is enabled, temperature and top_p must not be set
        let (temperature, top_p) = if request.thinking.is_enabled() {
            (None, None)
        } else {
            (request.temperature, request.top_p)
        };

        Self::builder()
            .model(
                request
                    .model
                    .clone()
                    .unwrap_or_else(|| config.model.clone()),
            )
            .messages(anthropic_messages)
            .max_tokens(max_tokens)
            .system(system)
            .temperature(temperature)
            .top_p(top_p)
            .stop_sequences(request.stop.clone())
            .stream(Some(request.stream))
            .tools(tools)
            .tool_choice(tool_choice)
            .thinking(thinking)
            .build()
    }
}

/// Accumulator for building tool calls from streaming `input_json_delta` events.
#[derive(Debug, Clone)]
pub struct StreamingToolCall {
    /// Tool use ID.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Accumulated JSON string.
    pub accumulated_json: String,
}

impl StreamingToolCall {
    /// Creates a new streaming tool call accumulator.
    #[must_use]
    pub const fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            accumulated_json: String::new(),
        }
    }

    /// Appends a delta to the accumulated JSON.
    pub fn append_delta(&mut self, partial_json: &str) {
        self.accumulated_json.push_str(partial_json);
    }

    /// Finalizes the tool call, parsing the accumulated JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated JSON cannot be parsed.
    pub fn finalize(self) -> Result<ToolCall, serde_json::Error> {
        // Parse the accumulated JSON
        let input: serde_json::Value = if self.accumulated_json.is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&self.accumulated_json)?
        };

        Ok(ToolCall {
            id: self.id,
            call_type: "function".to_string(),
            function: FunctionCall {
                name: self.name,
                arguments: vec![input.to_string()],
            },
        })
    }
}
