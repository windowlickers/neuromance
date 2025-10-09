//! OpenAI API types and client implementation.
//!
//! This module provides types for the OpenAI chat completions API
//! and a client implementation that works with any OpenAI-compatible endpoint.

use std::borrow::Cow;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use typed_builder::TypedBuilder;

use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatRequest, Config, Usage};
use neuromance_common::tools::{FunctionCall, Tool, ToolCall};

pub mod client;
pub use client::{OpenAIClient, convert_chunk_to_chat_chunk};

/// A single choice from a chat completion response.
///
/// When `n > 1`, the API returns multiple choices. Each choice has an index
/// and its own message and finish reason.
#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    /// The index of this choice in the response array.
    pub index: u32,
    /// The generated message for this choice.
    pub message: OpenAIMessage,
    /// Why generation stopped for this choice.
    ///
    /// Common values: "stop", "length", "tool_calls", "content_filter"
    pub finish_reason: Option<String>,
}

/// OpenAI-compatible message format.
///
/// Wrapper type for serializing/deserializing messages to the OpenAI API format.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
pub struct OpenAIMessage {
    /// The role of the message author (user, assistant, system, or tool).
    pub role: MessageRole,
    /// The text content of the message (optional for tool calls).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Optional name of the message author.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls requested by the assistant (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<SmallVec<[OpenAIToolCall; 2]>>,
    /// ID of the tool call this message is responding to (for tool messages).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl From<&Message> for OpenAIMessage {
    fn from(message: &Message) -> Self {
        let tool_calls = if !message.tool_calls.is_empty() {
            Some(
                message
                    .tool_calls
                    .iter()
                    .map(OpenAIToolCall::from)
                    .collect(),
            )
        } else {
            None
        };

        // Only include content if it's non-empty
        let content = if !message.content.is_empty() {
            Some(message.content.clone())
        } else {
            None
        };

        OpenAIMessage::builder()
            .role(message.role)
            .content(content)
            .name(message.name.clone())
            .tool_calls(tool_calls)
            .tool_call_id(message.tool_call_id.clone())
            .build()
    }
}

/// OpenAI-compatible tool call format.
///
/// Represents a request from the model to call a function/tool.
///
/// Uses `Cow<'static, str>` to avoid allocations for static strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    /// Unique identifier for this tool call.
    pub id: Cow<'static, str>,
    /// Type of the tool call, typically "function".
    #[serde(rename = "type", default = "default_tool_call_type")]
    pub r#type: Cow<'static, str>,
    /// The function to call with its arguments.
    pub function: OpenAIFunction,
}

/// Conversion from a generic `ToolCall` to OpenAI-specific format.
impl From<&ToolCall> for OpenAIToolCall {
    fn from(tool_call: &ToolCall) -> Self {
        Self {
            id: Cow::Owned(tool_call.id.clone()),
            r#type: Cow::Owned(tool_call.call_type.clone()),
            function: OpenAIFunction::from(&tool_call.function),
        }
    }
}

fn default_tool_call_type() -> Cow<'static, str> {
    Cow::Borrowed("function")
}

/// OpenAI-compatible function call format.
///
/// Contains the function name and JSON-serialized arguments.
///
/// Uses `Cow<'static, str>` to avoid allocations for static strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    /// The name of the function to call.
    pub name: Cow<'static, str>,
    /// The arguments as a JSON-serialized string.
    pub arguments: Cow<'static, str>,
}

impl From<&FunctionCall> for OpenAIFunction {
    fn from(function_call: &FunctionCall) -> Self {
        Self {
            name: Cow::Owned(function_call.name.clone()),
            // OpenAI expects a single JSON string, take the first argument (which should be the JSON string)
            arguments: function_call
                .arguments
                .first()
                .map(|s| Cow::Owned(s.clone()))
                .unwrap_or(Cow::Borrowed("")),
        }
    }
}

/// Request for a chat completion.
///
/// Contains all parameters for the OpenAI chat completions API.
///
/// # Examples
///
/// ```no_run
/// use neuromance_client::openai::ChatCompletionRequest;
/// # use neuromance_client::openai::OpenAIMessage;
/// # use neuromance_common::MessageRole;
///
/// let request = ChatCompletionRequest::builder()
///     .model("gpt-4".to_string())
///     .messages(vec![])
///     .temperature(Some(0.7))
///     .max_tokens(Some(1000))
///     .build();
/// ```
#[derive(Debug, Clone, Serialize, TypedBuilder)]
pub struct ChatCompletionRequest {
    /// The model identifier to use (e.g., "gpt-4", "gpt-3.5-turbo").
    pub model: String,
    /// The conversation messages in OpenAI format.
    pub messages: Vec<OpenAIMessage>,
    /// Maximum tokens to generate (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature 0.0 to 2.0 (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold 0.0 to 1.0 (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Number of completions to generate (optional, default 1).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Stop sequences (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Presence penalty -2.0 to 2.0 (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty -2.0 to 2.0 (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Token bias map for modifying likelihoods (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// End-user identifier for tracking (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Whether to stream the response (optional, default false).
    #[builder(default = Some(false))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Tools available for function calling (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool selection strategy (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Whether to enable thinking mode (vendor-specific, optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    /// Stream options for controlling streaming behavior (optional).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<serde_json::Value>,
}

/// Conversion from a generic `ChatRequest` to OpenAI-specific format.
///
/// Maps common request parameters to the OpenAI API format, using the provided
/// configuration for defaults like the model name.
impl From<(&ChatRequest, &Config)> for ChatCompletionRequest {
    fn from((request, config): (&ChatRequest, &Config)) -> Self {
        let openai_messages: Vec<OpenAIMessage> =
            request.messages.iter().map(OpenAIMessage::from).collect();

        let tools: Option<Vec<Tool>> = request.tools.clone();

        ChatCompletionRequest::builder()
            .model(
                request
                    .model
                    .clone()
                    .unwrap_or_else(|| config.model.clone()),
            )
            .messages(openai_messages)
            .max_tokens(request.max_tokens)
            .temperature(request.temperature)
            .top_p(request.top_p)
            .stop(request.stop.clone())
            .presence_penalty(request.presence_penalty)
            .frequency_penalty(request.frequency_penalty)
            .user(request.user.clone())
            .stream(Some(request.stream))
            .tools(tools)
            .tool_choice(request.tool_choice.as_ref().map(|tc| tc.clone().into()))
            .enable_thinking(request.enable_thinking)
            .build()
    }
}

impl ChatCompletionRequest {
    /// Add logit bias to the request.
    ///
    /// Modifies the likelihood of specified tokens appearing in the completion.
    pub fn with_logit_bias(mut self, logit_bias: HashMap<String, f32>) -> Self {
        self.logit_bias = Some(logit_bias);
        self
    }
}

/// Response from a chat completion request.
///
/// Contains the model's response along with usage statistics and metadata.
///
/// # Examples
///
/// ```no_run
/// # use neuromance_client::openai::ChatCompletionResponse;
/// # let response: ChatCompletionResponse = serde_json::from_str("{}").unwrap();
/// // Access the first choice's message
/// if let Some(choice) = response.choices.first() {
///     println!("Response: {:?}", choice.message);
/// }
///
/// // Check token usage
/// if let Some(usage) = &response.usage {
///     println!("Tokens used: {}", usage.total_tokens);
/// }
/// ```
#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique identifier for this completion.
    pub id: String,
    /// Object type, typically "chat.completion".
    pub object: String,
    /// Unix timestamp of when the completion was created.
    pub created: u64,
    /// The model that generated this completion.
    pub model: String,
    /// Array of generated completions (length equals `n` parameter).
    pub choices: Vec<ChatChoice>,
    /// Token usage statistics (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A single choice from a streaming chat completion chunk.
///
/// Each chunk contains a delta with incremental updates to the message.
#[derive(Debug, Deserialize)]
pub struct ChatStreamChoice {
    /// The index of this choice in the response array.
    pub index: u32,
    /// Incremental message delta for this chunk.
    pub delta: OpenAIMessageDelta,
    /// Why generation stopped (only present in final chunk).
    pub finish_reason: Option<String>,
}

/// Delta representing incremental changes to a message.
///
/// Used in streaming responses to communicate partial updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessageDelta {
    /// The role (only present in first chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,
    /// Incremental content added in this chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Incremental tool calls (for function calling).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<SmallVec<[OpenAIToolCallDelta; 2]>>,
}

/// Delta representing incremental changes to a tool call.
///
/// Tool calls may be built across multiple streaming chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCallDelta {
    /// Index of this tool call in the array.
    pub index: u32,
    /// Unique identifier (only present in first chunk for this tool call).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Type of tool call (only present in first chunk).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// Incremental function call data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIFunctionDelta>,
}

/// Delta representing incremental changes to a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionDelta {
    /// Function name (only present in first chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Incremental arguments added in this chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Chunk from a streaming chat completion.
///
/// Each chunk represents an incremental update to the response.
///
/// # Examples
///
/// ```no_run
/// # use neuromance_client::openai::ChatCompletionChunk;
/// # let chunk: ChatCompletionChunk = serde_json::from_str("{}").unwrap();
/// // Process streaming delta
/// if let Some(choice) = chunk.choices.first() {
///     if let Some(content) = &choice.delta.content {
///         print!("{}", content);
///     }
/// }
/// ```
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique identifier for this completion stream.
    pub id: String,
    /// Object type, typically "chat.completion.chunk".
    pub object: String,
    /// Unix timestamp of when this chunk was created.
    pub created: u64,
    /// The model generating this stream.
    pub model: String,
    /// Array of delta choices.
    pub choices: Vec<ChatStreamChoice>,
    /// Token usage (only present in final chunk for some providers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
