//! `OpenAI` Responses API types and client implementation.
//!
//! This module provides types for the `OpenAI` Responses API (openresponses.org spec)
//! and a client implementation with support for stateless mode, streaming, and tool calling.

use std::collections::HashMap;

use log::warn;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use typed_builder::TypedBuilder;

use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatRequest, Config, Usage};
use neuromance_common::features::ReasoningLevel;
use neuromance_common::tools::{FunctionCall, Tool, ToolCall};

pub mod client;
pub use client::ResponsesClient;

// ============================================================================
// Role Types
// ============================================================================

/// Role of message author in the Responses API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponsesRole {
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// System message.
    System,
}

impl From<MessageRole> for ResponsesRole {
    fn from(role: MessageRole) -> Self {
        match role {
            MessageRole::Assistant => Self::Assistant,
            MessageRole::System => Self::System,
            MessageRole::User => Self::User,
            // Tool messages should be converted to FunctionCallOutput input items,
            // not mapped to a role. This fallback exists for safety but indicates
            // a bug in the calling code if reached.
            other => {
                log::warn!(
                    "Unexpected role {other:?} converted to ResponsesRole::User; \
                     Tool messages should use InputItem::FunctionCallOutput instead"
                );
                Self::User
            }
        }
    }
}

// ============================================================================
// Content Types
// ============================================================================

/// Content for a message - can be simple text or multimodal parts.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Array of content parts for multimodal content.
    Parts(Vec<ContentPart>),
}

/// A part of multimodal content.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content part.
    InputText {
        /// The text content.
        text: String,
    },
    /// Image content part.
    InputImage {
        /// Image data or URL.
        #[serde(flatten)]
        image: ImageData,
    },
    /// File content part.
    InputFile {
        /// File data.
        #[serde(flatten)]
        file: FileData,
    },
}

/// Image data for multimodal content.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ImageData {
    /// URL-referenced image.
    Url {
        /// URL of the image.
        url: String,
        /// Optional detail level.
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// Base64-encoded image.
    Base64 {
        /// Base64-encoded data with data URL prefix.
        data: String,
    },
}

/// File data for multimodal content.
#[derive(Debug, Clone, Serialize)]
pub struct FileData {
    /// File ID or URL.
    pub file_id: Option<String>,
    /// File URL.
    pub url: Option<String>,
}

// ============================================================================
// Input Item Types
// ============================================================================

/// Input items that can be sent to the Responses API.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    /// A message in the conversation.
    Message {
        /// Role of the message author.
        role: ResponsesRole,
        /// Content of the message.
        content: MessageContent,
    },
    /// A function call made by the assistant.
    FunctionCall {
        /// Unique ID for this function call.
        call_id: String,
        /// Name of the function to call.
        name: String,
        /// Arguments as a JSON string.
        arguments: String,
    },
    /// Output from a function call.
    FunctionCallOutput {
        /// ID of the function call this is responding to.
        call_id: String,
        /// Output from the function.
        output: String,
    },
    /// Reasoning content from the model.
    Reasoning {
        /// The reasoning content.
        content: Vec<ReasoningContentBlock>,
    },
}

/// A block of reasoning content.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningContentBlock {
    /// Summary text of reasoning.
    SummaryText {
        /// The summary text.
        text: String,
    },
}

// ============================================================================
// Tool Types
// ============================================================================

/// A tool definition for the Responses API.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTool {
    /// A function tool.
    Function {
        /// The function definition.
        #[serde(flatten)]
        function: FunctionDefinition,
    },
}

/// Definition of a function tool.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionDefinition {
    /// Name of the function.
    pub name: String,
    /// Description of what the function does.
    pub description: String,
    /// JSON Schema for the function parameters.
    pub parameters: serde_json::Value,
    /// Whether the function can run without approval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl From<&Tool> for ResponsesTool {
    fn from(tool: &Tool) -> Self {
        Self::Function {
            function: FunctionDefinition {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
                strict: None,
            },
        }
    }
}

/// Tool choice configuration for the Responses API.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    /// String-based choice (auto, none, required).
    Mode(String),
    /// Force a specific function.
    Function {
        /// Type, always "function".
        #[serde(rename = "type")]
        choice_type: String,
        /// Name of the function to call.
        name: String,
    },
}

impl From<&neuromance_common::client::ToolChoice> for ResponsesToolChoice {
    fn from(choice: &neuromance_common::client::ToolChoice) -> Self {
        match choice {
            neuromance_common::client::ToolChoice::None => Self::Mode("none".to_string()),
            neuromance_common::client::ToolChoice::Required => Self::Mode("required".to_string()),
            neuromance_common::client::ToolChoice::Function { name } => Self::Function {
                choice_type: "function".to_string(),
                name: name.clone(),
            },
            // Auto and any future variants default to auto
            _ => Self::Mode("auto".to_string()),
        }
    }
}

// ============================================================================
// Reasoning Configuration
// ============================================================================

/// Reasoning configuration for models that support it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Effort level for reasoning.
    pub effort: ReasoningEffort,
    /// Whether to generate a summary of reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

/// Reasoning effort level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
}

/// Convert a `ReasoningLevel` to an optional `ReasoningEffort`.
const fn reasoning_level_to_effort(level: ReasoningLevel) -> Option<ReasoningEffort> {
    match level {
        ReasoningLevel::Minimal | ReasoningLevel::Low => Some(ReasoningEffort::Low),
        ReasoningLevel::Medium => Some(ReasoningEffort::Medium),
        ReasoningLevel::High | ReasoningLevel::Maximum => Some(ReasoningEffort::High),
        // Default and any future variants return None
        _ => None,
    }
}

/// Reasoning summary configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    /// Generate a concise summary.
    Concise,
    /// Generate a detailed summary.
    Detailed,
    /// Do not generate a summary.
    None,
}

// ============================================================================
// Request Types
// ============================================================================

/// Request for the Responses API.
#[derive(Debug, Clone, Serialize, TypedBuilder)]
pub struct ResponsesRequest {
    /// Model identifier.
    pub model: String,
    /// Input items for the conversation.
    pub input: Vec<InputItem>,
    /// Whether to store the response for later retrieval.
    #[builder(default = Some(false))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Maximum output tokens to generate.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Instructions for the model (system prompt).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Sampling temperature (0.0-2.0).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold (0.0-1.0).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Tools available for the model.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    /// Tool selection strategy.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    /// Whether to truncate input if too long.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    /// Previous response ID for stateful conversations.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Reasoning configuration.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// Whether to stream the response.
    #[builder(default = Some(false))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Request metadata.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl From<(&ChatRequest, &Config)> for ResponsesRequest {
    fn from((request, config): (&ChatRequest, &Config)) -> Self {
        let mut input_items: Vec<InputItem> = Vec::new();
        let mut instructions: Option<String> = None;

        // Convert messages to input items
        for message in request.messages.iter() {
            match message.role {
                MessageRole::System => {
                    // System messages become instructions
                    if let Some(ref mut inst) = instructions {
                        inst.push_str("\n\n");
                        inst.push_str(&message.content);
                    } else {
                        instructions = Some(message.content.clone());
                    }
                }
                MessageRole::Tool => {
                    // Tool messages become FunctionCallOutput
                    if let Some(call_id) = &message.tool_call_id {
                        input_items.push(InputItem::FunctionCallOutput {
                            call_id: call_id.clone(),
                            output: message.content.clone(),
                        });
                    } else {
                        warn!(
                            "Tool message without tool_call_id was skipped; this likely indicates a bug in the calling code"
                        );
                    }
                }
                MessageRole::Assistant if !message.tool_calls.is_empty() => {
                    // Assistant message with tool calls
                    // First add the message content if present
                    if !message.content.is_empty() {
                        input_items.push(InputItem::Message {
                            role: ResponsesRole::Assistant,
                            content: MessageContent::Text(message.content.clone()),
                        });
                    }
                    // Then add each tool call as a separate FunctionCall item
                    for tool_call in &message.tool_calls {
                        input_items.push(InputItem::FunctionCall {
                            call_id: tool_call.id.clone(),
                            name: tool_call.function.name.clone(),
                            arguments: tool_call.function.arguments_json().to_owned(),
                        });
                    }
                }
                role => {
                    // Regular message
                    input_items.push(InputItem::Message {
                        role: ResponsesRole::from(role),
                        content: MessageContent::Text(message.content.clone()),
                    });
                }
            }
        }

        // Convert tools
        let tools: Option<Vec<ResponsesTool>> = request
            .tools
            .as_ref()
            .map(|t| t.iter().map(ResponsesTool::from).collect());

        // Convert tool choice
        let tool_choice: Option<ResponsesToolChoice> =
            request.tool_choice.as_ref().map(ResponsesToolChoice::from);

        // Convert reasoning level
        // Reasoning summary can be configured via metadata["reasoning_summary"] with values:
        // "concise" (default), "detailed", or "none"
        let reasoning: Option<ReasoningConfig> = reasoning_level_to_effort(request.reasoning_level)
            .map(|effort| {
                let summary = request
                    .metadata
                    .get("reasoning_summary")
                    .and_then(|v| serde_json::from_value::<ReasoningSummary>(v.clone()).ok())
                    .unwrap_or(ReasoningSummary::Concise);
                ReasoningConfig {
                    effort,
                    summary: Some(summary),
                }
            });

        // Get previous_response_id from metadata if present
        let previous_response_id = request
            .metadata
            .get("previous_response_id")
            .and_then(|v| v.as_str())
            .map(String::from);

        let store = request
            .metadata
            .get("store")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        Self::builder()
            .model(
                request
                    .model
                    .clone()
                    .unwrap_or_else(|| config.model.clone()),
            )
            .input(input_items)
            .store(Some(store))
            .max_output_tokens(request.max_tokens.or(request.max_completion_tokens))
            .instructions(instructions)
            .temperature(request.temperature)
            .top_p(request.top_p)
            .tools(tools)
            .tool_choice(tool_choice)
            .previous_response_id(previous_response_id)
            .reasoning(reasoning)
            .stream(Some(request.stream))
            .build()
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Response from the Responses API.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesResponse {
    /// Unique response ID.
    pub id: String,
    /// Object type, typically "response".
    pub object: String,
    /// Unix timestamp of creation.
    pub created_at: i64,
    /// Model that generated the response.
    pub model: String,
    /// Status of the response.
    pub status: ResponseStatus,
    /// Output items from the model.
    pub output: Vec<OutputItem>,
    /// Error details, present when status is "failed".
    #[serde(default)]
    pub error: Option<ResponseError>,
    /// Details about why the response is incomplete.
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    /// Token usage statistics.
    pub usage: Option<ResponsesUsage>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Output items from the Responses API.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    /// A message from the assistant.
    Message {
        /// Role, always "assistant".
        role: String,
        /// Content of the message (may be empty during streaming).
        #[serde(default)]
        content: Vec<OutputContentBlock>,
    },
    /// A function call from the assistant.
    ///
    /// Note: During streaming, `call_id` and `arguments` may be empty initially
    /// and filled in by `FunctionCallArgumentsDone` event.
    FunctionCall {
        /// Unique ID for this function call (may be empty during streaming).
        #[serde(default)]
        call_id: String,
        /// Name of the function to call.
        name: String,
        /// Arguments as a JSON string (may be empty during streaming).
        #[serde(default)]
        arguments: String,
    },
    /// Reasoning content from the model.
    Reasoning {
        /// The reasoning content (may be empty during streaming).
        #[serde(default)]
        content: Vec<ReasoningOutputBlock>,
    },
}

/// Content blocks in output messages.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentBlock {
    /// Text output.
    OutputText {
        /// The text content.
        text: String,
    },
    /// Refusal message.
    Refusal {
        /// The refusal message.
        refusal: String,
    },
}

/// Reasoning output blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningOutputBlock {
    /// Summary text of reasoning.
    SummaryText {
        /// The summary text.
        text: String,
    },
}

/// Status of a response from the Responses API.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Response completed successfully.
    Completed,
    /// Response generation failed.
    Failed,
    /// Response is being generated.
    InProgress,
    /// Response was cancelled.
    Cancelled,
    /// Response is queued for processing.
    Queued,
    /// Response is incomplete (e.g., hit max tokens or content filter).
    Incomplete,
}

/// Details about why a response is incomplete.
#[derive(Debug, Clone, Deserialize)]
pub struct IncompleteDetails {
    /// The reason the response is incomplete.
    pub reason: IncompleteReason,
}

/// Reason why a response is incomplete.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteReason {
    /// Maximum output tokens reached.
    MaxOutputTokens,
    /// Content was filtered.
    ContentFilter,
}

/// Error returned when the model fails to generate a response.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseError {
    /// The error code.
    pub code: String,
    /// A human-readable description of the error.
    pub message: String,
}

/// Derive a `FinishReason` from the response status and incomplete details.
#[must_use]
pub const fn finish_reason_from_status(
    status: &ResponseStatus,
    incomplete_details: Option<&IncompleteDetails>,
    has_tool_calls: bool,
) -> Option<neuromance_common::client::FinishReason> {
    match status {
        ResponseStatus::Completed => {
            if has_tool_calls {
                Some(neuromance_common::client::FinishReason::ToolCalls)
            } else {
                Some(neuromance_common::client::FinishReason::Stop)
            }
        }
        ResponseStatus::Incomplete => match incomplete_details {
            Some(details) => match details.reason {
                IncompleteReason::MaxOutputTokens => {
                    Some(neuromance_common::client::FinishReason::Length)
                }
                IncompleteReason::ContentFilter => {
                    Some(neuromance_common::client::FinishReason::ContentFilter)
                }
            },
            None => Some(neuromance_common::client::FinishReason::Length),
        },
        // Failed, cancelled, in_progress, queued don't map to a finish reason
        _ => None,
    }
}

/// Token usage statistics from the Responses API.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesUsage {
    /// Input tokens consumed.
    pub input_tokens: u32,
    /// Output tokens generated.
    pub output_tokens: u32,
    /// Total tokens used.
    #[serde(default)]
    pub total_tokens: u32,
    /// Detailed input token breakdown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    /// Detailed output token breakdown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// Detailed input token breakdown.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct InputTokensDetails {
    /// Cached tokens.
    #[serde(default)]
    pub cached_tokens: u32,
}

/// Detailed output token breakdown.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens.
    #[serde(default)]
    pub reasoning_tokens: u32,
}

impl From<ResponsesUsage> for Usage {
    fn from(usage: ResponsesUsage) -> Self {
        let total = if usage.total_tokens > 0 {
            usage.total_tokens
        } else {
            usage.input_tokens + usage.output_tokens
        };

        Self {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: total,
            cost: None,
            input_tokens_details: usage.input_tokens_details.map(|d| {
                neuromance_common::client::InputTokensDetails {
                    cached_tokens: d.cached_tokens,
                    cache_creation_tokens: 0,
                }
            }),
            output_tokens_details: usage.output_tokens_details.map(|d| {
                neuromance_common::client::OutputTokensDetails {
                    reasoning_tokens: d.reasoning_tokens,
                }
            }),
        }
    }
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Streaming events from the Responses API.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Response creation started.
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// The initial response object.
        response: PartialResponse,
    },
    /// Response is in progress.
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        /// The response object.
        response: PartialResponse,
    },
    /// Response completed.
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        /// The final response object.
        response: ResponsesResponse,
    },
    /// Response incomplete (e.g., hit max tokens or content filter).
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        /// The final response object (includes usage).
        response: ResponsesResponse,
    },
    /// Response failed.
    #[serde(rename = "response.failed")]
    ResponseFailed {
        /// The response object (contains the error in `response.error`).
        response: PartialResponse,
    },
    /// Output item added.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        /// Index of the output item.
        output_index: u32,
        /// The output item.
        item: OutputItem,
    },
    /// Output item completed.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        /// Index of the output item.
        output_index: u32,
        /// The completed output item.
        item: OutputItem,
    },
    /// Content part added.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        /// Index of the output item.
        output_index: u32,
        /// Index of the content part.
        content_index: u32,
        /// The content part.
        part: OutputContentBlock,
    },
    /// Content part completed.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        /// Index of the output item.
        output_index: u32,
        /// Index of the content part.
        content_index: u32,
        /// The completed content part.
        part: OutputContentBlock,
    },
    /// Text delta.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        /// Index of the output item.
        output_index: u32,
        /// Index of the content part.
        content_index: u32,
        /// The text delta.
        delta: String,
    },
    /// Text completed.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        /// Index of the output item.
        output_index: u32,
        /// Index of the content part.
        content_index: u32,
        /// The final text.
        text: String,
    },
    /// Function call arguments delta.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        /// Index of the output item.
        output_index: u32,
        /// The call ID (may be present in some API versions).
        #[serde(default)]
        call_id: String,
        /// The arguments delta.
        delta: String,
    },
    /// Function call arguments completed.
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        /// Index of the output item.
        output_index: u32,
        /// The item ID (used as `call_id` for tool calls).
        item_id: String,
        /// The final arguments.
        arguments: String,
        /// Sequence number (optional, for ordering).
        #[serde(default)]
        sequence_number: u32,
    },
    /// Reasoning summary text delta.
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        /// Index of the output item.
        output_index: u32,
        /// Index within the reasoning content.
        summary_index: u32,
        /// The text delta.
        delta: String,
    },
    /// Reasoning summary text completed.
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        /// Index of the output item.
        output_index: u32,
        /// Index within the reasoning content.
        summary_index: u32,
        /// The final text.
        text: String,
    },
    /// Error event.
    Error {
        /// Error details.
        error: ApiError,
    },
    /// Unknown event type - catch-all for forward compatibility.
    #[serde(other)]
    Unknown,
}

/// Partial response during streaming.
#[derive(Debug, Clone, Deserialize)]
pub struct PartialResponse {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created_at: i64,
    /// Model identifier.
    pub model: String,
    /// Status of the response.
    pub status: ResponseStatus,
    /// Output items so far.
    #[serde(default)]
    pub output: Vec<OutputItem>,
    /// Error details, present when status is "failed".
    #[serde(default)]
    pub error: Option<ResponseError>,
    /// Details about why the response is incomplete.
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    /// Usage if available.
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
}

/// API error details.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiError {
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error code.
    #[serde(default)]
    pub code: Option<String>,
    /// Error message.
    pub message: String,
}

// ============================================================================
// Streaming Accumulator
// ============================================================================

/// Accumulator for building function calls from streaming events.
#[derive(Debug, Clone)]
pub struct StreamingFunctionCall {
    /// Function call ID.
    pub call_id: String,
    /// Function name.
    pub name: String,
    /// Accumulated arguments JSON.
    pub accumulated_arguments: String,
}

impl StreamingFunctionCall {
    /// Creates a new streaming function call accumulator.
    #[must_use]
    pub const fn new(call_id: String, name: String) -> Self {
        Self {
            call_id,
            name,
            accumulated_arguments: String::new(),
        }
    }

    /// Appends a delta to the accumulated arguments.
    pub fn append_delta(&mut self, delta: &str) {
        self.accumulated_arguments.push_str(delta);
    }

    /// Finalizes the function call into a `ToolCall`.
    #[must_use]
    pub fn finalize(self) -> ToolCall {
        let arguments = if self.accumulated_arguments.is_empty() {
            "{}".to_string()
        } else {
            self.accumulated_arguments
        };

        ToolCall {
            id: self.call_id,
            call_type: "function".to_string(),
            function: FunctionCall {
                name: self.name,
                arguments,
            },
        }
    }
}

// ============================================================================
// Conversion Helpers
// ============================================================================

/// Convert a Responses API response to our internal Message format.
pub fn convert_response_to_message(
    response: &ResponsesResponse,
    conversation_id: uuid::Uuid,
) -> Message {
    let mut content = String::new();
    let mut tool_calls: SmallVec<[ToolCall; 2]> = SmallVec::new();
    let mut reasoning_content: Option<String> = None;

    for item in &response.output {
        match item {
            OutputItem::Message {
                content: blocks, ..
            } => {
                for block in blocks {
                    match block {
                        OutputContentBlock::OutputText { text } => {
                            if !content.is_empty() {
                                content.push('\n');
                            }
                            content.push_str(text);
                        }
                        OutputContentBlock::Refusal { refusal } => {
                            if !content.is_empty() {
                                content.push('\n');
                            }
                            content.push_str(refusal);
                        }
                    }
                }
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                tool_calls.push(ToolCall {
                    id: call_id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                });
            }
            OutputItem::Reasoning {
                content: reasoning_blocks,
            } => {
                for block in reasoning_blocks {
                    match block {
                        ReasoningOutputBlock::SummaryText { text } => {
                            if let Some(ref mut rc) = reasoning_content {
                                rc.push('\n');
                                rc.push_str(text);
                            } else {
                                reasoning_content = Some(text.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    Message {
        id: uuid::Uuid::new_v4(),
        conversation_id,
        role: MessageRole::Assistant,
        content,
        tool_calls,
        tool_call_id: None,
        name: None,
        timestamp: chrono::Utc::now(),
        metadata: HashMap::new(),
        reasoning: reasoning_content.map(neuromance_common::ReasoningContent::new),
    }
}
