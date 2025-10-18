use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::chat::Message;
use crate::tools::Tool;

/// Controls how the model selects which tool to call, if any.
///
/// This enum provides fine-grained control over tool selection behavior,
/// from fully automatic to forcing a specific function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolChoice {
    /// Let the model automatically decide whether to call a tool and which one.
    ///
    /// This is the default behavior for most use cases.
    #[serde(rename = "auto")]
    Auto,
    /// Disable tool calling for this request.
    ///
    /// The model will not call any tools and will respond directly.
    #[serde(rename = "none")]
    None,
    /// Require the model to call at least one tool.
    ///
    /// The model must call a tool rather than responding directly.
    #[serde(rename = "required")]
    Required,
    /// Force the model to call a specific function by name.
    ///
    /// # Example
    ///
    /// ```
    /// use neuromance_common::ToolChoice;
    ///
    /// let choice = ToolChoice::Function {
    ///     name: "get_weather".to_string(),
    /// };
    /// ```
    Function {
        /// The name of the function to call
        name: String,
    },
}

impl fmt::Display for ToolChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::None => write!(f, "none"),
            Self::Required => write!(f, "required"),
            Self::Function { name } => write!(f, "{name}"),
        }
    }
}

impl From<ToolChoice> for serde_json::Value {
    fn from(tool_choice: ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::Auto => Self::String("auto".to_string()),
            ToolChoice::None => Self::String("none".to_string()),
            ToolChoice::Required => Self::String("required".to_string()),
            ToolChoice::Function { name } => {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": name
                    }
                })
            }
        }
    }
}

/// Indicates why the model stopped generating tokens.
///
/// This enum provides information about whether generation completed naturally,
/// was truncated, or was interrupted for another reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Copy)]
#[non_exhaustive]
pub enum FinishReason {
    /// Generation completed naturally at a stop sequence or end of response.
    ///
    /// This is the most common finish reason for successful completions.
    #[serde(rename = "stop")]
    Stop,
    /// Generation was truncated because the maximum token limit was reached.
    ///
    /// Consider increasing `max_tokens` if the response appears incomplete.
    #[serde(rename = "length")]
    Length,
    /// Generation stopped because the model requested tool calls.
    ///
    /// The response contains tool calls that should be executed.
    #[serde(rename = "tool_calls")]
    ToolCalls,
    /// Generation was stopped by the content filter.
    ///
    /// The response may have been blocked due to safety policies.
    #[serde(rename = "content_filter")]
    ContentFilter,
    /// Generation failed due to a model error.
    ///
    /// This typically indicates an internal error in the model.
    #[serde(rename = "model_error")]
    ModelError,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::Length => write!(f, "length"),
            Self::ToolCalls => write!(f, "tool_calls"),
            Self::ContentFilter => write!(f, "content_filter"),
            Self::ModelError => write!(f, "model_error"),
        }
    }
}

impl FromStr for FinishReason {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "stop" => Ok(Self::Stop),
            "length" => Ok(Self::Length),
            "tool_calls" => Ok(Self::ToolCalls),
            "content_filter" => Ok(Self::ContentFilter),
            "model_error" => Ok(Self::ModelError),
            _ => anyhow::bail!("Unknown finish reason: {s}"),
        }
    }
}

/// Configuration for exponential backoff retry behavior.
///
/// This struct controls how failed requests are retried with increasing delays
/// between attempts. Supports optional jitter to avoid thundering herd problems.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use neuromance_common::client::RetryConfig;
///
/// // Conservative retry policy
/// let config = RetryConfig {
///     max_retries: 5,
///     initial_delay: Duration::from_millis(500),
///     max_delay: Duration::from_secs(60),
///     backoff_multiplier: 2.0,
///     jitter: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts before failing.
    pub max_retries: usize,
    /// Initial delay before the first retry attempt.
    pub initial_delay: Duration,
    /// Maximum delay between retry attempts (caps exponential growth).
    pub max_delay: Duration,
    /// Multiplier for exponential backoff (typically 2.0 for doubling).
    pub backoff_multiplier: f64,
    /// Whether to add random jitter to retry delays to prevent thundering herd.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Token usage statistics for a completion request.
///
/// Tracks the number of tokens consumed by the prompt and completion,
/// along with optional cost information and detailed breakdowns.
///
/// # Note
///
/// Different providers may count tokens differently. The `total_tokens`
/// should always equal `prompt_tokens + completion_tokens`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the input prompt.
    #[serde(alias = "input_tokens")]
    pub prompt_tokens: u32,
    /// Number of tokens generated in the completion.
    #[serde(alias = "output_tokens")]
    pub completion_tokens: u32,
    /// Total tokens used (prompt + completion).
    pub total_tokens: u32,
    /// Estimated cost in USD for this request (if available).
    pub cost: Option<f64>,
    /// Detailed breakdown of input token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    /// Detailed breakdown of output token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// Detailed breakdown of input token usage.
///
/// Provides additional information about how input tokens were processed,
/// including cache utilization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Number of tokens served from cache rather than processed fresh.
    ///
    /// Cached tokens are typically cheaper and faster to process.
    pub cached_tokens: u32,
}

/// Detailed breakdown of output token usage.
///
/// Provides additional information about token usage in the model's response,
/// including reasoning tokens for models that support chain-of-thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Number of tokens used for internal reasoning (e.g., chain-of-thought).
    ///
    /// Some models generate reasoning tokens that are not part of the final response.
    pub reasoning_tokens: u32,
}

/// A request for a chat completion from an LLM.
///
/// This struct encapsulates all parameters needed to request a completion,
/// including conversation history, sampling parameters, and tool configuration.
///
/// # Examples
///
/// ```
/// use neuromance_common::{ChatRequest, Message, MessageRole};
/// use uuid::Uuid;
///
/// let message = Message::new(Uuid::new_v4(), MessageRole::User, "Hello!");
/// let request = ChatRequest::new(vec![message])
///     .with_model("gpt-4")
///     .with_temperature(0.7)
///     .with_max_tokens(1000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The conversation messages to send to the model.
    pub messages: Arc<[Message]>,
    /// The model identifier to use for generation.
    pub model: Option<String>,
    /// Sampling temperature controlling randomness (0.0 to 2.0).
    ///
    /// Lower values make output more focused and deterministic.
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate in the response.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling threshold (0.0 to 1.0).
    ///
    /// Only tokens comprising the top `top_p` probability mass are considered.
    pub top_p: Option<f32>,
    /// Penalty for token frequency in the output (-2.0 to 2.0).
    ///
    /// Positive values decrease likelihood of repeating tokens.
    pub frequency_penalty: Option<f32>,
    /// Penalty for token presence in the output (-2.0 to 2.0).
    ///
    /// Positive values encourage discussing new topics.
    pub presence_penalty: Option<f32>,
    /// Sequences that will stop generation when encountered.
    pub stop: Option<Vec<String>>,
    /// Tools available for the model to call.
    pub tools: Option<Vec<Tool>>,
    /// Strategy for tool selection.
    pub tool_choice: Option<ToolChoice>,
    /// Whether to stream the response incrementally.
    pub stream: bool,
    /// End-user identifier for tracking and abuse prevention.
    pub user: Option<String>,
    /// Whether to enable thinking mode (vendor-specific, e.g., Qwen models).
    pub enable_thinking: Option<bool>,
    /// Additional metadata to attach to this request.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl fmt::Display for ChatRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string(self) {
            Ok(json) => write!(f, "{json}"),
            Err(_) => write!(f, "Error serializing ChatRequest to JSON"),
        }
    }
}

impl ChatRequest {
    /// Creates a new chat request with the given messages.
    ///
    /// All optional parameters are set to `None` and must be configured
    /// using the builder methods.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history to send to the model
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::{ChatRequest, Message, MessageRole};
    /// use uuid::Uuid;
    ///
    /// let msg = Message::new(Uuid::new_v4(), MessageRole::User, "Hello!");
    /// let request = ChatRequest::new(vec![msg]);
    /// ```
    pub fn new(messages: impl Into<Arc<[Message]>>) -> Self {
        Self {
            messages: messages.into(),
            model: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            tools: None,
            tool_choice: None,
            stream: false,
            user: None,
            enable_thinking: None,
            metadata: HashMap::new(),
        }
    }
}

impl From<(&Config, Vec<Message>)> for ChatRequest {
    fn from((config, messages): (&Config, Vec<Message>)) -> Self {
        Self {
            messages: messages.into(),
            model: Some(config.model.clone()),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            stop: config.stop_sequences.clone(),
            tools: None,
            tool_choice: None,
            stream: false,
            user: None,
            enable_thinking: None,
            metadata: HashMap::new(),
        }
    }
}

impl From<(&Config, Arc<[Message]>)> for ChatRequest {
    fn from((config, messages): (&Config, Arc<[Message]>)) -> Self {
        Self {
            messages,
            model: Some(config.model.clone()),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            stop: config.stop_sequences.clone(),
            tools: None,
            tool_choice: None,
            stream: false,
            user: None,
            enable_thinking: None,
            metadata: HashMap::new(),
        }
    }
}

impl ChatRequest {
    /// Sets the model to use for this request.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "gpt-4", "claude-3-opus")
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the sampling temperature.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Value between 0.0 and 2.0
    ///
    /// Higher values produce more random output.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum number of tokens to generate.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens in the response
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the nucleus sampling threshold.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Value between 0.0 and 1.0
    ///
    /// Lower values make output more focused.
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the frequency penalty.
    ///
    /// # Arguments
    ///
    /// * `frequency_penalty` - Value between -2.0 and 2.0
    ///
    /// Positive values discourage token repetition.
    #[must_use]
    pub const fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Sets the presence penalty.
    ///
    /// # Arguments
    ///
    /// * `presence_penalty` - Value between -2.0 and 2.0
    ///
    /// Positive values encourage discussing new topics.
    #[must_use]
    pub const fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Sets stop sequences that will halt generation.
    ///
    /// # Arguments
    ///
    /// * `stop_sequences` - An iterable of strings to use as stop sequences
    #[must_use]
    pub fn with_stop_sequences(
        mut self,
        stop_sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.stop = Some(stop_sequences.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the tools available for the model to call.
    ///
    /// # Arguments
    ///
    /// * `tools` - Vector of tool definitions
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the tool selection strategy.
    ///
    /// # Arguments
    ///
    /// * `tool_choice` - Strategy for how the model should select tools
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Enables or disables streaming for this request.
    ///
    /// # Arguments
    ///
    /// * `stream` - Whether to stream the response
    #[must_use]
    pub const fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Sets custom metadata for this request.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Key-value pairs of metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Enables or disables thinking mode (vendor-specific).
    ///
    /// # Arguments
    ///
    /// * `enable_thinking` - Whether to enable thinking mode
    #[must_use]
    pub const fn with_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }

    /// Validate that this request has at least one message.
    ///
    /// # Errors
    ///
    /// Returns an error if the messages vector is empty.
    pub fn validate_has_messages(&self) -> anyhow::Result<()> {
        if self.messages.is_empty() {
            anyhow::bail!("Chat request must have at least one message");
        }
        Ok(())
    }

    /// Validates all configuration parameters
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails (empty messages, invalid `temperature`/`top_p` ranges).
    pub fn validate(&self) -> anyhow::Result<()> {
        self.validate_has_messages()?;

        if let Some(temp) = self.temperature
            && !(0.0..=2.0).contains(&temp)
        {
            anyhow::bail!("Temperature must be between 0.0 and 2.0, got {temp}");
        }

        if let Some(top_p) = self.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            anyhow::bail!("top_p must be between 0.0 and 1.0, got {top_p}");
        }

        if let Some(freq_penalty) = self.frequency_penalty
            && !(-2.0..=2.0).contains(&freq_penalty)
        {
            anyhow::bail!("frequency_penalty must be between -2.0 and 2.0, got {freq_penalty}");
        }

        if let Some(pres_penalty) = self.presence_penalty
            && !(-2.0..=2.0).contains(&pres_penalty)
        {
            anyhow::bail!("presence_penalty must be between -2.0 and 2.0, got {pres_penalty}");
        }

        Ok(())
    }

    /// Returns whether this request has tools configured.
    ///
    /// # Returns
    ///
    /// `true` if tools are present and non-empty, `false` otherwise.
    #[must_use]
    pub fn has_tools(&self) -> bool {
        self.tools.as_ref().is_some_and(|t| !t.is_empty())
    }

    /// Returns whether this request uses streaming.
    ///
    /// # Returns
    ///
    /// `true` if streaming is enabled, `false` otherwise.
    #[must_use]
    pub const fn is_streaming(&self) -> bool {
        self.stream
    }
}

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

/// Configuration for an LLM client.
///
/// This struct holds both connection details (API keys, URLs) and default
/// generation parameters that will be applied to all requests unless overridden.
///
/// # Security
///
/// The `api_key` field uses `SecretString` to prevent accidental logging or
/// display of sensitive credentials.
///
/// # Examples
///
/// ```
/// use neuromance_common::Config;
///
/// let config = Config::new("openai", "gpt-4")
///     .with_api_key("sk-...")
///     .with_temperature(0.7)
///     .with_max_tokens(1000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// The LLM provider name (e.g., "openai", "anthropic").
    pub provider: String,
    /// The default model identifier to use.
    pub model: String,
    /// Optional custom base URL for API requests.
    ///
    /// Override this for self-hosted deployments or custom endpoints.
    pub base_url: Option<String>,
    /// API key for authentication (stored securely).
    ///
    /// Will not be serialized to prevent accidental exposure.
    #[serde(skip_serializing, default)]
    pub api_key: Option<SecretString>,
    /// Optional organization identifier.
    pub organization: Option<String>,
    /// Request timeout in seconds.
    pub timeout_seconds: Option<u64>,
    /// Configuration for retry behavior with exponential backoff.
    #[serde(skip)]
    pub retry_config: RetryConfig,
    /// Default sampling temperature (0.0 to 2.0).
    pub temperature: Option<f32>,
    /// Default maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Default nucleus sampling threshold (0.0 to 1.0).
    pub top_p: Option<f32>,
    /// Default frequency penalty (-2.0 to 2.0).
    pub frequency_penalty: Option<f32>,
    /// Default presence penalty (-2.0 to 2.0).
    pub presence_penalty: Option<f32>,
    /// Default stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Additional metadata to attach to all requests.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "gpt-oss:20b".to_string(),
            base_url: None,
            api_key: None,
            organization: None,
            timeout_seconds: None,
            retry_config: RetryConfig::default(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            metadata: HashMap::new(),
        }
    }
}

impl Config {
    /// Creates a new configuration with the specified provider and model.
    ///
    /// All optional fields are initialized to their defaults.
    ///
    /// # Arguments
    ///
    /// * `provider` - The LLM provider name
    /// * `model` - The model identifier
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::Config;
    ///
    /// let config = Config::new("openai", "gpt-4");
    /// ```
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Sets a custom base URL for API requests.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL for the API
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the API key for authentication.
    ///
    /// The key is stored securely using `SecretString`.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::new(api_key.into().into()));
        self
    }

    /// Sets the organization identifier.
    ///
    /// # Arguments
    ///
    /// * `organization` - The organization ID
    #[must_use]
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Sets the request timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout_seconds` - Timeout in seconds
    #[must_use]
    pub const fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = Some(timeout_seconds);
        self
    }

    /// Sets the default sampling temperature.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Value between 0.0 and 2.0
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the default maximum tokens to generate.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens in responses
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the default nucleus sampling threshold.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Value between 0.0 and 1.0
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the default frequency penalty.
    ///
    /// # Arguments
    ///
    /// * `frequency_penalty` - Value between -2.0 and 2.0
    #[must_use]
    pub const fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Sets the default presence penalty.
    ///
    /// # Arguments
    ///
    /// * `presence_penalty` - Value between -2.0 and 2.0
    #[must_use]
    pub const fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Sets the default stop sequences.
    ///
    /// # Arguments
    ///
    /// * `stop_sequences` - An iterable of stop sequences
    #[must_use]
    pub fn with_stop_sequences(
        mut self,
        stop_sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.stop_sequences = Some(stop_sequences.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the default metadata.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Key-value pairs of metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the retry configuration.
    ///
    /// # Arguments
    ///
    /// * `retry_config` - The retry configuration
    #[must_use]
    pub const fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }
}

impl From<(Config, Vec<Message>)> for ChatRequest {
    fn from((config, messages): (Config, Vec<Message>)) -> Self {
        let mut request = Self::new(messages).with_model(&config.model);

        if let Some(temperature) = config.temperature {
            request = request.with_temperature(temperature);
        }

        if let Some(max_tokens) = config.max_tokens {
            request = request.with_max_tokens(max_tokens);
        }

        if let Some(top_p) = config.top_p {
            request.top_p = Some(top_p);
        }

        if let Some(frequency_penalty) = config.frequency_penalty {
            request.frequency_penalty = Some(frequency_penalty);
        }

        if let Some(presence_penalty) = config.presence_penalty {
            request.presence_penalty = Some(presence_penalty);
        }

        if let Some(stop_sequences) = config.stop_sequences {
            request.stop = Some(stop_sequences);
        }

        request.metadata = config.metadata;

        request
    }
}

impl Config {
    /// Converts this configuration and messages into a chat request.
    ///
    /// This is a convenience method that creates a `ChatRequest` with all
    /// default parameters from this configuration.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation messages
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::{Config, Message, MessageRole};
    /// use uuid::Uuid;
    ///
    /// let config = Config::new("openai", "gpt-4").with_temperature(0.7);
    /// let msg = Message::new(Uuid::new_v4(), MessageRole::User, "Hello!");
    /// let request = config.into_chat_request(vec![msg]);
    /// ```
    #[must_use]
    pub fn into_chat_request(self, messages: Vec<Message>) -> ChatRequest {
        (self, messages).into()
    }

    /// Validates the configuration parameters.
    ///
    /// Checks that all numeric parameters are within their valid ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of range:
    /// - `temperature` must be between 0.0 and 2.0
    /// - `top_p` must be between 0.0 and 1.0
    /// - `frequency_penalty` must be between -2.0 and 2.0
    /// - `presence_penalty` must be between -2.0 and 2.0
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(temp) = self.temperature
            && !(0.0..=2.0).contains(&temp)
        {
            anyhow::bail!("Temperature must be between 0.0 and 2.0, got {temp}");
        }

        if let Some(top_p) = self.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            anyhow::bail!("top_p must be between 0.0 and 1.0, got {top_p}");
        }

        if let Some(freq_penalty) = self.frequency_penalty
            && !(-2.0..=2.0).contains(&freq_penalty)
        {
            anyhow::bail!("frequency_penalty must be between -2.0 and 2.0, got {freq_penalty}");
        }

        if let Some(pres_penalty) = self.presence_penalty
            && !(-2.0..=2.0).contains(&pres_penalty)
        {
            anyhow::bail!("presence_penalty must be between -2.0 and 2.0, got {pres_penalty}");
        }

        Ok(())
    }
}

#[cfg(test)]
mod proptests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn temperature_validation(temp in -10.0f32..10.0f32) {
            let config = Config::new("openai", "gpt-4").with_temperature(temp);
            let is_valid = (0.0..=2.0).contains(&temp);
            assert_eq!(config.validate().is_ok(), is_valid);
        }

        #[test]
        fn top_p_validation(top_p in -5.0f32..5.0f32) {
            let config = Config::new("openai", "gpt-4").with_top_p(top_p);
            let is_valid = (0.0..=1.0).contains(&top_p);
            assert_eq!(config.validate().is_ok(), is_valid);
        }

        #[test]
        fn frequency_penalty_validation(penalty in -10.0f32..10.0f32) {
            let config = Config::new("openai", "gpt-4").with_frequency_penalty(penalty);
            let is_valid = (-2.0..=2.0).contains(&penalty);
            assert_eq!(config.validate().is_ok(), is_valid);
        }

        #[test]
        fn presence_penalty_validation(penalty in -10.0f32..10.0f32) {
            let config = Config::new("openai", "gpt-4").with_presence_penalty(penalty);
            let is_valid = (-2.0..=2.0).contains(&penalty);
            assert_eq!(config.validate().is_ok(), is_valid);
        }

        #[test]
        fn max_tokens_validation(tokens in 0u32..1000000u32) {
            let config = Config::new("openai", "gpt-4").with_max_tokens(tokens);
            // max_tokens can be 0 (infinite) or any positive value
            assert!(config.validate().is_ok());
        }

        #[test]
        fn config_builder_with_string_slice(
            provider in ".*",
            model in ".*",
            base_url in ".*",
        ) {
            let config = Config::new(provider.as_str(), model.as_str())
                .with_base_url(base_url.as_str());

            // Should compile and work with &str
            assert_eq!(config.provider, provider);
            assert_eq!(config.model, model);
            assert_eq!(config.base_url, Some(base_url));
        }

        #[test]
        fn config_builder_with_owned_string(
            provider in ".*",
            model in ".*",
        ) {
            let config = Config::new(provider.clone(), model.clone());

            // Should compile and work with String
            assert_eq!(config.provider, provider);
            assert_eq!(config.model, model);
        }

        #[test]
        fn stop_sequences_accepts_various_types(
            sequences in prop::collection::vec(".*", 0..10),
        ) {
            // Test with Vec<String>
            let config1 = Config::new("openai", "gpt-4")
                .with_stop_sequences(sequences.clone());
            assert_eq!(config1.stop_sequences, Some(sequences.clone()));

            // Test with Vec<&str>
            let str_refs: Vec<&str> = sequences.iter().map(|s| s.as_str()).collect();
            let config2 = Config::new("openai", "gpt-4")
                .with_stop_sequences(str_refs);
            assert_eq!(config2.stop_sequences, Some(sequences.clone()));

            // Test with array of &str
            if sequences.len() <= 3 {
                let arr: Vec<&str> = sequences.iter().map(|s| s.as_str()).collect();
                let config3 = Config::new("openai", "gpt-4")
                    .with_stop_sequences(arr);
                assert_eq!(config3.stop_sequences, Some(sequences));
            }
        }

        #[test]
        fn builder_chain_preserves_all_values(
            provider in ".*",
            model in ".*",
            temp in 0.0f32..2.0f32,
            max_tokens in 0u32..100000u32,
        ) {
            let config = Config::new(provider.as_str(), model.as_str())
                .with_temperature(temp)
                .with_max_tokens(max_tokens);

            assert_eq!(config.provider, provider);
            assert_eq!(config.model, model);
            assert_eq!(config.temperature, Some(temp));
            assert_eq!(config.max_tokens, Some(max_tokens));
            assert!(config.validate().is_ok());
        }

        // ChatRequest property tests
        #[test]
        fn chat_request_temperature_validation(
            temp in -10.0f32..10.0f32,
            msg_count in 1usize..10,
        ) {
            use crate::chat::{Message, MessageRole};
            use uuid::Uuid;

            let messages: Vec<Message> = (0..msg_count)
                .map(|i| Message::new(Uuid::new_v4(), MessageRole::User, format!("message {}", i)))
                .collect();

            let request = ChatRequest::new(messages).with_temperature(temp);
            let is_valid = (0.0..=2.0).contains(&temp);
            assert_eq!(request.validate().is_ok(), is_valid);
        }

        #[test]
        fn chat_request_with_string_types(
            model in ".*",
        ) {
            use crate::chat::{Message, MessageRole};
            use uuid::Uuid;

            let msg = Message::new(Uuid::new_v4(), MessageRole::User, "test");

            // Test with &str
            let request1 = ChatRequest::new(vec![msg.clone()])
                .with_model(model.as_str());
            assert_eq!(request1.model, Some(model.clone()));

            // Test with String
            let request2 = ChatRequest::new(vec![msg])
                .with_model(model.clone());
            assert_eq!(request2.model, Some(model));
        }

        #[test]
        fn chat_request_stop_sequences_ergonomics(
            sequences in prop::collection::vec(".*", 1..5),
        ) {
            use crate::chat::{Message, MessageRole};
            use uuid::Uuid;

            let msg = Message::new(Uuid::new_v4(), MessageRole::User, "test");

            // Test with Vec<String>
            let request1 = ChatRequest::new(vec![msg.clone()])
                .with_stop_sequences(sequences.clone());
            assert_eq!(request1.stop, Some(sequences.clone()));

            // Test with &[&str]
            let str_refs: Vec<&str> = sequences.iter().map(|s| s.as_str()).collect();
            let request2 = ChatRequest::new(vec![msg])
                .with_stop_sequences(str_refs);
            assert_eq!(request2.stop, Some(sequences));
        }

        #[test]
        fn chat_request_builder_chain(
            model in ".*",
            temp in 0.0f32..2.0f32,
            max_tokens in 0u32..100000u32,
            top_p in 0.0f32..1.0f32,
        ) {
            use crate::chat::{Message, MessageRole};
            use uuid::Uuid;

            let msg = Message::new(Uuid::new_v4(), MessageRole::User, "test");
            let request = ChatRequest::new(vec![msg])
                .with_model(model.as_str())
                .with_temperature(temp)
                .with_max_tokens(max_tokens)
                .with_top_p(top_p);

            assert_eq!(request.model, Some(model));
            assert_eq!(request.temperature, Some(temp));
            assert_eq!(request.max_tokens, Some(max_tokens));
            assert_eq!(request.top_p, Some(top_p));
            assert!(request.validate().is_ok());
        }
    }

    #[test]
    fn chat_request_validates_empty_messages() {
        let request = ChatRequest::new(vec![]);
        assert!(request.validate().is_err());
        assert!(request.validate_has_messages().is_err());
    }

    #[test]
    fn chat_request_has_tools() {
        use crate::chat::{Message, MessageRole};
        use crate::tools::{Function, Tool};
        use uuid::Uuid;

        let msg = Message::new(Uuid::new_v4(), MessageRole::User, "test");

        // Test with no tools (None)
        let request_no_tools = ChatRequest::new(vec![msg.clone()]);
        assert!(!request_no_tools.has_tools());

        // Test with empty tools vector
        let request_empty_tools = ChatRequest::new(vec![msg.clone()]).with_tools(vec![]);
        assert!(!request_empty_tools.has_tools());

        // Test with tools present
        let function = Function {
            name: "test_function".to_string(),
            description: "A test function".to_string(),
            parameters: serde_json::json!({}),
        };
        let tool = Tool::builder().function(function).build();
        let request_with_tools = ChatRequest::new(vec![msg]).with_tools(vec![tool]);
        assert!(request_with_tools.has_tools());
    }
}
