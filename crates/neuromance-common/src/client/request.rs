use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::config::Config;
use super::enums::ToolChoice;
use crate::chat::Message;
use crate::features::{ReasoningLevel, ThinkingMode};
use crate::tools::Tool;

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
    ///
    /// For reasoning models (o1, o3, GPT-5+), use [`max_completion_tokens`](Self::max_completion_tokens)
    /// instead, which includes both output and reasoning tokens.
    pub max_tokens: Option<u32>,
    /// Maximum completion tokens including reasoning tokens (for reasoning models).
    ///
    /// This replaces `max_tokens` for reasoning models (o1, o3, GPT-5+) and includes
    /// both visible output tokens and internal reasoning tokens.
    pub max_completion_tokens: Option<u32>,
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
    /// Thinking/reasoning mode configuration.
    ///
    /// Controls extended thinking capabilities across providers:
    /// - **Anthropic**: Maps to `thinking.budget_tokens` and interleaved-thinking beta
    /// - **`OpenAI`**: Maps to `max_completion_tokens` for reasoning models
    ///
    /// See [`ThinkingMode`] for available options.
    #[serde(default)]
    pub thinking: ThinkingMode,
    /// Reasoning effort level for models that support it.
    ///
    /// Controls how much compute the model spends on reasoning:
    /// - **`OpenAI`**: Maps to `reasoning_effort` parameter (o1, o3 models)
    /// - **Anthropic**: Can influence thinking budget heuristics
    #[serde(default)]
    pub reasoning_level: ReasoningLevel,
    /// Additional metadata to attach to this request.
    pub metadata: HashMap<String, serde_json::Value>,
}

pub(super) fn validate_sampling_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> anyhow::Result<()> {
    if let Some(temp) = temperature
        && !(0.0..=2.0).contains(&temp)
    {
        anyhow::bail!("Temperature must be between 0.0 and 2.0, got {temp}");
    }

    if let Some(top_p) = top_p
        && !(0.0..=1.0).contains(&top_p)
    {
        anyhow::bail!("top_p must be between 0.0 and 1.0, got {top_p}");
    }

    if let Some(freq) = frequency_penalty
        && !(-2.0..=2.0).contains(&freq)
    {
        anyhow::bail!("frequency_penalty must be between -2.0 and 2.0, got {freq}");
    }

    if let Some(pres) = presence_penalty
        && !(-2.0..=2.0).contains(&pres)
    {
        anyhow::bail!("presence_penalty must be between -2.0 and 2.0, got {pres}");
    }

    Ok(())
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
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            tools: None,
            tool_choice: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        }
    }
}

impl ChatRequest {
    fn from_config(config: &Config, messages: Arc<[Message]>) -> Self {
        Self {
            messages,
            model: Some(config.model.clone()),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            max_completion_tokens: None,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            stop: config.stop_sequences.clone(),
            tools: None,
            tool_choice: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: config.metadata.clone(),
        }
    }
}

impl From<(&Config, Vec<Message>)> for ChatRequest {
    fn from((config, messages): (&Config, Vec<Message>)) -> Self {
        Self::from_config(config, messages.into())
    }
}

impl From<(&Config, Arc<[Message]>)> for ChatRequest {
    fn from((config, messages): (&Config, Arc<[Message]>)) -> Self {
        Self::from_config(config, messages)
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
    /// For reasoning models (o1, o3, GPT-5+), use [`with_max_completion_tokens`](Self::with_max_completion_tokens)
    /// instead, which includes both output and reasoning tokens.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens in the response
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the maximum completion tokens for reasoning models.
    ///
    /// This includes both visible output tokens and internal reasoning tokens.
    /// Use this instead of `max_tokens` for reasoning models (o1, o3, GPT-5+).
    ///
    /// # Arguments
    ///
    /// * `max_completion_tokens` - Maximum tokens including reasoning
    #[must_use]
    pub const fn with_max_completion_tokens(mut self, max_completion_tokens: u32) -> Self {
        self.max_completion_tokens = Some(max_completion_tokens);
        self
    }

    /// Sets the reasoning effort level for models that support it.
    ///
    /// Controls how much compute the model spends on reasoning before responding.
    /// Higher effort levels may produce better results for complex problems.
    ///
    /// # Arguments
    ///
    /// * `level` - The reasoning effort level
    #[must_use]
    pub const fn with_reasoning_level(mut self, level: ReasoningLevel) -> Self {
        self.reasoning_level = level;
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

    /// Sets the thinking/reasoning mode.
    ///
    /// This is the primary way to configure extended thinking capabilities.
    /// See [`ThinkingMode`] for available options.
    ///
    /// # Arguments
    ///
    /// * `mode` - The thinking mode configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::{ChatRequest, ThinkingMode, Message, MessageRole};
    /// use uuid::Uuid;
    ///
    /// let msg = Message::new(Uuid::new_v4(), MessageRole::User, "Hello!");
    /// let request = ChatRequest::new(vec![msg])
    ///     .with_thinking_mode(ThinkingMode::Extended { budget_tokens: 10000 });
    /// ```
    #[must_use]
    pub const fn with_thinking_mode(mut self, mode: ThinkingMode) -> Self {
        self.thinking = mode;
        self
    }

    /// Sets the token budget for extended thinking.
    ///
    /// This is a convenience method for `with_thinking_mode(ThinkingMode::Extended { budget_tokens })`.
    ///
    /// When set, enables extended thinking with the specified budget.
    /// The budget determines the maximum tokens the model can use for internal reasoning.
    /// - **Anthropic**: Minimum is 1024 tokens
    /// - **`OpenAI`**: Maps to `max_completion_tokens` for reasoning models
    ///
    /// # Arguments
    ///
    /// * `budget` - The thinking budget in tokens
    #[must_use]
    pub const fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking = ThinkingMode::Extended {
            budget_tokens: budget,
        };
        self
    }

    /// Enables interleaved thinking with the given budget.
    ///
    /// This is a convenience method for `with_thinking_mode(ThinkingMode::Interleaved { budget_tokens })`.
    ///
    /// When enabled, the model can think between tool calls, allowing more sophisticated
    /// reasoning after receiving tool results.
    /// - **Anthropic**: Enables the interleaved-thinking beta feature
    /// - **`OpenAI`**: Falls back to extended thinking behavior (not directly supported)
    ///
    /// # Arguments
    ///
    /// * `budget` - The thinking budget in tokens
    #[must_use]
    pub const fn with_interleaved_thinking(mut self, budget: u32) -> Self {
        self.thinking = ThinkingMode::Interleaved {
            budget_tokens: budget,
        };
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
        validate_sampling_params(
            self.temperature,
            self.top_p,
            self.frequency_penalty,
            self.presence_penalty,
        )
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn chat_request_temperature_validation(
            temp in -10.0f32..10.0f32,
            msg_count in 1usize..10,
        ) {
            use crate::chat::{Message, MessageRole};
            use uuid::Uuid;

            let messages: Vec<Message> = (0..msg_count)
                .map(|i| Message::new(Uuid::new_v4(), MessageRole::User, format!("message {i}")))
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
            let str_refs: Vec<&str> = sequences.iter().map(std::string::String::as_str).collect();
            let request2 = ChatRequest::new(vec![msg])
                .with_stop_sequences(str_refs);
            assert_eq!(request2.stop, Some(sequences));
        }

        #[test]
        fn chat_request_builder_chain(
            model in ".*",
            temp in 0.0f32..2.0f32,
            max_tokens in 0u32..100_000_u32,
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
