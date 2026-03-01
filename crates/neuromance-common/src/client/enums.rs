use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// LLM provider identifier.
///
/// Used in `ModelProfile` to select the correct client implementation.
/// Deserializes from lowercase strings (e.g., `"anthropic"`, `"openai"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// Anthropic (Claude models)
    Anthropic,
    /// `OpenAI` Chat Completions API
    OpenAI,
    /// `OpenAI` Responses API
    Responses,
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAI => write!(f, "openai"),
            Self::Responses => write!(f, "responses"),
        }
    }
}

impl FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "anthropic" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAI),
            "responses" => Ok(Self::Responses),
            other => Err(format!("unknown provider: {other}")),
        }
    }
}

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

/// Reasoning effort level for thinking models (with support).
///
/// Controls how much compute the model spends on reasoning before responding.
/// Higher effort levels may produce better results for complex problems but
/// use more tokens and take longer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// No reasoning (GPT-5.1+ only).
    None,
    /// Minimal reasoning effort.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort (default for most models).
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort (GPT-5.2+).
    #[serde(rename = "xhigh")]
    XHigh,
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
