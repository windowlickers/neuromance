//! Feature abstractions for cross-provider capabilities.
//!
//! These types represent high-level features that different providers
//! implement in their own way. The client layer maps these to native APIs.

use serde::{Deserialize, Serialize};

/// Configuration for model thinking/reasoning capabilities.
///
/// Different providers implement thinking differently:
/// - **Anthropic**: Extended thinking with token budget, interleaved thinking between tool calls
/// - **OpenAI**: Reasoning effort levels (o1, o3 models), max_completion_tokens
/// - **Other providers**: May use temperature/sampling adjustments
///
/// # Examples
///
/// ```
/// use neuromance_common::features::ThinkingMode;
///
/// // Extended thinking with 10k token budget
/// let mode = ThinkingMode::Extended { budget_tokens: 10000 };
/// assert_eq!(mode.budget(), Some(10000));
///
/// // Interleaved thinking (between tool calls)
/// let mode = ThinkingMode::Interleaved { budget_tokens: 8000 };
/// assert!(mode.is_interleaved());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ThinkingMode {
    /// No explicit thinking mode (provider default behavior).
    #[default]
    Default,

    /// Extended thinking with a token budget.
    ///
    /// - **Anthropic**: Maps to `thinking.budget_tokens`
    /// - **OpenAI**: Maps to `max_completion_tokens` (for reasoning models)
    Extended {
        /// Maximum tokens the model can use for thinking/reasoning.
        /// Anthropic minimum is 1024 tokens.
        budget_tokens: u32,
    },

    /// Interleaved thinking between tool calls.
    ///
    /// Allows the model to think/reason after receiving tool results,
    /// enabling more sophisticated multi-step reasoning.
    ///
    /// - **Anthropic**: Enables the interleaved-thinking beta feature
    /// - **OpenAI**: Falls back to `Extended` behavior (not directly supported)
    Interleaved {
        /// Maximum tokens the model can use for thinking/reasoning.
        budget_tokens: u32,
    },
}

impl ThinkingMode {
    /// Returns the thinking budget in tokens, if thinking is enabled.
    ///
    /// Returns `None` for `Default` mode.
    #[must_use]
    pub const fn budget(&self) -> Option<u32> {
        match self {
            Self::Extended { budget_tokens } | Self::Interleaved { budget_tokens } => {
                Some(*budget_tokens)
            }
            Self::Default => None,
        }
    }

    /// Whether interleaved thinking mode is enabled.
    ///
    /// When true, the model can think between tool calls.
    #[must_use]
    pub const fn is_interleaved(&self) -> bool {
        matches!(self, Self::Interleaved { .. })
    }

    /// Whether any thinking mode is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        !matches!(self, Self::Default)
    }

    /// Creates an extended thinking mode with the given budget.
    #[must_use]
    pub const fn extended(budget_tokens: u32) -> Self {
        Self::Extended { budget_tokens }
    }

    /// Creates an interleaved thinking mode with the given budget.
    #[must_use]
    pub const fn interleaved(budget_tokens: u32) -> Self {
        Self::Interleaved { budget_tokens }
    }
}

/// Reasoning effort level for models that support it.
///
/// This is an abstraction over provider-specific reasoning controls:
/// - **OpenAI**: Direct mapping to `reasoning_effort` parameter (o1, o3 models)
/// - **Anthropic**: Can influence thinking budget selection heuristics
///
/// # Examples
///
/// ```
/// use neuromance_common::features::ReasoningLevel;
///
/// let level = ReasoningLevel::High;
/// assert!(level > ReasoningLevel::Low);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningLevel {
    /// No explicit reasoning configuration (provider default).
    #[default]
    Default,
    /// Minimal reasoning effort - fastest responses.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort (balanced).
    Medium,
    /// High reasoning effort.
    High,
    /// Maximum reasoning effort - most thorough but slowest.
    Maximum,
}

impl ReasoningLevel {
    /// Whether a non-default reasoning level is set.
    #[must_use]
    pub const fn is_set(&self) -> bool {
        !matches!(self, Self::Default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_mode_budget() {
        assert_eq!(ThinkingMode::Default.budget(), None);
        assert_eq!(ThinkingMode::Extended { budget_tokens: 5000 }.budget(), Some(5000));
        assert_eq!(ThinkingMode::Interleaved { budget_tokens: 8000 }.budget(), Some(8000));
    }

    #[test]
    fn test_thinking_mode_is_interleaved() {
        assert!(!ThinkingMode::Default.is_interleaved());
        assert!(!ThinkingMode::Extended { budget_tokens: 5000 }.is_interleaved());
        assert!(ThinkingMode::Interleaved { budget_tokens: 8000 }.is_interleaved());
    }

    #[test]
    fn test_thinking_mode_is_enabled() {
        assert!(!ThinkingMode::Default.is_enabled());
        assert!(ThinkingMode::Extended { budget_tokens: 5000 }.is_enabled());
        assert!(ThinkingMode::Interleaved { budget_tokens: 8000 }.is_enabled());
    }

    #[test]
    fn test_thinking_mode_constructors() {
        assert_eq!(ThinkingMode::extended(1000), ThinkingMode::Extended { budget_tokens: 1000 });
        assert_eq!(ThinkingMode::interleaved(2000), ThinkingMode::Interleaved { budget_tokens: 2000 });
    }

    #[test]
    fn test_reasoning_level_ordering() {
        assert!(ReasoningLevel::Minimal < ReasoningLevel::Low);
        assert!(ReasoningLevel::Low < ReasoningLevel::Medium);
        assert!(ReasoningLevel::Medium < ReasoningLevel::High);
        assert!(ReasoningLevel::High < ReasoningLevel::Maximum);
    }

    #[test]
    fn test_reasoning_level_is_set() {
        assert!(!ReasoningLevel::Default.is_set());
        assert!(ReasoningLevel::Minimal.is_set());
        assert!(ReasoningLevel::High.is_set());
    }

    #[test]
    fn test_thinking_mode_serde() {
        let mode = ThinkingMode::Extended { budget_tokens: 5000 };
        let json = serde_json::to_string(&mode).unwrap();
        assert!(json.contains("extended"));
        assert!(json.contains("5000"));

        let deserialized: ThinkingMode = serde_json::from_str(&json).unwrap();
        assert_eq!(mode, deserialized);
    }

    #[test]
    fn test_reasoning_level_serde() {
        let level = ReasoningLevel::High;
        let json = serde_json::to_string(&level).unwrap();
        assert_eq!(json, "\"high\"");

        let deserialized: ReasoningLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(level, deserialized);
    }
}
