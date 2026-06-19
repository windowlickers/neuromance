//! # neuromance-context
//!
//! Context management for LLM conversations, built around a typestate state
//! machine.
//!
//! [`Context<S>`] moves a conversation through `Raw → Filtered → Transformed →
//! Ready`, with each transition enforced at compile time (see the [`state`] and
//! [`transforms`] modules). [`Compactor`] plugs into the `Transformed` step to
//! summarize history when the window fills up. The [`tokens`] module provides
//! the [`TokenCounter`] that measures when that is needed, and the [`skills`]
//! and [`rules`] modules inject additional context.
//!
//! ## Example
//!
//! ```no_run
//! use neuromance_context::Context;
//! use neuromance_context::transforms::FilterCriteria;
//! use neuromance_common::Conversation;
//!
//! # fn example() {
//! let conversation = Conversation::new();
//!
//! // Drive the conversation through the state machine.
//! let ready = Context::new(conversation)
//!     .filter(FilterCriteria::default())
//!     .transform()
//!     .ready();
//!
//! let prepared = ready.into_conversation();
//! # let _ = prepared;
//! # }
//! ```

mod error;
pub use error::TokenCounterError;

pub mod compaction;
pub mod compaction_hook;
pub mod context;
pub mod metadata;
pub mod rules;
pub mod skills;
pub mod state;
pub mod tokens;
pub mod transforms;

pub use compaction::{CompactionConfig, CompactionResult, CompactionStrategy, Compactor};
pub use compaction_hook::{CompactionHook, ContextConfig, TokenSource};
pub use context::Context;
pub use tokens::{ModelConfig, SearchMatch, TokenCounter, TokenInfo, TokenizedText};

/// Shared test fixtures used across the crate's unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    #![allow(clippy::unwrap_used)]

    use tokenizers::Tokenizer;

    /// Creates a minimal `WordLevel` tokenizer that doesn't require network access.
    pub(crate) fn create_test_tokenizer() -> Tokenizer {
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[UNK]": 0,
                    "hello": 1,
                    ",": 2,
                    "world": 3,
                    "!": 4,
                    "how": 5,
                    "are": 6,
                    "you": 7,
                    "?": 8,
                    "the": 9,
                    "weather": 10,
                    "is": 11,
                    "sunny": 12,
                    "today": 13,
                    "what": 14
                },
                "unk_token": "[UNK]"
            },
            "pre_tokenizer": {
                "type": "Whitespace"
            }
        });

        Tokenizer::from_bytes(tokenizer_json.to_string()).unwrap()
    }
}
