//! # neuromance-context
//!
//! Context management for LLM conversations.
//!
//! The edit ledger itself — [`ContextLedger`](neuromance_common::context::ContextLedger),
//! the single funnel through which a conversation's history is mutated — lives in
//! [`neuromance_common`]. This crate provides the heavier operations layered on
//! top: the [`context`] module's batch [`filter`](context::filter),
//! [`transform`](context::transform), and [`compact`](context::compact)
//! functions, the [`Compactor`] that summarizes history when the window fills up,
//! the [`tokens`] module's [`TokenCounter`], and the [`skills`] and [`rules`]
//! modules that inject additional context.
//!
//! ## Example
//!
//! ```
//! use neuromance_context::context::filter;
//! use neuromance_context::transforms::FilterCriteria;
//! use neuromance_common::context::{ContextLedger, EditSource, Operation};
//! use neuromance_common::Conversation;
//! use neuromance_common::chat::{Message, MessageRole};
//!
//! let conversation = Conversation::new();
//! let id = conversation.id;
//! let mut ledger = ContextLedger::new(conversation);
//!
//! // Every edit is recorded with its provenance.
//! ledger.append(EditSource::core(), [Message::user(id, "hi"), Message::system(id, "sys")]);
//! filter(&mut ledger, &FilterCriteria::default().with_roles(vec![MessageRole::User]));
//!
//! assert_eq!(ledger.messages().len(), 1);
//! assert!(ledger.metadata().has_operation(Operation::Replace));
//! ```

mod error;
pub use error::TokenCounterError;

pub mod compaction;
pub mod compaction_hook;
pub mod context;
pub mod rules;
pub mod skills;
pub mod tokens;
pub mod transforms;

pub use compaction::{CompactionConfig, CompactionResult, CompactionStrategy, Compactor};
pub use compaction_hook::{CompactionHook, ContextConfig, TokenSource};
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
