//! Error types for the neuromance-context crate.

use thiserror::Error;

/// Errors that can occur during context management operations.
#[derive(Error, Debug)]
pub enum TokenCounterError {
    /// Failed to load the tokenizer
    #[error("Failed to load tokenizer: {0}")]
    TokenizerLoad(String),

    /// Failed to tokenize text
    #[error("Failed to tokenize text: {0}")]
    Tokenization(String),

    /// Failed to download from Hugging Face
    #[error("Failed to download from Hugging Face: {0}")]
    HuggingFaceDownload(String),

    /// Failed to read GGUF file
    #[cfg(feature = "gguf")]
    #[error("Failed to read GGUF file: {0}")]
    GGUFRead(String),

    /// Failed to extract tokenizer from GGUF
    #[cfg(feature = "gguf")]
    #[error("Failed to extract tokenizer from GGUF: {0}")]
    GGUFTokenizerExtraction(String),

    /// Chat template error (missing or failed to render)
    #[error("Chat template error: {0}")]
    Template(String),

    /// Invalid regex pattern
    #[error("Invalid regex pattern: {0}")]
    Regex(#[from] regex::Error),

    /// Token range out of bounds or invalid
    #[error("Token range error: {0}")]
    TokenRange(String),

    /// Compaction failed (LLM call or summary generation)
    #[error("Compaction failed: {0}")]
    Compaction(String),

    /// Rebuilding the compacted conversation violated a conversation invariant
    #[error("Failed to rebuild compacted conversation: {0}")]
    ConversationRebuild(String),
}
