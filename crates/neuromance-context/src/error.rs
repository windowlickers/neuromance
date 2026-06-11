//! Error types for the neuromance-context crate.

use thiserror::Error;

/// Errors that can occur during context management operations.
#[derive(Error, Debug)]
pub enum TokenCounterError {
    /// Could not determine a cache directory for tokenizer files.
    #[error("{0}")]
    CacheDir(String),

    /// A filesystem operation on a cache or tokenizer file failed.
    #[error("{context}")]
    Io {
        /// Which operation failed (includes the path when known).
        context: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The tokenizers library failed to load, encode, or build a tokenizer.
    #[error("{context}")]
    Tokenizer {
        /// Which tokenizer operation failed.
        context: String,
        /// The underlying tokenizers error.
        #[source]
        source: tokenizers::Error,
    },

    /// Failed to download from Hugging Face.
    #[error("Failed to download from Hugging Face")]
    HuggingFaceDownload(#[from] hf_hub::api::tokio::ApiError),

    /// Failed to open or read a GGUF file.
    #[cfg(feature = "gguf")]
    #[error("Failed to read GGUF file {path}")]
    GgufReadIo {
        /// Path to the GGUF file.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to parse the contents of a GGUF file.
    #[cfg(feature = "gguf")]
    #[error("Failed to parse GGUF file {path}")]
    GgufParse {
        /// Path to the GGUF file.
        path: String,
        /// The underlying parse error.
        #[source]
        source: candle_core::Error,
    },

    /// Failed to extract tokenizer vocabulary from GGUF metadata.
    #[cfg(feature = "gguf")]
    #[error("Failed to extract tokenizer from GGUF: {0}")]
    GgufTokenizerExtraction(String),

    /// Chat template missing, or a template operation failed.
    #[error("Chat template error: {0}")]
    Template(String),

    /// Invalid regex pattern.
    #[error("Invalid regex pattern")]
    Regex(#[from] regex::Error),

    /// Token range out of bounds or invalid.
    #[error("Token range error: {0}")]
    TokenRange(String),

    /// Compaction failed (LLM call or conversation rebuild).
    #[error("Compaction failed: {0}")]
    Compaction(String),

    /// Rebuilding the compacted conversation violated a conversation invariant.
    #[error("Failed to rebuild compacted conversation")]
    ConversationRebuild(#[from] anyhow::Error),
}
