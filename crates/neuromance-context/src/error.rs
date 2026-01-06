//! Error types for token counting operations.

use thiserror::Error;

/// Errors that can occur during token counting operations.
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
    #[error("Failed to read GGUF file: {0}")]
    GGUFRead(String),

    /// Failed to extract tokenizer from GGUF
    #[error("Failed to extract tokenizer from GGUF: {0}")]
    GGUFTokenizerExtraction(String),
}
