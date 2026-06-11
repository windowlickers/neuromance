//! GGUF model metadata extraction and tokenizer support.
//!
//! This module provides utilities for extracting metadata and tokenizer information
//! from GGUF model files without loading the full model tensors.
//!
//! ## Example
//!
//! ```no_run
//! use neuromance_context::gguf::GGUFModelInfo;
//!
//! # fn example() -> Result<(), neuromance_context::TokenCounterError> {
//! let info = GGUFModelInfo::from_file("model.gguf")?;
//! println!("Model: {:?}", info.model_name);
//! println!("Context length: {:?}", info.context_length);
//! println!("Vocab size: {:?}", info.vocab_size);
//! # Ok(())
//! # }
//! ```

use crate::TokenCounterError;
use candle_core::quantized::gguf_file;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::{debug, warn};

/// GGUF model metadata extracted without loading tensor data.
///
/// This struct contains common metadata fields found in GGUF files.
/// Fields are optional because not all models include all metadata keys.
#[derive(Debug, Clone)]
pub struct GGUFModelInfo {
    /// Model name (e.g., "unsloth/gpt-oss-20b-GGUF")
    pub model_name: Option<String>,

    /// Maximum context length in tokens
    pub context_length: Option<u32>,

    /// Embedding dimension
    pub embedding_dim: Option<u32>,

    /// Vocabulary size
    pub vocab_size: Option<u32>,

    /// Beginning-of-sequence token ID
    pub bos_token_id: Option<u32>,

    /// End-of-sequence token ID
    pub eos_token_id: Option<u32>,

    /// Chat template (Jinja2 format)
    pub chat_template: Option<String>,

    /// RoPE frequency base
    pub rope_freq_base: Option<f32>,

    /// Number of layers/blocks
    pub num_layers: Option<u32>,

    /// Number of attention heads
    pub num_attention_heads: Option<u32>,

    /// Raw metadata for custom access
    pub raw_metadata: HashMap<String, MetadataValue>,
}

/// Simplified metadata value type for easier access.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// String value
    String(String),
    /// Unsigned 32-bit integer
    U32(u32),
    /// Unsigned 64-bit integer
    U64(u64),
    /// Signed 32-bit integer
    I32(i32),
    /// Signed 64-bit integer
    I64(i64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// Boolean value
    Bool(bool),
    /// Array of values
    Array(Vec<MetadataValue>),
}

impl GGUFModelInfo {
    /// Extracts metadata from a GGUF file without loading tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or is not a valid GGUF file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenCounterError> {
        let mut file = File::open(path.as_ref()).map_err(|e| {
            TokenCounterError::GGUFRead(format!("Failed to open {:?}: {e}", path.as_ref()))
        })?;

        let content = gguf_file::Content::read(&mut file).map_err(|e| {
            TokenCounterError::GGUFRead(format!("Failed to parse {:?}: {e}", path.as_ref()))
        })?;

        debug!(
            "Loaded GGUF metadata with {} entries",
            content.metadata.len()
        );

        // TODO:
        //
        // Convert raw metadata to our simplified format
        let raw_metadata = Self::convert_metadata(&content);

        Ok(Self {
            model_name: Self::get_string(&content, "general.name")
                .or_else(|| Self::get_string(&content, "general.basename")),

            // TODO
            // Could probably look for second layer xyz.context_length
            // Or is llama, qwen2, phi2 a fixed list of N we can build?
            context_length: Self::get_u32(&content, "llama.context_length")
                .or_else(|| Self::get_u32(&content, "qwen2.context_length"))
                .or_else(|| Self::get_u32(&content, "phi2.context_length")),

            // TODO
            // Same thing as above for embedding_length
            embedding_dim: Self::get_u32(&content, "llama.embedding_length")
                .or_else(|| Self::get_u32(&content, "qwen2.embedding_length")),

            vocab_size: Self::get_u32(&content, "tokenizer.ggml.vocab_size"),

            bos_token_id: Self::get_u32(&content, "tokenizer.ggml.bos_token_id"),

            eos_token_id: Self::get_u32(&content, "tokenizer.ggml.eos_token_id"),

            chat_template: Self::get_string(&content, "tokenizer.chat_template")
                .or_else(|| Self::get_string(&content, "chat_template")),

            // TODO
            // Same thing as above for rope
            rope_freq_base: Self::get_f32(&content, "llama.rope.freq_base")
                .or_else(|| Self::get_f32(&content, "qwen2.rope.freq_base")),

            // TODO
            // Same thing as above for num_layers
            num_layers: Self::get_u32(&content, "llama.block_count")
                .or_else(|| Self::get_u32(&content, "qwen2.block_count")),

            // TODO
            // Same thing as above for attention
            num_attention_heads: Self::get_u32(&content, "llama.attention.head_count")
                .or_else(|| Self::get_u32(&content, "qwen2.attention.head_count")),

            raw_metadata,
        })
    }

    /// Extracts tokenizer vocabulary from GGUF metadata.
    ///
    /// Attempt to build a tokenizer from the vocabulary embedded in the GGUF file.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file doesn't contain tokenizer vocabulary data
    /// or if the tokenizer cannot be constructed.
    pub fn extract_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer, TokenCounterError> {
        let mut file = File::open(path.as_ref()).map_err(|e| {
            TokenCounterError::GGUFRead(format!("Failed to open {:?}: {e}", path.as_ref()))
        })?;

        let content = gguf_file::Content::read(&mut file).map_err(|e| {
            TokenCounterError::GGUFRead(format!("Failed to parse {:?}: {e}", path.as_ref()))
        })?;

        // Extract vocabulary tokens
        let tokens = Self::extract_vocab_tokens(&content)?;
        debug!("Extracted {} tokens from GGUF", tokens.len());

        // Extract merges if present (for BPE tokenizers)
        let merges = Self::extract_merges(&content);

        // Determine tokenizer model type
        let model_type =
            Self::get_string(&content, "tokenizer.ggml.model").unwrap_or_else(|| "bpe".to_string());

        debug!("Tokenizer model type: {}", model_type);

        // Only "bpe"/"gpt2" are byte-level BPE. Other types (notably "llama",
        // which is SentencePiece-based) have no dedicated builder yet, so they
        // fall back to BPE and produce approximate token counts.
        if !matches!(model_type.as_str(), "bpe" | "gpt2") {
            warn!(
                "Tokenizer model type '{}' is not byte-level BPE; falling back to a BPE \
                 tokenizer, so token counts will be approximate",
                model_type
            );
        }

        Self::build_bpe_tokenizer(tokens, merges)
    }

    /// Prints a human-readable summary of the model metadata.
    pub fn print_summary(&self) {
        println!("GGUF Model Information:");
        println!("=======================");

        if let Some(name) = &self.model_name {
            println!("Model: {}", name);
        }
        if let Some(ctx) = self.context_length {
            println!("Context length: {} tokens", ctx);
        }
        if let Some(dim) = self.embedding_dim {
            println!("Embedding dimension: {}", dim);
        }
        if let Some(vocab) = self.vocab_size {
            println!("Vocabulary size: {}", vocab);
        }
        if let Some(layers) = self.num_layers {
            println!("Number of layers: {}", layers);
        }
        if let Some(heads) = self.num_attention_heads {
            println!("Attention heads: {}", heads);
        }
        if let Some(bos) = self.bos_token_id {
            println!("BOS token ID: {}", bos);
        }
        if let Some(eos) = self.eos_token_id {
            println!("EOS token ID: {}", eos);
        }
        if let Some(rope) = self.rope_freq_base {
            println!("RoPE frequency base: {}", rope);
        }
        if self.chat_template.is_some() {
            println!("Has chat template: yes");
        }
    }

    // Helper methods for extracting typed values from GGUF metadata
    fn get_string(content: &gguf_file::Content, key: &str) -> Option<String> {
        content
            .metadata
            .get(key)?
            .to_string()
            .ok()
            .map(|s| s.to_string())
    }

    fn get_u32(content: &gguf_file::Content, key: &str) -> Option<u32> {
        content.metadata.get(key)?.to_u32().ok()
    }

    fn get_f32(content: &gguf_file::Content, key: &str) -> Option<f32> {
        content.metadata.get(key)?.to_f32().ok()
    }

    fn convert_metadata(content: &gguf_file::Content) -> HashMap<String, MetadataValue> {
        content
            .metadata
            .iter()
            .filter_map(|(k, v)| {
                let value = match v {
                    // TODO could we just use gguf_file::Value in our MetadataValue?
                    gguf_file::Value::String(s) => MetadataValue::String(s.to_string()),
                    gguf_file::Value::U8(n) => MetadataValue::U32(*n as u32),
                    gguf_file::Value::U16(n) => MetadataValue::U32(*n as u32),
                    gguf_file::Value::U32(n) => MetadataValue::U32(*n),
                    gguf_file::Value::U64(n) => MetadataValue::U64(*n),
                    gguf_file::Value::I8(n) => MetadataValue::I32(*n as i32),
                    gguf_file::Value::I16(n) => MetadataValue::I32(*n as i32),
                    gguf_file::Value::I32(n) => MetadataValue::I32(*n),
                    gguf_file::Value::I64(n) => MetadataValue::I64(*n),
                    gguf_file::Value::F32(f) => MetadataValue::F32(*f),
                    gguf_file::Value::F64(f) => MetadataValue::F64(*f),
                    gguf_file::Value::Bool(b) => MetadataValue::Bool(*b),
                    gguf_file::Value::Array(_) => return None, // Skip arrays for now
                };
                Some((k.clone(), value))
            })
            .collect()
    }

    fn extract_vocab_tokens(
        content: &gguf_file::Content,
    ) -> Result<Vec<String>, TokenCounterError> {
        let tokens_value = content
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| {
                TokenCounterError::GGUFTokenizerExtraction(
                    "No tokenizer.ggml.tokens in GGUF metadata".to_string(),
                )
            })?;

        if let gguf_file::Value::Array(arr) = tokens_value {
            arr.iter()
                .map(|v| {
                    if let gguf_file::Value::String(s) = v {
                        Ok(s.to_string())
                    } else {
                        Err(TokenCounterError::GGUFTokenizerExtraction(format!(
                            "Expected string in tokens array, \
                                 got {:?}",
                            v
                        )))
                    }
                })
                .collect()
        } else {
            Err(TokenCounterError::GGUFTokenizerExtraction(
                "tokenizer.ggml.tokens is not an array".to_string(),
            ))
        }
    }

    fn extract_merges(content: &gguf_file::Content) -> Vec<(String, String)> {
        // Try to extract BPE merges if present
        if let Some(gguf_file::Value::Array(arr)) = content.metadata.get("tokenizer.ggml.merges") {
            arr.iter()
                .filter_map(|v| {
                    if let gguf_file::Value::String(s) = v {
                        // Merges are typically in format "token1 token2"
                        let parts: Vec<&str> = s.split_whitespace().collect();
                        if parts.len() == 2 {
                            return Some((parts[0].to_string(), parts[1].to_string()));
                        }
                    }
                    None
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    fn build_bpe_tokenizer(
        tokens: Vec<String>,
        merges: Vec<(String, String)>,
    ) -> Result<Tokenizer, TokenCounterError> {
        use tokenizers::models::bpe::BpeBuilder;

        let vocab: ahash::AHashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(idx, token)| (token.clone(), idx as u32))
            .collect();

        let bpe = BpeBuilder::new()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| {
                TokenCounterError::GGUFTokenizerExtraction(format!(
                    "Failed to build BPE tokenizer from GGUF \
                     vocabulary: {e}"
                ))
            })?;

        Ok(Tokenizer::new(bpe))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires a GGUF file
    fn test_gguf_metadata_extraction() {
        // This test requires a GGUF file to be present
        // Run with: cargo test test_gguf_metadata_extraction -- --ignored
        let info = GGUFModelInfo::from_file("test.gguf").expect("Failed to extract GGUF metadata");

        info.print_summary();
        assert!(info.vocab_size.is_some());
    }

    #[test]
    #[ignore] // Requires a GGUF file
    fn test_gguf_tokenizer_extraction() {
        let tokenizer =
            GGUFModelInfo::extract_tokenizer("test.gguf").expect("Failed to extract tokenizer");

        let encoding = tokenizer
            .encode("Hello, world!", false)
            .expect("Failed to encode text");

        assert!(!encoding.get_ids().is_empty());
    }
}
