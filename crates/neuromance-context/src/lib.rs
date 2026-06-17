//! # neuromance-context
//!
//! Token counting and context management for LLM conversations using Candle.
//!
//! This crate provides functionality to calculate token counts for strings and conversations,
//! enabling better context window management when working with LLMs.
//!
//! ## Example
//!
//! ```no_run
//! use neuromance_context::{TokenCounter, ModelConfig};
//! use neuromance_common::Conversation;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Initialize the token counter with a model configuration
//! let config = ModelConfig::gpt_oss_20b()
//!     .with_hf_token(std::env::var("HF_TOKEN")?);
//! let counter = TokenCounter::new(config).await?;
//!
//! // Count tokens in a string
//! let text = "Hello, how many tokens is this?";
//! let count = counter.count_tokens(text)?;
//! println!("Token count: {}", count);
//!
//! // Count tokens in a conversation
//! let mut conv = Conversation::new();
//! conv.add_message(conv.user_message("What's the weather?"))?;
//! let conv_tokens = counter.count_conversation_tokens(&conv)?;
//! println!("Conversation tokens: {}", conv_tokens);
//! # Ok(())
//! # }
//! ```

use hf_hub::{Repo, RepoType, api::tokio::ApiBuilder};
use neuromance_common::{Conversation, Message};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tracing::debug;

mod error;
pub use error::TokenCounterError;

pub mod compaction;
pub mod compaction_hook;
pub mod context;
#[cfg(feature = "gguf")]
pub mod gguf;
mod jinja_compat;
pub mod metadata;
mod navigation;
pub mod rules;
pub mod skills;
pub mod state;
mod template;
pub mod transforms;

pub use compaction::{CompactionConfig, CompactionResult, CompactionStrategy, Compactor};
pub use compaction_hook::{CompactionHook, ContextConfig, TokenSource};
pub use context::Context;
pub use navigation::{SearchMatch, TokenInfo, TokenizedText};

/// Configuration for a specific model's tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// The Hugging Face model repository (e.g., "openai/gpt-oss-20b")
    pub model_repo: String,

    /// Optional Hugging Face API token for accessing gated models.
    ///
    /// Wrapped in `SecretString` to prevent accidental logging; skipped during
    /// serialization so the credential is never written to config output.
    #[serde(skip_serializing, default)]
    pub hf_token: Option<SecretString>,

    /// Optional local path to a tokenizer.json file (bypasses HF download)
    pub local_tokenizer_path: Option<PathBuf>,

    /// Custom cache directory for downloaded tokenizers (defaults to ~/.cache/neuromance/tokenizers)
    pub cache_dir: Option<PathBuf>,
}

impl ModelConfig {
    /// Creates a configuration for the GPT-OSS-20B model.
    #[must_use]
    pub fn gpt_oss_20b() -> Self {
        Self {
            model_repo: "openai/gpt-oss-20b".to_string(),
            hf_token: None,
            local_tokenizer_path: None,
            cache_dir: None,
        }
    }

    /// Creates a configuration for the GPT-OSS-120B model.
    #[must_use]
    pub fn gpt_oss_120b() -> Self {
        Self {
            model_repo: "openai/gpt-oss-120b".to_string(),
            hf_token: None,
            local_tokenizer_path: None,
            cache_dir: None,
        }
    }

    /// Creates a configuration for Qwen/Qwen3-30B-A3B-Instruct-2507
    #[must_use]
    pub fn qwen3_30b_a3b_instruct() -> Self {
        Self {
            model_repo: "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string(),
            hf_token: None,
            local_tokenizer_path: None,
            cache_dir: None,
        }
    }

    /// Creates a configuration for a custom model repository.
    #[must_use]
    pub fn custom(model_repo: impl Into<String>) -> Self {
        Self {
            model_repo: model_repo.into(),
            hf_token: None,
            local_tokenizer_path: None,
            cache_dir: None,
        }
    }

    /// Creates a configuration from a GGUF file.
    ///
    /// This extracts metadata from the GGUF file including model name, chat template,
    /// and other information. The GGUF file path is stored as the local tokenizer source.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file cannot be read or parsed.
    #[cfg(feature = "gguf")]
    pub fn from_gguf(gguf_path: impl Into<PathBuf>) -> Result<Self, TokenCounterError> {
        let path = gguf_path.into();
        let info = gguf::GGUFModelInfo::from_file(&path)?;

        Ok(Self {
            model_repo: info
                .model_name
                .clone()
                .unwrap_or_else(|| "local-gguf".to_string()),
            hf_token: None,
            local_tokenizer_path: Some(path),
            cache_dir: None,
        })
    }

    /// Sets the Hugging Face API token.
    #[must_use]
    pub fn with_hf_token(mut self, token: impl Into<String>) -> Self {
        self.hf_token = Some(SecretString::new(token.into().into()));
        self
    }

    /// Sets a local path to a tokenizer.json file.
    #[must_use]
    pub fn with_local_tokenizer(mut self, path: impl Into<PathBuf>) -> Self {
        self.local_tokenizer_path = Some(path.into());
        self
    }

    /// Sets a custom cache directory for downloaded tokenizers.
    #[must_use]
    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Returns the cache directory, defaulting to ~/.cache/neuromance/tokenizers if not set.
    fn get_cache_dir(&self) -> Result<PathBuf, TokenCounterError> {
        if let Some(cache_dir) = &self.cache_dir {
            return Ok(cache_dir.clone());
        }

        let cache_home = dirs::cache_dir().ok_or_else(|| {
            TokenCounterError::CacheDir("Failed to determine cache directory".to_string())
        })?;

        Ok(cache_home.join("neuromance").join("tokenizers"))
    }

    /// Returns the path to the cached tokenizer file for this model.
    fn get_cached_tokenizer_path(&self) -> Result<PathBuf, TokenCounterError> {
        let cache_dir = self.get_cache_dir()?;

        // Create subdirectory structure: cache_dir/repo/model/tokenizer.json
        // e.g., ~/.cache/neuromance/tokenizers/openai/gpt-oss-20b/tokenizer.json
        let repo_path = self.model_repo.replace('/', "-");
        let tokenizer_path = cache_dir.join(&repo_path).join("tokenizer.json");

        Ok(tokenizer_path)
    }

    /// Returns the path to the cached chat template file for this model.
    fn get_cached_chat_template_path(&self) -> Result<PathBuf, TokenCounterError> {
        let cache_dir = self.get_cache_dir()?;
        let repo_path = self.model_repo.replace('/', "-");
        let template_path = cache_dir.join(&repo_path).join("chat_template.jinja");

        Ok(template_path)
    }
}

/// Token counter for calculating token counts using Candle and Hugging Face tokenizers.
#[derive(Debug)]
pub struct TokenCounter {
    tokenizer: Tokenizer,
    config: ModelConfig,
    chat_template: Option<String>,
}

impl TokenCounter {
    /// Creates a token counter directly from a pre-loaded tokenizer.
    ///
    /// This is useful for testing or when you have already loaded a tokenizer
    /// through other means.
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            config: ModelConfig::custom("local"),
            chat_template: None,
        }
    }

    /// Creates a new token counter with the specified model configuration.
    ///
    /// This will download the tokenizer from Hugging Face if a local path is not provided.
    /// It also attempts to download a chat template file if available.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer cannot be downloaded or loaded.
    pub async fn new(config: ModelConfig) -> Result<Self, TokenCounterError> {
        let tokenizer = if let Some(local_path) = &config.local_tokenizer_path {
            Self::load_local_tokenizer(local_path)?
        } else {
            Self::download_and_load_tokenizer(&config).await?
        };

        // Try to load chat template (either from tokenizer or separate file)
        let chat_template = Self::load_chat_template(&config, &tokenizer).await;

        Ok(Self {
            tokenizer,
            config,
            chat_template,
        })
    }

    /// Loads a tokenizer from a local file path.
    ///
    /// If the `gguf` feature is enabled and the path ends with .gguf, attempts to
    /// extract the tokenizer from GGUF metadata. Otherwise, loads a standard
    /// tokenizer.json file.
    fn load_local_tokenizer(path: &Path) -> Result<Tokenizer, TokenCounterError> {
        #[cfg(feature = "gguf")]
        if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            debug!("Detected GGUF file, attempting to extract tokenizer");
            return gguf::GGUFModelInfo::extract_tokenizer(path);
        }

        debug!("Loading standard tokenizer.json");
        Tokenizer::from_file(path).map_err(|e| TokenCounterError::Tokenizer {
            context: format!("Failed to load tokenizer from {}", path.display()),
            source: e,
        })
    }

    /// Downloads and loads a tokenizer from Hugging Face.
    ///
    /// This method checks the cache first and only downloads if the tokenizer is not cached.
    async fn download_and_load_tokenizer(
        config: &ModelConfig,
    ) -> Result<Tokenizer, TokenCounterError> {
        let cached_path = config.get_cached_tokenizer_path()?;

        if cached_path.exists() {
            return Tokenizer::from_file(&cached_path).map_err(|e| TokenCounterError::Tokenizer {
                context: format!("Failed to load cached tokenizer {}", cached_path.display()),
                source: e,
            });
        }

        let mut api_builder = ApiBuilder::new();
        if let Some(token) = &config.hf_token {
            api_builder = api_builder.with_token(Some(token.expose_secret().to_string()));
        }
        let api = api_builder.build()?;

        let repo = api.repo(Repo::new(config.model_repo.clone(), RepoType::Model));

        let tokenizer_path = repo.get("tokenizer.json").await?;

        if let Some(parent) = cached_path.parent() {
            fs::create_dir_all(parent).map_err(|e| TokenCounterError::Io {
                context: format!("Failed to create cache directory {}", parent.display()),
                source: e,
            })?;
        }

        fs::copy(&tokenizer_path, &cached_path).map_err(|e| TokenCounterError::Io {
            context: format!(
                "Failed to cache tokenizer file to {}",
                cached_path.display()
            ),
            source: e,
        })?;

        Tokenizer::from_file(&cached_path).map_err(|e| TokenCounterError::Tokenizer {
            context: format!("Failed to load cached tokenizer {}", cached_path.display()),
            source: e,
        })
    }

    /// Counts the number of tokens in a string.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn count_tokens(&self, text: &str) -> Result<usize, TokenCounterError> {
        let encoding =
            self.tokenizer
                .encode(text, false)
                .map_err(|e| TokenCounterError::Tokenizer {
                    context: "Failed to tokenize text".to_string(),
                    source: e,
                })?;

        Ok(encoding.get_ids().len())
    }

    /// Counts the total number of tokens in a conversation.
    ///
    /// This iterates through all messages and sums their token counts.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails for any message.
    pub fn count_conversation_tokens(
        &self,
        conversation: &Conversation,
    ) -> Result<usize, TokenCounterError> {
        conversation
            .get_messages()
            .iter()
            .map(|message| self.count_message_tokens(message))
            .sum()
    }

    /// Counts the number of tokens in a single message.
    ///
    /// This includes the message content and accounts for role markers and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn count_message_tokens(&self, message: &Message) -> Result<usize, TokenCounterError> {
        // Count the main content
        let mut total = self.count_tokens(&message.content)?;

        // Add tokens for role (approximate overhead: role name + formatting)
        // Most chat formats add ~4 tokens per message for role markers
        total += 4;

        // Count tool call tokens if present
        for tool_call in &message.tool_calls {
            // Count function name
            total += self.count_tokens(&tool_call.function.name)?;

            // Count arguments
            total += self.count_tokens(tool_call.function.arguments_json())?;

            // Overhead for tool call structure (~5 tokens for formatting)
            total += 5;
        }

        // Count tool metadata if present
        if let Some(tool_call_id) = &message.tool_call_id {
            total += self.count_tokens(tool_call_id)?;
        }

        if let Some(name) = &message.name {
            total += self.count_tokens(name)?;
        }

        Ok(total)
    }

    /// Returns a reference to the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns a reference to the underlying tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::create_test_tokenizer;
    use neuromance_common::MessageRole;

    #[test]
    fn test_count_tokens_offline() {
        let tokenizer = create_test_tokenizer();
        let counter = TokenCounter::from_tokenizer(tokenizer);

        let count = counter.count_tokens("hello world").unwrap();
        assert!(count > 0, "Expected at least 1 token");
    }

    #[test]
    fn test_count_tokens_empty_string() {
        let tokenizer = create_test_tokenizer();
        let counter = TokenCounter::from_tokenizer(tokenizer);

        let count = counter.count_tokens("").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_conversation_token_counting_offline() {
        let tokenizer = create_test_tokenizer();
        let counter = TokenCounter::from_tokenizer(tokenizer);

        let mut conv = Conversation::new();
        conv.add_message(conv.user_message("hello world")).unwrap();
        conv.add_message(conv.assistant_message("hello")).unwrap();

        let count = counter.count_conversation_tokens(&conv).unwrap();
        // Each message has content tokens + 4 role overhead
        assert!(count > 0, "Expected positive token count");
    }

    #[test]
    fn test_message_token_counting_offline() {
        let tokenizer = create_test_tokenizer();
        let counter = TokenCounter::from_tokenizer(tokenizer);

        let msg = Message::new(uuid::Uuid::new_v4(), MessageRole::User, "hello world");
        let count = counter.count_message_tokens(&msg).unwrap();

        // Should be content tokens + 4 overhead
        let content_tokens = counter.count_tokens("hello world").unwrap();
        assert_eq!(count, content_tokens + 4);
    }

    #[test]
    fn test_model_config_constructors() {
        let config = ModelConfig::gpt_oss_20b();
        assert_eq!(config.model_repo, "openai/gpt-oss-20b");
        assert!(config.hf_token.is_none());

        let config = ModelConfig::custom("test/model").with_hf_token("token123");
        assert_eq!(config.model_repo, "test/model");
        assert_eq!(
            config.hf_token.as_ref().map(|t| t.expose_secret()),
            Some("token123")
        );
    }

    #[test]
    fn test_from_tokenizer_constructor() {
        let tokenizer = create_test_tokenizer();
        let counter = TokenCounter::from_tokenizer(tokenizer);
        assert_eq!(counter.config().model_repo, "local");
        assert!(counter.get_chat_template().is_none());
    }

    #[cfg(feature = "online-tests")]
    #[tokio::test]
    #[ignore = "requires HF_TOKEN and network access"]
    async fn test_token_counter_creation() {
        let token = std::env::var("HF_TOKEN").expect("HF_TOKEN not set");
        let config = ModelConfig::gpt_oss_20b().with_hf_token(token);
        let counter = TokenCounter::new(config)
            .await
            .expect("Failed to create counter");

        let count = counter
            .count_tokens("Hello, world!")
            .expect("Failed to count tokens");
        assert!(count > 0);
    }

    #[cfg(feature = "online-tests")]
    #[tokio::test]
    #[ignore = "requires HF_TOKEN and network access"]
    async fn test_conversation_token_counting() {
        let token = std::env::var("HF_TOKEN").expect("HF_TOKEN not set");
        let config = ModelConfig::gpt_oss_20b().with_hf_token(token);
        let counter = TokenCounter::new(config)
            .await
            .expect("Failed to create counter");

        let mut conv = Conversation::new();
        conv.add_message(conv.user_message("What's the weather?"))
            .expect("Failed to add message");
        conv.add_message(conv.assistant_message("It's sunny today!"))
            .expect("Failed to add message");

        let count = counter
            .count_conversation_tokens(&conv)
            .expect("Failed to count conversation tokens");
        assert!(count > 0);
    }
}
