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
use minijinja::Environment;
use neuromance_common::{Conversation, Message, MessageRole};
use regex::Regex;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tracing::{debug, error};

mod error;
pub use error::TokenCounterError;

pub mod compaction;
pub mod context;
#[cfg(feature = "gguf")]
pub mod gguf;
mod jinja_compat;
pub mod metadata;
pub mod state;
pub mod transforms;

pub use compaction::{CompactionConfig, CompactionResult, CompactionStrategy, Compactor};

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
    pub fn gpt_oss_20b() -> Self {
        Self {
            model_repo: "openai/gpt-oss-20b".to_string(),
            hf_token: None,
            local_tokenizer_path: None,
            cache_dir: None,
        }
    }

    /// Creates a configuration for the GPT-OSS-120B model.
    pub fn gpt_oss_120b() -> Self {
        Self {
            model_repo: "openai/gpt-oss-120b".to_string(),
            hf_token: None,
            local_tokenizer_path: None,
            cache_dir: None,
        }
    }

    /// Creates a configuration for Qwen/Qwen3-30B-A3B-Instruct-2507
    pub fn qwen3_30b_a3b_instruct() -> Self {
        Self {
            model_repo: "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string(),
            hf_token: None,
            local_tokenizer_path: None,
            cache_dir: None,
        }
    }

    /// Creates a configuration for a custom model repository.
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
    pub fn with_hf_token(mut self, token: impl Into<String>) -> Self {
        self.hf_token = Some(SecretString::new(token.into().into()));
        self
    }

    /// Sets a local path to a tokenizer.json file.
    pub fn with_local_tokenizer(mut self, path: impl Into<PathBuf>) -> Self {
        self.local_tokenizer_path = Some(path.into());
        self
    }

    /// Sets a custom cache directory for downloaded tokenizers.
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
            TokenCounterError::TokenizerLoad("Failed to determine cache directory".to_string())
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

/// Information about a single token including its position in the text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    /// The index of this token in the sequence (0-based)
    pub index: usize,

    /// The string representation of this token
    pub token: String,

    /// The token ID from the tokenizer vocabulary
    pub token_id: u32,

    /// Starting character position in the original text
    pub char_start: usize,

    /// Ending character position in the original text
    pub char_end: usize,
}

/// Tokenized text with full position mapping information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedText {
    /// The original text
    pub text: String,

    /// Information about each token
    pub tokens: Vec<TokenInfo>,
}

impl TokenizedText {
    /// Returns the token at the given index.
    pub fn get_token(&self, index: usize) -> Option<&TokenInfo> {
        self.tokens.get(index)
    }

    /// Returns the token that contains the given character position.
    pub fn token_at_char_position(&self, char_pos: usize) -> Option<&TokenInfo> {
        self.tokens
            .iter()
            .find(|t| t.char_start <= char_pos && char_pos < t.char_end)
    }

    /// Returns the total number of tokens.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Returns the character range for a token range.
    pub fn char_range_for_tokens(
        &self,
        start_token: usize,
        end_token: usize,
    ) -> Option<(usize, usize)> {
        if start_token >= self.tokens.len()
            || end_token > self.tokens.len()
            || start_token >= end_token
        {
            return None;
        }

        Some((
            self.tokens[start_token].char_start,
            self.tokens[end_token - 1].char_end,
        ))
    }
}

/// A search match result with both character and token position information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMatch {
    /// The matched text
    pub matched_text: String,

    /// Starting character position in the text
    pub char_start: usize,

    /// Ending character position in the text
    pub char_end: usize,

    /// Starting token index (if the match starts within a token boundary)
    pub token_start: Option<usize>,

    /// Ending token index (if the match ends within a token boundary)
    pub token_end: Option<usize>,
}

impl SearchMatch {
    /// Returns the token range if both start and end are available.
    pub fn token_range(&self) -> Option<(usize, usize)> {
        match (self.token_start, self.token_end) {
            (Some(start), Some(end)) => Some((start, end + 1)),
            _ => None,
        }
    }

    /// Returns the character length of the match.
    pub fn char_length(&self) -> usize {
        self.char_end - self.char_start
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
        Tokenizer::from_file(path).map_err(|e| TokenCounterError::TokenizerLoad(e.to_string()))
    }

    /// Downloads and loads a tokenizer from Hugging Face.
    ///
    /// This method checks the cache first and only downloads if the tokenizer is not cached.
    async fn download_and_load_tokenizer(
        config: &ModelConfig,
    ) -> Result<Tokenizer, TokenCounterError> {
        let cached_path = config.get_cached_tokenizer_path()?;

        if cached_path.exists() {
            return Tokenizer::from_file(&cached_path)
                .map_err(|e| TokenCounterError::TokenizerLoad(e.to_string()));
        }

        let mut api_builder = ApiBuilder::new();
        if let Some(token) = &config.hf_token {
            api_builder = api_builder.with_token(Some(token.expose_secret().to_string()));
        }
        let api = api_builder.build().map_err(|e| {
            TokenCounterError::HuggingFaceDownload(format!("Failed to build HF API client: {e}"))
        })?;

        let repo = api.repo(Repo::new(config.model_repo.clone(), RepoType::Model));

        let tokenizer_path = repo.get("tokenizer.json").await.map_err(|e| {
            TokenCounterError::HuggingFaceDownload(format!(
                "Failed to download tokenizer.json: {e}"
            ))
        })?;

        if let Some(parent) = cached_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                TokenCounterError::TokenizerLoad(format!("Failed to create cache directory: {e}"))
            })?;
        }

        fs::copy(&tokenizer_path, &cached_path).map_err(|e| {
            TokenCounterError::TokenizerLoad(format!("Failed to cache tokenizer file: {e}"))
        })?;

        Tokenizer::from_file(&cached_path)
            .map_err(|e| TokenCounterError::TokenizerLoad(e.to_string()))
    }

    /// Loads the chat template for a model.
    ///
    /// Checks in order:
    /// 1. GGUF metadata (if using a GGUF file)
    /// 2. Embedded in tokenizer.json
    /// 3. Separate chat_template.jinja file from HuggingFace
    async fn load_chat_template(config: &ModelConfig, tokenizer: &Tokenizer) -> Option<String> {
        #[cfg(feature = "gguf")]
        if let Some(template) = Self::extract_gguf_template(config) {
            return Some(template);
        }

        if let Some(template) = Self::extract_tokenizer_template(tokenizer) {
            return Some(template);
        }

        debug!(
            "No chat template found in tokenizer.json, attempting to download chat_template.jinja"
        );

        let template_path = match config.get_cached_chat_template_path() {
            Ok(path) => path,
            Err(e) => {
                error!("Failed to get cache path for chat template: {}", e);
                return None;
            }
        };

        debug!("Chat template cache path: {:?}", template_path);

        if template_path.exists() {
            debug!("Loading chat template from cache");
            return fs::read_to_string(&template_path).ok();
        }

        Self::download_chat_template(config, &template_path).await
    }

    /// Extracts a chat template embedded in a GGUF file's metadata, if any.
    #[cfg(feature = "gguf")]
    fn extract_gguf_template(config: &ModelConfig) -> Option<String> {
        let path = config.local_tokenizer_path.as_ref()?;
        if path.extension().and_then(|s| s.to_str()) != Some("gguf") {
            return None;
        }

        let info = gguf::GGUFModelInfo::from_file(path).ok()?;
        let template = info.chat_template?;
        debug!("Found chat template in GGUF metadata");
        Some(template)
    }

    /// Extracts a chat template embedded in a tokenizer.json, if any.
    fn extract_tokenizer_template(tokenizer: &Tokenizer) -> Option<String> {
        let tokenizer_json = serde_json::to_value(tokenizer).ok()?;
        let template_str = tokenizer_json.get("chat_template")?.as_str()?;
        debug!("Found chat template embedded in tokenizer.json");
        Some(template_str.to_string())
    }

    /// Downloads `chat_template.jinja` from HuggingFace and caches it locally.
    ///
    /// Returns the template content on success, or `None` if the file is
    /// absent or the request fails. Caching failures are logged but do not
    /// prevent the content from being returned.
    async fn download_chat_template(config: &ModelConfig, template_path: &Path) -> Option<String> {
        // Use the /raw/ endpoint which serves the file directly without redirects.
        let url = format!(
            "https://huggingface.co/{}/raw/main/chat_template.jinja",
            config.model_repo
        );
        debug!("Attempting to download chat template from: {}", url);

        let client = match reqwest::Client::builder()
            .user_agent(concat!("neuromance-context/", env!("CARGO_PKG_VERSION")))
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                error!("Failed to build HTTP client: {}", e);
                return None;
            }
        };

        let mut request = client.get(&url);
        if let Some(token) = &config.hf_token {
            request = request.header("Authorization", format!("Bearer {}", token.expose_secret()));
        }

        let response = match request.send().await {
            Ok(response) => response,
            Err(e) => {
                debug!(
                    "Could not download chat template: {} (file may not exist in this model)",
                    e
                );
                return None;
            }
        };

        if response.status().as_u16() == 404 {
            debug!("Chat template file not found in repository (this is normal for some models)");
            return None;
        }
        if !response.status().is_success() {
            debug!(
                "Failed to download chat template: HTTP {}",
                response.status()
            );
            return None;
        }

        let content = match response.text().await {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read response body: {}", e);
                return None;
            }
        };
        debug!(
            "Successfully downloaded chat_template.jinja ({} bytes)",
            content.len()
        );

        Self::cache_chat_template(template_path, &content);
        Some(content)
    }

    /// Writes a downloaded chat template to the cache, logging any failure.
    fn cache_chat_template(template_path: &Path, content: &str) {
        if let Some(parent) = template_path.parent()
            && let Err(e) = fs::create_dir_all(parent)
        {
            error!("Failed to create cache directory: {}", e);
            return;
        }

        if let Err(e) = fs::write(template_path, content) {
            error!("Failed to cache chat template: {}", e);
        } else {
            debug!("Cached chat template to: {:?}", template_path);
        }
    }

    /// Counts the number of tokens in a string.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn count_tokens(&self, text: &str) -> Result<usize, TokenCounterError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| TokenCounterError::Tokenization(e.to_string()))?;

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
        let mut total = 0;

        for message in conversation.get_messages() {
            total += self.count_message_tokens(message)?;
        }

        Ok(total)
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

    /// Tokenizes text and returns detailed token information with position mappings.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn tokenize_with_positions(&self, text: &str) -> Result<TokenizedText, TokenCounterError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| TokenCounterError::Tokenization(e.to_string()))?;

        let tokens = encoding
            .get_tokens()
            .iter()
            .enumerate()
            .map(|(idx, token)| {
                let offsets = encoding.get_offsets()[idx];
                TokenInfo {
                    index: idx,
                    token: token.clone(),
                    token_id: encoding.get_ids()[idx],
                    char_start: offsets.0,
                    char_end: offsets.1,
                }
            })
            .collect();

        Ok(TokenizedText {
            text: text.to_string(),
            tokens,
        })
    }

    /// Searches for a pattern in text and returns matches with their token positions.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or regex compilation fails.
    pub fn search_with_token_positions(
        &self,
        text: &str,
        pattern: &str,
    ) -> Result<Vec<SearchMatch>, TokenCounterError> {
        let tokenized = self.tokenize_with_positions(text)?;
        let regex = Regex::new(pattern)?;

        let mut matches = Vec::new();

        for match_result in regex.find_iter(text) {
            let char_start = match_result.start();
            let char_end = match_result.end();
            let matched_text = match_result.as_str().to_string();

            // Find token range that covers this match
            let token_start = tokenized
                .tokens
                .iter()
                .find(|t| t.char_start <= char_start && char_start < t.char_end)
                .map(|t| t.index);

            let token_end = tokenized
                .tokens
                .iter()
                .rev()
                .find(|t| t.char_start < char_end && char_end <= t.char_end)
                .map(|t| t.index);

            matches.push(SearchMatch {
                matched_text,
                char_start,
                char_end,
                token_start,
                token_end,
            });
        }

        Ok(matches)
    }

    /// Extracts the chat template from the tokenizer configuration.
    ///
    /// Returns None if no chat template is defined in the tokenizer.
    pub fn get_chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Formats a conversation using the tokenizer's chat template.
    ///
    /// This applies the model's chat template to format messages exactly as they would
    /// appear when sent to the model, enabling accurate token counting.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tokenizer has no chat template defined
    /// - The template rendering fails
    pub fn format_conversation_with_template(
        &self,
        conversation: &Conversation,
    ) -> Result<String, TokenCounterError> {
        let template_str = self.get_chat_template().ok_or_else(|| {
            TokenCounterError::Template("No chat template found in tokenizer".to_string())
        })?;

        let mut env = Environment::new();
        jinja_compat::configure_environment(&mut env);

        env.add_function("strftime_now", |format: String| -> String {
            use chrono::Local;
            let now = Local::now();
            now.format(&format).to_string()
        });

        env.add_template("chat", template_str).map_err(|e| {
            TokenCounterError::Template(format!("Failed to add chat template: {e}"))
        })?;

        let messages: Vec<serde_json::Value> = conversation
            .get_messages()
            .iter()
            .map(|msg| {
                let role_str = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                    _ => "user",
                };

                let mut message_obj = serde_json::json!({
                    "role": role_str,
                    "content": msg.content.clone(),
                });

                if let Some(tool_call_id) = &msg.tool_call_id {
                    message_obj["tool_call_id"] = serde_json::json!(tool_call_id);
                }
                if let Some(name) = &msg.name {
                    message_obj["name"] = serde_json::json!(name);
                }

                if !msg.tool_calls.is_empty() {
                    let tool_calls: Vec<serde_json::Value> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": tc.call_type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments_json(),
                                }
                            })
                        })
                        .collect();
                    message_obj["tool_calls"] = serde_json::json!(tool_calls);
                }

                message_obj
            })
            .collect();

        let tmpl = env
            .get_template("chat")
            .map_err(|e| TokenCounterError::Template(e.to_string()))?;

        let bos_token = self.get_special_token("bos_token").unwrap_or_default();
        let eos_token = self.get_special_token("eos_token").unwrap_or_default();

        let context = serde_json::json!({
            "messages": messages,
            "add_generation_prompt": false,
            "tools": serde_json::Value::Array(vec![]),
            "builtin_tools": serde_json::Value::Array(vec![]),
            "bos_token": bos_token,
            "eos_token": eos_token,
        });

        let rendered = tmpl.render(&context).map_err(|e| {
            TokenCounterError::Template(format!("Failed to render chat template: {e}"))
        })?;

        Ok(rendered)
    }

    /// Get a special token from the tokenizer by name.
    ///
    /// Common special tokens include: bos_token, eos_token, pad_token, unk_token
    fn get_special_token(&self, name: &str) -> Option<String> {
        // Try to get the token from the tokenizer's added tokens
        // This is a best-effort extraction since tokenizers store this differently
        let tokenizer = &self.tokenizer;

        // Map common names to token IDs we might know
        let token_id = match name {
            "bos_token" => tokenizer
                .token_to_id("<|begin_of_text|>")
                .or_else(|| tokenizer.token_to_id("<s>"))
                .or_else(|| tokenizer.token_to_id("<bos>"))
                .or_else(|| tokenizer.token_to_id("[BOS]")),
            "eos_token" => tokenizer
                .token_to_id("<|end_of_text|>")
                .or_else(|| tokenizer.token_to_id("</s>"))
                .or_else(|| tokenizer.token_to_id("<eos>"))
                .or_else(|| tokenizer.token_to_id("[EOS]"))
                .or_else(|| tokenizer.token_to_id("<|eot_id|>")),
            "pad_token" => tokenizer
                .token_to_id("<pad>")
                .or_else(|| tokenizer.token_to_id("[PAD]")),
            "unk_token" => tokenizer
                .token_to_id("<unk>")
                .or_else(|| tokenizer.token_to_id("[UNK]")),
            _ => None,
        };

        // If we found an ID, convert it back to the token string
        token_id.and_then(|id| tokenizer.id_to_token(id).map(|s| s.to_string()))
    }

    /// Counts tokens in a conversation using the chat template.
    ///
    /// This is the most accurate way to count tokens as it formats the conversation
    /// exactly as the model would see it, including all special tokens and formatting.
    ///
    /// # Errors
    ///
    /// Returns an error if template formatting or tokenization fails.
    pub fn count_conversation_tokens_with_template(
        &self,
        conversation: &Conversation,
    ) -> Result<usize, TokenCounterError> {
        let formatted = self.format_conversation_with_template(conversation)?;
        self.count_tokens(&formatted)
    }

    /// Extracts a token range from text and returns the corresponding substring.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails or token range is invalid.
    pub fn extract_token_range(
        &self,
        text: &str,
        start_token: usize,
        end_token: usize,
    ) -> Result<String, TokenCounterError> {
        let tokenized = self.tokenize_with_positions(text)?;

        if start_token >= tokenized.tokens.len() || end_token > tokenized.tokens.len() {
            return Err(TokenCounterError::TokenRange(format!(
                "{}-{} out of bounds (text has {} tokens)",
                start_token,
                end_token,
                tokenized.tokens.len()
            )));
        }

        if start_token >= end_token {
            return Err(TokenCounterError::TokenRange(format!(
                "start ({}) must be less than end ({})",
                start_token, end_token
            )));
        }

        let char_start = tokenized.tokens[start_token].char_start;
        let char_end = tokenized.tokens[end_token - 1].char_end;

        Ok(text[char_start..char_end].to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    /// Creates a simple test tokenizer that doesn't require network access.
    fn create_test_tokenizer() -> Tokenizer {
        // Construct a minimal WordLevel tokenizer from JSON
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
    fn test_tokenize_with_positions_offline() {
        let tokenizer = create_test_tokenizer();
        let counter = TokenCounter::from_tokenizer(tokenizer);

        let result = counter.tokenize_with_positions("hello world").unwrap();
        assert!(!result.tokens.is_empty());
        assert_eq!(result.text, "hello world");
        assert_eq!(result.token_count(), result.tokens.len());
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

    #[test]
    fn test_tokenized_text_helpers() {
        let tokenizer = create_test_tokenizer();
        let counter = TokenCounter::from_tokenizer(tokenizer);

        let result = counter.tokenize_with_positions("hello world").unwrap();

        // Test get_token
        assert!(result.get_token(0).is_some());
        assert!(result.get_token(999).is_none());
    }

    #[test]
    fn test_token_at_char_position_boundaries() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let tokenized = counter.tokenize_with_positions("hello world").unwrap();

        // A position inside the first token resolves to that token.
        let first_start = tokenized.tokens[0].char_start;
        assert!(tokenized.token_at_char_position(first_start).is_some());

        // A position past the end of the text maps to no token.
        assert!(
            tokenized
                .token_at_char_position("hello world".len())
                .is_none()
        );
    }

    #[test]
    fn test_char_range_for_tokens_invalid() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let tokenized = counter.tokenize_with_positions("hello world").unwrap();
        let n = tokenized.token_count();

        assert!(
            tokenized.char_range_for_tokens(1, 1).is_none(),
            "start == end"
        );
        assert!(
            tokenized.char_range_for_tokens(0, n + 1).is_none(),
            "end past len"
        );
        assert!(
            tokenized.char_range_for_tokens(n, n).is_none(),
            "start at len"
        );
        assert!(
            tokenized.char_range_for_tokens(0, n).is_some(),
            "valid range"
        );
    }

    #[test]
    fn test_search_with_invalid_regex() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let err = counter
            .search_with_token_positions("hello world", "[invalid")
            .unwrap_err();
        assert!(matches!(err, TokenCounterError::Regex(_)));
    }

    #[test]
    fn test_format_conversation_without_template_errors() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let mut conv = Conversation::new();
        conv.add_message(conv.user_message("hello")).unwrap();

        let err = counter
            .format_conversation_with_template(&conv)
            .unwrap_err();
        assert!(matches!(err, TokenCounterError::Template(_)));
    }

    #[test]
    fn test_extract_token_range_errors() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let text = "hello world";
        let n = counter.tokenize_with_positions(text).unwrap().token_count();

        let out_of_bounds = counter.extract_token_range(text, 0, n + 1).unwrap_err();
        assert!(matches!(out_of_bounds, TokenCounterError::TokenRange(_)));

        let inverted = counter.extract_token_range(text, 1, 1).unwrap_err();
        assert!(matches!(inverted, TokenCounterError::TokenRange(_)));

        // A valid range round-trips back to the original text.
        let extracted = counter.extract_token_range(text, 0, n).unwrap();
        assert_eq!(extracted, text);
    }

    #[cfg(feature = "online-tests")]
    #[tokio::test]
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
