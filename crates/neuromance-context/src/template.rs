//! Chat-template loading, rendering, and template-based token counting.

use std::fs;
use std::path::Path;

use minijinja::Environment;
use neuromance_common::{Conversation, MessageRole};
use secrecy::ExposeSecret;
use tokenizers::Tokenizer;
use tracing::{debug, error};

use crate::error::TokenCounterError;
use crate::{ModelConfig, TokenCounter, jinja_compat};

impl TokenCounter {
    /// Loads the chat template for a model.
    ///
    /// Checks in order:
    /// 1. GGUF metadata (if using a GGUF file)
    /// 2. Embedded in tokenizer.json
    /// 3. Separate chat_template.jinja file from HuggingFace
    pub(crate) async fn load_chat_template(
        config: &ModelConfig,
        tokenizer: &Tokenizer,
    ) -> Option<String> {
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

        let info = crate::gguf::GGUFModelInfo::from_file(path).ok()?;
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

        env.add_template("chat", template_str)
            .map_err(|e| TokenCounterError::TemplateRender {
                model: self.config.model_repo.clone(),
                source: e,
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
            .map_err(|e| TokenCounterError::TemplateRender {
                model: self.config.model_repo.clone(),
                source: e,
            })?;

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

        let rendered = tmpl
            .render(&context)
            .map_err(|e| TokenCounterError::TemplateRender {
                model: self.config.model_repo.clone(),
                source: e,
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
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::create_test_tokenizer;

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
}
