//! # neuromance-client
//!
//! Client library for interacting with LLM inference providers.
//!
//! Provides a unified `LLMClient` trait for various LLM providers. Currently supports
//! Chat Completions-compatible APIs with tool/function calling and streaming.

// ClientError contains EventSourceError (~176 bytes), but this is acceptable
// for network-bound code where HTTP latency dwarfs any stack size concerns.
#![allow(clippy::result_large_err)]

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use reqwest_middleware::ClientWithMiddleware;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use reqwest_retry_after::RetryAfterMiddleware;

use neuromance_common::client::{ChatChunk, Provider, ProxyConfig, resolve_model_prefix};
use neuromance_common::{ChatRequest, ChatResponse, Config};
use secrecy::SecretString;

pub mod anthropic;
pub mod chat_completions;
pub mod embedding;
mod error;
pub(crate) mod message;
pub mod responses;
pub(crate) mod retry_logging;
pub(crate) mod streaming;
pub(crate) mod transport;

pub use anthropic::AnthropicClient;
pub use chat_completions::{ChatCompletionsClient, OpenAIEmbedding};
pub use embedding::{
    EmbeddingClient, EmbeddingConfig, EmbeddingInput, EmbeddingRequest, EmbeddingResponse,
};
pub use error::ClientError;
pub use responses::ResponsesClient;

/// Shared resources produced by client constructor logic.
///
/// All provider clients need the same set of HTTP plumbing: a retry-aware
/// middleware client, a raw streaming client, the resolved base URL, API key,
/// and optional proxy configuration. This struct bundles them so each provider
/// only needs to call [`build_client_resources`] instead of duplicating the
/// setup.
pub(crate) struct ClientResources {
    pub client: ClientWithMiddleware,
    pub streaming_client: reqwest::Client,
    pub api_key: Arc<SecretString>,
    pub base_url: String,
    pub config: Arc<Config>,
    pub proxy_config: Option<ProxyConfig>,
}

/// Build the shared HTTP client resources from a [`Config`].
///
/// Extracts the API key, resolves the base URL, builds the retry policy, and
/// constructs both the middleware-wrapped and raw reqwest clients.
///
/// When a [`ProxyConfig`] is set, the reqwest client is configured as an HTTP
/// forward proxy client: the base URL's scheme is rewritten to `http://` so
/// reqwest emits absolute-form requests in cleartext to the proxy (RFC 7230
/// §5.3.2), and the proxy is attached via [`reqwest::Proxy::http`]. The proxy
/// then terminates the connection, validates the sealed token, and originates
/// a fresh upstream connection (scheme controlled by the proxy / its token)
/// to the real provider. The original upstream authority and path are carried
/// in the request URL, so no `X-Target-Host` side-band header is needed.
///
/// # Errors
///
/// Returns `ClientError::ConfigurationError` if the API key is missing, the
/// base URL is unparseable, or the proxy URL cannot be parsed by reqwest.
pub(crate) fn build_client_resources(
    config: Config,
    default_base_url: &str,
) -> Result<ClientResources, ClientError> {
    let api_key = config
        .api_key
        .clone()
        .ok_or_else(|| ClientError::ConfigurationError("API key is required".to_string()))?;

    let original_url = config
        .base_url
        .clone()
        .unwrap_or_else(|| default_base_url.to_string());

    // In proxy mode, rewrite the base URL scheme to plaintext HTTP. reqwest
    // tunnels HTTPS proxies via CONNECT — that hides the request from the
    // proxy and prevents token injection — so the upstream URL must travel
    // in cleartext absolute-form. The proxy upgrades to TLS itself when
    // dialing the real upstream.
    let (base_url, proxy_config) = match config.proxy.as_ref() {
        Some(proxy) => {
            let mut url = url::Url::parse(&original_url).map_err(|e| {
                ClientError::ConfigurationError(format!("invalid base URL '{original_url}': {e}"))
            })?;
            url.set_scheme("http").map_err(|()| {
                ClientError::ConfigurationError(format!(
                    "cannot rewrite base URL '{original_url}' to http scheme",
                ))
            })?;
            let mut as_str = url.to_string();
            // url::Url always preserves the trailing slash on the path; trim
            // it so callers building `{base_url}/{endpoint}` don't double-slash.
            if as_str.ends_with('/') && url.path() == "/" {
                as_str.pop();
            }
            (as_str, Some(proxy.clone()))
        }
        None => (original_url, None),
    };

    let retry_policy = ExponentialBackoff::builder()
        .retry_bounds(
            config.retry_config.initial_delay,
            config.retry_config.max_delay,
        )
        .build_with_max_retries(config.retry_config.max_retries);

    let mut client_builder = reqwest::Client::builder();
    if let Some(timeout) = config.timeout_seconds {
        client_builder = client_builder.timeout(Duration::from_secs(timeout));
    }
    if let Some(ref proxy) = proxy_config {
        let proxy = reqwest::Proxy::http(&proxy.proxy_url).map_err(|e| {
            ClientError::ConfigurationError(format!("invalid proxy URL '{}': {e}", proxy.proxy_url))
        })?;
        client_builder = client_builder.proxy(proxy);
    }
    let reqwest_client = client_builder.build().map_err(ClientError::NetworkError)?;

    // Create client with retry middleware.
    // RetryAfterMiddleware is added before RetryTransientMiddleware
    // so that Retry-After headers are respected before falling back to exponential backoff.
    // RetryLoggingMiddleware sits before the retry middleware so it observes
    // every attempt (including the original).
    let client = reqwest_middleware::ClientBuilder::new(reqwest_client.clone())
        .with(retry_logging::RetryLoggingMiddleware)
        .with(RetryAfterMiddleware::new())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build();

    Ok(ClientResources {
        client,
        streaming_client: reqwest_client,
        api_key: Arc::new(api_key),
        base_url,
        config: Arc::new(config),
        proxy_config,
    })
}

/// A retry policy for SSE streams that never retries.
///
/// Useful for handling retries at a higher level rather than automatic reconnection.
pub struct NoRetryPolicy;

impl reqwest_eventsource::retry::RetryPolicy for NoRetryPolicy {
    fn retry(
        &self,
        _error: &reqwest_eventsource::Error,
        _last_retry: Option<(usize, std::time::Duration)>,
    ) -> Option<std::time::Duration> {
        // Never retry - return None to indicate no retry should happen
        None
    }

    fn set_reconnection_time(&mut self, _duration: std::time::Duration) {
        // Ignored - we never retry anyway
    }
}

/// Builds a boxed [`LLMClient`] from a [`Config`], dispatching by provider prefix.
///
/// The `config.provider` string is resolved via
/// [`resolve_model_prefix`](neuromance_common::client::resolve_model_prefix):
/// friendly aliases like `"openai"`, `"anthropic"`, `"groq"`, or `"ollama"` pick
/// the correct client implementation and base URL.
///
/// This is the ergonomic entry point when you don't want to hard-wire a specific
/// client type — pair it with [`Config::from_model`] for a one-liner:
///
/// ```no_run
/// use neuromance_client::build_client;
/// use neuromance_common::Config;
///
/// # fn example() -> anyhow::Result<()> {
/// let config = Config::from_model("openai:gpt-4o")?.with_api_key("sk-...");
/// let client = build_client(config)?;
/// # let _ = client;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`ClientError::ConfigurationError`] if the provider prefix is unknown,
/// or propagates any error from the concrete client constructor.
pub fn build_client(config: Config) -> Result<Box<dyn LLMClient>, ClientError> {
    let (provider, _) = resolve_model_prefix(&config.provider).ok_or_else(|| {
        ClientError::ConfigurationError(format!(
            "unknown provider '{}'. Use Config::from_model(\"openai:gpt-4o\") \
             or one of: openai, openai-responses, anthropic, ollama, groq, \
             openrouter, together, mistral, deepseek, xai, chat_completions, responses",
            config.provider
        ))
    })?;

    match provider {
        Provider::Anthropic => Ok(Box::new(AnthropicClient::new(config)?)),
        Provider::ChatCompletions => Ok(Box::new(ChatCompletionsClient::new(config)?)),
        Provider::Responses => Ok(Box::new(ResponsesClient::new(config)?)),
    }
}

/// Trait for LLM client implementations.
///
/// Provides a unified interface for interacting with different LLM providers.
/// Implementations must be async and thread-safe (Send + Sync).
#[must_use = "LLMClient must be used to make requests"]
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Get the client's configuration.
    fn config(&self) -> &Config;

    /// Send a chat completion request to the LLM.
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError>;

    /// Send a streaming chat completion request to the LLM.
    ///
    /// Returns a stream of incremental response chunks for real-time display.
    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>;

    /// Check if the client supports tool/function calling.
    fn supports_tools(&self) -> bool;

    /// Check if the client supports streaming responses.
    fn supports_streaming(&self) -> bool;

    /// Validate a configuration object.
    ///
    /// Checks parameter ranges: `temperature` (0.0-2.0), `top_p` (0.0-1.0), `frequency_penalty` (-2.0-2.0).
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of range.
    fn validate_config(&self, config: Config) -> Result<(), ClientError> {
        if config
            .temperature
            .is_some_and(|t| !(0.0..=2.0).contains(&t))
        {
            return Err(ClientError::InvalidTemperature);
        }

        if config.top_p.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
            return Err(ClientError::InvalidTopP);
        }

        if config
            .frequency_penalty
            .is_some_and(|f| !(-2.0..=2.0).contains(&f))
        {
            return Err(ClientError::InvalidFrequencyPenalty);
        }

        Ok(())
    }

    /// Validate a chat request before sending.
    ///
    /// Checks messages exist and that tools/streaming are supported if requested.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    fn validate_request(&self, request: &ChatRequest) -> Result<(), ClientError> {
        request
            .validate_has_messages()
            .map_err(|e| ClientError::InvalidRequest(e.to_string()))?;

        if !self.supports_tools() && request.has_tools() {
            return Err(ClientError::ToolsNotSupported);
        }

        if !self.supports_streaming() && request.is_streaming() {
            return Err(ClientError::StreamingNotSupported);
        }

        Ok(())
    }
}

/// Blanket impl so [`build_client`] output (`Box<dyn LLMClient>`) plugs into
/// any API that is generic over `C: LLMClient`, such as `Core<C>`.
#[async_trait]
impl<T: LLMClient + ?Sized> LLMClient for Box<T> {
    fn config(&self) -> &Config {
        (**self).config()
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
        (**self).chat(request).await
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
    {
        (**self).chat_stream(request).await
    }

    fn supports_tools(&self) -> bool {
        (**self).supports_tools()
    }

    fn supports_streaming(&self) -> bool {
        (**self).supports_streaming()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_common::chat::Message;
    use neuromance_common::client::{ChatResponse, ToolChoice};
    use neuromance_common::features::{ReasoningLevel, ThinkingMode};
    use neuromance_common::tools::Tool;
    use std::collections::HashMap;
    use uuid::Uuid;

    // Mock implementation for testing
    struct MockLLMClient {
        config: Config,
        supports_tools: bool,
        supports_streaming: bool,
    }

    impl MockLLMClient {
        fn new() -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                supports_tools: true,
                supports_streaming: true,
            }
        }

        fn without_tools() -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                supports_tools: false,
                supports_streaming: true,
            }
        }

        fn without_streaming() -> Self {
            Self {
                config: Config::new("mock", "mock-model"),
                supports_tools: true,
                supports_streaming: false,
            }
        }
    }

    #[async_trait]
    impl LLMClient for MockLLMClient {
        fn config(&self) -> &Config {
            &self.config
        }

        async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            Ok(ChatResponse {
                message: create_test_message(),
                model: "mock-model".to_string(),
                usage: None,
                finish_reason: None,
                created_at: chrono::Utc::now(),
                response_id: Some("test-response".to_string()),
                metadata: HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
        {
            use futures::stream;

            let chunk = ChatChunk {
                model: "mock-model".to_string(),
                delta_content: Some("Hello".to_string()),
                delta_reasoning_content: None,
                delta_role: Some(neuromance_common::chat::MessageRole::Assistant),
                delta_tool_calls: None,
                finish_reason: None,
                usage: None,
                response_id: Some("test-response".to_string()),
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };

            Ok(Box::pin(stream::iter(vec![Ok(chunk)])))
        }

        fn supports_tools(&self) -> bool {
            self.supports_tools
        }

        fn supports_streaming(&self) -> bool {
            self.supports_streaming
        }
    }

    fn create_test_message() -> Message {
        Message::user(Uuid::new_v4(), "Test message")
    }

    fn create_test_tool() -> Tool {
        use neuromance_common::tools::{Function, Parameters, Property};
        use std::collections::HashMap;

        let mut properties = HashMap::new();
        properties.insert("arg".to_string(), Property::string("A test argument"));

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "test_function".to_string(),
                description: "A test function".to_string(),
                parameters: Parameters::new(properties, vec!["arg".into()]).into(),
            },
        }
    }

    #[test]
    fn test_validate_request_empty_messages() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(error, ClientError::InvalidRequest(_)));
    }

    #[test]
    fn test_validate_request_valid() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_request_tools_not_supported() {
        let client = MockLLMClient::without_tools();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: Some(vec![create_test_tool()]),
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(error, ClientError::ToolsNotSupported));
    }

    #[test]
    fn test_validate_request_tools_supported() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: Some(vec![create_test_tool()]),
            tool_choice: Some(ToolChoice::Auto),
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_request_streaming_not_supported() {
        let client = MockLLMClient::without_streaming();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: true,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(error, ClientError::StreamingNotSupported));
    }

    #[test]
    fn test_validate_request_streaming_supported() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: true,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.validate_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_client_config_access() {
        let client = MockLLMClient::new();
        let config = client.config();
        assert_eq!(config.provider, "mock");
        assert_eq!(config.model, "mock-model");
    }

    #[test]
    fn test_supports_tools() {
        let client_with_tools = MockLLMClient::new();
        let client_without_tools = MockLLMClient::without_tools();

        assert!(client_with_tools.supports_tools());
        assert!(!client_without_tools.supports_tools());
    }

    #[test]
    fn test_supports_streaming() {
        let client_with_streaming = MockLLMClient::new();
        let client_without_streaming = MockLLMClient::without_streaming();

        assert!(client_with_streaming.supports_streaming());
        assert!(!client_without_streaming.supports_streaming());
    }

    #[tokio::test]
    async fn test_chat_method() {
        let client = MockLLMClient::new();
        let request = ChatRequest {
            model: Some("test-model".to_string()),
            messages: vec![create_test_message()].into(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
            user: None,
            thinking: ThinkingMode::Default,
            reasoning_level: ReasoningLevel::Default,
            metadata: HashMap::new(),
        };

        let result = client.chat(&request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.response_id, Some("test-response".to_string()));
        assert_eq!(response.model, "mock-model");
    }

    #[test]
    fn build_client_dispatches_openai_to_chat_completions() {
        let config = Config::from_model("openai:gpt-4o")
            .unwrap()
            .with_api_key("sk-test");
        let client = build_client(config).unwrap();
        assert!(client.supports_tools());
        assert_eq!(client.config().model, "gpt-4o");
    }

    #[test]
    fn build_client_dispatches_anthropic() {
        let config = Config::from_model("anthropic:claude-sonnet-4-5-20250929")
            .unwrap()
            .with_api_key("sk-ant-test");
        let client = build_client(config).unwrap();
        assert_eq!(client.config().provider, "anthropic");
    }

    #[test]
    fn build_client_rejects_unknown_provider() {
        let config = Config::new("totally-fake", "some-model").with_api_key("k");
        let result = build_client(config);
        assert!(result.is_err(), "unknown provider should fail");
        let err = result.err().unwrap();
        assert!(matches!(err, ClientError::ConfigurationError(_)));
        assert!(err.to_string().contains("totally-fake"));
    }

    #[test]
    fn boxed_client_impls_llmclient() {
        // Compile-time check: Box<dyn LLMClient> implements LLMClient via blanket impl
        // so it plugs into any API that's generic over `C: LLMClient`.
        fn assert_llm<C: LLMClient>(_: &C) {}
        let config = Config::from_model("openai:gpt-4o")
            .unwrap()
            .with_api_key("k");
        let boxed: Box<dyn LLMClient> = build_client(config).unwrap();
        assert_llm(&boxed);
    }
}
