//! The attribute set shared by a GenAI span and its metrics.
//!
//! Both come from one [`GenAiAttrs`] value so a span and the histogram
//! recorded beside it can never disagree about which model or provider the
//! operation hit.

use opentelemetry::{Array, KeyValue, StringValue, Value};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use neuromance_common::client::{ChatRequest, Config, resolve_model_prefix};
use neuromance_common::telemetry::genai;

/// Request-side identity of one GenAI operation.
///
/// Every field is known before the request leaves the process, so the whole
/// set can be applied to a span at creation and reused verbatim as metric
/// attributes when the operation ends.
#[derive(Debug, Clone)]
pub struct GenAiAttrs {
    /// `gen_ai.operation.name`; see [`genai::op`].
    pub operation: &'static str,
    /// `gen_ai.provider.name`.
    pub provider: &'static str,
    /// `gen_ai.request.model`.
    pub request_model: String,
    /// `server.address`, absent when the base URL is unparseable.
    pub server_address: Option<String>,
    /// `server.port`.
    pub server_port: Option<i64>,
}

impl GenAiAttrs {
    /// Identity of a chat operation.
    pub fn chat(config: &Config, request: &ChatRequest) -> Self {
        Self::new(genai::op::CHAT, config, request_model(config, request))
    }

    fn new(operation: &'static str, config: &Config, request_model: String) -> Self {
        let upstream = upstream_base_url(config);
        let (server_address, server_port) = server_attrs(upstream.as_deref());
        Self {
            operation,
            provider: genai::provider_name(&config.provider, upstream.as_deref()),
            request_model,
            server_address,
            server_port,
        }
    }

    /// The span name the conventions require: `{operation} {model}`.
    pub fn span_name(&self) -> String {
        format!("{} {}", self.operation, self.request_model)
    }

    /// Attach the request-side attributes to `span`.
    ///
    /// The span declares these as `tracing` fields, so this is a `record`
    /// rather than a `set_attribute`: a field always supersedes an attribute
    /// of the same name.
    pub fn apply_to_span(&self, span: &Span) {
        span.record(genai::OPERATION_NAME, self.operation);
        span.record(genai::PROVIDER_NAME, self.provider);
        span.record(genai::REQUEST_MODEL, self.request_model.as_str());
        if let Some(ref address) = self.server_address {
            span.record(genai::SERVER_ADDRESS, address.as_str());
        }
        if let Some(port) = self.server_port {
            span.record(genai::SERVER_PORT, port);
        }
    }

    /// The metric attribute set, matching what [`Self::apply_to_span`] wrote.
    ///
    /// `response_model` and `error_type` are the only parts not known until
    /// the operation finishes, so they arrive as arguments rather than fields.
    pub fn metric_kv(
        &self,
        response_model: Option<&str>,
        error_type: Option<&str>,
    ) -> Vec<KeyValue> {
        let mut kv = vec![
            KeyValue::new(genai::OPERATION_NAME, self.operation),
            KeyValue::new(genai::PROVIDER_NAME, self.provider),
            KeyValue::new(
                genai::REQUEST_MODEL,
                Value::from(self.request_model.clone()),
            ),
        ];
        if let Some(ref address) = self.server_address {
            kv.push(KeyValue::new(
                genai::SERVER_ADDRESS,
                Value::from(address.clone()),
            ));
        }
        if let Some(port) = self.server_port {
            kv.push(KeyValue::new(genai::SERVER_PORT, port));
        }
        if let Some(model) = response_model {
            kv.push(KeyValue::new(
                genai::RESPONSE_MODEL,
                Value::from(model.to_owned()),
            ));
        }
        if let Some(error) = error_type {
            kv.push(KeyValue::new(
                genai::ERROR_TYPE,
                Value::from(error.to_owned()),
            ));
        }
        kv
    }
}

/// The model the request asks for, falling back to the client's default.
fn request_model(config: &Config, request: &ChatRequest) -> String {
    request
        .model
        .clone()
        .unwrap_or_else(|| config.model.clone())
}

/// The URL of the provider the request is logically addressed to.
///
/// This is the configured upstream, never the client's rewritten `base_url`
/// and never the tokenizer proxy, for three reasons:
///
/// 1. `normalize_base_url` rewrites the upstream scheme to plain `http` in
///    proxy mode, so the client's own `base_url` reports port 80 for what is
///    really an HTTPS endpoint.
/// 2. The proxy is transport plumbing attached as a `reqwest` forward proxy.
///    The request still names the upstream authority on the wire, and the
///    conventions want the server the request is logically sent to.
/// 3. Bucketing latency by proxy address would make every provider look like
///    one server, which is the opposite of what the attribute is for.
fn upstream_base_url(config: &Config) -> Option<String> {
    config.base_url.clone().or_else(|| {
        resolve_model_prefix(&config.provider)
            .and_then(|(_, default)| default)
            .map(str::to_owned)
    })
}

/// Split a base URL into `server.address` and `server.port`.
fn server_attrs(base_url: Option<&str>) -> (Option<String>, Option<i64>) {
    let Some(url) = base_url.and_then(|raw| url::Url::parse(raw).ok()) else {
        return (None, None);
    };
    let port = url.port_or_known_default().map(i64::from);
    (url.host_str().map(str::to_owned), port)
}

/// Set an array-valued attribute that cannot be a `tracing` field.
///
/// `tracing` fields hold scalars only, and a field always supersedes a
/// same-named attribute, so a key routed through here must never also be
/// declared on the span.
pub fn set_string_array(span: &Span, key: &'static str, values: Vec<String>) {
    if values.is_empty() {
        return;
    }
    let strings: Vec<StringValue> = values.into_iter().map(StringValue::from).collect();
    span.set_attribute(key, Value::Array(Array::String(strings)));
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_common::client::ProxyConfig;

    fn proxied(base_url: &str) -> Config {
        Config::new("anthropic", "claude-sonnet-4")
            .with_base_url(base_url)
            .with_proxy(
                ProxyConfig::new("http://tokenizer.internal:8080").expect("valid proxy url"),
            )
    }

    /// The proxy is transport plumbing. Reporting its address would collapse
    /// every provider into one server and would report port 80 for an HTTPS
    /// endpoint, because proxy mode rewrites the upstream scheme.
    #[test]
    fn test_server_attrs_uses_upstream_not_proxy() {
        let config = proxied("https://api.anthropic.com/v1");
        let attrs = GenAiAttrs::chat(&config, &ChatRequest::new(Vec::new()));

        assert_eq!(attrs.server_address.as_deref(), Some("api.anthropic.com"));
        assert_eq!(attrs.server_port, Some(443));
    }

    #[test]
    fn test_server_attrs_falls_back_to_the_provider_default() {
        let config = Config::new("openai", "gpt-4o");
        let attrs = GenAiAttrs::chat(&config, &ChatRequest::new(Vec::new()));

        assert_eq!(attrs.server_address.as_deref(), Some("api.openai.com"));
        assert_eq!(attrs.server_port, Some(443));
        assert_eq!(attrs.provider, "openai");
    }

    #[test]
    fn test_server_attrs_are_absent_for_an_unparseable_base_url() {
        let config = Config::new("chat_completions", "local").with_base_url("not a url");
        let attrs = GenAiAttrs::chat(&config, &ChatRequest::new(Vec::new()));

        assert_eq!(attrs.server_address, None);
        assert_eq!(attrs.server_port, None);
    }

    #[test]
    fn test_span_name_is_operation_then_model() {
        let config = Config::new("openai", "gpt-4o");
        let attrs = GenAiAttrs::chat(&config, &ChatRequest::new(Vec::new()));

        assert_eq!(attrs.span_name(), "chat gpt-4o");
    }

    /// A per-request model override wins over the client default; otherwise a
    /// multi-model deployment reports every call under one model name.
    #[test]
    fn test_request_model_prefers_the_per_request_override() {
        let config = Config::new("openai", "gpt-4o");
        let request = ChatRequest::new(Vec::new()).with_model("gpt-4o-mini");

        assert_eq!(
            GenAiAttrs::chat(&config, &request).request_model,
            "gpt-4o-mini"
        );
    }

    /// The span and the metrics must describe the same operation. A key set
    /// here is what makes a trace joinable to a histogram in the backend.
    #[test]
    fn test_metric_kv_carries_the_same_request_attributes_as_the_span() {
        let config = Config::new("anthropic", "claude-sonnet-4");
        let attrs = GenAiAttrs::chat(&config, &ChatRequest::new(Vec::new()));

        let kv = attrs.metric_kv(None, None);
        let keys: Vec<&str> = kv.iter().map(|entry| entry.key.as_str()).collect();

        assert_eq!(
            keys,
            vec![
                genai::OPERATION_NAME,
                genai::PROVIDER_NAME,
                genai::REQUEST_MODEL,
                genai::SERVER_ADDRESS,
                genai::SERVER_PORT,
            ]
        );
    }

    #[test]
    fn test_metric_kv_appends_the_outcome_when_known() {
        let config = Config::new("openai", "gpt-4o");
        let attrs = GenAiAttrs::chat(&config, &ChatRequest::new(Vec::new()));

        let kv = attrs.metric_kv(Some("gpt-4o-2024-08-06"), Some("rate_limited"));
        let keys: Vec<&str> = kv.iter().map(|k| k.key.as_str()).collect();

        assert!(keys.contains(&genai::RESPONSE_MODEL));
        assert!(keys.contains(&genai::ERROR_TYPE));
    }
}
