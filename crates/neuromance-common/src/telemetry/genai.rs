//! OpenTelemetry GenAI semantic-convention attribute keys and values.
//!
//! Pinned to **OpenTelemetry Semantic Conventions v1.38.0**, GenAI group.
//!
//! These are deliberately not taken from `opentelemetry-semantic-conventions`
//! 0.31: that crate gates its `GEN_AI_*` constants behind the
//! `semconv_experimental` feature, and it predates both the
//! `gen_ai.system` → `gen_ai.provider.name` rename and the
//! `gen_ai.input.messages` family. Sourcing half the keys from the crate and
//! defining the rest here would leave two vocabularies to keep in step.
//!
//! When upgrading to a semconv release that covers the whole GenAI group,
//! update this file and the version pin above together.
//!
//! The keys are plain `&'static str` so they work everywhere a name is
//! needed: as a `tracing` field name (via the macros' `{ expr } = value`
//! form), as a `Span::record` key, and as an OpenTelemetry `Key` or
//! `KeyValue`.

// --- Operation identity ---

/// The operation being performed. See [`op`] for the values.
pub const OPERATION_NAME: &str = "gen_ai.operation.name";
/// The GenAI provider serving the request. See [`provider_name`].
pub const PROVIDER_NAME: &str = "gen_ai.provider.name";
/// Identifier tying every span of one conversation together.
pub const CONVERSATION_ID: &str = "gen_ai.conversation.id";

// --- Request parameters ---

/// Model name as requested by the client.
pub const REQUEST_MODEL: &str = "gen_ai.request.model";
/// Sampling temperature.
pub const REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
/// Nucleus-sampling cutoff.
pub const REQUEST_TOP_P: &str = "gen_ai.request.top_p";
/// Upper bound on generated tokens.
pub const REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
/// Frequency penalty.
pub const REQUEST_FREQUENCY_PENALTY: &str = "gen_ai.request.frequency_penalty";
/// Presence penalty.
pub const REQUEST_PRESENCE_PENALTY: &str = "gen_ai.request.presence_penalty";
/// Sequences that stop generation. Array-valued.
pub const REQUEST_STOP_SEQUENCES: &str = "gen_ai.request.stop_sequences";
/// Number of completions requested.
pub const REQUEST_CHOICE_COUNT: &str = "gen_ai.request.choice.count";
/// Requested output shape. See [`output_type`].
pub const OUTPUT_TYPE: &str = "gen_ai.output.type";

// --- Response ---

/// Provider-assigned identifier for the response.
pub const RESPONSE_ID: &str = "gen_ai.response.id";
/// Model name as reported by the provider, which may differ from the request.
pub const RESPONSE_MODEL: &str = "gen_ai.response.model";
/// Why generation stopped. Array-valued, one entry per choice.
pub const RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";

// --- Usage ---

/// Tokens consumed by the prompt.
pub const USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
/// Tokens produced by the model.
pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
/// Which side of the exchange a token-usage measurement counts.
/// See [`token_type`].
pub const TOKEN_TYPE: &str = "gen_ai.token.type";

// --- Tools ---

/// Name of the tool being executed.
pub const TOOL_NAME: &str = "gen_ai.tool.name";
/// Kind of tool. See [`tool_type`].
pub const TOOL_TYPE: &str = "gen_ai.tool.type";
/// Human-readable description of the tool.
pub const TOOL_DESCRIPTION: &str = "gen_ai.tool.description";
/// Identifier the model assigned to this tool call.
pub const TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
/// Arguments the model passed. Opt-in content; see [`super::capture`].
pub const TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
/// Value the tool returned. Opt-in content; see [`super::capture`].
pub const TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";

// --- Agents ---

/// Stable identifier of the agent.
pub const AGENT_ID: &str = "gen_ai.agent.id";
/// Human-readable name of the agent.
pub const AGENT_NAME: &str = "gen_ai.agent.name";
/// Description of what the agent does.
pub const AGENT_DESCRIPTION: &str = "gen_ai.agent.description";

// --- Opt-in message content ---

/// Messages sent to the model. Opt-in content; see [`super::capture`].
pub const INPUT_MESSAGES: &str = "gen_ai.input.messages";
/// Messages returned by the model. Opt-in content; see [`super::capture`].
pub const OUTPUT_MESSAGES: &str = "gen_ai.output.messages";
/// System-role instructions. Opt-in content; see [`super::capture`].
pub const SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";

// --- Non-GenAI keys set on GenAI spans ---

/// Host of the provider endpoint.
pub const SERVER_ADDRESS: &str = "server.address";
/// Port of the provider endpoint.
pub const SERVER_PORT: &str = "server.port";
/// Low-cardinality class of failure.
pub const ERROR_TYPE: &str = "error.type";

// --- Metric names ---

/// Histogram of tokens used per operation, split by [`TOKEN_TYPE`].
pub const METRIC_TOKEN_USAGE: &str = "gen_ai.client.token.usage";
/// Histogram of client operation duration, in seconds.
pub const METRIC_OPERATION_DURATION: &str = "gen_ai.client.operation.duration";

/// Values for [`OPERATION_NAME`].
pub mod op {
    /// A multi-turn chat completion.
    pub const CHAT: &str = "chat";
    /// An embedding request.
    pub const EMBEDDINGS: &str = "embeddings";
    /// Execution of a single tool call.
    pub const EXECUTE_TOOL: &str = "execute_tool";
    /// A whole agent run.
    pub const INVOKE_AGENT: &str = "invoke_agent";
}

/// Values for [`TOKEN_TYPE`].
pub mod token_type {
    /// Tokens consumed by the prompt.
    pub const INPUT: &str = "input";
    /// Tokens produced by the model.
    pub const OUTPUT: &str = "output";
}

/// Values for [`TOOL_TYPE`].
pub mod tool_type {
    /// A tool the application itself implements.
    pub const FUNCTION: &str = "function";
    /// A tool provided by an external component, such as an MCP server.
    pub const EXTENSION: &str = "extension";
}

/// Values for [`OUTPUT_TYPE`].
pub mod output_type {
    /// Free-form text.
    pub const TEXT: &str = "text";
    /// A JSON value, usually schema-constrained.
    pub const JSON: &str = "json";
}

/// Advised bucket boundaries for [`METRIC_TOKEN_USAGE`], semconv v1.38.0.
pub const TOKEN_USAGE_BUCKETS: [f64; 14] = [
    1.0,
    4.0,
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
    16384.0,
    65536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
];

/// Advised bucket boundaries for [`METRIC_OPERATION_DURATION`], semconv v1.38.0.
pub const OPERATION_DURATION_BUCKETS: [f64; 14] = [
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];

/// Map a neuromance provider prefix to a [`PROVIDER_NAME`] value.
///
/// `gen_ai.provider.name` is an open enum, so self-hosted and aggregator
/// providers keep their own identity instead of being flattened into `openai`
/// merely because they speak its wire protocol. Latency and cost split by
/// provider are the whole point of the attribute.
///
/// `chat_completions` and `responses` are the escape-hatch prefixes for any
/// compatible endpoint, so the only signal available there is the host.
///
/// An unrecognised prefix cannot reach a live request — `build_client` rejects
/// it — so it falls back to `openai`, the protocol such a client would speak.
#[must_use]
pub fn provider_name(provider: &str, base_url: Option<&str>) -> &'static str {
    match provider {
        "anthropic" => "anthropic",
        "deepseek" => "deepseek",
        "groq" => "groq",
        "mistral" => "mistral_ai",
        "xai" => "x_ai",
        "ollama" => "ollama",
        "openrouter" => "openrouter",
        "together" => "together_ai",
        _ => compatible_provider_name(base_url),
    }
}

/// Identify an OpenAI-compatible endpoint from its host.
fn compatible_provider_name(base_url: Option<&str>) -> &'static str {
    let Some(url) = base_url else {
        return "openai";
    };
    if url.contains(".openai.azure.com") {
        "azure.ai.openai"
    } else if url.contains("generativelanguage.googleapis.com") {
        "gcp.gemini"
    } else if url.contains("aiplatform.googleapis.com") {
        "gcp.vertex_ai"
    } else {
        "openai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name_maps_known_prefixes() {
        for (prefix, expected) in [
            ("openai", "openai"),
            ("openai-responses", "openai"),
            ("anthropic", "anthropic"),
            ("deepseek", "deepseek"),
            ("groq", "groq"),
            ("mistral", "mistral_ai"),
            ("xai", "x_ai"),
            ("ollama", "ollama"),
            ("openrouter", "openrouter"),
            ("together", "together_ai"),
        ] {
            assert_eq!(
                provider_name(prefix, None),
                expected,
                "prefix {prefix} mapped wrong"
            );
        }
    }

    #[test]
    fn test_provider_name_detects_hosted_openai_variants_from_base_url() {
        for (url, expected) in [
            ("https://contoso.openai.azure.com/", "azure.ai.openai"),
            (
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "gcp.gemini",
            ),
            (
                "https://us-central1-aiplatform.googleapis.com/v1",
                "gcp.vertex_ai",
            ),
            ("https://llm.internal.example/v1", "openai"),
        ] {
            assert_eq!(
                provider_name("chat_completions", Some(url)),
                expected,
                "url {url} mapped wrong"
            );
        }
    }

    /// A named provider owns its identity; a `base_url` override does not turn
    /// Anthropic into Azure.
    #[test]
    fn test_provider_name_ignores_base_url_for_named_providers() {
        assert_eq!(
            provider_name("anthropic", Some("https://contoso.openai.azure.com/")),
            "anthropic"
        );
    }

    #[test]
    fn test_provider_name_falls_back_for_unknown_prefixes() {
        assert_eq!(provider_name("wat", None), "openai");
    }

    /// A transposed digit in a boundary literal silently corrupts a histogram,
    /// and the exporter will not complain.
    #[test]
    fn test_bucket_boundaries_are_monotonic() {
        assert!(TOKEN_USAGE_BUCKETS.is_sorted(), "{TOKEN_USAGE_BUCKETS:?}");
        assert!(
            OPERATION_DURATION_BUCKETS.is_sorted(),
            "{OPERATION_DURATION_BUCKETS:?}"
        );
    }
}
