//! Prompt Caching Demo
//!
//! Demonstrates prompt caching across multiple LLM providers and uses
//! `CacheMetrics` to track cache hit rates and token savings.
//!
//! Supports three providers:
//! - **Anthropic** (Claude) — automatic prefix caching for >= 1024 tokens
//! - **OpenAI Chat Completions** — automatic prompt caching
//! - **OpenAI Responses** — automatic prompt caching via Responses API
//!
//! # Usage
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-ant-..."
//! export OPENAI_API_KEY="sk-..."
//!
//! # Run all providers (default)
//! cargo run -p neuromance-context --example prompt_caching
//!
//! # Run a single provider
//! cargo run -p neuromance-context --example prompt_caching -- \
//!     --provider anthropic
//!
//! # Override base URLs
//! cargo run -p neuromance-context --example prompt_caching -- \
//!     --provider openai --openai-base-url http://localhost:8080/v1
//! ```

use std::collections::HashMap;
use std::io::Write;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use log::info;
use uuid::Uuid;

use neuromance_client::{AnthropicClient, ChatCompletionsClient, LLMClient, ResponsesClient};
use neuromance_common::{CacheMetrics, ChatRequest, Config, Message, Usage};

#[derive(Debug, Clone, ValueEnum)]
enum Provider {
    Anthropic,
    Openai,
    Responses,
    All,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Prompt caching demo across LLM providers")]
struct Args {
    /// Which provider(s) to test
    #[arg(long, value_enum, default_value = "all")]
    provider: Provider,

    /// Anthropic API key (or set ANTHROPIC_API_KEY env var)
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    anthropic_api_key: Option<String>,

    /// Anthropic base URL override
    #[arg(long)]
    anthropic_base_url: Option<String>,

    /// Anthropic model
    #[arg(long, default_value = "claude-sonnet-4-5-20250929")]
    anthropic_model: String,

    /// OpenAI API key (or set OPENAI_API_KEY env var)
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

    /// OpenAI base URL override (used for both chat and responses)
    #[arg(long)]
    openai_base_url: Option<String>,

    /// OpenAI model (used for both chat completions and responses)
    #[arg(long, default_value = "gpt-4o-mini")]
    openai_model: String,

    /// Maximum tokens per response
    #[arg(long, default_value = "512")]
    max_tokens: u32,

    /// Disable streaming (use synchronous API)
    #[arg(long)]
    no_stream: bool,
}

/// Build a long system prompt that exceeds caching thresholds.
///
/// Anthropic caches prompt prefixes >= 1024 tokens. OpenAI uses
/// automatic caching for repeated prefixes. We construct a detailed
/// persona + reference material block that stays constant across
/// all turns, making it an ideal caching candidate.
fn system_prompt() -> String {
    let preamble = "\
You are an expert software architect specializing in distributed \
systems, database design, and cloud-native applications. You have \
deep expertise in Rust, Go, TypeScript, and Python. When answering \
questions you should:

1. Start with a concise summary (2-3 sentences)
2. Provide detailed technical explanation
3. Include code examples when relevant
4. Discuss trade-offs and alternatives
5. Reference industry best practices

Your knowledge covers:
- Microservices architecture patterns (saga, CQRS, event sourcing)
- Database technologies (PostgreSQL, Redis, DynamoDB, CockroachDB)
- Message queues and streaming (Kafka, NATS, RabbitMQ)
- Container orchestration (Kubernetes, Nomad)
- Infrastructure as code (Terraform, Pulumi, Nix)
- Observability (OpenTelemetry, Prometheus, Grafana)
- Security best practices (zero-trust, mTLS, RBAC)
- Performance optimization and profiling
- CI/CD pipelines and deployment strategies";

    // Detailed reference material to guarantee we cross the 1024-token
    // caching threshold.  Each entry is ~40-60 tokens.
    let patterns: &[(&str, &str)] = &[
        (
            "Circuit Breaker",
            "Prevents cascading failures by wrapping calls to \
             external services. States: Closed (requests flow \
             normally), Open (requests fail immediately without \
             calling the downstream service), Half-Open (a limited \
             number of probe requests are allowed through to test \
             recovery). Implement with exponential backoff, \
             configurable failure thresholds, and health-check \
             probes. Libraries: Resilience4j (Java), Polly (.NET), \
             tower (Rust). Metrics to track: failure rate, state \
             transitions, recovery time.",
        ),
        (
            "Bulkhead",
            "Isolates components so that failure in one subsystem \
             does not cascade to others. Two main strategies: \
             thread-pool isolation (each dependency gets its own \
             thread pool with bounded queue) and semaphore isolation \
             (lightweight permits limiting concurrent calls). \
             Choose thread-pool when calls are blocking I/O; prefer \
             semaphore for async workloads. Combine with circuit \
             breakers for defense in depth.",
        ),
        (
            "Saga",
            "Manages distributed transactions through a sequence \
             of local transactions with compensating actions. Two \
             coordination approaches: choreography (event-driven, \
             each service publishes domain events that trigger the \
             next step) and orchestration (a central coordinator \
             directs each participant). Choreography offers loose \
             coupling but harder debugging; orchestration provides \
             clear control flow but introduces a coordinator \
             dependency. Always design compensating actions to be \
             idempotent.",
        ),
        (
            "CQRS (Command Query Responsibility Segregation)",
            "Separates read and write models for independent \
             scaling. Commands mutate state through domain events; \
             queries read from denormalized projections optimized \
             for specific access patterns. Benefits: read and write \
             stores can use different technologies (e.g., PostgreSQL \
             for writes, Elasticsearch for reads), independent \
             scaling, optimized query performance. Costs: eventual \
             consistency, increased complexity, projection rebuild \
             time. Often paired with event sourcing.",
        ),
        (
            "Event Sourcing",
            "Stores state changes as an append-only sequence of \
             domain events rather than mutable current state. Every \
             state transition is captured as an immutable fact. \
             Enables temporal queries (what was the state at time T?), \
             complete audit trails, and event replay for debugging \
             or migration. Use snapshotting every N events to bound \
             replay time. Storage: EventStoreDB, Kafka with \
             compaction, PostgreSQL with JSONB.",
        ),
        (
            "Strangler Fig",
            "Incrementally migrates a legacy system by routing \
             traffic through a facade that delegates to either the \
             old or new implementation based on feature flags or \
             routing rules. Start with low-risk, well-understood \
             endpoints. Gradually shift traffic as confidence \
             grows. Roll back individual routes without affecting \
             the rest. Named after strangler fig trees that grow \
             around host trees and eventually replace them.",
        ),
        (
            "Sidecar",
            "Deploys auxiliary processes alongside the main \
             application container in the same pod or host. Used \
             for cross-cutting concerns: logging agents, metrics \
             collectors, configuration watchers, and service mesh \
             proxies (Envoy, Linkerd). Benefits: language-agnostic \
             infrastructure, independent lifecycle, separation of \
             concerns. The application communicates with the sidecar \
             over localhost, avoiding network hops.",
        ),
        (
            "Ambassador",
            "A helper service that sends network requests on \
             behalf of a consumer service. Offloads cross-cutting \
             network concerns: retries with backoff, circuit \
             breaking, routing, TLS termination, and protocol \
             translation. Typically deployed as a sidecar proxy. \
             Differs from API gateway in that it runs per-service \
             rather than at the edge.",
        ),
        (
            "Backends for Frontends (BFF)",
            "Creates separate backend services tailored for each \
             frontend type (web SPA, mobile app, CLI, IoT device). \
             Each BFF aggregates calls to downstream microservices \
             and shapes the response for its specific client. \
             Avoids one-size-fits-all APIs that over-fetch for some \
             clients and under-fetch for others. Trade-off: more \
             services to maintain, but better client experience \
             and independent evolution.",
        ),
        (
            "Leader Election",
            "Coordinates action among distributed instances by \
             electing a single leader responsible for coordination \
             tasks (cron jobs, partition assignment, schema \
             migrations). Consensus algorithms: Raft (etcd, Consul), \
             Paxos (Chubby), ZAB (ZooKeeper). Practical \
             implementations: etcd lease with TTL, Redis SETNX \
             with expiry, PostgreSQL advisory locks. Leader must \
             heartbeat; followers monitor and trigger re-election \
             on timeout.",
        ),
        (
            "Outbox Pattern",
            "Ensures reliable event publishing from a service \
             that writes to both a database and a message broker. \
             Instead of dual-writing (which risks inconsistency), \
             the service writes the domain event to an outbox table \
             in the same database transaction as the state change. \
             A separate relay process (or CDC connector like \
             Debezium) reads the outbox and publishes to the \
             broker. Guarantees at-least-once delivery without \
             distributed transactions.",
        ),
        (
            "Service Mesh",
            "Infrastructure layer that handles service-to-service \
             communication transparently. Provides mTLS encryption, \
             traffic management (canary deployments, traffic \
             splitting), observability (distributed tracing, \
             metrics), and resilience (retries, circuit breaking, \
             rate limiting). Implementations: Istio (Envoy-based), \
             Linkerd (ultra-lightweight Rust proxy), Consul Connect. \
             Adds latency overhead (~1-2ms per hop) but removes \
             networking logic from application code.",
        ),
    ];

    let mut buf = String::with_capacity(8192);
    buf.push_str(preamble);
    buf.push_str("\n\n## Reference: Common Architecture Patterns\n\n");
    for (name, desc) in patterns {
        buf.push_str(&format!("### {name}\n{desc}\n\n"));
    }
    buf
}

fn print_usage(label: &str, usage: &Usage) {
    println!("  {label}:");
    println!("    Input tokens:  {}", usage.prompt_tokens);
    println!("    Output tokens: {}", usage.completion_tokens);
    if let Some(ref d) = usage.input_tokens_details {
        if d.cached_tokens > 0 {
            println!(
                "    Cached tokens: {} ({:.0}% of input)",
                d.cached_tokens,
                usage.cache_hit_ratio().unwrap_or(0.0) * 100.0,
            );
        }
        if d.cache_creation_tokens > 0 {
            println!("    Cache write:   {} tokens", d.cache_creation_tokens);
        }
    }
}

fn print_metrics(label: &str, metrics: &CacheMetrics) {
    println!("\n--- {label}: Aggregate Cache Metrics ---");
    println!("  Total requests:      {}", metrics.total_requests);
    println!("  Total input tokens:  {}", metrics.total_input_tokens);
    println!("  Total cached tokens: {}", metrics.total_cached_tokens);
    println!(
        "  Cache creation:      {} tokens",
        metrics.total_cache_creation_tokens
    );
    if let Some(ratio) = metrics.cache_hit_ratio() {
        println!("  Cache hit ratio:     {:.1}%", ratio * 100.0);
    }
    if let Some(rate) = metrics.request_hit_rate() {
        println!(
            "  Request hit rate:    {:.0}% ({}/{})",
            rate * 100.0,
            metrics.requests_with_cache_hits,
            metrics.total_requests,
        );
    }
    println!();
}

async fn send_streaming(
    client: &dyn LLMClient,
    request: &ChatRequest,
) -> Result<(String, Option<Usage>, Option<String>)> {
    let mut stream = client.chat_stream(request).await?;
    let mut content = String::new();
    let mut response_id: Option<String> = None;
    // Accumulate usage across chunks: providers may split
    // input and output token data across different events.
    let mut usage: Option<Usage> = None;

    print!("  Assistant: ");
    std::io::stdout().flush()?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(ref text) = chunk.delta_content {
            print!("{text}");
            content.push_str(text);
            std::io::stdout().flush()?;
        }
        if let Some(ref chunk_usage) = chunk.usage {
            usage = Some(match usage {
                None => chunk_usage.clone(),
                Some(mut acc) => {
                    acc.prompt_tokens = acc.prompt_tokens.max(chunk_usage.prompt_tokens);
                    acc.completion_tokens =
                        acc.completion_tokens.max(chunk_usage.completion_tokens);
                    acc.total_tokens = acc.prompt_tokens + acc.completion_tokens;
                    if acc.input_tokens_details.is_none() {
                        acc.input_tokens_details
                            .clone_from(&chunk_usage.input_tokens_details);
                    }
                    if acc.output_tokens_details.is_none() {
                        acc.output_tokens_details
                            .clone_from(&chunk_usage.output_tokens_details);
                    }
                    acc
                }
            });
        }
        if response_id.is_none() {
            response_id = chunk.response_id.clone();
        }
    }
    println!("\n");

    Ok((content, usage, response_id))
}

async fn send_non_streaming(
    client: &dyn LLMClient,
    request: &ChatRequest,
) -> Result<(String, Option<Usage>, Option<String>)> {
    let response = client.chat(request).await?;
    let content = response.message.content.clone();
    println!("  Assistant: {content}\n");
    Ok((content, response.usage, response.response_id))
}

/// Run a multi-turn conversation against a single provider.
async fn run_provider(
    label: &str,
    client: &dyn LLMClient,
    model: &str,
    max_tokens: u32,
    no_stream: bool,
    use_prev_response_id: bool,
) -> Result<CacheMetrics> {
    let bar = "=".repeat(60);
    println!("\n{bar}");
    println!("  Provider: {label}");
    println!("  Model:    {model}");
    println!("  Stream:   {}", !no_stream);
    println!("{bar}\n");

    let conversation_id = Uuid::new_v4();
    let system = system_prompt();

    info!("[{label}] System prompt length: ~{} chars", system.len());

    let questions = [
        "Briefly compare the Saga pattern (choreography vs \
         orchestration) for a payment processing pipeline.",
        "Now explain how you'd add observability to that \
         saga implementation using OpenTelemetry.",
        "What failure modes should we test for, and how \
         would you set up chaos engineering for this system?",
    ];

    let mut messages = vec![Message::system(conversation_id, &system)];
    let mut cache_metrics = CacheMetrics::default();
    let mut prev_response_id: Option<String> = None;

    for (i, question) in questions.iter().enumerate() {
        let turn = i + 1;
        println!("--- [{label}] Turn {turn}/{} ---", questions.len());
        println!("  User: {question}");
        println!();

        messages.push(Message::user(conversation_id, *question));

        let request = match (use_prev_response_id, prev_response_id.as_ref()) {
            (true, Some(prev_id)) => {
                println!("  (using previous_response_id)\n");
                let mut metadata = HashMap::new();
                metadata.insert(
                    "previous_response_id".to_string(),
                    serde_json::Value::String(prev_id.clone()),
                );
                metadata.insert("store".to_string(), serde_json::Value::Bool(true));
                ChatRequest::new(vec![Message::user(conversation_id, *question)])
                    .with_model(model)
                    .with_max_tokens(max_tokens)
                    .with_metadata(metadata)
            }
            (true, None) => {
                // First turn: store response for
                // subsequent previous_response_id use
                let mut metadata = HashMap::new();
                metadata.insert("store".to_string(), serde_json::Value::Bool(true));
                ChatRequest::new(messages.clone())
                    .with_model(model)
                    .with_max_tokens(max_tokens)
                    .with_metadata(metadata)
            }
            _ => ChatRequest::new(messages.clone())
                .with_model(model)
                .with_max_tokens(max_tokens),
        };

        let (response_text, usage, resp_id) = if no_stream {
            send_non_streaming(client, &request).await?
        } else {
            send_streaming(client, &request).await?
        };

        if use_prev_response_id {
            prev_response_id = resp_id;
        }

        if let Some(ref u) = usage {
            let ulabel = format!("[{label}] Turn {turn} usage");
            print_usage(&ulabel, u);
            cache_metrics.record(u);
        }

        messages.push(Message::assistant(conversation_id, &response_text));
    }

    print_metrics(label, &cache_metrics);
    Ok(cache_metrics)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Prompt Caching Demo");
    println!("===================\n");

    let run_anthropic = matches!(args.provider, Provider::Anthropic | Provider::All);
    let run_openai = matches!(args.provider, Provider::Openai | Provider::All);
    let run_responses = matches!(args.provider, Provider::Responses | Provider::All);

    // -- Anthropic --
    if run_anthropic {
        let api_key = args.anthropic_api_key.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "Anthropic API key required \
                 (--anthropic-api-key or ANTHROPIC_API_KEY)"
            )
        })?;

        let mut config = Config::new("anthropic", &args.anthropic_model)
            .with_api_key(api_key)
            .with_max_tokens(args.max_tokens);
        if let Some(ref url) = args.anthropic_base_url {
            config = config.with_base_url(url);
        }

        let client = AnthropicClient::new(config)?;
        run_provider(
            "Anthropic",
            &client,
            &args.anthropic_model,
            args.max_tokens,
            args.no_stream,
            false,
        )
        .await?;
    }

    // -- OpenAI Chat Completions --
    if run_openai {
        let api_key = args.openai_api_key.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "OpenAI API key required \
                 (--openai-api-key or OPENAI_API_KEY)"
            )
        })?;

        let mut config = Config::new("openai", &args.openai_model)
            .with_api_key(api_key)
            .with_max_tokens(args.max_tokens);
        if let Some(ref url) = args.openai_base_url {
            config = config.with_base_url(url);
        }

        let client = ChatCompletionsClient::new(config)?;
        run_provider(
            "OpenAI Chat Completions",
            &client,
            &args.openai_model,
            args.max_tokens,
            args.no_stream,
            false,
        )
        .await?;
    }

    // -- OpenAI Responses --
    if run_responses {
        let api_key = args.openai_api_key.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "OpenAI API key required for Responses API \
                 (--openai-api-key or OPENAI_API_KEY)"
            )
        })?;

        let mut config = Config::new("responses", &args.openai_model)
            .with_api_key(api_key)
            .with_max_tokens(args.max_tokens);
        if let Some(ref url) = args.openai_base_url {
            config = config.with_base_url(url);
        }

        let client = ResponsesClient::new(config)?;
        run_provider(
            "OpenAI Responses",
            &client,
            &args.openai_model,
            args.max_tokens,
            args.no_stream,
            true,
        )
        .await?;
    }

    println!("Done!");
    Ok(())
}
