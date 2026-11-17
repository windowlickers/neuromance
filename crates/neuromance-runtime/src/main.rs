use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, Registry, fmt, layer::SubscriberExt, util::SubscriberInitExt};

use neuromance::{Core, build_client};
use neuromance_agent::Agent;
use neuromance_client::LLMClient;
use neuromance_common::FnReviewHook;
use neuromance_context::CompactionHook;
use neuromance_context::rules::RulesHook;
use neuromance_db::{PersistenceHook, PgConversationStore};
use neuromance_runtime::{
    AgentBuilder, AgentOverrides, ApprovalMode, Mode, RuntimeConfig, RuntimeError, SkillRuntime,
    ToolsetParams,
    approval::WebhookApprover,
    bootstrap, build_parent_toolset,
    health::{ReadinessGate, router as health_router},
    lifecycle::shutdown_handler,
    mcp, metrics as runtime_metrics, oneshot,
    proxy::build_provider_config,
    rules, serve, skills,
    telemetry::{self, BoxedLayer},
};
use neuromance_tools::ToolImplementation;

#[derive(Debug, Parser)]
#[command(name = "neuromance-runtime", about, version)]
struct Cli {}

#[tokio::main]
async fn main() -> Result<()> {
    Cli::parse();

    let config = RuntimeConfig::load_default().map_err(anyhow::Error::from)?;

    let (telemetry_guard, otel_layer) =
        match telemetry::try_init::<Registry>(&config.agent.id).map_err(anyhow::Error::from)? {
            Some((guard, layer)) => (Some(guard), Some(layer)),
            None => (None, None),
        };
    init_tracing(otel_layer);

    if telemetry_guard.is_some() {
        info!("OTLP telemetry export enabled for traces, logs, and gen_ai metrics");
    }

    let cancel = CancellationToken::new();
    shutdown_handler(cancel.clone()).context("install shutdown handler")?;

    let result = run_orchestrator(&config, cancel.clone()).await;

    cancel.cancel();

    if let Err(ref e) = result {
        error!(error=%e, "runtime exited with error");
    }

    if let Some(guard) = telemetry_guard {
        guard.shutdown();
    }

    result
}

/// Run the orchestrator role: metrics, health server, agent, and the
/// configured serving mode.
async fn run_orchestrator(config: &RuntimeConfig, cancel: CancellationToken) -> Result<()> {
    let prometheus_handle = runtime_metrics::init().map_err(anyhow::Error::from)?;
    info!(
        mode = ?config.mode,
        agent_id = %config.agent.id,
        provider = %config.agent.provider,
        model = config.agent_model().unwrap_or("<unset>"),
        "neuromance-runtime starting"
    );

    let readiness = Arc::new(ReadinessGate::new());
    let health_handle = spawn_health_server(
        config,
        Arc::clone(&readiness),
        prometheus_handle,
        cancel.clone(),
    )
    .await
    .context("start health server")?;

    let store = init_store(config)
        .await
        .context("initialize database store")?;

    let mcp_tools = mcp::connect(&config.mcp_servers)
        .await
        .map_err(anyhow::Error::from)?;

    let skills = skills::build(config.skills.as_ref()).await;

    let rules = rules::build(config.rules.as_ref()).await;

    // The factory owns every input `build_agent` needs so serve mode can rebuild
    // a fresh agent for a task that overrides the model. The initial shared agent
    // is built through it too, so there is a single construction path.
    let factory = Arc::new(AgentFactory {
        config: Arc::new(config.clone()),
        store: store.clone(),
        mcp_tools,
        skills: skills.clone(),
        rules: rules.clone(),
        cancel: cancel.clone(),
    });
    // A oneshot `output_schema` constrains the single run, so it is applied when the shared
    // agent is built; serve mode carries the schema per task instead.
    let initial_overrides = AgentOverrides {
        output_schema: config.oneshot_output_schema()?,
        ..AgentOverrides::default()
    };
    let agent = factory
        .build(&initial_overrides)
        .await
        .map_err(anyhow::Error::from)?;

    // Best-effort one-time tool setup before tasks (the pod has no persistent
    // storage to cache credentials a tool writes to disk).
    bootstrap::run(&config.bootstrap).await;

    readiness.set_ready(true);

    // The menu lists each skill's materialized file path; fold it into the seed
    // system prompt so the agent reads skills from disk.
    let skills_menu = skills.as_ref().and_then(|s| s.menu());

    let result = match config.mode {
        Mode::Oneshot => {
            let mut agent = agent;
            oneshot::run(config, &mut agent, skills_menu.as_deref(), cancel.clone()).await
        }
        Mode::Serve => {
            serve::run(
                config,
                agent,
                factory,
                store,
                skills_menu.map(Arc::from),
                cancel.clone(),
            )
            .await
        }
    };

    readiness.set_ready(false);
    cancel.cancel();

    if let Err(e) = tokio::time::timeout(
        Duration::from_secs(config.runtime.shutdown_grace_seconds),
        health_handle,
    )
    .await
    {
        warn!(error=%e, "health server did not shut down within grace period");
    }

    result
}

fn init_tracing(extra_layer: Option<BoxedLayer<Registry>>) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let json = matches!(std::env::var("RUST_LOG_FORMAT").as_deref(), Ok("json"));
    // The OTLP layer is parameterized over `Registry` and must sit directly on
    // the Registry; later `.with()` calls add Layered<...> wrappers that the
    // boxed layer's type can't accept. Option<L> is used over Vec<L> because
    // an empty Vec reports Interest::never() and silences every event.
    let base = Registry::default().with(extra_layer);
    if json {
        // `with_current_span(false)` drops the per-event `span` object, which only
        // duplicates the leaf of the `spans` stack already on every line. The stack
        // is kept for task/conversation correlation.
        base.with(filter)
            .with(
                fmt::layer()
                    .json()
                    .flatten_event(true)
                    .with_current_span(false),
            )
            .init();
    } else {
        base.with(filter)
            .with(fmt::layer().with_target(true))
            .init();
    }
}

async fn spawn_health_server(
    config: &RuntimeConfig,
    readiness: Arc<ReadinessGate>,
    prometheus_handle: metrics_exporter_prometheus::PrometheusHandle,
    cancel: CancellationToken,
) -> Result<tokio::task::JoinHandle<()>> {
    let addr: SocketAddr = config
        .runtime
        .health_addr
        .parse()
        .with_context(|| format!("invalid health_addr: {}", config.runtime.health_addr))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind health server to {addr}"))?;
    info!(%addr, "health server listening");

    let app = health_router(readiness).merge(runtime_metrics::router(prometheus_handle));
    let handle = tokio::spawn(async move {
        let result = axum::serve(listener, app)
            .with_graceful_shutdown(async move { cancel.cancelled().await })
            .await;
        if let Err(e) = result {
            warn!(error=%e, "health server exited with error");
        }
    });
    Ok(handle)
}

/// Connect to postgres when `[database]` is configured, running the embedded
/// migrations unless `run_migrations = false` hands schema ownership to an
/// external manager.
///
/// Failures here abort startup: the readiness gate stays unready and the
/// orchestrator restarts the pod. A silently-disabled database would record
/// nothing while looking healthy — worse than a crash loop.
async fn init_store(config: &RuntimeConfig) -> Result<Option<Arc<PgConversationStore>>> {
    let Some(database) = &config.database else {
        return Ok(None);
    };
    let url = std::env::var(&database.url_env).with_context(|| {
        format!(
            "database.url_env: environment variable '{}' is not set",
            database.url_env
        )
    })?;
    let store = PgConversationStore::connect(
        &url,
        database.max_connections,
        Duration::from_secs(database.acquire_timeout_seconds),
    )
    .await
    .with_context(|| format!("connect to postgres (env: {})", database.url_env))?;
    if database.run_migrations {
        store.migrate().await.context("run database migrations")?;
    } else {
        warn!(
            "skipping database migrations (run_migrations = false); schema is externally managed"
        );
    }
    info!(
        max_connections = database.max_connections,
        run_migrations = database.run_migrations,
        "database persistence enabled"
    );
    Ok(Some(Arc::new(store)))
}

/// Wire usage-driven context compaction onto `core` when `[context]` is set.
///
/// Builds a separate summarization client from the same provider config so it
/// inherits the proxy / sealed-token setup. Returns `core` unchanged when no
/// `[context]` section is present.
fn apply_context_compaction(
    core: Core<Box<dyn LLMClient>>,
    config: &RuntimeConfig,
    llm_config: neuromance::Config,
) -> Result<Core<Box<dyn LLMClient>>, RuntimeError> {
    let Some(ctx) = &config.context else {
        return Ok(core);
    };
    let compaction_client = build_client(llm_config)
        .map_err(|e| RuntimeError::Config(format!("build compaction client: {e}")))?;
    let core = core.with_hook(Arc::new(CompactionHook::new(
        compaction_client,
        &ctx.to_context_config(),
    )));
    info!(
        window = ctx.context_window_size,
        threshold_ratio = ctx.compaction_threshold_ratio,
        target_ratio = ctx.target_ratio,
        "context compaction enabled (usage-driven)"
    );
    Ok(core)
}

fn build_agent(
    inputs: &AgentFactory,
    overrides: &AgentOverrides,
) -> Result<Agent<Box<dyn LLMClient>>, RuntimeError> {
    let config: &RuntimeConfig = &inputs.config;
    let store = inputs.store.as_ref();
    let skills = inputs.skills.as_ref();
    let rules = inputs.rules.as_ref();
    let cancel = &inputs.cancel;
    let (provider, model) = config
        .resolve_provider_and_model(overrides.provider.as_deref(), overrides.model.as_deref())?;
    let llm_config = build_provider_config(provider, model)?;
    let client = build_client(llm_config.clone())
        .map_err(|e| RuntimeError::Config(format!("build client: {e}")))?;

    let mut core = Core::new(client);
    if config.agent.streaming {
        core = core.with_streaming();
    }
    if let Some(max) = config.agent.max_turns {
        core.max_turns = Some(max);
    }
    core.empty_turn_retries = config.agent.empty_turn_retries;
    core.output_schema.clone_from(&overrides.output_schema);
    if let Some(store) = store {
        let sink: Arc<PgConversationStore> = Arc::clone(store);
        core = core.with_hook(Arc::new(PersistenceHook::new(sink)));
    }

    // The main agent's toolset, including delegate tools for every configured
    // subagent and the delegation tower beneath them (bounded by
    // runtime.max_delegation_depth). The tower gets the same startup handles the
    // main agent does, and the provider/model resolved above — so a subagent
    // that pins neither follows this task's override too.
    let tools = build_parent_toolset(
        config,
        &ToolsetParams {
            store,
            skills,
            rules,
            mcp_tools: &inputs.mcp_tools,
            parent_provider: provider,
            parent_model: model,
            cancel,
        },
    )?;

    reject_unapproved_tools_under_auto_mode(config, &tools)?;

    for tool in tools {
        core.tool_executor.add_tool_arc(tool);
    }

    core = apply_approval(core, config)?;
    core = apply_context_compaction(core, config, llm_config)?;

    // The skills menu is folded into the system prompt at seed time; the hook
    // only expands `$mention`ed bodies from the latest user message in-loop.
    if let Some(skills) = skills {
        core = core.with_hook(skills.hook() as Arc<dyn neuromance_common::hook::Hook>);
    }

    // Rules inject always-apply guidance at conversation start and glob-matched
    // guidance after a tool touches a matching path, entirely inside the loop.
    if let Some(rules) = rules {
        core = core.with_hook(Arc::clone(rules) as Arc<dyn neuromance_common::hook::Hook>);
    }

    Ok(Agent::new(config.agent.id.clone(), core))
}

/// Fail startup when `approval.mode = "auto"` would silently auto-approve tools
/// that ask for explicit approval.
///
/// The check runs before any tool is added to the executor, so an unsafe
/// combination never reaches a run. `approval.allow_unsafe_tools` downgrades it
/// to a warning for operators who accept the tradeoff.
///
/// # Errors
/// Returns [`RuntimeError::Config`] naming every offending tool and the ways out.
fn reject_unapproved_tools_under_auto_mode(
    config: &RuntimeConfig,
    tools: &[Arc<dyn ToolImplementation>],
) -> Result<(), RuntimeError> {
    if !matches!(config.approval.mode, ApprovalMode::Auto) {
        return Ok(());
    }
    let mut needs_approval: Vec<String> = tools
        .iter()
        .filter(|tool| !tool.is_auto_approved())
        .map(|tool| tool.get_definition().function.name)
        .collect();
    if needs_approval.is_empty() {
        return Ok(());
    }
    needs_approval.sort();
    if config.approval.allow_unsafe_tools {
        warn!(
            tools = %needs_approval.join(", "),
            "approval.allow_unsafe_tools is set: auto-approving tools that would \
             otherwise require explicit approval, bypassing the startup safety check"
        );
        return Ok(());
    }
    Err(RuntimeError::Config(format!(
        "approval.mode = \"auto\" but the following tools require explicit approval: [{}]. \
         Either remove them from [[tools]], set approval.mode = \"async\" with an \
         approval.webhook_url, or set approval.allow_unsafe_tools = true to opt out of this \
         safety check.",
        needs_approval.join(", ")
    )))
}

/// Wire the agent's tool-approval policy: blanket auto-approval, or a webhook
/// approver consulted per tool call.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if `async` mode has no `webhook_url`, or if
/// the approver rejects the configured URL or timeout.
fn apply_approval(
    mut core: Core<Box<dyn LLMClient>>,
    config: &RuntimeConfig,
) -> Result<Core<Box<dyn LLMClient>>, RuntimeError> {
    match config.approval.mode {
        ApprovalMode::Auto => {
            core.auto_approve_tools = true;
        }
        ApprovalMode::Async => {
            let url = config.approval.webhook_url.clone().ok_or_else(|| {
                RuntimeError::Config(
                    "approval.mode = \"async\" requires approval.webhook_url".to_string(),
                )
            })?;
            let approver = WebhookApprover::new(
                config.agent.id.clone(),
                url,
                Duration::from_secs(config.approval.timeout_seconds),
            )?;
            core = core.with_hook(Arc::new(FnReviewHook::new(move |tc| approver.approve(tc))));
        }
    }
    Ok(core)
}

/// Owns the startup inputs `build_agent` needs so the serve worker can build a
/// fresh agent per task when a task carries a model override. Every input is
/// already cheap to share (`Arc`, a token), so cloning
/// it into the factory costs nothing meaningful.
struct AgentFactory {
    config: Arc<RuntimeConfig>,
    store: Option<Arc<PgConversationStore>>,
    /// Adapters for every `[[mcp_servers]]` tool, connected once at startup.
    mcp_tools: Vec<Arc<dyn ToolImplementation>>,
    skills: Option<Arc<SkillRuntime>>,
    rules: Option<Arc<RulesHook>>,
    cancel: CancellationToken,
}

#[async_trait::async_trait]
impl AgentBuilder for AgentFactory {
    async fn build(
        &self,
        overrides: &AgentOverrides,
    ) -> Result<Agent<Box<dyn LLMClient>>, RuntimeError> {
        build_agent(self, overrides)
    }
}
