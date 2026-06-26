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
use neuromance_context::rules::RulesHook;
use neuromance_context::{CompactionHook, ContextConfig};
use neuromance_db::{PersistenceHook, PgConversationStore};
use neuromance_runtime::{
    AgentBuilder, ApprovalMode, Mode, RuntimeConfig, RuntimeError, SessionReset, SkillRuntime,
    approval::WebhookApprover,
    bootstrap, build_parent_toolset,
    health::{ReadinessGate, router as health_router},
    lifecycle::shutdown_handler,
    metrics as runtime_metrics, oneshot,
    proxy::build_provider_config,
    rules, sandbox, serve, skills,
    telemetry::{self, BoxedLayer},
};

/// Process role. Defaults to the orchestrator when no subcommand is given, so
/// existing no-argument invocations keep working.
#[derive(Debug, Clone, Copy, Default, clap::Subcommand)]
enum Command {
    /// Run the orchestrator: HTTP task API, agent loop, and persistence.
    #[default]
    Run,
    /// Run the sandbox tool executor: execute capability tools over gRPC on
    /// behalf of an orchestrator, with no database access.
    Sandbox,
}

#[derive(Debug, Parser)]
#[command(name = "neuromance-runtime", about, version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let config = RuntimeConfig::load_default().map_err(anyhow::Error::from)?;

    let (telemetry_guard, otel_layer) =
        match telemetry::try_init::<Registry>(&config.agent.id).map_err(anyhow::Error::from)? {
            Some((guard, layer)) => (Some(guard), Some(layer)),
            None => (None, None),
        };
    init_tracing(otel_layer);

    if telemetry_guard.is_some() {
        info!("OTLP telemetry export enabled");
    }

    let cancel = CancellationToken::new();
    shutdown_handler(cancel.clone()).context("install shutdown handler")?;

    let result = match cli.command.unwrap_or_default() {
        Command::Run => run_orchestrator(&config, cancel.clone()).await,
        Command::Sandbox => run_sandbox(&config, cancel.clone()).await,
    };

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

    // One sandbox client, shared by tool execution and (in serve mode) per-task
    // session cleanup. The channel connects lazily, so this never blocks on a
    // not-yet-ready sandbox.
    let sandbox_client = match config.sandbox.as_ref().and_then(|s| s.endpoint.as_ref()) {
        Some(endpoint) => {
            Some(sandbox::SandboxClient::connect(endpoint).map_err(anyhow::Error::from)?)
        }
        None => None,
    };

    let skills = skills::build(config.skills.as_ref()).await;

    let rules = rules::build(config.rules.as_ref()).await;

    // The factory owns every input `build_agent` needs so serve mode can rebuild
    // a fresh agent for a task that overrides the model. The initial shared agent
    // is built through it too, so there is a single construction path.
    let factory = Arc::new(AgentFactory {
        config: Arc::new(config.clone()),
        store: store.clone(),
        sandbox_client: sandbox_client.clone(),
        skills: skills.clone(),
        rules: rules.clone(),
        cancel: cancel.clone(),
    });
    let (agent, local_python) = factory.build(None).await.map_err(anyhow::Error::from)?;

    // Best-effort: run one-time tool setup before tasks, since the pod has no
    // persistent storage to cache credentials a tool writes to disk.
    bootstrap::run(&config.bootstrap).await;

    readiness.set_ready(true);

    // The menu lists each skill's materialized file path; fold it into the seed
    // system prompt so the agent reads skills from disk.
    let skills_menu = skills.as_ref().and_then(|s| s.menu());

    let result = match config.mode {
        Mode::Oneshot => run_oneshot(config, agent, skills_menu.as_deref(), cancel.clone()).await,
        Mode::Serve => {
            run_serve(
                config,
                agent,
                factory,
                store,
                sandbox_client,
                local_python,
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

/// Run the sandbox role: serve the gRPC tool service until shutdown. No agent,
/// database, or HTTP task API — just tool execution.
async fn run_sandbox(config: &RuntimeConfig, cancel: CancellationToken) -> Result<()> {
    let settings = config.sandbox.as_ref().ok_or_else(|| {
        anyhow::anyhow!("the sandbox subcommand requires a [sandbox] config section")
    })?;
    let addr: SocketAddr = settings
        .listen_addr
        .parse()
        .with_context(|| format!("invalid sandbox.listen_addr: {}", settings.listen_addr))?;

    let toolset = Arc::new(sandbox::server::build_sandbox_toolset(&config.tools)?);
    info!(%addr, tools = config.tools.len(), "neuromance-runtime sandbox starting");

    sandbox::server::serve(toolset, addr, cancel)
        .await
        .map_err(anyhow::Error::from)
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
    let addr: std::net::SocketAddr = config
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
    let context_config = ContextConfig::new(ctx.context_window_size)
        .with_compaction_threshold_ratio(ctx.compaction_threshold_ratio)
        .with_target_ratio(ctx.target_ratio)
        .with_preserve_recent_turns(ctx.preserve_recent_turns)
        .with_strategy(ctx.strategy);
    let core = core.with_hook(Arc::new(CompactionHook::new(
        compaction_client,
        &context_config,
    )));
    info!(
        window = ctx.context_window_size,
        threshold_ratio = ctx.compaction_threshold_ratio,
        target_ratio = ctx.target_ratio,
        "context compaction enabled (usage-driven)"
    );
    Ok(core)
}

async fn build_agent(
    config: &RuntimeConfig,
    store: Option<&Arc<PgConversationStore>>,
    sandbox_client: Option<&sandbox::SandboxClient>,
    skills: Option<&Arc<SkillRuntime>>,
    rules: Option<&Arc<RulesHook>>,
    cancel: &CancellationToken,
    model_override: Option<&str>,
) -> Result<(Agent<Box<dyn LLMClient>>, Option<SessionReset>), RuntimeError> {
    let provider = config.provider(&config.agent.provider).ok_or_else(|| {
        RuntimeError::Config(format!(
            "agent.provider '{}' does not match any [[providers]] entry",
            config.agent.provider
        ))
    })?;
    // A per-task override swaps only the model string; the configured provider
    // still supplies the credential and endpoint. The override's `provider:`
    // prefix selects the client family (see `Config::from_model`).
    let model = match model_override {
        Some(model) => model,
        None => config.agent_model().ok_or_else(|| {
            RuntimeError::Config(format!(
                "agent has no model: set agent.model or provider '{}' model",
                config.agent.provider
            ))
        })?,
    };
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
    if let Some(store) = store {
        let sink: Arc<PgConversationStore> = Arc::clone(store);
        core = core.with_hook(Arc::new(PersistenceHook::new(sink)));
    }

    // When a sandbox endpoint is configured, capability tools execute in the
    // sandbox process: fetch their definitions once and reuse the adapters
    // across the main agent and every subagent.
    let remote_capabilities = match sandbox_client {
        Some(client) => Some(sandbox::connect_tools(client).await?),
        None => None,
    };

    // The main agent's toolset, including delegate tools for every configured
    // subagent and the delegation tower beneath them (bounded by
    // runtime.max_delegation_depth). The store is threaded through so subagent
    // conversations persist and record their parent link too.
    let (tools, local_python) =
        build_parent_toolset(config, store, cancel, remote_capabilities.as_deref())?;

    if matches!(config.approval.mode, ApprovalMode::Auto) {
        let mut needs_approval: Vec<String> = tools
            .iter()
            .filter(|tool| !tool.is_auto_approved())
            .map(|tool| tool.get_definition().function.name)
            .collect();
        if !needs_approval.is_empty() {
            needs_approval.sort();
            if config.approval.allow_unsafe_tools {
                warn!(
                    tools = %needs_approval.join(", "),
                    "approval.allow_unsafe_tools is set: auto-approving tools that would \
                     otherwise require explicit approval, bypassing the startup safety check"
                );
            } else {
                return Err(RuntimeError::Config(format!(
                    "approval.mode = \"auto\" but the following tools require explicit approval: \
                     [{}]. Either remove them from [[tools]], set approval.mode = \"async\" with \
                     an approval.webhook_url, or set approval.allow_unsafe_tools = true to opt out \
                     of this safety check.",
                    needs_approval.join(", ")
                )));
            }
        }
    }

    for tool in tools {
        core.tool_executor.add_tool_arc(tool);
    }

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

    Ok((Agent::new(config.agent.id.clone(), core), local_python))
}

/// Owns the startup inputs `build_agent` needs so the serve worker can build a
/// fresh agent per task when a task carries a model override. Every input is
/// already cheap to share (`Arc`, a lazy sandbox channel, a token), so cloning
/// it into the factory costs nothing meaningful.
struct AgentFactory {
    config: Arc<RuntimeConfig>,
    store: Option<Arc<PgConversationStore>>,
    sandbox_client: Option<sandbox::SandboxClient>,
    skills: Option<Arc<SkillRuntime>>,
    rules: Option<Arc<RulesHook>>,
    cancel: CancellationToken,
}

#[async_trait::async_trait]
impl AgentBuilder for AgentFactory {
    async fn build(
        &self,
        model_override: Option<&str>,
    ) -> Result<(Agent<Box<dyn LLMClient>>, Option<SessionReset>), RuntimeError> {
        build_agent(
            &self.config,
            self.store.as_ref(),
            self.sandbox_client.as_ref(),
            self.skills.as_ref(),
            self.rules.as_ref(),
            &self.cancel,
            model_override,
        )
        .await
    }
}

async fn run_oneshot(
    config: &RuntimeConfig,
    mut agent: Agent<Box<dyn LLMClient>>,
    skills_menu: Option<&str>,
    cancel: CancellationToken,
) -> Result<()> {
    oneshot::run(config, &mut agent, skills_menu, cancel).await
}

#[allow(clippy::too_many_arguments)]
async fn run_serve(
    config: &RuntimeConfig,
    agent: Agent<Box<dyn LLMClient>>,
    builder: Arc<dyn AgentBuilder>,
    store: Option<Arc<PgConversationStore>>,
    sandbox_client: Option<sandbox::SandboxClient>,
    local_python: Option<SessionReset>,
    skills_menu: Option<Arc<str>>,
    cancel: CancellationToken,
) -> Result<()> {
    serve::run(
        config,
        agent,
        builder,
        store,
        sandbox_client,
        local_python,
        skills_menu,
        cancel,
    )
    .await
}
