use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, Registry, fmt, layer::SubscriberExt, util::SubscriberInitExt};

use neuromance::{Core, build_client};
use neuromance_agent::{Agent, Subagent, SubagentTool};
use neuromance_client::LLMClient;
use neuromance_db::PgConversationStore;
use neuromance_runtime::{
    ApprovalMode, Mode, RuntimeConfig, RuntimeError,
    approval::WebhookApprover,
    build_subagent_registry,
    health::{ReadinessGate, router as health_router},
    lifecycle::shutdown_handler,
    metrics as runtime_metrics, oneshot,
    proxy::build_llm_config,
    serve,
    telemetry::{self, BoxedLayer},
};
use neuromance_tools::{ToolConfig, ToolFactoryRegistry, ToolRegistry};

/// Tool name the runtime takes over to expose subagents in Python.
const EXECUTE_PYTHON: &str = "execute_python";

#[tokio::main]
async fn main() -> Result<()> {
    let prometheus_handle = runtime_metrics::init().map_err(anyhow::Error::from)?;

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
    info!(
        mode = ?config.mode,
        agent_id = %config.agent.id,
        model = %config.agent.model,
        "neuromance-runtime starting"
    );

    let cancel = CancellationToken::new();
    shutdown_handler(cancel.clone()).context("install shutdown handler")?;

    let readiness = Arc::new(ReadinessGate::new());
    let health_handle = spawn_health_server(
        &config,
        Arc::clone(&readiness),
        prometheus_handle,
        cancel.clone(),
    )
    .await
    .context("start health server")?;

    let store = init_store(&config)
        .await
        .context("initialize database store")?;
    let agent = build_agent(&config, store.clone(), &cancel).map_err(anyhow::Error::from)?;
    readiness.set_ready(true);

    let result = match config.mode {
        Mode::Oneshot => run_oneshot(&config, agent, cancel.clone()).await,
        Mode::Serve => run_serve(&config, agent, store, cancel.clone()).await,
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

    if let Err(ref e) = result {
        error!(error=%e, "runtime exited with error");
    }

    if let Some(guard) = telemetry_guard {
        guard.shutdown();
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
        base.with(filter)
            .with(fmt::layer().json().flatten_event(true))
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

/// Connect to postgres and run migrations when `[database]` is configured.
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
    store.migrate().await.context("run database migrations")?;
    info!(
        max_connections = database.max_connections,
        "database persistence enabled"
    );
    Ok(Some(Arc::new(store)))
}

fn build_agent(
    config: &RuntimeConfig,
    store: Option<Arc<PgConversationStore>>,
    cancel: &CancellationToken,
) -> Result<Agent<Box<dyn LLMClient>>, RuntimeError> {
    let llm_config = build_llm_config(config)?;
    let client =
        build_client(llm_config).map_err(|e| RuntimeError::Config(format!("build client: {e}")))?;

    let mut core = Core::new(client);
    if config.agent.streaming {
        core = core.with_streaming();
    }
    if let Some(max) = config.agent.max_turns {
        core.max_turns = Some(max);
    }
    if let Some(store) = store {
        core = core.with_persistence(store);
    }

    let subagents = build_subagent_registry(config)?;

    // When subagents exist and an execute_python tool is configured, the runtime
    // builds the REPL itself (with the subagents bridged in) so the factory must
    // not also build a plain one under the same name.
    let bridge_python = bridge_python_repl(config, &subagents);
    let factory_configs: Vec<ToolConfig> = if bridge_python {
        config
            .tools
            .iter()
            .filter(|t| t.name != EXECUTE_PYTHON)
            .cloned()
            .collect()
    } else {
        config.tools.clone()
    };

    #[cfg_attr(not(feature = "python-repl"), allow(unused_mut))]
    let mut factories = ToolFactoryRegistry::with_builtin();
    #[cfg(feature = "python-repl")]
    factories.register(neuromance_repl::python::PythonReplToolFactory);
    let staged = factories.build_all(&factory_configs)?;

    register_subagent_tools(config, &subagents, &staged, cancel)?;
    #[cfg(feature = "python-repl")]
    if bridge_python {
        register_subagent_repl(config, &subagents, &staged, cancel)?;
    }

    if matches!(config.approval.mode, ApprovalMode::Auto) {
        let mut needs_approval: Vec<String> = staged
            .tool_names()
            .into_iter()
            .filter(|name| !staged.is_tool_auto_approved(name))
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

    for name in staged.tool_names() {
        if let Some(tool) = staged.get(&name) {
            core.tool_executor.add_tool_arc(tool);
        }
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
            core = core.with_tool_approval_callback(move |tc| approver.approve(tc));
        }
    }

    Ok(Agent::new(config.agent.id.clone(), core))
}

/// Whether the runtime should take over `execute_python` to bridge subagents
/// into Python. True only when subagents are configured, the python-repl
/// feature is built in, and an `execute_python` tool entry is present.
#[cfg(feature = "python-repl")]
fn bridge_python_repl(
    config: &RuntimeConfig,
    subagents: &HashMap<String, Arc<dyn Subagent>>,
) -> bool {
    !subagents.is_empty() && config.tools.iter().any(|t| t.name == EXECUTE_PYTHON)
}

#[cfg(not(feature = "python-repl"))]
fn bridge_python_repl(
    _config: &RuntimeConfig,
    _subagents: &HashMap<String, Arc<dyn Subagent>>,
) -> bool {
    false
}

/// Register one [`SubagentTool`] per configured subagent into `staged`, so the
/// main agent can delegate to each by its id. Registering into `staged` (rather
/// than directly onto the executor) means the same startup approval gate that
/// covers factory tools also covers delegate tools.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if a subagent id collides with an
/// already-registered tool name.
fn register_subagent_tools(
    config: &RuntimeConfig,
    subagents: &HashMap<String, Arc<dyn Subagent>>,
    staged: &ToolRegistry,
    cancel: &CancellationToken,
) -> Result<(), RuntimeError> {
    for sub in &config.subagents {
        if staged.contains(&sub.id) {
            return Err(RuntimeError::Config(format!(
                "subagent id '{}' collides with a configured tool of the same name",
                sub.id
            )));
        }
        let Some(inner) = subagents.get(&sub.id).map(Arc::clone) else {
            continue;
        };
        let description = sub
            .description
            .clone()
            .unwrap_or_else(|| format!("Delegate a task to the '{}' subagent.", sub.id));
        let tool = SubagentTool::new(inner, sub.id.clone(), description, cancel.clone());
        staged.register(Arc::new(tool));
    }
    Ok(())
}

/// Build the subagent-enabled Python REPL and register it as the
/// `execute_python` tool. The bridge exposes `run_subagent`/`spawn_agents` over
/// the same subagent registry the delegate tools use.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if the `execute_python` entry requests
/// unrestricted mode (the bridge supports restricted mode only) or if building
/// the REPL or bridge fails.
#[cfg(feature = "python-repl")]
fn register_subagent_repl(
    config: &RuntimeConfig,
    subagents: &HashMap<String, Arc<dyn Subagent>>,
    staged: &ToolRegistry,
    cancel: &CancellationToken,
) -> Result<(), RuntimeError> {
    use neuromance_repl::python::{PythonRepl, SubagentRepl};

    let entry = config
        .tools
        .iter()
        .find(|t| t.name == EXECUTE_PYTHON)
        .ok_or_else(|| RuntimeError::Config("execute_python tool entry missing".to_string()))?;
    if entry.config.get("restricted") == Some(&serde_json::Value::Bool(false)) {
        return Err(RuntimeError::Config(
            "the subagent Python REPL bridge supports restricted mode only; remove \
             restricted = false from the execute_python tool config"
                .to_string(),
        ));
    }

    let repl = Arc::new(
        PythonRepl::new().map_err(|e| RuntimeError::Config(format!("build python repl: {e}")))?,
    );
    let bridge = SubagentRepl::new(repl, subagents.clone(), cancel.clone())
        .map_err(|e| RuntimeError::Config(format!("build subagent repl bridge: {e}")))?;
    staged.register(Arc::new(bridge.into_tool()));
    Ok(())
}

async fn run_oneshot(
    config: &RuntimeConfig,
    mut agent: Agent<Box<dyn LLMClient>>,
    cancel: CancellationToken,
) -> Result<()> {
    oneshot::run(config, &mut agent, cancel).await
}

async fn run_serve(
    config: &RuntimeConfig,
    agent: Agent<Box<dyn LLMClient>>,
    store: Option<Arc<PgConversationStore>>,
    cancel: CancellationToken,
) -> Result<()> {
    serve::run(config, agent, store, cancel).await
}
