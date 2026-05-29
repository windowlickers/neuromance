use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, Registry, fmt, layer::SubscriberExt, util::SubscriberInitExt};

use neuromance::{Core, build_client};
use neuromance_agent::Agent;
use neuromance_client::LLMClient;
use neuromance_runtime::{
    ApprovalMode, Mode, RuntimeConfig, RuntimeError,
    approval::WebhookApprover,
    health::{ReadinessGate, router as health_router},
    lifecycle::shutdown_handler,
    metrics as runtime_metrics, oneshot,
    proxy::build_llm_config,
    serve,
    telemetry::{self, BoxedLayer},
};
use neuromance_tools::ToolFactoryRegistry;

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

    let agent = build_agent(&config).map_err(anyhow::Error::from)?;
    readiness.set_ready(true);

    let result = match config.mode {
        Mode::Oneshot => run_oneshot(&config, agent, cancel.clone()).await,
        Mode::Serve => run_serve(&config, agent, cancel.clone()).await,
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

fn build_agent(config: &RuntimeConfig) -> Result<Agent<Box<dyn LLMClient>>, RuntimeError> {
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

    #[cfg_attr(not(feature = "python-repl"), allow(unused_mut))]
    let mut factories = ToolFactoryRegistry::with_builtin();
    #[cfg(feature = "python-repl")]
    factories.register(neuromance_repl::python::PythonReplToolFactory);
    let staged = factories.build_all(&config.tools)?;

    if matches!(config.approval.mode, ApprovalMode::Auto) && !config.approval.allow_unsafe_tools {
        let mut needs_approval: Vec<String> = staged
            .tool_names()
            .into_iter()
            .filter(|name| !staged.is_tool_auto_approved(name))
            .collect();
        if !needs_approval.is_empty() {
            needs_approval.sort();
            return Err(RuntimeError::Config(format!(
                "approval.mode = \"auto\" but the following tools require explicit approval: \
                 [{}]. Either remove them from [[tools]], set approval.mode = \"async\" with \
                 an approval.webhook_url, or set approval.allow_unsafe_tools = true to opt out \
                 of this safety check.",
                needs_approval.join(", ")
            )));
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
    cancel: CancellationToken,
) -> Result<()> {
    serve::run(config, agent, cancel).await
}
