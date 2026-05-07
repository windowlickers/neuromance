use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, fmt};

use neuromance::{Config, Core, build_client};
use neuromance_agent::Agent;
use neuromance_client::LLMClient;
use neuromance_runtime::{
    ApprovalMode, Mode, RuntimeConfig, RuntimeError,
    approval::WebhookApprover,
    health::{ReadinessGate, router as health_router},
    lifecycle::shutdown_handler,
    oneshot, serve,
};
use neuromance_tools::ToolFactoryRegistry;

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let config = RuntimeConfig::load_default().map_err(anyhow::Error::from)?;
    info!(
        mode = ?config.mode,
        agent_id = %config.agent.id,
        model = %config.agent.model,
        "neuromance-runtime starting"
    );

    let cancel = CancellationToken::new();
    shutdown_handler(cancel.clone()).context("install shutdown handler")?;

    let readiness = Arc::new(ReadinessGate::new());
    let health_handle = spawn_health_server(&config, Arc::clone(&readiness), cancel.clone())
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
    result
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = fmt().with_env_filter(filter).with_target(true);
    match std::env::var("RUST_LOG_FORMAT").as_deref() {
        Ok("json") => builder.json().flatten_event(true).init(),
        _ => builder.init(),
    }
}

async fn spawn_health_server(
    config: &RuntimeConfig,
    readiness: Arc<ReadinessGate>,
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

    let app = health_router(readiness);
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
    let api_key = std::env::var(&config.agent.api_key_env)
        .map_err(|_| RuntimeError::MissingEnv(config.agent.api_key_env.clone()))?;

    let llm_config = Config::from_model(&config.agent.model)
        .map_err(|e| RuntimeError::Config(format!("model '{}': {e}", config.agent.model)))?
        .with_api_key(api_key);
    let client =
        build_client(llm_config).map_err(|e| RuntimeError::Config(format!("build client: {e}")))?;

    let mut core = Core::new(client);
    if config.agent.streaming {
        core = core.with_streaming();
    }
    if let Some(max) = config.agent.max_turns {
        core.max_turns = Some(max);
    }

    let factories = ToolFactoryRegistry::with_builtin();
    let staged = factories
        .build_all(&config.tools)
        .map_err(RuntimeError::Other)?;

    if matches!(config.approval.mode, ApprovalMode::Auto) {
        let mut needs_approval: Vec<String> = staged
            .tool_names()
            .into_iter()
            .filter(|name| !staged.is_tool_auto_approved(name))
            .collect();
        if !needs_approval.is_empty() {
            needs_approval.sort();
            return Err(RuntimeError::Config(format!(
                "approval.mode = \"auto\" but the following tools require explicit approval: \
                 [{}]. Either remove them from [[tools]] or set approval.mode = \"async\" \
                 with an approval.webhook_url.",
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
