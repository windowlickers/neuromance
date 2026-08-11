//! Oneshot mode: run one task, write the result, exit.

use anyhow::{Context, Result};
use metrics::counter;
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use neuromance::CoreError;
use neuromance_agent::Agent;
use neuromance_client::LLMClient;
use neuromance_common::chat::Message;
use neuromance_common::delegation::{self, DelegationContext};

use crate::config::RuntimeConfig;
use crate::skills::fold_menu;
use crate::workspace::{self, WorkspaceManager};

#[derive(Debug, Serialize)]
pub struct OneshotOutput {
    pub agent_id: String,
    pub conversation_id: Uuid,
    pub content: String,
    /// The response parsed as JSON when `[oneshot].output_schema` is set; omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured: Option<serde_json::Value>,
    pub tool_responses: usize,
    pub success: bool,
    pub error: Option<String>,
}

/// Execute the agent on the configured oneshot input, write the result to
/// `output_path` (or stdout), and return success/failure.
///
/// # Errors
/// Returns an error if the config has no `[oneshot]` section, the agent
/// execution fails, or writing the output fails.
pub async fn run<C: LLMClient + Send + Sync>(
    config: &RuntimeConfig,
    agent: &mut Agent<C>,
    skills_menu: Option<&str>,
    workspace: Option<&WorkspaceManager>,
    cancel: CancellationToken,
) -> Result<()> {
    let oneshot = config
        .oneshot
        .as_ref()
        .context("oneshot mode requires [oneshot] section")?;

    let conversation_id = agent.conversation_id;

    // Prepare the workspace before the run: seed from the definition named by
    // `[oneshot].workspace`, or the sole definition when exactly one exists.
    let workspace_dir = match workspace {
        Some(mgr) => {
            let definition = match oneshot.workspace.as_deref() {
                Some(name) => Some(config.workspace_definition(name).with_context(|| {
                    format!("[oneshot].workspace names no [[workspace.definitions]]: {name}")
                })?),
                None => config.sole_workspace_definition(),
            };
            Some(
                mgr.prepare(conversation_id, definition)
                    .await
                    .context("prepare workspace")?,
            )
        }
        None => None,
    };

    let mut system_prompt = fold_menu(&config.agent.system_prompt, skills_menu);
    if let Some(dir) = &workspace_dir {
        system_prompt = format!("{system_prompt}\n\n{}", workspace::note(dir));
    }
    let messages = vec![
        Message::system(conversation_id, system_prompt),
        Message::user(conversation_id, &oneshot.input),
    ];

    info!(agent=%agent.id, conversation_id=%conversation_id, "running oneshot");
    let scope_ctx = DelegationContext {
        workspace_dir,
        ..DelegationContext::default()
    };
    // Keep the typed error rather than erasing it to `anyhow` here: the outcome
    // metric below needs `CoreError::reason()`. The conversion to `anyhow`
    // happens at the `bail!` at the end.
    let result: Result<_, CoreError> = tokio::select! {
        biased;
        () = cancel.cancelled() => Err(CoreError::Cancelled("oneshot cancelled".to_string())),
        res = delegation::scope(
            scope_ctx,
            agent.execute(Some(messages), cancel.child_token()),
        ) => res,
    };

    // Best-effort final snapshot so a follow-up conversation on another pod
    // can pick the files up. No-op when persistence is off.
    if let Some(mgr) = workspace
        && let Err(e) = mgr.snapshot(conversation_id).await
    {
        error!(conversation_id = %conversation_id, error = %e, "workspace snapshot failed");
    }

    let output = record_outcome(&agent.id, conversation_id, result);

    let json = serde_json::to_string_pretty(&output)?;

    if let Some(path) = &oneshot.output_path {
        tokio::fs::write(path, &json)
            .await
            .with_context(|| format!("write {}", path.display()))?;
        info!(path=%path.display(), "oneshot output written");
    } else {
        println!("{json}");
    }

    if !output.success {
        anyhow::bail!(
            "agent execution failed: {}",
            output.error.unwrap_or_default()
        );
    }
    Ok(())
}

/// Turn a finished run into its output record and emit the outcome metric.
///
/// Oneshot reports the same `neuromance_tasks_total{outcome,reason}` series as
/// serve mode, so a `Job` and a `Deployment` running the same agent aggregate
/// together. `reason` comes from [`CoreError::reason`] — a fixed slug, never the
/// error's `Display` text, which would put provider messages into a label.
fn record_outcome(
    agent_id: &str,
    conversation_id: Uuid,
    result: Result<neuromance_agent::AgentResponse, CoreError>,
) -> OneshotOutput {
    let failed = |error: String, outcome: &'static str, reason: &'static str| {
        counter!("neuromance_tasks_total", "outcome" => outcome, "reason" => reason).increment(1);
        OneshotOutput {
            agent_id: agent_id.to_string(),
            conversation_id,
            content: String::new(),
            structured: None,
            tool_responses: 0,
            success: false,
            error: Some(error),
        }
    };

    match result {
        Ok(response) => {
            counter!("neuromance_tasks_total", "outcome" => "succeeded", "reason" => "none")
                .increment(1);
            OneshotOutput {
                agent_id: agent_id.to_string(),
                conversation_id,
                content: response.content.content,
                structured: response.structured,
                tool_responses: response.tool_responses.len(),
                success: true,
                error: None,
            }
        }
        Err(CoreError::Cancelled(reason)) => {
            error!(agent = %agent_id, %reason, "oneshot cancelled");
            failed(reason, "cancelled", "cancelled")
        }
        Err(e) => {
            let reason = e.reason();
            error!(agent = %agent_id, reason, error = %e, "oneshot execution failed");
            failed(e.to_string(), "failed", reason)
        }
    }
}
