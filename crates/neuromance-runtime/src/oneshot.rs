//! Oneshot mode: run one task, write the result, exit.

use anyhow::{Context, Result};
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use neuromance_agent::Agent;
use neuromance_client::LLMClient;
use neuromance_common::chat::Message;

use crate::config::RuntimeConfig;
use crate::skills::SkillRuntime;

#[derive(Debug, Serialize)]
pub struct OneshotOutput {
    pub agent_id: String,
    pub conversation_id: Uuid,
    pub content: String,
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
    skills: Option<&SkillRuntime>,
    cancel: CancellationToken,
) -> Result<()> {
    let oneshot = config
        .oneshot
        .as_ref()
        .context("oneshot mode requires [oneshot] section")?;

    let conversation_id = agent.conversation_id;
    let mut messages = vec![Message::system(
        conversation_id,
        &config.agent.system_prompt,
    )];
    if let Some(skills) = skills {
        messages.extend(skills.menu_message(conversation_id));
        messages.extend(
            skills
                .mention_messages(conversation_id, &oneshot.input)
                .await,
        );
    }
    messages.push(Message::user(conversation_id, &oneshot.input));

    info!(agent=%agent.id, conversation_id=%conversation_id, "running oneshot");
    let result = tokio::select! {
        biased;
        () = cancel.cancelled() => Err(anyhow::anyhow!("oneshot cancelled")),
        res = agent.execute(Some(messages), cancel.child_token()) => res.map_err(anyhow::Error::from),
    };

    let output = match result {
        Ok(response) => OneshotOutput {
            agent_id: agent.id.clone(),
            conversation_id,
            content: response.content.content,
            tool_responses: response.tool_responses.len(),
            success: true,
            error: None,
        },
        Err(e) => {
            error!(agent=%agent.id, error=%e, "oneshot execution failed");
            OneshotOutput {
                agent_id: agent.id.clone(),
                conversation_id,
                content: String::new(),
                tool_responses: 0,
                success: false,
                error: Some(e.to_string()),
            }
        }
    };

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
