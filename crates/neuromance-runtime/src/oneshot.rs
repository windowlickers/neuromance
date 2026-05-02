//! Oneshot mode: run one task, write the result, exit.

use anyhow::{Context, Result};
use serde::Serialize;
use tracing::{error, info};
use uuid::Uuid;

use neuromance_agent::{Agent, BaseAgent};
use neuromance_client::LLMClient;
use neuromance_common::chat::Message;

use crate::config::RuntimeConfig;

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
    agent: &mut BaseAgent<C>,
) -> Result<()> {
    let oneshot = config
        .oneshot
        .as_ref()
        .context("oneshot mode requires [oneshot] section")?;

    let conversation_id = agent.conversation_id;
    let messages = vec![
        Message::system(conversation_id, &config.agent.system_prompt),
        Message::user(conversation_id, &oneshot.input),
    ];

    info!(agent=%agent.id, conversation_id=%conversation_id, "running oneshot");
    let result = agent.execute(Some(messages)).await;

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
        std::fs::write(path, &json)
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
