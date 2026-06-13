//! Build leaf subagents from `[[subagents]]` config.
//!
//! Each entry becomes a [`LocalSubagent`] over an `Arc`-shared LLM client: the
//! client (and its connection pool) is built once at startup, and every
//! delegated run constructs a fresh [`Agent`] around a clone of that `Arc`.
//! Fresh-agent-per-run keeps concurrent runs of one subagent parallel (see
//! [`LocalSubagent`]); the shared client avoids rebuilding a reqwest pool on
//! every run.
//!
//! The returned registry is shared by both delegation surfaces: the
//! [`SubagentTool`](neuromance_agent::SubagentTool) wrappers exposed on the main
//! agent, and the Python REPL bridge (`run_subagent`/`spawn_agents`). Because it
//! is keyed by id and holds `Arc<dyn Subagent>`, both surfaces see the same set.

use std::collections::HashMap;
use std::sync::Arc;

use neuromance::Core;
use neuromance_agent::{Agent, LocalSubagent, Subagent};
use neuromance_client::{LLMClient, build_client};

use crate::config::RuntimeConfig;
use crate::error::RuntimeError;
use crate::proxy::build_llm_config_for;

/// Build the named subagent registry from `config.subagents`.
///
/// # Errors
/// Returns [`RuntimeError::Config`] (or the credential errors surfaced by
/// [`build_llm_config_for`]) if a subagent's model or credentials fail to
/// resolve into an LLM client.
pub fn build_subagent_registry(
    config: &RuntimeConfig,
) -> Result<HashMap<String, Arc<dyn Subagent>>, RuntimeError> {
    let mut registry: HashMap<String, Arc<dyn Subagent>> = HashMap::new();

    for sub in &config.subagents {
        let model = sub.model.as_deref().unwrap_or(&config.agent.model);
        let base_url = sub.base_url.as_deref().or(config.agent.base_url.as_deref());
        let llm_config = build_llm_config_for(config, model, base_url)?;

        let client: Arc<dyn LLMClient> = build_client(llm_config)
            .map_err(|e| RuntimeError::Config(format!("subagent '{}': build client: {e}", sub.id)))?
            .into();

        let id = sub.id.clone();
        let max_turns = sub.max_turns;
        let build_agent = move || {
            let mut core = Core::new(Arc::clone(&client));
            if let Some(max) = max_turns {
                core.max_turns = Some(max);
            }
            // Subagents carry no tools, so there is nothing to gate; the loop
            // only ever calls the model.
            core.auto_approve_tools = true;
            Agent::new(id.clone(), core)
        };

        let subagent = LocalSubagent::new(sub.id.clone(), sub.system_prompt.clone(), build_agent);
        registry.insert(sub.id.clone(), Arc::new(subagent));
    }

    Ok(registry)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::config::SubagentConfig;
    use crate::{AgentConfig, ApprovalConfig, Mode, RuntimeSettings};

    fn config_with_subagents(subagents: Vec<SubagentConfig>) -> RuntimeConfig {
        RuntimeConfig {
            mode: Mode::Serve,
            agent: AgentConfig {
                id: "manager".to_string(),
                model: "openai:gpt-4o".to_string(),
                api_key_env: Some("OPENAI_API_KEY".to_string()),
                system_prompt: "be helpful".to_string(),
                base_url: None,
                max_turns: None,
                streaming: false,
            },
            runtime: RuntimeSettings::default(),
            approval: ApprovalConfig::default(),
            tools: Vec::new(),
            oneshot: None,
            proxy: None,
            database: None,
            subagents,
        }
    }

    fn subagent(id: &str) -> SubagentConfig {
        SubagentConfig {
            id: id.to_string(),
            system_prompt: "you are a worker".to_string(),
            description: None,
            model: None,
            base_url: None,
            max_turns: None,
        }
    }

    /// The build path resolves credentials through `agent.api_key_env`. With
    /// that variable unset, the build fails naming it rather than silently
    /// dropping a subagent. (Env mutation is forbidden by `unsafe_code`, so the
    /// populated-key path is exercised via the proxy/integration tests.)
    #[test]
    fn test_build_surfaces_missing_credential_env() {
        let config = config_with_subagents(vec![subagent("alpha"), subagent("beta")]);
        let err = build_subagent_registry(&config)
            .err()
            .expect("build should fail without the credential env var set");
        assert!(
            matches!(err, RuntimeError::MissingEnv(ref v) if v == "OPENAI_API_KEY"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn test_empty_subagents_yields_empty_registry() {
        let config = config_with_subagents(vec![]);
        let registry = build_subagent_registry(&config).unwrap();
        assert!(registry.is_empty());
    }
}
