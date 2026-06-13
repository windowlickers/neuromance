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
use crate::proxy::build_provider_config;

/// Build the named subagent registry from `config.subagents`.
///
/// Each subagent inherits the parent agent's provider unless it names its own.
/// The effective model is the subagent's `model` override, then the chosen
/// provider's default `model`, then the agent's effective model.
///
/// # Errors
/// Returns [`RuntimeError::Config`] (or the credential errors surfaced by
/// [`build_provider_config`]) if a subagent's provider, model, or credentials
/// fail to resolve into an LLM client.
pub fn build_subagent_registry(
    config: &RuntimeConfig,
) -> Result<HashMap<String, Arc<dyn Subagent>>, RuntimeError> {
    let mut registry: HashMap<String, Arc<dyn Subagent>> = HashMap::new();
    let agent_model = config.agent_model();

    for sub in &config.subagents {
        let provider_name = sub.provider.as_deref().unwrap_or(&config.agent.provider);
        let provider = config.provider(provider_name).ok_or_else(|| {
            RuntimeError::Config(format!(
                "subagent '{}' provider '{provider_name}' does not match any [[providers]] entry",
                sub.id
            ))
        })?;
        let model = sub
            .model
            .as_deref()
            .or(provider.model.as_deref())
            .or(agent_model)
            .ok_or_else(|| {
                RuntimeError::Config(format!(
                    "subagent '{}' has no model: set subagent.model, provider '{provider_name}' \
                     model, or agent.model",
                    sub.id
                ))
            })?;
        let llm_config = build_provider_config(provider, model)?;

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
    use crate::config::{ProviderConfig, SubagentConfig};
    use crate::{AgentConfig, ApprovalConfig, Mode, RuntimeSettings};

    /// A single-provider config whose `[agent]` points at provider "primary"
    /// (env-var credential `OPENAI_API_KEY`, model `openai:gpt-4o`).
    fn config_with_subagents(subagents: Vec<SubagentConfig>) -> RuntimeConfig {
        config_with_providers(
            vec![provider("primary", "OPENAI_API_KEY", "openai:gpt-4o")],
            "primary",
            subagents,
        )
    }

    fn config_with_providers(
        providers: Vec<ProviderConfig>,
        agent_provider: &str,
        subagents: Vec<SubagentConfig>,
    ) -> RuntimeConfig {
        RuntimeConfig {
            mode: Mode::Serve,
            agent: AgentConfig {
                id: "manager".to_string(),
                provider: agent_provider.to_string(),
                model: None,
                system_prompt: "be helpful".to_string(),
                max_turns: None,
                streaming: false,
            },
            runtime: RuntimeSettings::default(),
            approval: ApprovalConfig::default(),
            tools: Vec::new(),
            oneshot: None,
            providers,
            database: None,
            subagents,
        }
    }

    fn provider(name: &str, api_key_env: &str, model: &str) -> ProviderConfig {
        ProviderConfig {
            name: name.to_string(),
            model: Some(model.to_string()),
            base_url: None,
            api_key_env: Some(api_key_env.to_string()),
            proxy: None,
        }
    }

    fn subagent(id: &str) -> SubagentConfig {
        SubagentConfig {
            id: id.to_string(),
            system_prompt: "you are a worker".to_string(),
            description: None,
            provider: None,
            model: None,
            max_turns: None,
        }
    }

    /// The build path resolves credentials through the inherited provider's
    /// `api_key_env`. With that variable unset, the build fails naming it
    /// rather than silently dropping a subagent. (Env mutation is forbidden by
    /// `unsafe_code`, so the populated-key path is exercised via the
    /// proxy/integration tests.)
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

    /// A subagent with no `provider` inherits the agent's provider, so its
    /// credential resolves through that provider's `api_key_env`.
    #[test]
    fn test_subagent_inherits_parent_provider_credential() {
        let config = config_with_subagents(vec![subagent("worker")]);
        let err = build_subagent_registry(&config)
            .err()
            .expect("build should fail on the inherited provider's unset env var");
        assert!(
            matches!(err, RuntimeError::MissingEnv(ref v) if v == "OPENAI_API_KEY"),
            "unexpected error: {err}",
        );
    }

    /// A subagent's `provider` override switches the credential path: with two
    /// providers keyed to different env vars, the override decides which unset
    /// variable surfaces.
    #[test]
    fn test_subagent_provider_override_switches_credential() {
        let config = config_with_providers(
            vec![
                provider("primary", "PRIMARY_KEY", "openai:gpt-4o"),
                provider("secondary", "SECONDARY_KEY", "openai:gpt-4o-mini"),
            ],
            "primary",
            vec![SubagentConfig {
                provider: Some("secondary".to_string()),
                ..subagent("worker")
            }],
        );
        let err = build_subagent_registry(&config)
            .err()
            .expect("build should fail on the overridden provider's unset env var");
        assert!(
            matches!(err, RuntimeError::MissingEnv(ref v) if v == "SECONDARY_KEY"),
            "expected the overridden provider's env var, got: {err}",
        );
    }

    /// An unknown `subagent.provider` is reported by name. (Reached only if it
    /// slips past `validate`; the build path defends against it directly.)
    #[test]
    fn test_subagent_unknown_provider_is_named() {
        let config = config_with_subagents(vec![SubagentConfig {
            provider: Some("ghost".to_string()),
            ..subagent("worker")
        }]);
        let err = build_subagent_registry(&config)
            .err()
            .expect("build should fail for an unknown provider");
        assert!(
            matches!(err, RuntimeError::Config(ref msg)
                if msg.contains("provider 'ghost'") && msg.contains("worker")),
            "unexpected error: {err}",
        );
    }
}
