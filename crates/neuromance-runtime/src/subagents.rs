//! Build the main agent's toolset and the subagent delegation tower from
//! `[[subagents]]` config.
//!
//! A subagent is provisioned with the *same* toolset as the main agent:
//! capability tools (the built-in factories plus anything in `[[tools]]`), the
//! `execute_python` bridge, and the delegate tools that let it hand work to
//! other subagents. Delegation is bounded by `runtime.max_delegation_depth`,
//! and that bound is enforced *structurally* by building a finite tower of
//! subagent instances rather than threading a runtime counter:
//!
//! - Every configured subagent exists at each depth level.
//! - A subagent at depth *k* gets delegate tools wired to the subagents at
//!   depth *k+1*.
//! - The deepest level holds no delegate tools, which terminates the recursion
//!   and breaks the otherwise-circular wiring (a subagent's toolset would
//!   otherwise contain delegate tools that wrap that same subagent).
//!
//! Each [`LocalSubagent`] holds an `Arc`-shared LLM client (built once at
//! startup) and a factory that constructs a fresh [`Agent`] per run, so
//! concurrent runs of one subagent stay parallel. The factory reassembles the
//! whole toolset on every run rather than cloning a shared template, so each
//! run — including each concurrent sibling run in a `spawn_agents` fan-out —
//! gets its own Python interpreter. No interpreter state bleeds across runs.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use neuromance::Core;
use neuromance_agent::{Agent, LocalSubagent, Subagent, SubagentError, SubagentTool};
use neuromance_client::{LLMClient, build_client};
use neuromance_db::PgConversationStore;
use neuromance_tools::{ToolConfig, ToolFactoryRegistry, ToolImplementation, ToolRegistry};

use crate::config::RuntimeConfig;
use crate::error::RuntimeError;
use crate::proxy::build_provider_config;
use crate::sandbox::EXECUTE_PYTHON;

/// A per-task cleanup handle for the main agent's in-process `execute_python`
/// interpreter. Calling it clears the interpreter's user namespace so state
/// from one serve task never bleeds into the next.
///
/// `None` accompanies a toolset with no resettable interpreter: `execute_python`
/// is unconfigured, runs in the sandbox (keyed by task there instead), or the
/// `python-repl` feature is disabled.
pub type SessionReset = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// An assembled toolset paired with the optional reset handle for its
/// in-process `execute_python` interpreter (see [`SessionReset`]).
type Toolset = (Vec<Arc<dyn ToolImplementation>>, Option<SessionReset>);

/// Build the main agent's toolset, including delegate tools for every
/// configured subagent and (when `execute_python` is configured) the Python
/// delegation bridge.
///
/// When no subagents are configured this is just the capability toolset built
/// from `[[tools]]`. Otherwise it also builds the delegation tower down to
/// `runtime.max_delegation_depth` and wires the main agent's delegate tools to
/// the top of that tower.
///
/// `store`, when present, is wired into every subagent's `Core` so child
/// conversations persist (and record their parent/child lineage) just like the
/// main agent's.
///
/// `remote_capabilities`, when present, are the sandbox-backed capability tools
/// (`bash`, file tools, `execute_python`, …) that replace the locally-built
/// ones. They are shared (cloned) across the main agent and every subagent;
/// delegate tools are still built locally per level.
///
/// # Errors
/// Returns [`RuntimeError`] if a subagent's provider/model/credentials fail to
/// resolve, a tool factory fails, or a subagent id collides with a configured
/// tool name.
pub fn build_parent_toolset(
    config: &RuntimeConfig,
    store: Option<&Arc<PgConversationStore>>,
    cancel: &CancellationToken,
    remote_capabilities: Option<&[Arc<dyn ToolImplementation>]>,
) -> Result<Toolset, RuntimeError> {
    let children = if config.subagents.is_empty() {
        HashMap::new()
    } else {
        // The main agent is depth 0; its children may delegate `depth - 1`
        // further hops.
        let remaining = config.runtime.max_delegation_depth.saturating_sub(1);
        build_subagents_at_depth(config, remaining, store, cancel, remote_capabilities)?
    };
    assemble_toolset(config, &children, cancel, remote_capabilities)
}

/// Build the subagent registry for one tower level.
///
/// `remaining` is the number of delegation hops still allowed *below* this
/// level: at `0` the level is a leaf (its subagents get no delegate tools); above
/// `0` each subagent can delegate to the level built with `remaining - 1`.
fn build_subagents_at_depth(
    config: &RuntimeConfig,
    remaining: u32,
    store: Option<&Arc<PgConversationStore>>,
    cancel: &CancellationToken,
    remote_capabilities: Option<&[Arc<dyn ToolImplementation>]>,
) -> Result<HashMap<String, Arc<dyn Subagent>>, RuntimeError> {
    let children = if remaining == 0 {
        HashMap::new()
    } else {
        build_subagents_at_depth(config, remaining - 1, store, cancel, remote_capabilities)?
    };
    // Shared across every subagent at this level and captured by each run's
    // builder. The toolset itself is *not* built here: each run reassembles it
    // (see the builder closure) so stateful tools like the Python interpreter
    // never bleed across concurrent sibling runs.
    let children = Arc::new(children);
    let config = Arc::new(config.clone());
    let agent_model = config.agent_model();

    let mut registry: HashMap<String, Arc<dyn Subagent>> = HashMap::new();
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
        let config = Arc::clone(&config);
        let children = Arc::clone(&children);
        let cancel = cancel.clone();
        let store = store.cloned();
        // Shared sandbox capability tools (stateless handles), cloned into each
        // run's toolset. Under the sandbox a subagent never carries
        // execute_python (rejected by config validation), so there is no
        // interpreter state to keep fresh across runs.
        let remote_capabilities = remote_capabilities.map(<[_]>::to_vec);
        let build_agent = move || {
            // Reassemble the toolset per run so a fresh Python interpreter is
            // built each time; nothing persists across runs of one subagent or
            // across concurrent sibling runs. A subagent rebuilds per run, so it
            // has no need of the parent's between-task reset handle.
            let (tools, _reset) =
                assemble_toolset(&config, &children, &cancel, remote_capabilities.as_deref())
                    .map_err(SubagentError::execution)?;
            let mut core = Core::new(Arc::clone(&client));
            if let Some(max) = max_turns {
                core.max_turns = Some(max);
            }
            // Subagent tool calls run autonomously inside one parent delegation,
            // with no interactive approver in the loop; the pod boundary (kata)
            // is the isolation. See the README Subagents section.
            core.auto_approve_tools = true;
            // Persist child conversations (and their parent link) when the
            // runtime has a store, matching the main agent.
            if let Some(store) = &store {
                let sink: Arc<PgConversationStore> = Arc::clone(store);
                core = core.with_persistence(sink);
            }
            for tool in tools {
                core.tool_executor.add_tool_arc(tool);
            }
            Ok(Agent::new(id.clone(), core))
        };

        let subagent = LocalSubagent::new(sub.id.clone(), sub.system_prompt.clone(), build_agent);
        registry.insert(sub.id.clone(), Arc::new(subagent));
    }

    Ok(registry)
}

/// Assemble the toolset for one agent level: the capability tools, a delegate
/// tool per `child`, and (when `execute_python` is configured and `children` is
/// non-empty) the Python delegation bridge over `children`.
///
/// When `remote_capabilities` is `Some`, the capability tools are the
/// sandbox-backed adapters rather than locally-built tools; the Python bridge
/// is never built in that case (config validation forbids the combination).
/// With empty `children` and no sandbox, this is the capability toolset only,
/// and any configured `execute_python` is built as a plain REPL (no bridge).
///
/// The second return value is a reset handle for the in-process
/// `execute_python` interpreter, or `None` when there is none to reset (no
/// local interpreter, or the sandbox hosts it). Callers that reuse one agent
/// across tasks (serve mode) call it between tasks; callers that rebuild per
/// run (subagents) ignore it.
fn assemble_toolset(
    config: &RuntimeConfig,
    children: &HashMap<String, Arc<dyn Subagent>>,
    cancel: &CancellationToken,
    remote_capabilities: Option<&[Arc<dyn ToolImplementation>]>,
) -> Result<Toolset, RuntimeError> {
    // The Python->subagent bridge runs the interpreter in-process and cannot
    // cross the sandbox boundary, so it is only ever built for the local path.
    #[cfg_attr(not(feature = "python-repl"), allow(unused_variables))]
    let bridge = remote_capabilities.is_none() && bridge_python(config, children);

    let staged = if let Some(remote) = remote_capabilities {
        let registry = ToolRegistry::new();
        for tool in remote {
            registry.register(Arc::clone(tool));
        }
        registry
    } else {
        // The runtime builds `execute_python` explicitly below (plain or
        // bridged) so it can hold a typed handle to reset the interpreter; keep
        // the factory from also building one. Without the python-repl feature
        // there is no explicit build, so leave it in for `build_all` to reject.
        let factory_configs: Vec<ToolConfig> = if cfg!(feature = "python-repl") {
            config
                .tools
                .iter()
                .filter(|t| t.name != EXECUTE_PYTHON)
                .cloned()
                .collect()
        } else {
            config.tools.clone()
        };

        let factories = ToolFactoryRegistry::with_builtin();
        factories.build_all(&factory_configs)?
    };

    register_child_delegates(config, children, &staged, cancel)?;

    // Only the in-process interpreter carries resettable state; the sandbox
    // path keys it by task instead, so no reset handle is produced there.
    #[cfg(feature = "python-repl")]
    let reset = if remote_capabilities.is_none() {
        register_local_python(config, children, &staged, cancel, bridge)?
    } else {
        None
    };
    #[cfg(not(feature = "python-repl"))]
    let reset = None;

    let tools = staged
        .tool_names()
        .into_iter()
        .filter_map(|name| staged.get(&name))
        .collect();
    Ok((tools, reset))
}

/// Register one [`SubagentTool`] per child subagent into `staged`, so an agent
/// at this level can delegate to each child by its id.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if a subagent id collides with a configured
/// tool name.
fn register_child_delegates(
    config: &RuntimeConfig,
    children: &HashMap<String, Arc<dyn Subagent>>,
    staged: &ToolRegistry,
    cancel: &CancellationToken,
) -> Result<(), RuntimeError> {
    for sub in &config.subagents {
        let Some(inner) = children.get(&sub.id).map(Arc::clone) else {
            continue;
        };
        if staged.contains(&sub.id) {
            return Err(RuntimeError::Config(format!(
                "subagent id '{}' collides with a configured tool of the same name",
                sub.id
            )));
        }
        let description = sub
            .description
            .clone()
            .unwrap_or_else(|| format!("Delegate a task to the '{}' subagent.", sub.id));
        let tool = SubagentTool::new(inner, sub.id.clone(), description, cancel.clone());
        staged.register(Arc::new(tool));
    }
    Ok(())
}

/// Build the local in-process `execute_python` tool, register it into `staged`,
/// and return a handle that resets its interpreter between runs.
///
/// In bridge mode (non-empty `children`) the tool exposes
/// `run_subagent`/`spawn_agents` over the children; otherwise it is a plain
/// REPL. Returns `None` when `[[tools]]` configures no `execute_python`.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if the `execute_python` entry is malformed,
/// requests unrestricted mode while bridging (the bridge supports restricted
/// mode only), or the interpreter fails to build.
#[cfg(feature = "python-repl")]
fn register_local_python(
    config: &RuntimeConfig,
    children: &HashMap<String, Arc<dyn Subagent>>,
    staged: &ToolRegistry,
    cancel: &CancellationToken,
    bridge: bool,
) -> Result<Option<SessionReset>, RuntimeError> {
    use neuromance_repl::python::PythonReplToolFactory;

    let Some(entry) = config.tools.iter().find(|t| t.name == EXECUTE_PYTHON) else {
        return Ok(None);
    };

    let tool = if bridge {
        build_child_repl(children, cancel, entry)?
    } else {
        PythonReplToolFactory::build_tool(&entry.config)
            .map_err(|e| RuntimeError::Config(format!("build execute_python tool: {e}")))?
    };
    let registered: Arc<dyn ToolImplementation> = tool.clone();
    staged.register(registered);
    Ok(Some(local_python_reset(tool)))
}

/// Build the subagent-enabled Python REPL over `children`, exposing
/// `run_subagent`/`spawn_agents`.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if `entry` requests unrestricted mode (the
/// bridge supports restricted mode only) or if building the REPL or bridge
/// fails.
#[cfg(feature = "python-repl")]
fn build_child_repl(
    children: &HashMap<String, Arc<dyn Subagent>>,
    cancel: &CancellationToken,
    entry: &ToolConfig,
) -> Result<Arc<neuromance_repl::python::PythonReplTool>, RuntimeError> {
    use neuromance_repl::python::{PythonRepl, SubagentRepl};

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
    let bridge = SubagentRepl::new(repl, children.clone(), cancel.clone())
        .map_err(|e| RuntimeError::Config(format!("build subagent repl bridge: {e}")))?;
    Ok(Arc::new(bridge.into_tool()))
}

/// Wrap a [`PythonReplTool`](neuromance_repl::python::PythonReplTool) handle in a
/// [`SessionReset`] closure that clears its interpreter, logging a warning if
/// the reset fails rather than failing the caller.
#[cfg(feature = "python-repl")]
fn local_python_reset(tool: Arc<neuromance_repl::python::PythonReplTool>) -> SessionReset {
    Arc::new(move || {
        let tool = Arc::clone(&tool);
        Box::pin(async move {
            if let Err(e) = tool.reset().await {
                tracing::warn!(error = %e, "failed to reset local execute_python interpreter");
            }
        })
    })
}

/// Whether `execute_python` should be bridged over `children` rather than built
/// as a plain REPL: true only with the python-repl feature, a non-empty child
/// set, and an `execute_python` entry in `[[tools]]`. The restricted-mode
/// requirement is enforced when the bridge is built (see [`register_child_repl`]).
#[cfg(feature = "python-repl")]
fn bridge_python(config: &RuntimeConfig, children: &HashMap<String, Arc<dyn Subagent>>) -> bool {
    !children.is_empty() && config.tools.iter().any(|t| t.name == EXECUTE_PYTHON)
}

#[cfg(not(feature = "python-repl"))]
fn bridge_python(_config: &RuntimeConfig, _children: &HashMap<String, Arc<dyn Subagent>>) -> bool {
    false
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use async_trait::async_trait;
    use neuromance_common::task::{Outcome, Task};

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
            bootstrap: Vec::new(),
            sandbox: None,
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

    fn read_tool() -> ToolConfig {
        ToolConfig {
            name: "read".to_string(),
            config: serde_json::Value::Null,
        }
    }

    /// A stand-in child subagent so toolset assembly can be exercised without
    /// building an LLM client.
    struct MockSubagent(&'static str);

    #[async_trait]
    impl Subagent for MockSubagent {
        fn id(&self) -> &str {
            self.0
        }

        async fn run(
            &self,
            task: Task,
            _cancel: CancellationToken,
        ) -> Result<Outcome, neuromance_agent::SubagentError> {
            Ok(Outcome::new(task.id, "ok".to_string()))
        }
    }

    fn mock_children(ids: &[&'static str]) -> HashMap<String, Arc<dyn Subagent>> {
        ids.iter()
            .map(|id| {
                let sub: Arc<dyn Subagent> = Arc::new(MockSubagent(id));
                ((*id).to_string(), sub)
            })
            .collect()
    }

    fn tool_names(tools: &[Arc<dyn ToolImplementation>]) -> Vec<String> {
        tools
            .iter()
            .map(|t| t.get_definition().function.name)
            .collect()
    }

    /// With no children, the toolset is exactly the configured capability tools
    /// — no delegate tools appear.
    #[test]
    fn test_assemble_toolset_capability_only_without_children() {
        let mut config = config_with_subagents(vec![subagent("worker")]);
        config.tools = vec![read_tool()];

        let (tools, _reset) =
            assemble_toolset(&config, &HashMap::new(), &CancellationToken::new(), None).unwrap();
        let names = tool_names(&tools);

        assert_eq!(names, vec!["read".to_string()]);
        assert!(!names.contains(&"worker".to_string()));
    }

    /// A non-empty child set adds one delegate tool per configured subagent,
    /// named by its id, alongside the capability tools.
    #[test]
    fn test_assemble_toolset_adds_delegate_per_child() {
        let mut config = config_with_subagents(vec![subagent("worker"), subagent("critic")]);
        config.tools = vec![read_tool()];

        let children = mock_children(&["worker", "critic"]);
        let (tools, _reset) =
            assemble_toolset(&config, &children, &CancellationToken::new(), None).unwrap();
        let mut names = tool_names(&tools);
        names.sort();

        assert_eq!(
            names,
            vec![
                "critic".to_string(),
                "read".to_string(),
                "worker".to_string()
            ]
        );
    }

    /// With children present and an `execute_python` entry configured, the
    /// toolset carries a single bridged `execute_python` alongside the delegate
    /// tools — the plain factory REPL is not also built under that name.
    #[cfg(feature = "python-repl")]
    #[test]
    fn test_assemble_toolset_bridges_python_over_children() {
        let mut config = config_with_subagents(vec![subagent("worker")]);
        config.tools = vec![ToolConfig {
            name: "execute_python".to_string(),
            config: serde_json::Value::Null,
        }];

        let children = mock_children(&["worker"]);
        let (tools, _reset) =
            assemble_toolset(&config, &children, &CancellationToken::new(), None).unwrap();
        let names = tool_names(&tools);

        assert_eq!(
            names.iter().filter(|n| *n == "execute_python").count(),
            1,
            "exactly one execute_python tool expected, got: {names:?}"
        );
        assert!(names.contains(&"worker".to_string()));
    }

    /// A subagent id that collides with a configured tool name is rejected when
    /// delegate tools are wired in.
    #[test]
    fn test_assemble_toolset_rejects_id_tool_collision() {
        let mut config = config_with_subagents(vec![subagent("read")]);
        config.tools = vec![read_tool()];

        let children = mock_children(&["read"]);
        let err = assemble_toolset(&config, &children, &CancellationToken::new(), None)
            .err()
            .expect("colliding subagent id must be rejected");
        assert!(
            matches!(err, RuntimeError::Config(ref msg) if msg.contains("collides")),
            "unexpected error: {err}",
        );
    }

    /// The build path resolves credentials through the inherited provider's
    /// `api_key_env`. With that variable unset, the build fails naming it
    /// rather than silently dropping a subagent. (Env mutation is forbidden by
    /// `unsafe_code`, so the populated-key path is exercised via the
    /// proxy/integration tests.)
    #[test]
    fn test_build_surfaces_missing_credential_env() {
        let config = config_with_subagents(vec![subagent("alpha"), subagent("beta")]);
        let err = build_parent_toolset(&config, None, &CancellationToken::new(), None)
            .err()
            .expect("build should fail without the credential env var set");
        assert!(
            matches!(err, RuntimeError::MissingEnv(ref v) if v == "OPENAI_API_KEY"),
            "unexpected error: {err}",
        );
    }

    /// A locally-built `execute_python` yields a [`SessionReset`] that clears
    /// the very interpreter registered in the toolset: after the reset, a
    /// variable a prior run defined is gone. This is what keeps serve-mode
    /// tasks from leaking interpreter state into one another.
    #[cfg(feature = "python-repl")]
    #[tokio::test]
    #[serial_test::serial]
    async fn test_session_reset_clears_registered_interpreter() {
        use serde_json::json;

        let mut config = config_with_subagents(vec![]);
        config.tools = vec![ToolConfig {
            name: "execute_python".to_string(),
            config: serde_json::Value::Null,
        }];

        let (tools, reset) =
            build_parent_toolset(&config, None, &CancellationToken::new(), None).unwrap();
        let reset = reset.expect("a local execute_python tool must yield a reset handle");

        let python = tools
            .iter()
            .find(|t| t.get_definition().function.name == "execute_python")
            .expect("execute_python must be registered")
            .clone();

        python
            .execute(&json!({ "code": "marker = 7" }))
            .await
            .unwrap();
        let before: serde_json::Value = serde_json::from_str(
            &python
                .execute(&json!({ "code": "print(marker)" }))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(before["status"], "success");

        reset().await;

        let after: serde_json::Value = serde_json::from_str(
            &python
                .execute(&json!({ "code": "print(marker)" }))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(after["status"], "error");
        assert!(after["stderr"].as_str().unwrap().contains("NameError"));
    }

    /// The bridged `execute_python` (built when subagents exist) yields a reset
    /// that clears user state while preserving the injected `run_subagent` and
    /// `spawn_agents` primitives. This is what lets serve-mode delegation keep
    /// working after the per-task reset; it would break if a future change moved
    /// the callbacks out of the interpreter globals that reset re-establishes.
    #[cfg(feature = "python-repl")]
    #[tokio::test]
    #[serial_test::serial]
    async fn test_bridge_session_reset_preserves_subagent_primitives() {
        use serde_json::json;

        let mut config = config_with_subagents(vec![subagent("worker")]);
        config.tools = vec![ToolConfig {
            name: "execute_python".to_string(),
            config: serde_json::Value::Null,
        }];
        let children = mock_children(&["worker"]);

        let (tools, reset) =
            assemble_toolset(&config, &children, &CancellationToken::new(), None).unwrap();
        let reset = reset.expect("a bridged execute_python must yield a reset handle");

        let python = tools
            .iter()
            .find(|t| t.get_definition().function.name == "execute_python")
            .expect("execute_python must be registered")
            .clone();

        // Define user state and prove the bridge is wired in before the reset.
        // MockSubagent echoes "ok", so a successful delegation reaches stdout.
        python
            .execute(&json!({ "code": "marker = 7" }))
            .await
            .unwrap();
        let before: serde_json::Value = serde_json::from_str(
            &python
                .execute(&json!({ "code": "print(run_subagent('worker', 'do x'))" }))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(before["status"], "success");
        assert!(before["stdout"].as_str().unwrap().contains("ok"));

        reset().await;

        // User state is cleared...
        let cleared: serde_json::Value = serde_json::from_str(
            &python
                .execute(&json!({ "code": "print(marker)" }))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(cleared["status"], "error");
        assert!(cleared["stderr"].as_str().unwrap().contains("NameError"));

        // ...but run_subagent and the spawn_agents prelude still resolve and run.
        let after: serde_json::Value = serde_json::from_str(
            &python
                .execute(&json!({
                    "code": "print(run_subagent('worker', 'again'))\n\
                             print(spawn_agents([Agent('worker', 'fan')]))"
                }))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            after["status"], "success",
            "delegation must survive reset: {after:?}"
        );
        assert!(
            after["stdout"].as_str().unwrap().contains("ok"),
            "run_subagent/spawn_agents output missing after reset: {after:?}"
        );
    }

    /// With no subagents configured, the toolset is the capability tools only
    /// and no client is built.
    #[test]
    fn test_no_subagents_yields_capability_tools_only() {
        let mut config = config_with_subagents(vec![]);
        config.tools = vec![read_tool()];

        let (tools, _reset) =
            build_parent_toolset(&config, None, &CancellationToken::new(), None).unwrap();
        assert_eq!(tool_names(&tools), vec!["read".to_string()]);
    }

    /// A subagent with no `provider` inherits the agent's provider, so its
    /// credential resolves through that provider's `api_key_env`.
    #[test]
    fn test_subagent_inherits_parent_provider_credential() {
        let config = config_with_subagents(vec![subagent("worker")]);
        let err = build_parent_toolset(&config, None, &CancellationToken::new(), None)
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
        let err = build_parent_toolset(&config, None, &CancellationToken::new(), None)
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
        let err = build_parent_toolset(&config, None, &CancellationToken::new(), None)
            .err()
            .expect("build should fail for an unknown provider");
        assert!(
            matches!(err, RuntimeError::Config(ref msg)
                if msg.contains("provider 'ghost'") && msg.contains("worker")),
            "unexpected error: {err}",
        );
    }
}
