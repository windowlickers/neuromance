//! Build the main agent's toolset and the subagent delegation tower from
//! `[[subagents]]` config.
//!
//! A subagent is provisioned with the *same* toolset as the main agent:
//! capability tools (the built-in factories plus anything in `[[tools]]`), the
//! `[[mcp_servers]]` tools, `load_skill` when `[skills]` is configured, and the
//! delegate tools that let it hand work to other subagents. Delegation is
//! bounded by `runtime.max_delegation_depth`, and that bound is enforced
//! *structurally* by building a finite tower of subagent instances rather than
//! threading a runtime counter:
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
//! concurrent runs of one subagent stay parallel.
//!
//! A subagent's `Core` is wired like the main agent's (`main.rs::build_agent`):
//! persistence, compaction, skills, and rules attach in the same order, and a
//! subagent that pins neither `provider` nor `model` runs whatever the parent is
//! running for this task, per-task override included. Two things are
//! deliberately left off:
//!
//! - **Streaming** — nothing consumes a subagent's token stream; a delegate tool
//!   returns only the final content.
//! - **Approval** — subagent tool calls auto-approve within one parent
//!   delegation, which has no interactive approver in the loop.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use neuromance::Core;
use neuromance_agent::{Agent, LocalSubagent, Subagent, SubagentError, SubagentTool};
use neuromance_client::{LLMClient, build_client};
use neuromance_common::hook::Hook;
use neuromance_context::CompactionHook;
use neuromance_context::rules::RulesHook;
use neuromance_context::skills::SkillsHook;
use neuromance_db::{PersistenceHook, PgConversationStore};
use neuromance_tools::{ToolFactoryRegistry, ToolImplementation, ToolRegistry};

use crate::config::{ProviderConfig, RuntimeConfig, SELF_DELEGATION_ID, SubagentConfig};
use crate::error::RuntimeError;
use crate::proxy::build_provider_config;
use crate::skills::{SkillRuntime, fold_menu};

type Toolset = Vec<Arc<dyn ToolImplementation>>;

/// Everything the delegation tower needs beyond [`RuntimeConfig`]: the shared
/// handles built at startup, plus the provider and model the parent agent is
/// actually running for this build.
pub struct ToolsetParams<'a> {
    /// Conversation store wired into every subagent's `Core`, when configured,
    /// so child conversations persist and record their parent/child lineage
    /// just like the main agent's.
    pub store: Option<&'a Arc<PgConversationStore>>,
    /// Skill catalog: its menu is folded into every subagent's system prompt,
    /// its `load_skill` tool joins every toolset, and its `$mention` hook is
    /// shared with them.
    pub skills: Option<&'a Arc<SkillRuntime>>,
    /// The main agent's rules hook, shared verbatim with every subagent.
    pub rules: Option<&'a Arc<RulesHook>>,
    /// Tools from the connected `[[mcp_servers]]`, shared (cloned) across the
    /// main agent and every subagent.
    pub mcp_tools: &'a [Arc<dyn ToolImplementation>],
    /// The provider the parent agent runs for this task, per-task override
    /// already applied (see `RuntimeConfig::resolve_provider_and_model`).
    pub parent_provider: &'a ProviderConfig,
    /// The parent agent's effective model, per-task override already applied.
    pub parent_model: &'a str,
    /// Cancellation token for delegate tools and subagent runs.
    pub cancel: &'a CancellationToken,
}

/// Resolved, tower-wide inputs threaded through every level of the recursion.
struct ChildContext<'a> {
    store: Option<&'a Arc<PgConversationStore>>,
    /// Tools every level carries unchanged: the `[[mcp_servers]]` adapters and,
    /// when `[skills]` is configured, `load_skill`.
    shared_tools: Vec<Arc<dyn ToolImplementation>>,
    cancel: &'a CancellationToken,
    /// `$mention`-expanding skills hook, built once for the whole tower: its
    /// injections are keyed by conversation, so one instance serves every level
    /// and every concurrent run. The menu rides the system prompt instead — see
    /// [`tower_config`].
    skills: Option<Arc<SkillsHook>>,
    /// The parent's rules hook, shared verbatim (also conversation-keyed).
    rules: Option<Arc<RulesHook>>,
    parent_provider: &'a ProviderConfig,
    parent_model: &'a str,
}

impl<'a> ChildContext<'a> {
    fn new(params: &ToolsetParams<'a>) -> Self {
        let mut shared_tools = params.mcp_tools.to_vec();
        shared_tools.extend(params.skills.map(|skills| skills.tool()));
        Self {
            store: params.store,
            shared_tools,
            cancel: params.cancel,
            skills: params.skills.map(|skills| skills.hook()),
            rules: params.rules.map(Arc::clone),
            parent_provider: params.parent_provider,
            parent_model: params.parent_model,
        }
    }

    /// The hook handles for one subagent's per-run `Core`, owned so the factory
    /// closure can hold them past this borrow.
    fn hooks(&self) -> ChildHooks {
        ChildHooks {
            store: self.store.cloned(),
            skills: self.skills.clone(),
            rules: self.rules.clone(),
        }
    }
}

/// The shared hook handles a child `Core` is built with.
struct ChildHooks {
    store: Option<Arc<PgConversationStore>>,
    skills: Option<Arc<SkillsHook>>,
    rules: Option<Arc<RulesHook>>,
}

/// Build the main agent's toolset, including delegate tools for every
/// configured subagent.
///
/// When no subagents are configured this is just the capability toolset built
/// from `[[tools]]`. Otherwise it also builds the delegation tower down to
/// `runtime.max_delegation_depth` and wires the main agent's delegate tools to
/// the top of that tower.
///
/// # Errors
/// Returns [`RuntimeError`] if a subagent's provider/model/credentials fail to
/// resolve, a tool factory fails, or a subagent id collides with a configured
/// tool name.
pub fn build_parent_toolset(
    config: &RuntimeConfig,
    params: &ToolsetParams<'_>,
) -> Result<Toolset, RuntimeError> {
    let menu = params.skills.and_then(|skills| skills.menu());
    let effective = tower_config(config, menu.as_deref());
    let config: &RuntimeConfig = &effective;

    let ctx = ChildContext::new(params);
    let children = if config.subagents.is_empty() {
        HashMap::new()
    } else {
        // The main agent is depth 0; its children may delegate `depth - 1`
        // further hops.
        let remaining = config.runtime.max_delegation_depth.saturating_sub(1);
        build_subagents_at_depth(config, remaining, &ctx)?
    };
    assemble_toolset(config, &children, params.cancel, &ctx.shared_tools)
}

/// The config the whole tower is built from: the synthesized self-delegation
/// subagent appended (when `runtime.self_delegation` is set), then `skills_menu`
/// folded into every subagent's system prompt.
///
/// Injecting the self-delegation entry into `subagents` lets the tower,
/// and `SubagentTool` registration treat it exactly like a
/// configured `[[subagents]]` entry. Appending it *before* the fold means the
/// self-clone's prompt ends up byte-identical to the main agent's seed, menu
/// included.
///
/// The menu is folded here rather than injected by the shared [`SkillsHook`],
/// so each subagent's prompt carries the same text the host folds into the main
/// agent's seed (`oneshot::run`, `serve::seed_new_conversation`). Every level
/// registers the `load_skill` tool the menu points to (see [`ChildContext`]).
///
/// Borrows `config` unchanged when neither transform applies.
fn tower_config<'a>(
    config: &'a RuntimeConfig,
    skills_menu: Option<&str>,
) -> Cow<'a, RuntimeConfig> {
    let mut effective = config.with_self_delegation();
    if let Some(menu) = skills_menu
        && !effective.subagents.is_empty()
    {
        for sub in &mut effective.to_mut().subagents {
            sub.system_prompt = fold_menu(&sub.system_prompt, Some(menu));
        }
    }
    effective
}

/// Build the subagent registry for one tower level.
///
/// `remaining` is the number of delegation hops still allowed *below* this
/// level: at `0` the level is a leaf (its subagents get no delegate tools); above
/// `0` each subagent can delegate to the level built with `remaining - 1`.
fn build_subagents_at_depth(
    config: &RuntimeConfig,
    remaining: u32,
    ctx: &ChildContext<'_>,
) -> Result<HashMap<String, Arc<dyn Subagent>>, RuntimeError> {
    let children = if remaining == 0 {
        HashMap::new()
    } else {
        build_subagents_at_depth(config, remaining - 1, ctx)?
    };
    // Shared across every subagent at this level and captured by each run's
    // builder; each run assembles its own toolset from these.
    let children = Arc::new(children);
    let config = Arc::new(config.clone());

    let mut registry: HashMap<String, Arc<dyn Subagent>> = HashMap::new();
    for sub in &config.subagents {
        let client: Arc<dyn LLMClient> = build_client(resolve_child_provider(&config, sub, ctx)?)
            .map_err(|e| RuntimeError::Config(format!("subagent '{}': build client: {e}", sub.id)))?
            .into();

        let id = sub.id.clone();
        let max_turns = sub.max_turns;
        let config = Arc::clone(&config);
        let children = Arc::clone(&children);
        let cancel = ctx.cancel.clone();
        let hooks = ctx.hooks();
        let shared_tools = ctx.shared_tools.clone();
        let build_agent = move || {
            let tools = assemble_toolset(&config, &children, &cancel, &shared_tools)
                .map_err(SubagentError::execution)?;
            let mut core = build_child_core(&config, Arc::clone(&client), max_turns, &hooks);
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

/// Resolve the LLM config one subagent runs on, credentials included.
///
/// # Errors
/// Returns [`RuntimeError`] if [`resolve_child_llm`] fails or the resolved
/// provider's credential env var is unset.
fn resolve_child_provider(
    config: &RuntimeConfig,
    sub: &SubagentConfig,
    ctx: &ChildContext<'_>,
) -> Result<neuromance::Config, RuntimeError> {
    let (provider, model) = resolve_child_llm(config, sub, ctx)?;
    build_provider_config(provider, model)
}

/// Pick the provider and model one subagent runs on.
///
/// A subagent that pins neither `provider` nor `model` runs exactly what the
/// parent runs for this task — per-task override included — so a self-delegated
/// clone really is a clone. Pinning either one keeps the subagent's configured
/// identity and ignores the task override; the pinned model then falls back to
/// the chosen provider's default, then the agent's effective model.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if a pinned provider names no `[[providers]]`
/// entry or no model can be resolved for the subagent.
fn resolve_child_llm<'a>(
    config: &'a RuntimeConfig,
    sub: &'a SubagentConfig,
    ctx: &ChildContext<'a>,
) -> Result<(&'a ProviderConfig, &'a str), RuntimeError> {
    if sub.provider.is_none() && sub.model.is_none() {
        return Ok((ctx.parent_provider, ctx.parent_model));
    }
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
        .or_else(|| config.agent_model())
        .ok_or_else(|| {
            RuntimeError::Config(format!(
                "subagent '{}' has no model: set subagent.model, provider '{provider_name}' \
                 model, or agent.model",
                sub.id
            ))
        })?;
    Ok((provider, model))
}

/// Build one subagent run's `Core` with the same lifecycle wiring the main agent
/// gets (`main.rs::build_agent`), minus streaming and approval (see the module
/// docs). Hooks attach in the parent's registration order — persistence,
/// compaction, skills, rules — because hooks dispatch in registration order, so
/// matching it keeps a subagent's start-of-conversation injections and
/// compaction timing identical to the parent's.
fn build_child_core(
    config: &RuntimeConfig,
    client: Arc<dyn LLMClient>,
    max_turns: Option<u32>,
    hooks: &ChildHooks,
) -> Core<Arc<dyn LLMClient>> {
    let mut core = Core::new(Arc::clone(&client));
    if let Some(max) = max_turns {
        core.max_turns = Some(max);
    }
    // An empty terminal turn is a provider hiccup, not a property of who is
    // asking, so a subagent retries on the same budget as the main agent
    // (`main.rs::build_agent`). Leaving this unset silently pinned every
    // subagent to `Core`'s default and dropped the operator's setting.
    core.empty_turn_retries = config.agent.empty_turn_retries;
    // Subagent tool calls run autonomously inside one parent delegation, with no
    // interactive approver in the loop; the pod boundary (kata) is the
    // isolation. See the README Subagents section.
    core.auto_approve_tools = true;

    // Persist child conversations (and their parent link) when the runtime has a
    // store, matching the main agent. Constructed inside the parent's delegation
    // scope, so the hook captures the lineage linking this child to its parent.
    if let Some(store) = &hooks.store {
        let sink: Arc<PgConversationStore> = Arc::clone(store);
        core = core.with_hook(Arc::new(PersistenceHook::new(sink)));
    }
    // Compaction summarizes through the chat client rather than a second one:
    // the parent needs a separate client only because its own was moved into
    // `Core` as a `Box`, while a subagent holds an `Arc` that is itself an
    // `LLMClient`. The hook is built per run by necessity — its reported-token
    // slot is one global cell, not keyed by conversation, so a shared instance
    // would let concurrent sibling runs drive each other's compaction.
    if let Some(context) = &config.context {
        core = core.with_hook(Arc::new(CompactionHook::new(
            client,
            &context.to_context_config(),
        )));
    }
    if let Some(skills) = &hooks.skills {
        core = core.with_hook(Arc::clone(skills) as Arc<dyn Hook>);
    }
    if let Some(rules) = &hooks.rules {
        core = core.with_hook(Arc::clone(rules) as Arc<dyn Hook>);
    }
    core
}

/// Assemble the toolset for one agent level: the capability tools, the
/// `shared_tools`, and a delegate tool per `child`.
fn assemble_toolset(
    config: &RuntimeConfig,
    children: &HashMap<String, Arc<dyn Subagent>>,
    cancel: &CancellationToken,
    shared_tools: &[Arc<dyn ToolImplementation>],
) -> Result<Toolset, RuntimeError> {
    let staged = ToolFactoryRegistry::with_builtin().build_all(&config.tools)?;

    register_shared_tools(shared_tools, &staged)?;
    register_child_delegates(config, children, &staged, cancel)?;

    Ok(staged
        .tool_names()
        .into_iter()
        .filter_map(|name| staged.get(&name))
        .collect())
}

/// Register the shared tools (MCP adapters and `load_skill`) into `staged`.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if a shared tool name collides with a tool
/// already in `staged`.
fn register_shared_tools(
    shared_tools: &[Arc<dyn ToolImplementation>],
    staged: &ToolRegistry,
) -> Result<(), RuntimeError> {
    for tool in shared_tools {
        let name = tool.get_definition().function.name;
        if staged.contains(&name) {
            return Err(RuntimeError::Config(format!(
                "tool '{name}' collides with a configured tool of the same name; rename \
                 the [[mcp_servers]] id or remove the [[tools]] entry"
            )));
        }
        staged.register(Arc::clone(tool));
    }
    Ok(())
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
                "subagent id '{}' collides with a configured tool of the same name; rename \
                 whichever you control (the id is synthesized when it is \
                 '{SELF_DELEGATION_ID}' and runtime.self_delegation is set)",
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::pin::Pin;
    use std::sync::OnceLock;

    use async_trait::async_trait;
    use futures::Stream;
    use neuromance_client::ClientError;
    use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config};
    use neuromance_common::task::{Outcome, Task};
    use neuromance_context::compaction::CompactionStrategy;
    use neuromance_context::rules::RuleCatalog;
    use neuromance_context::skills::SkillCatalog;

    use super::*;
    use crate::config::{ContextSettings, ProviderConfig, SELF_DELEGATION_ID, SubagentConfig};
    use crate::{AgentConfig, ApprovalConfig, Mode, RuntimeSettings};
    use neuromance_tools::ToolConfig;

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
                empty_turn_retries: 1,
            },
            runtime: RuntimeSettings::default(),
            approval: ApprovalConfig::default(),
            tools: Vec::new(),
            oneshot: None,
            providers,
            database: None,
            context: None,
            subagents,
            skills: None,
            rules: None,
            bootstrap: Vec::new(),
            mcp_servers: Vec::new(),
        }
    }

    /// Default tower params for `config`: no store, skills, rules, or MCP tools,
    /// with the parent's provider/model resolved from `[agent]` exactly as
    /// `build_agent` does for a task that carries no override.
    fn params<'a>(config: &'a RuntimeConfig, cancel: &'a CancellationToken) -> ToolsetParams<'a> {
        let (parent_provider, parent_model) =
            config.resolve_provider_and_model(None, None).unwrap();
        ToolsetParams {
            store: None,
            skills: None,
            rules: None,
            mcp_tools: &[],
            parent_provider,
            parent_model,
            cancel,
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

    fn context_settings() -> ContextSettings {
        ContextSettings {
            context_window_size: 128_000,
            compaction_threshold_ratio: 0.8,
            target_ratio: 0.5,
            preserve_recent_turns: 3,
            strategy: CompactionStrategy::OneShot,
        }
    }

    /// A stand-in LLM client, so a child `Core` can be wired without credentials
    /// or a network. Every wiring test inspects the built `Core` rather than
    /// running it, so the chat methods are never reached.
    struct StubClient;

    #[async_trait]
    impl LLMClient for StubClient {
        fn config(&self) -> &Config {
            static CONFIG: OnceLock<Config> = OnceLock::new();
            CONFIG.get_or_init(|| Config::new("mock", "mock-model"))
        }

        async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            unreachable!("child core wiring tests never chat")
        }

        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
        {
            unreachable!("child core wiring tests never stream")
        }

        fn supports_tools(&self) -> bool {
            true
        }

        fn supports_structured_output(&self) -> bool {
            false
        }

        fn supports_streaming(&self) -> bool {
            false
        }
    }

    fn stub_client() -> Arc<dyn LLMClient> {
        Arc::new(StubClient)
    }

    async fn rules_hook() -> Arc<RulesHook> {
        Arc::new(RulesHook::new(
            Arc::new(RuleCatalog::build(Vec::new()).await),
            4096,
        ))
    }

    async fn skills_hook() -> Arc<SkillsHook> {
        Arc::new(SkillsHook::new(
            Arc::new(SkillCatalog::build(Vec::new()).await),
            4096,
            4096,
            true,
            false,
        ))
    }

    async fn skill_runtime() -> Arc<SkillRuntime> {
        Arc::new(SkillRuntime {
            catalog: Arc::new(SkillCatalog::build(Vec::new()).await),
            menu_budget: 4096,
            body_budget: 4096,
            mention: true,
        })
    }

    fn child_hooks(skills: Option<Arc<SkillsHook>>, rules: Option<Arc<RulesHook>>) -> ChildHooks {
        ChildHooks {
            store: None,
            skills,
            rules,
        }
    }

    /// A tower context carrying the parent's effective provider/model and no
    /// shared hooks — enough to exercise child provider resolution.
    fn child_ctx<'a>(
        cancel: &'a CancellationToken,
        parent_provider: &'a ProviderConfig,
        parent_model: &'a str,
    ) -> ChildContext<'a> {
        ChildContext {
            store: None,
            shared_tools: Vec::new(),
            cancel,
            skills: None,
            rules: None,
            parent_provider,
            parent_model,
        }
    }

    fn hook_names(core: &Core<Arc<dyn LLMClient>>) -> Vec<&'static str> {
        core.hooks.iter().map(|h| h.name()).collect()
    }

    /// Two providers keyed to distinct env vars, with `[agent]` on "primary" and
    /// one inheriting subagent.
    fn two_provider_config(subagents: Vec<SubagentConfig>) -> RuntimeConfig {
        config_with_providers(
            vec![
                provider("primary", "PRIMARY_KEY", "openai:gpt-4o"),
                provider("fast", "FAST_KEY", "openai:gpt-4o-mini"),
            ],
            "primary",
            subagents,
        )
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

        let tools =
            assemble_toolset(&config, &HashMap::new(), &CancellationToken::new(), &[]).unwrap();
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
        let tools = assemble_toolset(&config, &children, &CancellationToken::new(), &[]).unwrap();
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

    /// With `runtime.self_delegation` set, the effective config gains a
    /// synthesized `task` subagent, so assembling its toolset over the matching
    /// child registers a `task` delegate tool alongside the capability tools —
    /// no `[[subagents]]` entry required.
    #[test]
    fn test_self_delegation_adds_task_delegate() {
        let mut config = config_with_subagents(vec![]);
        config.tools = vec![read_tool()];
        config.runtime.self_delegation = true;

        let effective = config.with_self_delegation();
        let children = mock_children(&[SELF_DELEGATION_ID]);
        let tools =
            assemble_toolset(&effective, &children, &CancellationToken::new(), &[]).unwrap();
        let mut names = tool_names(&tools);
        names.sort();

        assert_eq!(names, vec!["read".to_string(), "task".to_string()]);
    }

    /// A subagent id that collides with a configured tool name is rejected when
    /// delegate tools are wired in.
    #[test]
    fn test_assemble_toolset_rejects_id_tool_collision() {
        let mut config = config_with_subagents(vec![subagent("read")]);
        config.tools = vec![read_tool()];

        let children = mock_children(&["read"]);
        let err = assemble_toolset(&config, &children, &CancellationToken::new(), &[])
            .err()
            .expect("colliding subagent id must be rejected");
        assert!(
            matches!(err, RuntimeError::Config(ref msg) if msg.contains("collides")),
            "unexpected error: {err}",
        );
    }

    /// A stand-in MCP tool adapter: only its advertised name matters here.
    struct NamedTool(&'static str);

    #[async_trait]
    impl ToolImplementation for NamedTool {
        fn get_definition(&self) -> neuromance_common::tools::Tool {
            neuromance_common::tools::Tool::builder()
                .function(neuromance_common::tools::Function {
                    name: self.0.to_string(),
                    description: String::new(),
                    parameters: serde_json::json!({"type": "object"}),
                })
                .build()
        }

        async fn execute(
            &self,
            _args: &serde_json::Value,
        ) -> Result<String, neuromance_tools::ToolError> {
            Ok(String::new())
        }
    }

    /// MCP tools join the capability tools and delegate tools at every level.
    #[test]
    fn test_assemble_toolset_includes_mcp_tools() {
        let mut config = config_with_subagents(vec![subagent("worker")]);
        config.tools = vec![read_tool()];
        let mcp: Vec<Arc<dyn ToolImplementation>> = vec![Arc::new(NamedTool("fs__read_file"))];

        let children = mock_children(&["worker"]);
        let tools = assemble_toolset(&config, &children, &CancellationToken::new(), &mcp).unwrap();
        let mut names = tool_names(&tools);
        names.sort();

        assert_eq!(names, vec!["fs__read_file", "read", "worker"]);
    }

    /// An MCP tool that shadows a configured tool is rejected rather than
    /// silently replacing it.
    #[test]
    fn test_assemble_toolset_rejects_mcp_tool_collision() {
        let mut config = config_with_subagents(vec![]);
        config.tools = vec![read_tool()];
        let mcp: Vec<Arc<dyn ToolImplementation>> = vec![Arc::new(NamedTool("read"))];

        let err = assemble_toolset(&config, &HashMap::new(), &CancellationToken::new(), &mcp)
            .err()
            .expect("colliding MCP tool must be rejected");
        assert!(
            matches!(err, RuntimeError::Config(ref msg) if msg.contains("tool 'read' collides")),
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
        let err = build_parent_toolset(&config, &params(&config, &CancellationToken::new()))
            .err()
            .expect("build should fail without the credential env var set");
        assert!(
            matches!(err, RuntimeError::MissingEnv(ref v) if v == "OPENAI_API_KEY"),
            "unexpected error: {err}",
        );
    }

    /// With no subagents configured, the toolset is the capability tools only
    /// and no client is built.
    #[test]
    fn test_no_subagents_yields_capability_tools_only() {
        let mut config = config_with_subagents(vec![]);
        config.tools = vec![read_tool()];

        let tools =
            build_parent_toolset(&config, &params(&config, &CancellationToken::new())).unwrap();
        assert_eq!(tool_names(&tools), vec!["read".to_string()]);
    }

    /// With `[skills]` configured, the main agent gets `load_skill`, and it is
    /// auto-approved, so `approval.mode = "auto"` still boots.
    #[tokio::test]
    async fn test_skills_add_auto_approved_load_skill_to_parent_toolset() {
        let config = config_with_subagents(vec![]);
        assert!(matches!(config.approval.mode, crate::ApprovalMode::Auto));
        let skills = skill_runtime().await;
        let cancel = CancellationToken::new();

        let tools = build_parent_toolset(
            &config,
            &ToolsetParams {
                skills: Some(&skills),
                ..params(&config, &cancel)
            },
        )
        .unwrap();

        assert_eq!(tool_names(&tools), vec!["load_skill".to_string()]);
        assert!(tools.iter().all(|tool| tool.is_auto_approved()));
    }

    /// Every tower level assembles its toolset from the context's shared tools,
    /// so `load_skill` reaches each subagent beside the MCP tools.
    #[tokio::test]
    async fn test_child_context_shares_load_skill_with_every_level() {
        let config = config_with_subagents(vec![subagent("worker")]);
        let skills = skill_runtime().await;
        let cancel = CancellationToken::new();
        let mcp: Vec<Arc<dyn ToolImplementation>> = vec![Arc::new(NamedTool("fs__read_file"))];

        let ctx = ChildContext::new(&ToolsetParams {
            skills: Some(&skills),
            mcp_tools: &mcp,
            ..params(&config, &cancel)
        });

        assert_eq!(
            tool_names(&ctx.shared_tools),
            vec!["fs__read_file", "load_skill"]
        );
    }

    /// A subagent id that collides with `load_skill` is rejected rather than
    /// shadowing the skill tool.
    #[tokio::test]
    async fn test_subagent_named_load_skill_collides() {
        let config = config_with_subagents(vec![subagent("load_skill")]);
        let shared = vec![skill_runtime().await.tool()];

        let children = mock_children(&["load_skill"]);
        let err = assemble_toolset(&config, &children, &CancellationToken::new(), &shared)
            .err()
            .expect("a subagent named load_skill must be rejected");
        assert!(
            matches!(err, RuntimeError::Config(ref msg) if msg.contains("collides")),
            "unexpected error: {err}",
        );
    }

    /// A subagent with no `provider` inherits the agent's provider, so its
    /// credential resolves through that provider's `api_key_env`.
    #[test]
    fn test_subagent_inherits_parent_provider_credential() {
        let config = config_with_subagents(vec![subagent("worker")]);
        let err = build_parent_toolset(&config, &params(&config, &CancellationToken::new()))
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
        let err = build_parent_toolset(&config, &params(&config, &CancellationToken::new()))
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
        let err = build_parent_toolset(&config, &params(&config, &CancellationToken::new()))
            .err()
            .expect("build should fail for an unknown provider");
        assert!(
            matches!(err, RuntimeError::Config(ref msg)
                if msg.contains("provider 'ghost'") && msg.contains("worker")),
            "unexpected error: {err}",
        );
    }

    /// A subagent that pins neither `provider` nor `model` runs whatever the
    /// parent runs for this task — the per-task override included. Without this,
    /// a serve task that overrides the model leaves its subagents on the startup
    /// pair, so a "self-clone" is not one.
    #[test]
    fn test_subagent_without_pins_follows_task_override() {
        let config = two_provider_config(vec![subagent("worker")]);
        let (parent_provider, parent_model) = config
            .resolve_provider_and_model(Some("fast"), Some("openai:o3"))
            .unwrap();
        let cancel = CancellationToken::new();
        let ctx = child_ctx(&cancel, parent_provider, parent_model);

        let (provider, model) = resolve_child_llm(&config, &config.subagents[0], &ctx).unwrap();

        assert_eq!(provider.name, "fast");
        assert_eq!(model, "openai:o3");
    }

    /// Pinning `model` keeps a subagent's configured identity: the task override
    /// applies to the parent only, so a cheap reviewer stays cheap when the
    /// parent is bumped to a larger model.
    #[test]
    fn test_subagent_with_pinned_model_ignores_task_override() {
        let config = two_provider_config(vec![SubagentConfig {
            model: Some("openai:gpt-4o-mini".to_string()),
            ..subagent("worker")
        }]);
        let (parent_provider, parent_model) = config
            .resolve_provider_and_model(Some("fast"), Some("openai:o3"))
            .unwrap();
        let cancel = CancellationToken::new();
        let ctx = child_ctx(&cancel, parent_provider, parent_model);

        let (provider, model) = resolve_child_llm(&config, &config.subagents[0], &ctx).unwrap();

        assert_eq!(provider.name, "primary");
        assert_eq!(model, "openai:gpt-4o-mini");
    }

    /// The synthesized `task` subagent pins nothing, so it follows the parent's
    /// override too — the point of `runtime.self_delegation`.
    #[test]
    fn test_self_delegation_follows_task_override() {
        let mut config = two_provider_config(vec![]);
        config.runtime.self_delegation = true;
        let (parent_provider, parent_model) = config
            .resolve_provider_and_model(Some("fast"), Some("openai:o3"))
            .unwrap();
        let cancel = CancellationToken::new();
        let ctx = child_ctx(&cancel, parent_provider, parent_model);

        let effective = tower_config(&config, None);
        let task = effective
            .subagents
            .iter()
            .find(|s| s.id == SELF_DELEGATION_ID)
            .expect("self_delegation must synthesize a task subagent");
        let (provider, model) = resolve_child_llm(&effective, task, &ctx).unwrap();

        assert_eq!(provider.name, "fast");
        assert_eq!(model, "openai:o3");
    }

    /// With nothing to add, the tower builds from the config as-is rather than
    /// deep-cloning it — including when a menu exists but there are no subagents
    /// to fold it into.
    #[test]
    fn test_tower_config_borrows_when_nothing_to_add() {
        let with_subagents = config_with_subagents(vec![subagent("worker")]);
        assert!(matches!(
            tower_config(&with_subagents, None),
            Cow::Borrowed(_)
        ));

        let no_subagents = config_with_subagents(vec![]);
        assert!(matches!(
            tower_config(&no_subagents, Some("## Skills\n- alpha")),
            Cow::Borrowed(_)
        ));
    }

    /// The skills menu is folded into every subagent's system prompt. A subagent
    /// has no seed message for the host to fold it into, so without this it
    /// cannot see any skill.
    #[test]
    fn test_tower_config_folds_menu_into_every_subagent_prompt() {
        let config = config_with_subagents(vec![subagent("worker"), subagent("critic")]);

        let effective = tower_config(&config, Some("## Skills\n- alpha"));

        assert_eq!(effective.subagents.len(), 2);
        for sub in &effective.subagents {
            assert_eq!(sub.system_prompt, "you are a worker\n\n## Skills\n- alpha");
        }
    }

    /// The self-delegated clone's prompt matches the main agent's seed byte for
    /// byte, menu included — assembled the same way `oneshot::run` and
    /// `serve::seed_new_conversation` assemble the parent's.
    #[test]
    fn test_tower_config_self_clone_prompt_matches_parent_seed() {
        let mut config = config_with_subagents(vec![]);
        config.runtime.self_delegation = true;
        let menu = "## Skills\n- alpha";

        let effective = tower_config(&config, Some(menu));

        let task = effective
            .subagents
            .iter()
            .find(|s| s.id == SELF_DELEGATION_ID)
            .expect("self_delegation must synthesize a task subagent");
        assert_eq!(
            task.system_prompt,
            format!("{}\n\n{menu}", config.agent.system_prompt)
        );
    }

    /// A subagent compacts like the main agent: with `[context]` configured its
    /// `Core` carries a compaction hook, so a long delegated run summarizes
    /// instead of walking into the context window.
    #[test]
    fn test_child_core_compacts_when_context_configured() {
        let mut config = config_with_subagents(vec![subagent("worker")]);
        config.context = Some(context_settings());

        let core = build_child_core(&config, stub_client(), None, &child_hooks(None, None));

        assert_eq!(hook_names(&core), vec!["compaction"]);
    }

    /// Without `[context]` there is no window to compact against, so no hook is
    /// attached — compaction stays opt-in for subagents exactly as for the main
    /// agent.
    #[test]
    fn test_child_core_has_no_compaction_without_context() {
        let config = config_with_subagents(vec![subagent("worker")]);

        let core = build_child_core(&config, stub_client(), None, &child_hooks(None, None));

        assert!(hook_names(&core).is_empty(), "{:?}", hook_names(&core));
    }

    /// Each run builds its own compaction hook. The hook's reported-token slot is
    /// one global cell rather than keyed by conversation, so a shared instance
    /// would let concurrent sibling runs drive each other's compaction.
    #[test]
    fn test_child_core_builds_a_fresh_compaction_hook_per_run() {
        let mut config = config_with_subagents(vec![subagent("worker")]);
        config.context = Some(context_settings());

        let first = build_child_core(&config, stub_client(), None, &child_hooks(None, None));
        let second = build_child_core(&config, stub_client(), None, &child_hooks(None, None));

        assert!(!Arc::ptr_eq(&first.hooks[0], &second.hooks[0]));
    }

    /// Hooks attach in the parent's registration order, because `Core` dispatches
    /// them in that order — a different order would change when a subagent's
    /// start-of-conversation injections land and when it compacts. (The
    /// persistence hook needs a live store; it is covered in `neuromance-db`.)
    #[tokio::test]
    async fn test_child_core_hook_order_mirrors_parent() {
        let mut config = config_with_subagents(vec![subagent("worker")]);
        config.context = Some(context_settings());
        let hooks = child_hooks(Some(skills_hook().await), Some(rules_hook().await));

        let core = build_child_core(&config, stub_client(), None, &hooks);

        assert_eq!(hook_names(&core), vec!["compaction", "skills", "rules"]);
    }

    /// The child `Core` attaches the shared rules and skills hooks it is handed
    /// rather than rebuilding them per run — safe because both key their state by
    /// conversation, unlike the compaction hook above.
    #[tokio::test]
    async fn test_child_core_attaches_the_shared_rules_and_skills_hooks() {
        let config = config_with_subagents(vec![subagent("worker")]);
        let skills = skills_hook().await;
        let rules = rules_hook().await;
        let hooks = child_hooks(Some(Arc::clone(&skills)), Some(Arc::clone(&rules)));

        let core = build_child_core(&config, stub_client(), None, &hooks);

        let attached = |name: &str| core.hooks.iter().find(|h| h.name() == name).cloned();
        assert_eq!(
            attached("skills").map(|h| Arc::ptr_eq(&h, &(skills as Arc<dyn Hook>))),
            Some(true)
        );
        assert_eq!(
            attached("rules").map(|h| Arc::ptr_eq(&h, &(rules as Arc<dyn Hook>))),
            Some(true)
        );
    }

    /// One hook instance serves the whole tower: `ChildContext` builds it once
    /// and every level's `hooks()` hands out that same instance, instead of one
    /// per subagent per level per run.
    #[tokio::test]
    async fn test_child_context_shares_hooks_across_levels() {
        let cancel = CancellationToken::new();
        let parent_provider = provider("primary", "PRIMARY_KEY", "openai:gpt-4o");
        let ctx = ChildContext {
            skills: Some(skills_hook().await),
            rules: Some(rules_hook().await),
            ..child_ctx(&cancel, &parent_provider, "openai:gpt-4o")
        };

        let first = ctx.hooks();
        let second = ctx.hooks();

        assert!(Arc::ptr_eq(&first.skills.unwrap(), &second.skills.unwrap()));
        assert!(Arc::ptr_eq(&first.rules.unwrap(), &second.rules.unwrap()));
    }

    /// A subagent's tool calls auto-approve: it runs inside one parent delegation
    /// with no interactive approver in the loop.
    #[test]
    fn test_child_core_auto_approves_tools() {
        let config = config_with_subagents(vec![subagent("worker")]);

        let core = build_child_core(&config, stub_client(), Some(7), &child_hooks(None, None));

        assert!(core.auto_approve_tools);
        assert_eq!(core.max_turns, Some(7));
    }

    /// A subagent retries an empty terminal turn on the operator's configured
    /// budget, not `Core`'s default. The two agree by default, so only a
    /// non-default value proves the setting is carried across.
    #[test]
    fn test_child_core_carries_configured_empty_turn_retries() {
        let mut config = config_with_subagents(vec![subagent("worker")]);
        config.agent.empty_turn_retries = 4;

        let core = build_child_core(&config, stub_client(), None, &child_hooks(None, None));

        assert_eq!(core.empty_turn_retries, 4);
    }
}
