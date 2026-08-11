//! Runtime configuration parsed from a TOML file.
//!
//! The runtime loads `$NEUROMANCE_CONFIG` (default `/etc/neuromance/config.toml`)
//! at startup. API keys are *not* embedded in this file.
//!
//! Endpoints, credentials, and default models are grouped into named
//! `[[providers]]` entries. The `[agent]` references one by name; each subagent
//! inherits the agent's provider unless it names its own. A provider supplies
//! credentials via exactly one of two paths:
//!
//! - **Env var** — `api_key_env` names an environment variable whose value is
//!   the raw provider API key, read at startup.
//! - **Tokenizer proxy** — `[providers.proxy]` points at a sealed-token file on
//!   disk (typically a projected k8s `Secret` volume). The runtime forwards LLM
//!   requests through the tokenizer proxy, which injects the real credential
//!   server-side. The agent pod never holds the plaintext.
//!
//! Exactly one of these paths must be configured per provider.

use std::borrow::Cow;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use neuromance_common::client::OutputSchema;
use neuromance_context::ContextConfig;
use neuromance_context::compaction::CompactionStrategy;
use neuromance_tools::ToolConfig;

use crate::error::RuntimeError;
use crate::sandbox::EXECUTE_PYTHON;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Oneshot,
    Serve,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    #[default]
    Auto,
    Async,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeConfig {
    pub mode: Mode,
    pub agent: AgentConfig,
    #[serde(default)]
    pub runtime: RuntimeSettings,
    #[serde(default)]
    pub approval: ApprovalConfig,
    #[serde(default)]
    pub tools: Vec<ToolConfig>,
    #[serde(default)]
    pub oneshot: Option<OneshotConfig>,
    /// Named providers: each bundles an endpoint, a credential, and a default
    /// model. The `[agent]` and each subagent reference one by name. At least
    /// one entry is required.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    /// When set, conversation history is written through to postgres as
    /// tasks run. In-memory state stays authoritative for serving.
    #[serde(default)]
    pub database: Option<DatabaseSettings>,
    /// When set, the agent compacts conversation history once it grows past
    /// a ratio of the model's context window. Compaction triggers on
    /// provider-reported usage, so no tokenizer is downloaded at startup.
    #[serde(default)]
    pub context: Option<ContextSettings>,
    /// Leaf subagents the main agent can delegate to. Each is exposed as a
    /// delegate tool (named by its `id`) and, when an `execute_python` tool is
    /// configured, through the Python REPL's `run_subagent`/`spawn_agents`
    /// bridge.
    #[serde(default)]
    pub subagents: Vec<SubagentConfig>,
    /// When set, skills are discovered from on-host roots and/or a remote
    /// endpoint, materialized to local disk at startup, and their menu (listing
    /// each skill's on-disk path) is folded into the system prompt so the agent
    /// reads a skill's full instructions from a file on demand.
    #[serde(default)]
    pub skills: Option<SkillsSettings>,
    /// When set, rule files are discovered from on-host roots and/or a remote
    /// endpoint. `always_apply` rules are injected at conversation start, and
    /// glob-matched rules are injected when a tool touches a matching path.
    #[serde(default)]
    pub rules: Option<RulesSettings>,
    /// One-time tool setup run at container start, before tasks. Each entry
    /// spawns `command` with `args`; if `token_env` is set its value is fed on
    /// stdin. Best-effort — failures are logged, never fatal.
    #[serde(default)]
    pub bootstrap: Vec<BootstrapCommand>,
    /// Sandboxed tool-execution boundary. The `sandbox` subcommand binds
    /// `listen_addr`; the orchestrator routes capability tools to `endpoint`
    /// when it is set. See [`SandboxConfig`].
    #[serde(default)]
    pub sandbox: Option<SandboxConfig>,
    /// Per-conversation working directories under a shared root, with
    /// optional named seed definitions and snapshot persistence. The feature
    /// is fully off when the section is absent. See [`WorkspaceSettings`].
    #[serde(default)]
    pub workspace: Option<WorkspaceSettings>,
}

/// Configuration for the sandboxed tool-execution boundary.
///
/// Tool execution can run in a separate sandbox process (the `sandbox`
/// subcommand) under a restricted service account with no database access. The
/// orchestrator advertises the sandbox's tools and dispatches each approved
/// call over gRPC.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SandboxConfig {
    /// Address the sandbox process binds its gRPC server to. Loopback-only:
    /// the orchestrator and sandbox share the pod network namespace, so the
    /// channel never leaves the pod.
    #[serde(default = "default_sandbox_listen_addr")]
    pub listen_addr: String,
    /// gRPC endpoint the orchestrator dials (e.g. `http://127.0.0.1:50051`).
    /// When set, the orchestrator builds remote tool adapters instead of
    /// executing capability tools in-process.
    #[serde(default)]
    pub endpoint: Option<String>,
}

fn default_sandbox_listen_addr() -> String {
    "127.0.0.1:50051".to_string()
}

/// A command run once at container start to set up a tool.
///
/// The pod has no persistent storage, so tools that read auth from a config
/// file rather than the environment must be logged in each boot. Best-effort: a
/// failure is logged, not fatal.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BootstrapCommand {
    /// Human-readable label for logs.
    pub name: String,
    /// Executable to run; must be on `PATH`.
    pub command: String,
    /// Arguments. Must not contain secrets — `config.toml` ships in a
    /// `ConfigMap`. Sealed tokens go via `token_env` (stdin) instead.
    #[serde(default)]
    pub args: Vec<String>,
    /// Env var whose value is fed to the command on stdin (e.g. a sealed token).
    /// Never placed in argv. Skipped if unset or empty at runtime.
    #[serde(default)]
    pub token_env: Option<String>,
}

/// A named provider: an endpoint, a credential, and a default model bundled
/// under a name that `[agent]` and `[[subagents]]` reference.
///
/// Exactly one credential path must be set: either `api_key_env` (raw key from
/// the environment) or an inline `[providers.proxy]` table (sealed token routed
/// through a tokenizer proxy). The two are mutually exclusive.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderConfig {
    /// Unique name, referenced by `agent.provider` and `subagent.provider`.
    pub name: String,
    /// Default model string, e.g. `openai:gpt-4o`. The `provider:` prefix
    /// selects the client type and default endpoint (see `Config::from_model`).
    /// An agent or subagent may override it with its own `model`.
    #[serde(default)]
    pub model: Option<String>,
    /// Upstream LLM endpoint, e.g. `http://llama-server.windowlickers.svc:8080/v1`
    /// for an in-cluster `OpenAI`-compatible server. Falls back to the default
    /// for the model prefix; required when `model` uses the generic
    /// `chat_completions:` or `responses:` prefixes, which have no default.
    ///
    /// When `proxy` is set, this is still the *upstream* the proxy routes to —
    /// `proxy.base_url` is the forward proxy itself. The upstream URL travels in
    /// the absolute-form request URI sent through the proxy.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Environment variable holding the raw provider API key. Mutually exclusive
    /// with `proxy`.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Tokenizer proxy for this provider. Mutually exclusive with `api_key_env`.
    #[serde(default)]
    pub proxy: Option<ProviderProxyConfig>,
}

/// Automatic context-compaction settings.
///
/// Conversation size is measured from the provider-reported `Usage` of the
/// most recent response. One known lag: the first request of a resumed
/// conversation is sent uncompacted, because no usage exists yet in that run —
/// compaction at the end of the previous run keeps stored histories under
/// target, so the residual exposure is a single user message on a
/// near-threshold conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContextSettings {
    /// Model context window in tokens (e.g. `128000`).
    pub context_window_size: usize,
    /// Ratio of the window at which compaction triggers.
    #[serde(default = "default_compaction_threshold_ratio")]
    pub compaction_threshold_ratio: f64,
    /// Target ratio of the window after compaction.
    #[serde(default = "default_target_ratio")]
    pub target_ratio: f64,
    /// Number of recent user+assistant turns preserved verbatim.
    #[serde(default = "default_preserve_recent_turns")]
    pub preserve_recent_turns: usize,
    /// Compaction strategy: `one_shot`, `hierarchical`, or `truncate`.
    #[serde(default)]
    pub strategy: CompactionStrategy,
}

impl ContextSettings {
    /// The `neuromance-context` config this section describes.
    ///
    /// [`ContextConfig`] is not `Clone` and a `CompactionHook` must not be
    /// shared, so the main agent and every subagent run each build their own
    /// from here rather than duplicating the builder chain.
    #[must_use]
    pub const fn to_context_config(&self) -> ContextConfig {
        ContextConfig::new(self.context_window_size)
            .with_compaction_threshold_ratio(self.compaction_threshold_ratio)
            .with_target_ratio(self.target_ratio)
            .with_preserve_recent_turns(self.preserve_recent_turns)
            .with_strategy(self.strategy)
    }
}

/// Skill discovery settings.
///
/// At least one of `roots` or `endpoint` must be set for skills to be enabled;
/// an empty section disables skills. On-host `roots` take precedence over the
/// remote `endpoint` when a skill name appears in both.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SkillsSettings {
    /// On-host directories to discover skills in, highest precedence first.
    /// Each immediate subdirectory containing a `SKILL.md` is a skill.
    #[serde(default)]
    pub roots: Vec<PathBuf>,
    /// A corpus-shaped skills endpoint (e.g. `https://corpus/api/v1/skills`).
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Environment variable holding a bearer token for `endpoint`, if it
    /// requires authentication.
    #[serde(default)]
    pub endpoint_token_env: Option<String>,
    /// Whether a `$mention` in user input expands the skill's body inline
    /// (e.g. a `/thing-skill` slash command). The menu and on-demand bodies are
    /// always available as files on disk regardless (default: `true`).
    #[serde(default = "default_true")]
    pub mention: bool,
    /// Byte budget for the injected skills menu (default: 8192).
    #[serde(default = "default_skill_budget")]
    pub menu_budget_bytes: usize,
    /// Byte budget for each loaded skill body (default: 8192).
    #[serde(default = "default_skill_budget")]
    pub body_budget_bytes: usize,
}

/// Default for boolean fields that should be on unless explicitly disabled.
const fn default_true() -> bool {
    true
}

const fn default_skill_budget() -> usize {
    8192
}

/// Rule-file discovery settings.
///
/// At least one of `roots` or `endpoint` must be set for rules to be enabled;
/// an empty section disables them. On-host `roots` take precedence over the
/// remote `endpoint` when a rule id appears in both.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RulesSettings {
    /// On-host directories to discover rule files (`.md`/`.mdc`) in, highest
    /// precedence first; searched recursively.
    #[serde(default)]
    pub roots: Vec<PathBuf>,
    /// A corpus-shaped rules endpoint (e.g. `https://corpus/api/v1/rules`).
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Environment variable holding a bearer token for `endpoint`, if it
    /// requires authentication.
    #[serde(default)]
    pub endpoint_token_env: Option<String>,
    /// Byte budget for each injected rule body (default: 8192).
    #[serde(default = "default_skill_budget")]
    pub body_budget_bytes: usize,
}

const fn default_compaction_threshold_ratio() -> f64 {
    0.8
}
const fn default_target_ratio() -> f64 {
    0.5
}
const fn default_preserve_recent_turns() -> usize {
    3
}

/// A subagent: a named in-process agent the main agent can delegate to.
///
/// Subagents inherit the parent agent's provider unless they name their own; a
/// subagent may also override just the model. Each subagent is provisioned with
/// the same toolset as the main agent — capability tools, the `execute_python`
/// bridge, and the delegate tools — so it can both use tools and delegate
/// further, bounded by `runtime.max_delegation_depth`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubagentConfig {
    /// Stable identifier. Used as the delegate tool name and the
    /// `run_subagent`/`spawn_agents` registry key; must be unique.
    pub id: String,
    /// System prompt prepended to every delegated task.
    pub system_prompt: String,
    /// Description shown in the delegate tool schema. Defaults to
    /// "Delegate a task to the '<id>' subagent." when omitted.
    #[serde(default)]
    pub description: Option<String>,
    /// Provider override; defaults to the parent agent's provider.
    #[serde(default)]
    pub provider: Option<String>,
    /// Model override. Defaults to the chosen provider's `model`, then the
    /// parent agent's effective model.
    #[serde(default)]
    pub model: Option<String>,
    /// Maximum chat-loop turns; defaults to the `Core` default when unset.
    #[serde(default)]
    pub max_turns: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentConfig {
    pub id: String,
    /// Name of a `[[providers]]` entry supplying the endpoint, credential, and
    /// default model.
    pub provider: String,
    /// Model override. The effective model is this when set, otherwise the
    /// referenced provider's `model`; one of the two must be present.
    #[serde(default)]
    pub model: Option<String>,
    pub system_prompt: String,
    #[serde(default)]
    pub max_turns: Option<u32>,
    #[serde(default)]
    pub streaming: bool,
    /// How many times an empty terminal turn (no content, no tool calls) is
    /// silently resubmitted before the run fails with an `empty_response`
    /// error. The retry does not consume the turn budget. Defaults to 1.
    #[serde(default = "default_empty_turn_retries")]
    pub empty_turn_retries: u32,
}

const fn default_empty_turn_retries() -> u32 {
    1
}

/// Per-provider tokenizer-proxy settings.
///
/// When present on a provider, the runtime sends that provider's outbound LLM
/// requests to `base_url` with the sealed token from `token_file` carried in
/// the `token_header`. The proxy decrypts the sealed token server-side and
/// injects the real provider credential. The agent pod never sees the plaintext.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderProxyConfig {
    /// URL of the tokenizer-proxy Service (e.g.
    /// `http://tokenizer-proxy.windowlickers.svc.cluster.local:8080`). The
    /// runtime installs this as the HTTP forward proxy on the reqwest
    /// client; the upstream target host travels in the request URL
    /// (absolute form), not a side-band header.
    pub base_url: String,
    /// Absolute path to a file holding the sealed token. Typically a projected
    /// k8s `Secret` volume mount such as `/var/run/neuromance/tokens/llm`.
    pub token_file: PathBuf,
    /// Header name under which the sealed token is sent. Defaults to
    /// `X-Tokenizer-Token`.
    #[serde(default = "default_token_header")]
    pub token_header: String,
}

fn default_token_header() -> String {
    "X-Tokenizer-Token".to_string()
}

/// Postgres persistence settings.
///
/// The connection URL is a credential (it usually embeds a password), so it
/// is never written in the TOML file — `url_env` names the environment
/// variable that holds it, the same policy as a provider's `api_key_env`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatabaseSettings {
    /// Environment variable holding the postgres connection URL
    /// (e.g. `postgres://user:pass@host:5432/neuromance`).
    pub url_env: String,
    /// Maximum connections in the pool.
    #[serde(default = "default_db_max_connections")]
    pub max_connections: u32,
    /// Cap on how long a persistence call may wait for a pool connection.
    /// Bounds the stall a sick database can add to an agent turn.
    #[serde(default = "default_db_acquire_timeout")]
    pub acquire_timeout_seconds: u64,
    /// Whether the runtime applies neuromance's embedded schema migrations at
    /// startup.
    ///
    /// Defaults to `true`: the runtime owns and provisions its own tables. Set
    /// to `false` when an external owner (an operator or a shared schema
    /// service) manages the database — the runtime then uses the existing
    /// schema without attempting any DDL.
    #[serde(default = "default_db_run_migrations")]
    pub run_migrations: bool,
}

const fn default_db_max_connections() -> u32 {
    5
}
const fn default_db_acquire_timeout() -> u64 {
    5
}
const fn default_db_run_migrations() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeSettings {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
    #[serde(default = "default_health_addr")]
    pub health_addr: String,
    #[serde(default = "default_shutdown_grace")]
    pub shutdown_grace_seconds: u64,
    /// Maximum number of tasks buffered for the worker. `POST /tasks/new`
    /// returns `429` once the queue is at capacity. Sized for a single
    /// serial worker — buffering far beyond a handful is dishonest about
    /// what the runtime can actually do.
    #[serde(default = "default_max_queue_depth")]
    pub max_queue_depth: usize,
    /// Maximum length of a delegation chain, counting subagent hops from the
    /// main agent (which is depth 0). At `1` the main agent reaches subagents
    /// but those subagents hold no delegate tools; at `2` a subagent may
    /// delegate one further hop, and so on. The deepest subagents are still
    /// fully tool-capable — they just cannot delegate. Bounds startup cost and
    /// the *depth* of a delegation chain. It does not bound how many runs a
    /// chain produces: each delegated run carries its own `max_turns` budget, so
    /// an agent that delegates repeatedly can still fan out widely within the
    /// depth limit. Ignored when no delegation is configured (no `[[subagents]]`
    /// and `self_delegation` off).
    #[serde(default = "default_max_delegation_depth")]
    pub max_delegation_depth: u32,
    /// Give the main agent a `task` delegate tool that runs a fresh copy of
    /// itself — the configured system prompt, and the provider, model, and turn
    /// budget the parent is running — on a delegated subtask. A
    /// zero-boilerplate form of `[[subagents]]`: the synthesized subagent joins
    /// the same `max_delegation_depth`-bounded tower and, when an
    /// `execute_python` tool is configured, the same
    /// `run_subagent`/`spawn_agents` bridge. Off by default.
    ///
    /// One divergence: a serve task that supplies its own `system_prompt`
    /// overrides only the parent's seed message. The clone still carries
    /// `agent.system_prompt`, since a subagent's prompt is fixed when the tower
    /// is built and the per-task prompt is not known then.
    #[serde(default)]
    pub self_delegation: bool,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
            health_addr: default_health_addr(),
            shutdown_grace_seconds: default_shutdown_grace(),
            max_queue_depth: default_max_queue_depth(),
            max_delegation_depth: default_max_delegation_depth(),
            self_delegation: false,
        }
    }
}

/// Reserved id and delegate-tool name for the synthesized self-delegation
/// subagent (`runtime.self_delegation`).
pub const SELF_DELEGATION_ID: &str = "task";

fn default_listen_addr() -> String {
    "127.0.0.1:8080".to_string()
}
fn default_health_addr() -> String {
    "127.0.0.1:8081".to_string()
}
const fn default_shutdown_grace() -> u64 {
    30
}
const fn default_max_queue_depth() -> usize {
    8
}
const fn default_max_delegation_depth() -> u32 {
    2
}

/// Upper bound on `runtime.max_delegation_depth`. Each level multiplies the
/// number of subagent instances built at startup and widens the delegation
/// fan-out; deeper towers buy little and cost a lot.
const MAX_DELEGATION_DEPTH_CEILING: u32 = 5;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ApprovalConfig {
    #[serde(default)]
    pub mode: ApprovalMode,
    pub webhook_url: Option<String>,
    #[serde(default = "default_approval_timeout")]
    pub timeout_seconds: u64,
    /// Skip the startup safety gate that refuses to start when
    /// `mode = "auto"` is paired with tools whose `is_auto_approved()`
    /// returns false (`bash`, `edit`, `write`). The agentic loop already
    /// approves every tool under `Auto` — this flag opts out of the
    /// duplicate startup check, intended for deployments where the pod
    /// boundary itself provides isolation (e.g. kata containers).
    #[serde(default)]
    pub allow_unsafe_tools: bool,
}

const fn default_approval_timeout() -> u64 {
    60
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OneshotConfig {
    pub input: String,
    #[serde(default)]
    pub output_path: Option<PathBuf>,
    /// Names a `[[workspace.definitions]]` entry to seed the workspace from.
    /// With no name the sole definition (when exactly one is configured) is
    /// used; otherwise the workspace starts empty.
    #[serde(default)]
    pub workspace: Option<String>,
    /// Constrain the run's responses to this inline JSON Schema, enforced by the provider. The
    /// parsed object is written to the output file's `structured` field.
    ///
    /// The root must be `type: "object"` and every object node must set
    /// `additionalProperties: false`; anything else fails startup validation.
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
}

/// Per-conversation workspace settings.
///
/// Each conversation gets a working directory at `<root>/<conversation_id>`,
/// created on first use and reused by continuations. `root` is expected to be
/// a volume shared with the sandbox container (a k8s `emptyDir`) so tools
/// executed there see the same files the orchestrator seeds and snapshots.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkspaceSettings {
    /// Absolute directory under which per-conversation workspaces are
    /// created, e.g. the `emptyDir` mount `/workspace`.
    pub root: PathBuf,
    /// Named seed definitions (`[[workspace.definitions]]`). A task selects
    /// one by its `workspace` field; with no field, the sole definition (when
    /// exactly one is configured) is used. Seeding runs once, when the
    /// conversation's workspace is first created.
    #[serde(default)]
    pub definitions: Vec<WorkspaceDefinition>,
    /// S3-compatible object storage (garage) used by object seeds and
    /// snapshot persistence.
    #[serde(default)]
    pub storage: Option<WorkspaceStorageSettings>,
    /// Snapshot/restore of workspaces across replicas. Requires `storage`;
    /// absent means workspaces are purely local to the pod.
    #[serde(default)]
    pub persistence: Option<WorkspacePersistenceSettings>,
    /// Tokenizer-proxy auth for git seeds, shaped like a provider's
    /// `[providers.proxy]`. Absent means git seeds clone anonymously.
    #[serde(default)]
    pub git_proxy: Option<ProviderProxyConfig>,
}

/// A named workspace definition selecting seed content for new workspaces.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkspaceDefinition {
    /// Unique name, referenced by the task request's `workspace` field (serve)
    /// or `[oneshot].workspace`. When no name is given and exactly one
    /// definition is configured, that sole definition is used.
    pub name: String,
    /// Git repositories cloned into the workspace on first creation.
    #[serde(default)]
    pub git: Vec<GitSeed>,
    /// Object archives (tar.zst) unpacked into the workspace on first
    /// creation. Requires `[workspace.storage]`.
    #[serde(default)]
    pub objects: Vec<ObjectSeed>,
}

/// A git repository seeded into a fresh workspace.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GitSeed {
    /// Remote URL. Authenticated clones require an `http://` scheme so the
    /// tokenizer proxy can read the sealed header (it upgrades to TLS
    /// upstream).
    pub url: String,
    /// Branch, tag, or commit SHA to check out; defaults to the remote HEAD.
    #[serde(default, rename = "ref")]
    pub reference: Option<String>,
    /// Clone target relative to the workspace dir; must not be absolute or
    /// contain `..`. Defaults to the repository basename.
    #[serde(default)]
    pub dest: Option<String>,
}

/// An object-storage archive seeded into a fresh workspace.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ObjectSeed {
    /// Object key within `workspace.storage.bucket`; the object is a
    /// tar.zst archive.
    pub key: String,
    /// Unpack target relative to the workspace dir; must not be absolute or
    /// contain `..`. Defaults to the workspace dir itself.
    #[serde(default)]
    pub dest: Option<String>,
}

/// S3-compatible object storage for workspace seeds and snapshots.
///
/// Credentials come from the environment — the TOML ships in a `ConfigMap`
/// and never holds secrets, the same policy as a provider's `api_key_env`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkspaceStorageSettings {
    /// Endpoint URL, e.g. `http://garage.storage.svc.cluster.local:3900`.
    pub endpoint: String,
    /// Bucket holding seed archives and snapshots.
    pub bucket: String,
    /// Region string presented to the S3 API. Garage accepts its configured
    /// region name; defaults to `garage`.
    #[serde(default = "default_storage_region")]
    pub region: String,
    /// Environment variable holding the S3 access key id.
    pub access_key_id_env: String,
    /// Environment variable holding the S3 secret access key.
    pub secret_access_key_env: String,
}

fn default_storage_region() -> String {
    "garage".to_string()
}

/// Workspace snapshot/restore settings.
///
/// When present, the worker restores a conversation's workspace from the
/// last snapshot before a continuation run and snapshots it after every run.
/// Retention is expiry-based and external: point a garage lifecycle rule at
/// `prefix` rather than deleting from the runtime.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkspacePersistenceSettings {
    /// Object key prefix for snapshots. A conversation's snapshot lives at
    /// `<prefix><conversation_id>.tar.zst`.
    #[serde(default = "default_snapshot_prefix")]
    pub prefix: String,
    /// Glob patterns excluded from snapshots, matched against paths relative
    /// to the workspace dir (e.g. `target/**`, `node_modules/**`).
    #[serde(default)]
    pub exclude: Vec<String>,
}

fn default_snapshot_prefix() -> String {
    "snapshots/".to_string()
}

impl RuntimeConfig {
    /// Load from `$NEUROMANCE_CONFIG`, defaulting to `/etc/neuromance/config.toml`.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] if the file is missing, unreadable, or malformed.
    pub fn load_default() -> Result<Self, RuntimeError> {
        let path = std::env::var("NEUROMANCE_CONFIG")
            .unwrap_or_else(|_| "/etc/neuromance/config.toml".to_string());
        Self::load(&path)
    }

    /// Load and parse a config file at `path`.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] if the file cannot be read or parsed.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, RuntimeError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .map_err(|e| RuntimeError::Config(format!("read {}: {e}", path.display())))?;
        let config: Self = toml::from_str(&contents)
            .map_err(|e| RuntimeError::Config(format!("parse {}: {e}", path.display())))?;
        config.validate()?;
        Ok(config)
    }

    /// Look up a provider by name.
    #[must_use]
    pub fn provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// The agent's effective model: `agent.model` if set, otherwise the
    /// referenced provider's `model`.
    ///
    /// Returns `None` when neither is set or the agent's provider is unknown —
    /// both rejected by [`Self::validate`].
    #[must_use]
    pub fn agent_model(&self) -> Option<&str> {
        if let Some(model) = self.agent.model.as_deref() {
            return Some(model);
        }
        self.provider(&self.agent.provider)?.model.as_deref()
    }

    /// Validate and build the schema from `[oneshot].output_schema`.
    ///
    /// The schema's `title` names it for providers that want one, defaulting to `output`.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] when the schema is not one the providers can enforce.
    pub fn oneshot_output_schema(&self) -> Result<Option<OutputSchema>, RuntimeError> {
        let Some(schema) = self.oneshot.as_ref().and_then(|o| o.output_schema.as_ref()) else {
            return Ok(None);
        };
        OutputSchema::from_value(schema.clone())
            .map(Some)
            .map_err(|e| RuntimeError::Config(format!("[oneshot].output_schema: {e}")))
    }

    /// Whether any delegation tower must be built: either explicit
    /// `[[subagents]]` or the synthesized self-delegation subagent.
    #[must_use]
    pub const fn has_delegation(&self) -> bool {
        !self.subagents.is_empty() || self.runtime.self_delegation
    }

    /// The synthesized self-delegation subagent, when `runtime.self_delegation`
    /// is set: a fresh copy of the main agent (`agent.system_prompt`, and — via
    /// the inherit-when-`None` resolution — the provider, model, and turn budget
    /// the parent is running for this task), exposed as the
    /// [`SELF_DELEGATION_ID`] delegate tool. A per-task `system_prompt` override
    /// does not reach it; see [`RuntimeSettings::self_delegation`].
    #[must_use]
    pub fn self_delegation_subagent(&self) -> Option<SubagentConfig> {
        self.runtime.self_delegation.then(|| SubagentConfig {
            id: SELF_DELEGATION_ID.to_string(),
            system_prompt: self.agent.system_prompt.clone(),
            description: Some(
                "Delegate a self-contained subtask to a fresh instance of yourself — same \
                 instructions, model, and tools. The subagent runs autonomously and returns \
                 only its final result. Use it for well-scoped or parallelizable work; give it \
                 all the context it needs, since it does not see this conversation."
                    .to_string(),
            ),
            // Inherit the parent's provider/model through the existing
            // resolution (see `subagents.rs`); mirror its turn budget so a
            // self-delegated run behaves like a continuation of this agent.
            provider: None,
            model: None,
            max_turns: self.agent.max_turns,
        })
    }

    /// This config with the synthesized self-delegation subagent appended when
    /// `runtime.self_delegation` is set; borrowed unchanged otherwise. Injecting
    /// it into `subagents` lets the whole delegation tower, `SubagentTool`
    /// registration, and Python bridge treat self-delegation exactly like a
    /// configured `[[subagents]]` entry.
    #[must_use]
    pub fn with_self_delegation(&self) -> Cow<'_, Self> {
        self.self_delegation_subagent()
            .map_or(Cow::Borrowed(self), |sub| {
                let mut cfg = self.clone();
                cfg.subagents.push(sub);
                Cow::Owned(cfg)
            })
    }

    /// Resolve a task's provider and model from optional per-task overrides.
    ///
    /// Mirrors subagent resolution (see `subagents.rs`): the named provider
    /// falls back to `agent.provider`, and the model override's `provider:`
    /// prefix only selects the client family — the credential always comes
    /// from the resolved provider. With no overrides this reduces to
    /// `agent.provider` + [`Self::agent_model`], leaving the shared agent
    /// unchanged.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] when the provider names no
    /// `[[providers]]` entry or no effective model can be resolved.
    pub fn resolve_provider_and_model<'a>(
        &'a self,
        provider_override: Option<&str>,
        model_override: Option<&'a str>,
    ) -> Result<(&'a ProviderConfig, &'a str), RuntimeError> {
        let provider_name = provider_override.unwrap_or(&self.agent.provider);
        let provider = self.provider(provider_name).ok_or_else(|| {
            RuntimeError::Config(format!(
                "provider '{provider_name}' does not match any [[providers]] entry"
            ))
        })?;
        let model = match (model_override, provider_override) {
            (Some(model), _) => model,
            // No model override but a provider override: prefer that
            // provider's default model, then the agent's effective model.
            (None, Some(_)) => provider
                .model
                .as_deref()
                .or_else(|| self.agent_model())
                .ok_or_else(|| {
                    RuntimeError::Config(format!(
                        "agent has no model: set agent.model, provider '{provider_name}' \
                         model, or a per-task model override"
                    ))
                })?,
            // The shared agent path: identical to before — `agent.model` wins
            // over the agent provider's default.
            (None, None) => self.agent_model().ok_or_else(|| {
                RuntimeError::Config(format!(
                    "agent has no model: set agent.model or provider '{}' model",
                    self.agent.provider
                ))
            })?,
        };
        Ok((provider, model))
    }

    /// The single workspace definition, when exactly one is configured — the
    /// default seed for a task that names no `workspace`. With zero or several
    /// definitions there is no unambiguous default and the workspace starts
    /// empty; a task then picks one explicitly by name via
    /// [`Self::workspace_definition`].
    #[must_use]
    pub fn sole_workspace_definition(&self) -> Option<&WorkspaceDefinition> {
        match self.workspace.as_ref()?.definitions.as_slice() {
            [only] => Some(only),
            _ => None,
        }
    }

    /// Look up a workspace definition by name.
    #[must_use]
    pub fn workspace_definition(&self, name: &str) -> Option<&WorkspaceDefinition> {
        self.workspace
            .as_ref()?
            .definitions
            .iter()
            .find(|def| def.name == name)
    }

    /// Cross-field validation that serde alone cannot express.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] when:
    /// - `mode = "oneshot"` but no `[oneshot]` section is present
    /// - `approval.mode = "async"` but `approval.webhook_url` is unset
    /// - `approval.webhook_url` is set to a URL whose scheme is not
    ///   `http` or `https`
    /// - no `[[providers]]` entry is present
    /// - a provider has a duplicate name, both or neither credential path, or
    ///   an invalid `base_url`/proxy
    /// - `agent.provider` (or a `subagent.provider`) names no provider
    /// - the agent has no effective model (`agent.model` and the provider's
    ///   `model` are both unset)
    /// - `[oneshot].output_schema` is not an enforceable JSON Schema
    /// - a `[[bootstrap]]` entry has an empty `name` or `command`
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if matches!(self.mode, Mode::Oneshot) && self.oneshot.is_none() {
            return Err(RuntimeError::Config(
                "oneshot mode requires [oneshot] section".to_string(),
            ));
        }
        self.oneshot_output_schema()?;
        if matches!(self.approval.mode, ApprovalMode::Async) && self.approval.webhook_url.is_none()
        {
            return Err(RuntimeError::Config(
                "approval.mode = \"async\" requires approval.webhook_url".to_string(),
            ));
        }
        if let Some(url) = &self.approval.webhook_url {
            validate_http_url(url, "approval.webhook_url")?;
        }

        self.validate_providers()?;

        if self.provider(&self.agent.provider).is_none() {
            return Err(RuntimeError::Config(format!(
                "agent.provider '{}' does not match any [[providers]] entry",
                self.agent.provider
            )));
        }
        if self.agent_model().is_none() {
            return Err(RuntimeError::Config(format!(
                "agent has no model: set agent.model or provider '{}' model",
                self.agent.provider
            )));
        }

        if self.runtime.max_queue_depth == 0 {
            return Err(RuntimeError::Config(
                "runtime.max_queue_depth must be at least 1".to_string(),
            ));
        }

        if let Some(database) = &self.database {
            if database.url_env.trim().is_empty() {
                return Err(RuntimeError::Config(
                    "database.url_env must not be empty".to_string(),
                ));
            }
            if database.max_connections == 0 {
                return Err(RuntimeError::Config(
                    "database.max_connections must be at least 1".to_string(),
                ));
            }
            if database.acquire_timeout_seconds == 0 {
                return Err(RuntimeError::Config(
                    "database.acquire_timeout_seconds must be at least 1".to_string(),
                ));
            }
        }

        if self.has_delegation() {
            let depth = self.runtime.max_delegation_depth;
            if !(1..=MAX_DELEGATION_DEPTH_CEILING).contains(&depth) {
                return Err(RuntimeError::Config(format!(
                    "runtime.max_delegation_depth must be between 1 and \
                     {MAX_DELEGATION_DEPTH_CEILING} when subagents or self_delegation are \
                     configured (got {depth})"
                )));
            }
        }

        self.validate_subagents()?;

        for entry in &self.bootstrap {
            if entry.name.trim().is_empty() {
                return Err(RuntimeError::Config(
                    "bootstrap entry name must not be empty".to_string(),
                ));
            }
            if entry.command.trim().is_empty() {
                return Err(RuntimeError::Config(format!(
                    "bootstrap entry '{}' command must not be empty",
                    entry.name
                )));
            }
        }

        if let Some(context) = &self.context {
            context.validate()?;
        }

        self.validate_sandbox()?;
        self.validate_workspace()?;
        Ok(())
    }

    /// Validate `[[subagents]]`: non-empty unique ids, non-empty system
    /// prompts, and that any `provider` override names a known provider.
    fn validate_subagents(&self) -> Result<(), RuntimeError> {
        let mut seen_ids = std::collections::HashSet::new();
        for sub in &self.subagents {
            if sub.id.trim().is_empty() {
                return Err(RuntimeError::Config(
                    "subagent id must not be empty".to_string(),
                ));
            }
            if !seen_ids.insert(sub.id.as_str()) {
                return Err(RuntimeError::Config(format!(
                    "duplicate subagent id '{}'",
                    sub.id
                )));
            }
            if sub.system_prompt.trim().is_empty() {
                return Err(RuntimeError::Config(format!(
                    "subagent '{}' system_prompt must not be empty",
                    sub.id
                )));
            }
            if let Some(provider) = &sub.provider
                && self.provider(provider).is_none()
            {
                return Err(RuntimeError::Config(format!(
                    "subagent '{}' provider '{provider}' does not match any [[providers]] entry",
                    sub.id
                )));
            }
        }
        if self.runtime.self_delegation && self.subagents.iter().any(|s| s.id == SELF_DELEGATION_ID)
        {
            return Err(RuntimeError::Config(format!(
                "subagent id '{SELF_DELEGATION_ID}' is reserved for runtime.self_delegation; \
                 rename the [[subagents]] entry or disable self_delegation"
            )));
        }
        // The synthesized clone takes `agent.system_prompt` verbatim and is
        // appended after this loop runs, so it never reaches the non-empty check
        // above. Without this, the same config written as an explicit
        // [[subagents]] entry is rejected while the synthesized one boots an
        // uninstructed agent holding the parent's whole toolset.
        if self.runtime.self_delegation && self.agent.system_prompt.trim().is_empty() {
            return Err(RuntimeError::Config(format!(
                "agent.system_prompt must not be empty when runtime.self_delegation is set: the \
                 synthesized '{SELF_DELEGATION_ID}' subagent runs on it. Set agent.system_prompt \
                 or disable self_delegation."
            )));
        }
        Ok(())
    }

    /// Validate `[sandbox]`: a loopback `listen_addr`, a well-formed `endpoint`,
    /// and that the orchestrator role does not request the unsupported
    /// Python-over-subagents bridge across the sandbox boundary.
    fn validate_sandbox(&self) -> Result<(), RuntimeError> {
        let Some(sandbox) = &self.sandbox else {
            return Ok(());
        };

        let addr: SocketAddr = sandbox.listen_addr.parse().map_err(|e| {
            RuntimeError::Config(format!(
                "sandbox.listen_addr '{}' is not a valid socket address: {e}",
                sandbox.listen_addr
            ))
        })?;
        if !addr.ip().is_loopback() {
            return Err(RuntimeError::Config(format!(
                "sandbox.listen_addr must be loopback (the sandbox shares the pod \
                 network namespace), got '{}'",
                sandbox.listen_addr
            )));
        }

        if let Some(endpoint) = &sandbox.endpoint {
            validate_http_url(endpoint, "sandbox.endpoint")?;
            // The Python run_subagent/spawn_agents bridge needs the interpreter
            // and the subagent tower in one process; it cannot cross the gRPC
            // boundary in this release. self_delegation builds the same bridge,
            // so it is rejected on the same terms.
            if self.has_delegation() && self.tools.iter().any(|t| t.name == EXECUTE_PYTHON) {
                return Err(RuntimeError::Config(
                    "sandbox.endpoint with delegation ([[subagents]] or self_delegation) and an \
                     execute_python tool is not yet supported: the Python \
                     run_subagent/spawn_agents bridge requires the interpreter and the subagent \
                     tower in the same process. Remove the execute_python tool or the delegation \
                     config, or unset sandbox.endpoint."
                        .to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate `[[providers]]`: at least one entry, unique names, exactly one
    /// credential path each, and well-formed URLs/proxy fields.
    fn validate_providers(&self) -> Result<(), RuntimeError> {
        if self.providers.is_empty() {
            return Err(RuntimeError::Config(
                "at least one [[providers]] entry is required".to_string(),
            ));
        }
        let mut seen_names = std::collections::HashSet::new();
        for provider in &self.providers {
            if provider.name.trim().is_empty() {
                return Err(RuntimeError::Config(
                    "provider name must not be empty".to_string(),
                ));
            }
            if !seen_names.insert(provider.name.as_str()) {
                return Err(RuntimeError::Config(format!(
                    "duplicate provider name '{}'",
                    provider.name
                )));
            }
            match (&provider.api_key_env, &provider.proxy) {
                (Some(_), Some(_)) => {
                    return Err(RuntimeError::Config(format!(
                        "provider '{}': api_key_env and proxy are mutually exclusive",
                        provider.name
                    )));
                }
                (None, None) => {
                    return Err(RuntimeError::Config(format!(
                        "provider '{}': must set either api_key_env or proxy",
                        provider.name
                    )));
                }
                _ => {}
            }
            if let Some(url) = &provider.base_url {
                validate_http_url(url, &format!("provider '{}' base_url", provider.name))?;
            }
            if let Some(proxy) = &provider.proxy {
                validate_proxy_config(proxy, &format!("provider '{}' proxy", provider.name))?;
            }
        }
        Ok(())
    }

    /// Validate `[workspace]`: an absolute root, unique definition names,
    /// contained seed destinations, storage present wherever object seeds or
    /// persistence need it, and compiling exclusion globs.
    fn validate_workspace(&self) -> Result<(), RuntimeError> {
        let Some(workspace) = &self.workspace else {
            return Ok(());
        };

        if !workspace.root.is_absolute() {
            return Err(RuntimeError::Config(format!(
                "workspace.root must be an absolute path, got '{}'",
                workspace.root.display()
            )));
        }

        let mut seen_names = std::collections::HashSet::new();
        for def in &workspace.definitions {
            if def.name.trim().is_empty() {
                return Err(RuntimeError::Config(
                    "workspace definition name must not be empty".to_string(),
                ));
            }
            if !seen_names.insert(def.name.as_str()) {
                return Err(RuntimeError::Config(format!(
                    "duplicate workspace definition name '{}'",
                    def.name
                )));
            }
            for seed in &def.git {
                if seed.url.trim().is_empty() {
                    return Err(RuntimeError::Config(format!(
                        "workspace definition '{}': git seed url must not be empty",
                        def.name
                    )));
                }
                // A proxied clone needs the request in the clear so the proxy
                // can read the sealed-token header, so the remote must be
                // http://. Catch it here rather than at first seed.
                if workspace.git_proxy.is_some() && !seed.url.starts_with("http://") {
                    return Err(RuntimeError::Config(format!(
                        "workspace definition '{}': git seed url '{}' must be http:// when \
                         [workspace.git_proxy] is configured, so the proxy can read the \
                         sealed-token header",
                        def.name, seed.url
                    )));
                }
                validate_seed_dest(seed.dest.as_deref(), &def.name)?;
            }
            for seed in &def.objects {
                if seed.key.trim().is_empty() {
                    return Err(RuntimeError::Config(format!(
                        "workspace definition '{}': object seed key must not be empty",
                        def.name
                    )));
                }
                validate_seed_dest(seed.dest.as_deref(), &def.name)?;
            }
            if !def.objects.is_empty() && workspace.storage.is_none() {
                return Err(RuntimeError::Config(format!(
                    "workspace definition '{}' has object seeds but [workspace.storage] is \
                     not configured",
                    def.name
                )));
            }
        }

        if let Some(storage) = &workspace.storage {
            validate_http_url(&storage.endpoint, "workspace.storage.endpoint")?;
            for (value, field) in [
                (&storage.bucket, "bucket"),
                (&storage.region, "region"),
                (&storage.access_key_id_env, "access_key_id_env"),
                (&storage.secret_access_key_env, "secret_access_key_env"),
            ] {
                if value.trim().is_empty() {
                    return Err(RuntimeError::Config(format!(
                        "workspace.storage.{field} must not be empty"
                    )));
                }
            }
        }

        if let Some(persistence) = &workspace.persistence {
            if workspace.storage.is_none() {
                return Err(RuntimeError::Config(
                    "[workspace.persistence] requires [workspace.storage]".to_string(),
                ));
            }
            for pattern in &persistence.exclude {
                globset::Glob::new(pattern).map_err(|e| {
                    RuntimeError::Config(format!(
                        "workspace.persistence.exclude pattern '{pattern}' is not a valid \
                         glob: {e}"
                    ))
                })?;
            }
        }

        if let Some(proxy) = &workspace.git_proxy {
            validate_proxy_config(proxy, "workspace.git_proxy")?;
        }

        Ok(())
    }
}

/// Validate a seed `dest`: relative and free of `..` components, so it cannot
/// escape the workspace dir it is joined onto.
fn validate_seed_dest(dest: Option<&str>, definition: &str) -> Result<(), RuntimeError> {
    let Some(dest) = dest else {
        return Ok(());
    };
    let path = Path::new(dest);
    if path.is_absolute() {
        return Err(RuntimeError::Config(format!(
            "workspace definition '{definition}': seed dest must be relative, got '{dest}'"
        )));
    }
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(RuntimeError::Config(format!(
            "workspace definition '{definition}': seed dest must not contain '..', got '{dest}'"
        )));
    }
    Ok(())
}

/// Validate a tokenizer-proxy table: http(s) `base_url`, absolute
/// `token_file`, non-empty `token_header`. `ctx` names the owning section in
/// error messages (e.g. `provider 'x' proxy`, `workspace.git_proxy`).
fn validate_proxy_config(proxy: &ProviderProxyConfig, ctx: &str) -> Result<(), RuntimeError> {
    validate_http_url(&proxy.base_url, &format!("{ctx}.base_url"))?;
    if !proxy.token_file.is_absolute() {
        return Err(RuntimeError::Config(format!(
            "{ctx}.token_file must be an absolute path, got '{}'",
            proxy.token_file.display()
        )));
    }
    if proxy.token_header.trim().is_empty() {
        return Err(RuntimeError::Config(format!(
            "{ctx}.token_header must not be empty"
        )));
    }
    Ok(())
}

impl ContextSettings {
    /// Cross-field validation for the `[context]` section.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] when `context_window_size` is zero,
    /// either ratio is outside `(0, 1]`, the target ratio is not below the
    /// threshold ratio, or `preserve_recent_turns` is zero.
    fn validate(&self) -> Result<(), RuntimeError> {
        if self.context_window_size == 0 {
            return Err(RuntimeError::Config(
                "context.context_window_size must be at least 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.compaction_threshold_ratio)
            || self.compaction_threshold_ratio == 0.0
        {
            return Err(RuntimeError::Config(format!(
                "context.compaction_threshold_ratio must be in (0, 1], got {}",
                self.compaction_threshold_ratio
            )));
        }
        if self.target_ratio <= 0.0 || self.target_ratio >= 1.0 {
            return Err(RuntimeError::Config(format!(
                "context.target_ratio must be in (0, 1), got {}",
                self.target_ratio
            )));
        }
        if self.target_ratio >= self.compaction_threshold_ratio {
            return Err(RuntimeError::Config(format!(
                "context.target_ratio ({}) must be below context.compaction_threshold_ratio ({})",
                self.target_ratio, self.compaction_threshold_ratio
            )));
        }
        if self.preserve_recent_turns == 0 {
            return Err(RuntimeError::Config(
                "context.preserve_recent_turns must be at least 1: \
                 0 would summarize the in-flight tool exchange"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_http_url(url: &str, field: &str) -> Result<(), RuntimeError> {
    let parsed = url::Url::parse(url)
        .map_err(|e| RuntimeError::Config(format!("{field} is not a valid URL: {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(RuntimeError::Config(format!(
            "{field} must use http or https, got scheme '{}'",
            parsed.scheme()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    fn minimal_oneshot_toml() -> &'static str {
        r#"
            mode = "oneshot"

            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            api_key_env = "OPENAI_API_KEY"

            [agent]
            id = "research"
            provider = "default"
            system_prompt = "Be helpful."

            [oneshot]
            input = "Hello, world."
        "#
    }

    /// A `[[providers]]` + `[agent]` preamble for `serve`-mode tests, leaving the
    /// caller to append the section under test.
    const SERVE_PREAMBLE: &str = r#"
            mode = "serve"

            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            api_key_env = "K"

            [agent]
            id = "manager"
            provider = "default"
            system_prompt = "be helpful"
    "#;

    fn serve_config(extra: &str) -> RuntimeConfig {
        toml::from_str(&format!("{SERVE_PREAMBLE}{extra}")).unwrap()
    }

    /// An unenforceable schema fails startup, not the first run.
    #[test]
    fn test_validate_rejects_an_open_oneshot_output_schema() {
        let mut config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.oneshot.as_mut().unwrap().output_schema = Some(serde_json::json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}}
        }));

        let err = config
            .validate()
            .expect_err("open object is not enforceable");

        assert!(
            err.to_string().contains("[oneshot].output_schema"),
            "error must name the setting: {err}"
        );
    }

    #[test]
    fn test_oneshot_output_schema_takes_its_name_from_title() {
        let mut config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.oneshot.as_mut().unwrap().output_schema = Some(serde_json::json!({
            "title": "verdict",
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": false
        }));

        let schema = config.oneshot_output_schema().unwrap().unwrap();

        assert_eq!(schema.name, "verdict");
    }

    #[test]
    fn test_minimal_oneshot_parses_with_defaults() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert_eq!(config.mode, Mode::Oneshot);
        assert_eq!(config.agent.id, "research");
        assert_eq!(config.agent.provider, "default");
        assert_eq!(config.agent_model(), Some("openai:gpt-4o"));
        assert_eq!(config.runtime.listen_addr, "127.0.0.1:8080");
        assert_eq!(config.runtime.shutdown_grace_seconds, 30);
        assert_eq!(config.runtime.max_queue_depth, 8);
        assert_eq!(config.approval.mode, ApprovalMode::Auto);
        assert!(config.tools.is_empty());
        assert!(config.oneshot.is_some());
    }

    #[test]
    fn test_max_queue_depth_round_trips_when_set() {
        let config = serve_config(
            r"
            [runtime]
            max_queue_depth = 64
        ",
        );
        config.validate().unwrap();
        assert_eq!(config.runtime.max_queue_depth, 64);
    }

    #[test]
    fn test_oneshot_without_section_fails_validation() {
        let toml_str = r#"
            mode = "oneshot"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            api_key_env = "K"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("oneshot mode requires"));
    }

    #[test]
    fn test_zero_max_queue_depth_fails_validation() {
        let config = serve_config(
            r"
            [runtime]
            max_queue_depth = 0
        ",
        );
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("max_queue_depth"));
    }

    const ONE_SUBAGENT: &str = r#"
            [[subagents]]
            id = "worker"
            system_prompt = "you work"
    "#;

    #[test]
    fn test_max_delegation_depth_defaults_to_two() {
        let config = serve_config(ONE_SUBAGENT);
        config.validate().unwrap();
        assert_eq!(config.runtime.max_delegation_depth, 2);
    }

    #[test]
    fn test_zero_max_delegation_depth_with_subagents_fails() {
        let config = serve_config(&format!(
            "{ONE_SUBAGENT}
            [runtime]
            max_delegation_depth = 0
        "
        ));
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("max_delegation_depth"));
    }

    #[test]
    fn test_excessive_max_delegation_depth_fails() {
        let config = serve_config(&format!(
            "{ONE_SUBAGENT}
            [runtime]
            max_delegation_depth = 99
        "
        ));
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("max_delegation_depth"));
    }

    /// Without subagents the depth bound is irrelevant, so an out-of-range value
    /// does not block startup.
    #[test]
    fn test_max_delegation_depth_ignored_without_subagents() {
        let config = serve_config(
            r"
            [runtime]
            max_delegation_depth = 0
        ",
        );
        config.validate().unwrap();
    }

    #[test]
    fn test_async_approval_without_webhook_fails_validation() {
        let config = serve_config(
            r#"
            [approval]
            mode = "async"
        "#,
        );
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("webhook_url"));
    }

    #[test]
    fn test_webhook_url_must_be_http_or_https() {
        let config = serve_config(
            r#"
            [approval]
            mode = "async"
            webhook_url = "file:///etc/passwd"
        "#,
        );
        let err = config.validate().err().unwrap();
        let msg = format!("{err}");
        assert!(
            msg.contains("http or https"),
            "expected scheme rejection, got: {msg}"
        );
    }

    #[test]
    fn test_webhook_url_https_validates() {
        let config = serve_config(
            r#"
            [approval]
            mode = "async"
            webhook_url = "https://approve.example.com/decide"
        "#,
        );
        config.validate().unwrap();
    }

    #[test]
    fn test_allow_unsafe_tools_defaults_to_false_when_omitted() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        assert!(!config.approval.allow_unsafe_tools);
    }

    #[test]
    fn test_allow_unsafe_tools_round_trips_when_set() {
        let config = serve_config(
            r#"
            [approval]
            mode = "auto"
            allow_unsafe_tools = true
        "#,
        );
        config.validate().unwrap();
        assert!(config.approval.allow_unsafe_tools);
    }

    #[test]
    fn test_serve_mode_does_not_require_oneshot() {
        let config = serve_config("");
        config.validate().unwrap();
        assert_eq!(config.mode, Mode::Serve);
    }

    #[test]
    fn test_no_providers_fails_validation() {
        let toml_str = r#"
            mode = "serve"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("at least one [[providers]]"));
    }

    #[test]
    fn test_duplicate_provider_name_fails_validation() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "dup"
            model = "openai:gpt-4o"
            api_key_env = "A"
            [[providers]]
            name = "dup"
            model = "openai:gpt-4o"
            api_key_env = "B"
            [agent]
            id = "x"
            provider = "dup"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("duplicate provider name 'dup'"));
    }

    #[test]
    fn test_agent_provider_unknown_fails_validation() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            api_key_env = "K"
            [agent]
            id = "x"
            provider = "nope"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("agent.provider 'nope' does not match"),
            "got: {err}",
        );
    }

    #[test]
    fn test_agent_no_model_anywhere_fails_validation() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            api_key_env = "K"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("agent has no model"),
            "got: {err}"
        );
    }

    #[test]
    fn test_agent_model_falls_back_to_provider_model() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert!(config.agent.model.is_none());
        assert_eq!(config.agent_model(), Some("openai:gpt-4o"));
    }

    #[test]
    fn test_agent_model_overrides_provider_model() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            api_key_env = "K"
            [agent]
            id = "x"
            provider = "default"
            model = "openai:gpt-4o-mini"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(config.agent_model(), Some("openai:gpt-4o-mini"));
    }

    #[test]
    fn test_provider_base_url_round_trips_from_toml() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "local"
            model = "chat_completions:qwen"
            base_url = "http://llama-server.windowlickers.svc.cluster.local:8080/v1"
            api_key_env = "OPENAI_API_KEY"
            [agent]
            id = "manager"
            provider = "local"
            system_prompt = "be helpful"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(
            config.provider("local").unwrap().base_url.as_deref(),
            Some("http://llama-server.windowlickers.svc.cluster.local:8080/v1"),
        );
    }

    #[test]
    fn test_provider_base_url_rejects_non_http_scheme() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            base_url = "file:///etc/passwd"
            api_key_env = "K"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("http or https"));
    }

    #[test]
    fn test_provider_proxy_parses_with_required_fields() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            [providers.proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/neuromance/tokens/llm"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        let proxy = config.provider("default").unwrap().proxy.as_ref().unwrap();
        assert_eq!(proxy.base_url, "http://tokenizer-proxy.svc:8080");
        assert_eq!(
            proxy.token_file,
            PathBuf::from("/var/run/neuromance/tokens/llm")
        );
        // Default applied.
        assert_eq!(proxy.token_header, "X-Tokenizer-Token");
        assert!(config.provider("default").unwrap().api_key_env.is_none());
    }

    #[test]
    fn test_provider_proxy_round_trips_custom_token_header() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            [providers.proxy]
            base_url = "https://tokenizer-proxy.example.com"
            token_file = "/var/run/tokens/llm"
            token_header = "X-Token"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        let proxy = config.provider("default").unwrap().proxy.as_ref().unwrap();
        assert_eq!(proxy.token_header, "X-Token");
    }

    #[test]
    fn test_provider_proxy_and_api_key_env_are_mutually_exclusive() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            api_key_env = "K"
            [providers.proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/tokens/llm"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("mutually exclusive"),
            "expected exclusivity error, got: {err}",
        );
    }

    #[test]
    fn test_provider_without_credential_fails() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("must set either api_key_env or proxy"),
            "expected missing-credential error, got: {err}",
        );
    }

    #[test]
    fn test_two_providers_with_distinct_credential_paths_coexist() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "keyed"
            model = "openai:gpt-4o"
            api_key_env = "OPENAI_API_KEY"
            [[providers]]
            name = "proxied"
            model = "openai:gpt-4o"
            [providers.proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/tokens/llm"
            [agent]
            id = "x"
            provider = "keyed"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert!(config.provider("keyed").unwrap().api_key_env.is_some());
        assert!(config.provider("proxied").unwrap().proxy.is_some());
    }

    #[test]
    fn test_provider_proxy_token_file_must_be_absolute() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            [providers.proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "relative/path/token"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("absolute path"),
            "expected absolute-path error, got: {err}",
        );
    }

    #[test]
    fn test_provider_proxy_base_url_rejects_non_http_scheme() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            [providers.proxy]
            base_url = "file:///etc/passwd"
            token_file = "/var/run/tokens/llm"
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("http or https"));
    }

    #[test]
    fn test_provider_proxy_token_header_must_not_be_empty() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "default"
            model = "openai:gpt-4o"
            [providers.proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/tokens/llm"
            token_header = "   "
            [agent]
            id = "x"
            provider = "default"
            system_prompt = "x"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("token_header"));
    }

    fn serve_toml_with_context(context_section: &str) -> String {
        format!("{SERVE_PREAMBLE}{context_section}")
    }

    #[test]
    fn test_context_absent_is_none() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert!(config.context.is_none());
    }

    #[test]
    fn test_context_section_parses_with_defaults() {
        let toml_str = serve_toml_with_context(
            r"
            [context]
            context_window_size = 128000
        ",
        );
        let config: RuntimeConfig = toml::from_str(&toml_str).unwrap();
        config.validate().unwrap();
        let context = config.context.expect("context section");
        assert_eq!(context.context_window_size, 128_000);
        assert!((context.compaction_threshold_ratio - 0.8).abs() < f64::EPSILON);
        assert!((context.target_ratio - 0.5).abs() < f64::EPSILON);
        assert_eq!(context.preserve_recent_turns, 3);
        assert_eq!(context.strategy, CompactionStrategy::OneShot);
    }

    /// Every `[context]` setting reaches the `ContextConfig` the main agent and
    /// every subagent run compact against. A field dropped here is a knob that
    /// silently does nothing.
    #[test]
    fn test_to_context_config_carries_every_setting() {
        let toml_str = serve_toml_with_context(
            r#"
            [context]
            context_window_size = 64000
            compaction_threshold_ratio = 0.7
            target_ratio = 0.4
            preserve_recent_turns = 5
            strategy = "hierarchical"
        "#,
        );
        let config: RuntimeConfig = toml::from_str(&toml_str).unwrap();
        config.validate().unwrap();

        let context = config.context.expect("context section").to_context_config();

        assert_eq!(context.context_window_size, 64_000);
        assert!((context.compaction_threshold_ratio - 0.7).abs() < f64::EPSILON);
        assert!((context.target_ratio - 0.4).abs() < f64::EPSILON);
        assert_eq!(context.preserve_recent_turns, 5);
        assert_eq!(context.strategy, CompactionStrategy::Hierarchical);
    }

    #[test]
    fn test_context_strategy_parses_snake_case() {
        let toml_str = serve_toml_with_context(
            r#"
            [context]
            context_window_size = 128000
            strategy = "truncate"
        "#,
        );
        let config: RuntimeConfig = toml::from_str(&toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(
            config.context.expect("context").strategy,
            CompactionStrategy::Truncate
        );
    }

    #[test]
    fn test_context_strategy_rejects_pascal_case() {
        let toml_str = serve_toml_with_context(
            r#"
            [context]
            context_window_size = 128000
            strategy = "OneShot"
        "#,
        );
        assert!(toml::from_str::<RuntimeConfig>(&toml_str).is_err());
    }

    #[test]
    fn test_context_zero_window_fails_validation() {
        let toml_str = serve_toml_with_context(
            r"
            [context]
            context_window_size = 0
        ",
        );
        let config: RuntimeConfig = toml::from_str(&toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("context_window_size"));
    }

    #[test]
    fn test_context_threshold_ratio_above_one_fails_validation() {
        let toml_str = serve_toml_with_context(
            r"
            [context]
            context_window_size = 128000
            compaction_threshold_ratio = 1.5
        ",
        );
        let config: RuntimeConfig = toml::from_str(&toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("compaction_threshold_ratio"));
    }

    #[test]
    fn test_context_target_at_or_above_threshold_fails_validation() {
        let toml_str = serve_toml_with_context(
            r"
            [context]
            context_window_size = 128000
            compaction_threshold_ratio = 0.8
            target_ratio = 0.9
        ",
        );
        let config: RuntimeConfig = toml::from_str(&toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("must be below"),
            "expected ordering error, got: {err}",
        );
    }

    #[test]
    fn test_context_zero_preserve_recent_turns_fails_validation() {
        let toml_str = serve_toml_with_context(
            r"
            [context]
            context_window_size = 128000
            preserve_recent_turns = 0
        ",
        );
        let config: RuntimeConfig = toml::from_str(&toml_str).unwrap();
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("preserve_recent_turns"));
    }

    #[test]
    fn test_database_section_absent_means_none() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert!(config.database.is_none());
    }

    #[test]
    fn test_database_section_applies_defaults() {
        let config = serve_config(
            r#"
            [database]
            url_env = "DATABASE_URL"
        "#,
        );
        config.validate().unwrap();
        let database = config.database.expect("database section");
        assert_eq!(database.url_env, "DATABASE_URL");
        assert_eq!(database.max_connections, 5);
        assert_eq!(database.acquire_timeout_seconds, 5);
        assert!(database.run_migrations);
    }

    #[test]
    fn test_database_run_migrations_can_be_disabled() {
        let config = serve_config(
            r#"
            [database]
            url_env = "DATABASE_URL"
            run_migrations = false
        "#,
        );
        config.validate().unwrap();

        // Round-trip through Serialize to confirm the flag survives.
        let reserialized = toml::to_string(&config).unwrap();
        let config: RuntimeConfig = toml::from_str(&reserialized).unwrap();

        let database = config.database.expect("database section");
        assert!(!database.run_migrations);
    }

    #[test]
    fn test_database_section_round_trips_custom_values() {
        let config = serve_config(
            r#"
            [database]
            url_env = "PG_URL"
            max_connections = 12
            acquire_timeout_seconds = 30
        "#,
        );
        config.validate().unwrap();

        // Serialize back out and reparse to exercise the Serialize impl.
        let reserialized = toml::to_string(&config).unwrap();
        let config: RuntimeConfig = toml::from_str(&reserialized).unwrap();
        config.validate().unwrap();

        let database = config.database.expect("database section");
        assert_eq!(database.url_env, "PG_URL");
        assert_eq!(database.max_connections, 12);
        assert_eq!(database.acquire_timeout_seconds, 30);
    }

    #[test]
    fn test_rules_section_round_trips_custom_values() {
        let config = serve_config(
            r#"
            [rules]
            roots = ["/etc/rules", "/home/me/rules"]
            endpoint = "https://corpus/api/v1/rules"
            endpoint_token_env = "RULES_TOKEN"
            body_budget_bytes = 4096
        "#,
        );
        config.validate().unwrap();

        // Serialize back out and reparse to exercise the Serialize impl.
        let reserialized = toml::to_string(&config).unwrap();
        let config: RuntimeConfig = toml::from_str(&reserialized).unwrap();
        config.validate().unwrap();

        let rules = config.rules.expect("rules section");
        assert_eq!(
            rules.roots,
            vec![PathBuf::from("/etc/rules"), PathBuf::from("/home/me/rules")]
        );
        assert_eq!(
            rules.endpoint.as_deref(),
            Some("https://corpus/api/v1/rules")
        );
        assert_eq!(rules.endpoint_token_env.as_deref(), Some("RULES_TOKEN"));
        assert_eq!(rules.body_budget_bytes, 4096);
    }

    #[test]
    fn test_database_zero_acquire_timeout_fails_validation() {
        let config = serve_config(
            r#"
            [database]
            url_env = "DATABASE_URL"
            acquire_timeout_seconds = 0
        "#,
        );
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("database.acquire_timeout_seconds"));
    }

    #[test]
    fn test_database_empty_url_env_fails_validation() {
        let config = serve_config(
            r#"
            [database]
            url_env = "  "
        "#,
        );
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("database.url_env"));
    }

    #[test]
    fn test_database_zero_max_connections_fails_validation() {
        let config = serve_config(
            r#"
            [database]
            url_env = "DATABASE_URL"
            max_connections = 0
        "#,
        );
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("database.max_connections"));
    }

    #[test]
    fn test_subagents_absent_means_empty() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert!(config.subagents.is_empty());
    }

    #[test]
    fn test_subagents_section_parses_with_defaults() {
        let config = serve_config(
            r#"
            [[subagents]]
            id = "researcher"
            system_prompt = "You research."

            [[subagents]]
            id = "critic"
            system_prompt = "You critique."
            model = "anthropic:claude-opus-4-8"
            description = "Critique a draft."
            max_turns = 4
        "#,
        );
        config.validate().unwrap();
        assert_eq!(config.subagents.len(), 2);
        assert_eq!(config.subagents[0].id, "researcher");
        assert!(config.subagents[0].provider.is_none());
        assert!(config.subagents[0].model.is_none());
        assert!(config.subagents[0].description.is_none());
        assert_eq!(
            config.subagents[1].model.as_deref(),
            Some("anthropic:claude-opus-4-8")
        );
        assert_eq!(config.subagents[1].max_turns, Some(4));
    }

    #[test]
    fn test_duplicate_subagent_id_fails_validation() {
        let config = serve_config(
            r#"
            [[subagents]]
            id = "worker"
            system_prompt = "a"

            [[subagents]]
            id = "worker"
            system_prompt = "b"
        "#,
        );
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("duplicate subagent id 'worker'"));
    }

    #[test]
    fn test_subagent_empty_system_prompt_fails_validation() {
        let config = serve_config(
            r#"
            [[subagents]]
            id = "worker"
            system_prompt = "   "
        "#,
        );
        let err = config.validate().err().unwrap();
        assert!(format!("{err}").contains("system_prompt must not be empty"));
    }

    #[test]
    fn test_subagent_provider_must_reference_existing_provider() {
        let config = serve_config(
            r#"
            [[subagents]]
            id = "worker"
            system_prompt = "a"
            provider = "ghost"
        "#,
        );
        let err = config.validate().err().unwrap();
        assert!(
            format!("{err}").contains("provider 'ghost' does not match"),
            "got: {err}",
        );
    }

    #[test]
    fn test_subagent_can_reference_a_second_provider() {
        let toml_str = r#"
            mode = "serve"
            [[providers]]
            name = "primary"
            model = "openai:gpt-4o"
            api_key_env = "A"
            [[providers]]
            name = "secondary"
            model = "anthropic:claude-opus-4-8"
            api_key_env = "B"
            [agent]
            id = "manager"
            provider = "primary"
            system_prompt = "be helpful"
            [[subagents]]
            id = "worker"
            system_prompt = "a"
            provider = "secondary"
        "#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        config.validate().unwrap();
        assert_eq!(config.subagents[0].provider.as_deref(), Some("secondary"));
    }

    #[test]
    fn test_tools_section_parses() {
        let config = serve_config(
            r#"
            [[tools]]
            name = "read"

            [[tools]]
            name = "bash"
        "#,
        );
        config.validate().unwrap();
        assert_eq!(config.tools.len(), 2);
        assert_eq!(config.tools[0].name, "read");
        assert_eq!(config.tools[1].name, "bash");
    }

    #[test]
    fn test_bootstrap_defaults_to_empty() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert!(config.bootstrap.is_empty());
    }

    #[test]
    fn test_bootstrap_entry_parses_and_validates() {
        let config = serve_config(
            r#"
            [[bootstrap]]
            name = "forgejo"
            command = "fj"
            args = ["--host", "git.example", "auth", "add-tokenizer"]
            token_env = "FORGEJO_TOKEN"
        "#,
        );
        config.validate().unwrap();
        assert_eq!(config.bootstrap.len(), 1);
        let entry = &config.bootstrap[0];
        assert_eq!(entry.command, "fj");
        assert_eq!(entry.args.len(), 4);
        assert_eq!(entry.token_env.as_deref(), Some("FORGEJO_TOKEN"));
    }

    #[test]
    fn test_bootstrap_token_env_is_optional() {
        let config = serve_config(
            r#"
            [[bootstrap]]
            name = "noop"
            command = "true"
        "#,
        );
        config.validate().unwrap();
        let entry = &config.bootstrap[0];
        assert!(entry.token_env.is_none());
        assert!(entry.args.is_empty());
    }

    #[test]
    fn test_bootstrap_empty_command_fails_validation() {
        let config = serve_config(
            r#"
            [[bootstrap]]
            name = "forgejo"
            command = ""
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(matches!(err, RuntimeError::Config(_)));
    }

    #[test]
    fn test_sandbox_defaults_listen_addr_to_loopback() {
        let config = serve_config(
            r"
            [sandbox]
        ",
        );
        config.validate().unwrap();
        let sandbox = config.sandbox.expect("sandbox section present");
        assert_eq!(sandbox.listen_addr, "127.0.0.1:50051");
        assert!(sandbox.endpoint.is_none());
    }

    #[test]
    fn test_sandbox_non_loopback_listen_addr_rejected() {
        let config = serve_config(
            r#"
            [sandbox]
            listen_addr = "0.0.0.0:50051"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(
            matches!(err, RuntimeError::Config(ref m) if m.contains("loopback")),
            "{err}"
        );
    }

    #[test]
    fn test_sandbox_malformed_listen_addr_rejected() {
        let config = serve_config(
            r#"
            [sandbox]
            listen_addr = "not-an-addr"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(matches!(err, RuntimeError::Config(_)));
    }

    #[test]
    fn test_sandbox_endpoint_must_be_http_url() {
        let config = serve_config(
            r#"
            [sandbox]
            endpoint = "127.0.0.1:50051"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(
            matches!(err, RuntimeError::Config(ref m) if m.contains("sandbox.endpoint")),
            "{err}"
        );
    }

    #[test]
    fn test_sandbox_endpoint_rejects_python_subagent_bridge() {
        let config = serve_config(
            r#"
            [[tools]]
            name = "execute_python"

            [[subagents]]
            id = "worker"
            system_prompt = "you are a worker"

            [sandbox]
            endpoint = "http://127.0.0.1:50051"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(
            matches!(err, RuntimeError::Config(ref m) if m.contains("not yet supported")),
            "{err}"
        );
    }

    #[test]
    fn test_sandbox_endpoint_allows_subagents_without_python() {
        let config = serve_config(
            r#"
            [[tools]]
            name = "bash"

            [[subagents]]
            id = "worker"
            system_prompt = "you are a worker"

            [sandbox]
            endpoint = "http://127.0.0.1:50051"
        "#,
        );
        config.validate().unwrap();
    }

    #[test]
    fn test_self_delegation_defaults_off() {
        let config = serve_config("");
        assert!(!config.runtime.self_delegation);
        assert!(!config.has_delegation());
        assert!(config.self_delegation_subagent().is_none());
        // Off means the effective config is the config unchanged.
        assert!(config.with_self_delegation().subagents.is_empty());
    }

    #[test]
    fn test_self_delegation_synthesizes_self_subagent() {
        let config = serve_config(
            r"
            [runtime]
            self_delegation = true
        ",
        );
        config.validate().unwrap();
        assert!(config.has_delegation());

        let sub = config
            .self_delegation_subagent()
            .expect("synthesized when self_delegation is on");
        assert_eq!(sub.id, SELF_DELEGATION_ID);
        assert_eq!(sub.system_prompt, config.agent.system_prompt);
        // Provider/model are inherited through resolution, not copied by name.
        assert!(sub.provider.is_none());
        assert!(sub.model.is_none());
        assert_eq!(sub.max_turns, config.agent.max_turns);
        assert!(sub.description.is_some());

        // The effective config injects exactly the synthesized subagent so the
        // tower treats it like a configured `[[subagents]]` entry.
        let effective = config.with_self_delegation();
        assert_eq!(effective.subagents.len(), 1);
        assert_eq!(effective.subagents[0].id, SELF_DELEGATION_ID);
    }

    /// The synthesized clone runs on `agent.system_prompt`, so an empty one is
    /// rejected exactly as the equivalent explicit `[[subagents]]` entry is.
    /// Otherwise `self_delegation` boots an uninstructed agent holding the
    /// parent's whole auto-approved toolset.
    #[test]
    fn test_self_delegation_with_empty_agent_prompt_fails() {
        let mut config = serve_config(
            r"
            [runtime]
            self_delegation = true
        ",
        );
        config.agent.system_prompt = "   ".to_string();

        let err = config
            .validate()
            .expect_err("an uninstructed self-clone must not boot");

        let msg = format!("{err}");
        assert!(msg.contains("agent.system_prompt"), "{msg}");
        assert!(msg.contains("self_delegation"), "{msg}");
    }

    /// The same empty prompt is fine without self-delegation: nothing
    /// synthesizes a subagent from it.
    #[test]
    fn test_empty_agent_prompt_is_allowed_without_self_delegation() {
        let mut config = serve_config("");
        config.agent.system_prompt = String::new();

        config.validate().unwrap();
    }

    #[test]
    fn test_self_delegation_with_zero_depth_fails() {
        let config = serve_config(
            r"
            [runtime]
            self_delegation = true
            max_delegation_depth = 0
        ",
        );
        let err = config
            .validate()
            .expect_err("depth 0 is invalid once self_delegation enables the tower");
        assert!(format!("{err}").contains("max_delegation_depth"));
    }

    #[test]
    fn test_self_delegation_reserved_id_collision_fails() {
        let config = serve_config(
            r#"
            [runtime]
            self_delegation = true

            [[subagents]]
            id = "task"
            system_prompt = "you are a worker"
        "#,
        );
        let err = config
            .validate()
            .expect_err("an explicit subagent named `task` collides with self_delegation");
        assert!(
            matches!(err, RuntimeError::Config(ref m) if m.contains("reserved")),
            "{err}"
        );
    }

    #[test]
    fn test_sandbox_endpoint_rejects_self_delegation_python_bridge() {
        let config = serve_config(
            r#"
            [runtime]
            self_delegation = true

            [[tools]]
            name = "execute_python"

            [sandbox]
            endpoint = "http://127.0.0.1:50051"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(
            matches!(err, RuntimeError::Config(ref m) if m.contains("not yet supported")),
            "{err}"
        );
    }

    /// A full `[workspace]` section: storage, persistence, git proxy, and one
    /// definition with both seed kinds.
    const FULL_WORKSPACE: &str = r#"
            [workspace]
            root = "/workspace"

            [workspace.storage]
            endpoint = "http://garage.storage.svc:3900"
            bucket = "workspaces"
            access_key_id_env = "GARAGE_KEY_ID"
            secret_access_key_env = "GARAGE_SECRET"

            [workspace.persistence]
            exclude = ["target/**", "node_modules/**"]

            [workspace.git_proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/tokens/git"

            [[workspace.definitions]]
            name = "org-dev"

            [[workspace.definitions.git]]
            url = "http://git.example/org/repo.git"
            ref = "main"
            dest = "repo"

            [[workspace.definitions.objects]]
            key = "seeds/scratch.tar.zst"
            dest = "scratch"
    "#;

    #[test]
    fn test_workspace_absent_is_none() {
        let config: RuntimeConfig = toml::from_str(minimal_oneshot_toml()).unwrap();
        config.validate().unwrap();
        assert!(config.workspace.is_none());
    }

    #[test]
    fn test_workspace_full_section_round_trips() {
        let config = serve_config(FULL_WORKSPACE);
        config.validate().unwrap();

        // Serialize back out and reparse to exercise the Serialize impl.
        let reserialized = toml::to_string(&config).unwrap();
        let config: RuntimeConfig = toml::from_str(&reserialized).unwrap();
        config.validate().unwrap();

        let workspace = config.workspace.expect("workspace section");
        assert_eq!(workspace.root, PathBuf::from("/workspace"));
        let storage = workspace.storage.expect("storage");
        assert_eq!(storage.region, "garage"); // default applied
        assert_eq!(storage.bucket, "workspaces");
        let persistence = workspace.persistence.expect("persistence");
        assert_eq!(persistence.prefix, "snapshots/"); // default applied
        assert_eq!(persistence.exclude.len(), 2);
        let def = &workspace.definitions[0];
        assert_eq!(def.name, "org-dev");
        assert_eq!(def.git[0].reference.as_deref(), Some("main"));
        assert_eq!(def.git[0].dest.as_deref(), Some("repo"));
        assert_eq!(def.objects[0].key, "seeds/scratch.tar.zst");
    }

    #[test]
    fn test_workspace_minimal_root_only_validates() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
        "#,
        );
        config.validate().unwrap();
    }

    #[test]
    fn test_workspace_relative_root_fails_validation() {
        let config = serve_config(
            r#"
            [workspace]
            root = "relative/workspace"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("workspace.root"), "{err}");
    }

    #[test]
    fn test_workspace_duplicate_definition_name_fails() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [[workspace.definitions]]
            name = "dup"
            [[workspace.definitions]]
            name = "dup"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("duplicate workspace definition name 'dup'"),
            "{err}"
        );
    }

    #[test]
    fn test_workspace_object_seed_without_storage_fails() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [[workspace.definitions]]
            name = "seeded"
            [[workspace.definitions.objects]]
            key = "seeds/scratch.tar.zst"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("[workspace.storage]"), "{err}");
    }

    #[test]
    fn test_workspace_persistence_without_storage_fails() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [workspace.persistence]
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("requires [workspace.storage]"),
            "{err}"
        );
    }

    #[test]
    fn test_workspace_invalid_exclude_glob_fails() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [workspace.storage]
            endpoint = "http://garage.svc:3900"
            bucket = "b"
            access_key_id_env = "K"
            secret_access_key_env = "S"
            [workspace.persistence]
            exclude = ["target/[!"]
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("not a valid glob"), "{err}");
    }

    #[test]
    fn test_workspace_seed_dest_rejects_traversal_and_absolute() {
        for dest in ["../escape", "/absolute"] {
            let config = serve_config(&format!(
                r#"
                [workspace]
                root = "/workspace"
                [[workspace.definitions]]
                name = "seeded"
                [[workspace.definitions.git]]
                url = "http://git.example/org/repo.git"
                dest = "{dest}"
            "#
            ));
            let err = config.validate().unwrap_err();
            assert!(format!("{err}").contains("seed dest"), "{dest}: {err}");
        }
    }

    #[test]
    fn test_workspace_git_proxy_relative_token_file_fails() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [workspace.git_proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "relative/token"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("workspace.git_proxy.token_file"),
            "{err}"
        );
    }

    /// A proxied clone must be http:// so the proxy can read the sealed
    /// header. Reject it at load rather than at first seed.
    #[test]
    fn test_workspace_https_git_seed_with_proxy_fails() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [workspace.git_proxy]
            base_url = "http://tokenizer-proxy.svc:8080"
            token_file = "/var/run/token"
            [[workspace.definitions]]
            name = "dev"
            [[workspace.definitions.git]]
            url = "https://git.windowlicke.rs/darkhorse/kb.git"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("must be http://"), "{err}");
    }

    /// The same seed is fine without a proxy: anonymous clones go through
    /// libgit2, which handles https itself.
    #[test]
    fn test_workspace_https_git_seed_without_proxy_validates() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [[workspace.definitions]]
            name = "dev"
            [[workspace.definitions.git]]
            url = "https://git.windowlicke.rs/darkhorse/kb.git"
        "#,
        );
        config.validate().unwrap();
    }

    #[test]
    fn test_workspace_storage_empty_bucket_fails() {
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [workspace.storage]
            endpoint = "http://garage.svc:3900"
            bucket = "  "
            access_key_id_env = "K"
            secret_access_key_env = "S"
        "#,
        );
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("workspace.storage.bucket"),
            "{err}"
        );
    }

    #[test]
    fn test_sole_workspace_definition_only_with_exactly_one() {
        // Exactly one definition: it is the default seed.
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [[workspace.definitions]]
            name = "only"
        "#,
        );
        config.validate().unwrap();
        assert_eq!(config.sole_workspace_definition().unwrap().name, "only");

        // Several definitions: no unambiguous default.
        let config = serve_config(
            r#"
            [workspace]
            root = "/workspace"
            [[workspace.definitions]]
            name = "a"
            [[workspace.definitions]]
            name = "b"
        "#,
        );
        config.validate().unwrap();
        assert!(config.sole_workspace_definition().is_none());
        // Lookup by name still resolves each, and rejects the unknown.
        assert!(config.workspace_definition("b").is_some());
        assert!(config.workspace_definition("ghost").is_none());
    }

    #[test]
    fn test_sole_workspace_definition_none_without_section() {
        let config = serve_config("");
        assert!(config.sole_workspace_definition().is_none());
        assert!(config.workspace_definition("any").is_none());
    }
}
