//! # neuromance-runtime
//!
//! Container runtime that boots a [`neuromance_agent::Agent`] from a TOML
//! config and runs it in one of two modes:
//!
//! - **`oneshot`** — execute a single configured input, write the result, exit.
//!   Designed for Kubernetes Jobs.
//! - **`serve`** — bind an HTTP intake (`POST /tasks/new`, `GET /tasks`,
//!   `GET /tasks/{id}`) and process tasks sequentially until SIGTERM.
//!   Designed for Deployments.
//!
//! Approval is `auto` (every tool call approved) or `async` (each
//! non-auto-approved tool call is `POST`ed to a webhook for an approve/deny
//! decision).
//!
//! Tools come from the `[[mcp_servers]]` entries, connected once at startup,
//! plus the native delegate tools and `load_skill`. Serving state is
//! in-memory; an optional `[database]` section additionally writes
//! conversation history through to postgres as tasks run (see
//! `neuromance-db`).

pub mod approval;
pub mod bootstrap;
pub mod config;
pub mod error;
pub mod factory;
pub mod health;
pub mod lifecycle;
pub mod mcp;
pub mod metrics;
pub mod oneshot;
pub mod proxy;
pub mod rules;
pub mod serve;
pub mod skills;
pub mod subagents;
pub mod task_store;
pub mod telemetry;

pub use config::{
    AgentConfig, ApprovalConfig, ApprovalMode, ContextSettings, DatabaseSettings, Mode,
    OneshotConfig, ProviderConfig, ProviderProxyConfig, RulesSettings, RuntimeConfig,
    RuntimeSettings, SkillsSettings, SubagentConfig,
};
pub use error::RuntimeError;
pub use factory::{AgentBuilder, AgentOverrides};
pub use skills::SkillRuntime;
pub use subagents::{ToolsetParams, build_parent_toolset};
