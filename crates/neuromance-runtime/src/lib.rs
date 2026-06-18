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
//! Tools are constructed at startup from `[[tools]]` entries via
//! [`neuromance_tools::ToolFactoryRegistry::with_builtin`]. Serving state is
//! in-memory; an optional `[database]` section additionally writes
//! conversation history through to postgres as tasks run (see
//! `neuromance-db`).

pub mod approval;
pub mod bootstrap;
pub mod config;
pub mod error;
pub mod health;
pub mod lifecycle;
pub mod metrics;
pub mod oneshot;
pub mod proxy;
pub mod sandbox;
pub mod serve;
pub mod subagents;
pub mod telemetry;

pub use config::{
    AgentConfig, ApprovalConfig, ApprovalMode, DatabaseSettings, Mode, OneshotConfig,
    ProviderConfig, ProviderProxyConfig, RuntimeConfig, RuntimeSettings, SandboxConfig,
    SubagentConfig,
};
pub use error::RuntimeError;
pub use subagents::{SessionReset, build_parent_toolset};
