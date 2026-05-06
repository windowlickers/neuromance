//! # neuromance-runtime
//!
//! Container runtime that boots a [`neuromance_agent::Agent`] from a TOML
//! config and runs it in one of two modes:
//!
//! - **`oneshot`** — execute a single configured input, write the result, exit.
//!   Designed for Kubernetes Jobs.
//! - **`serve`** — bind an HTTP intake (`POST /tasks`, `GET /tasks/{id}`) and
//!   process tasks sequentially until SIGTERM. Designed for Deployments.
//!
//! Approval is `auto` (every tool call approved) or `async` (each
//! non-auto-approved tool call is `POST`ed to a webhook for an approve/deny
//! decision).
//!
//! Tools are constructed at startup from `[[tools]]` entries via
//! [`neuromance_tools::ToolFactoryRegistry::with_builtin`]. State is in-memory
//! only; persistence (postgres) is future work.

pub mod approval;
pub mod config;
pub mod error;
pub mod health;
pub mod lifecycle;
pub mod oneshot;
pub mod serve;

pub use config::{
    AgentConfig, ApprovalConfig, ApprovalMode, Mode, OneshotConfig, RuntimeConfig, RuntimeSettings,
};
pub use error::RuntimeError;
