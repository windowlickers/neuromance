//! The subagent contract: a composable unit of delegated work.
//!
//! A [`Subagent`] turns a [`Task`] into an [`Outcome`]. The trait is object-safe
//! so subagents can be held as `Arc<dyn Subagent>` and composed. Concrete
//! implementations (in-process agents, remote runtimes, combinators) live in
//! higher-level crates; this crate owns only the abstract contract so any layer
//! can depend on it without pulling in an execution engine.

use async_trait::async_trait;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::task::{Outcome, Task};

/// Errors produced while running a [`Subagent`].
///
/// Deliberately free of any execution-engine type: implementations wrap their
/// own failures via [`SubagentError::execution`], keeping this crate independent
/// of the crates that run agents.
#[derive(Debug, Error)]
pub enum SubagentError {
    /// The underlying execution failed.
    #[error(transparent)]
    Execution(Box<dyn std::error::Error + Send + Sync>),

    /// A fanout completed but no member produced a successful outcome.
    #[error("fanout produced no successful member outcomes")]
    NoOutcomes,
}

impl SubagentError {
    /// Wrap an implementation-specific error as an execution failure.
    pub fn execution(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::Execution(error.into())
    }
}

/// A composable unit of delegated work: runs a [`Task`], yields an [`Outcome`].
#[async_trait]
pub trait Subagent: Send + Sync {
    /// Stable identifier for this subagent, used in logs and diagnostics.
    fn id(&self) -> &str;

    /// Run `task` to completion, honoring `cancel`.
    ///
    /// # Errors
    /// Returns [`SubagentError`] if the underlying execution fails or, for
    /// combinators, if no member produced an outcome.
    async fn run(&self, task: Task, cancel: CancellationToken) -> Result<Outcome, SubagentError>;
}
