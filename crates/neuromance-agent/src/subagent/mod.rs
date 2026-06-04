//! Subagents: composable units of delegated work.
//!
//! A [`Subagent`] turns a [`Task`] into an [`Outcome`]. The trait is object-safe
//! so subagents can be held as `Arc<dyn Subagent>` and composed. Combinators such
//! as [`FanoutVote`] are themselves subagents, which lets them nest and lets
//! [`SubagentTool`] expose any subagent — leaf or combinator — uniformly.
//!
//! [`LocalSubagent`] runs an in-process [`Agent`](crate::Agent). A remote variant
//! that calls out to another runtime is future work; it will be another
//! `impl Subagent` over the same [`Task`]/[`Outcome`] contract.

use async_trait::async_trait;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use neuromance::error::CoreError;
use neuromance_common::task::{Outcome, Task};

mod combinators;
mod local;
mod tool;

pub use combinators::FanoutVote;
pub use local::LocalSubagent;
pub use tool::SubagentTool;

/// Errors produced while running a subagent.
#[derive(Debug, Error)]
pub enum SubagentError {
    /// The underlying agent execution failed.
    ///
    /// Boxed because [`CoreError`] is large relative to the other variants.
    #[error(transparent)]
    Core(Box<CoreError>),

    /// A fanout completed but no member produced a successful outcome.
    #[error("fanout produced no successful member outcomes")]
    NoOutcomes,
}

impl From<CoreError> for SubagentError {
    fn from(err: CoreError) -> Self {
        Self::Core(Box::new(err))
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
