//! Subagents: composable units of delegated work.
//!
//! A [`Subagent`] turns a [`Task`] into an [`Outcome`]. The trait is object-safe
//! so subagents can be held as `Arc<dyn Subagent>` and composed. Combinators such
//! as [`FanoutVote`] are themselves subagents, which lets them nest and lets
//! [`SubagentTool`] expose any subagent — leaf or combinator — uniformly.
//!
//! [`LocalSubagent`] runs an in-process [`Agent`](crate::Agent). A remote variant
//! that calls out to another runtime is future work; it will be another
//! `impl Subagent` over the same [`Task`](neuromance_common::task::Task)/
//! [`Outcome`](neuromance_common::task::Outcome) contract.
//!
//! The [`Subagent`] trait and [`SubagentError`] live in `neuromance-common` so
//! crates below the agent layer (e.g. the Python REPL bridge) can depend on the
//! contract without depending on this crate. They are re-exported here for
//! consumers that already use `neuromance-agent`.

pub use neuromance_common::subagent::{Subagent, SubagentError};

mod combinators;
mod local;
mod tool;

pub use combinators::FanoutVote;
pub use local::LocalSubagent;
pub use tool::SubagentTool;
