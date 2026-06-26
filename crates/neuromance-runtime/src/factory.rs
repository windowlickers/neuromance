//! Per-task agent construction.
//!
//! The serve worker normally reuses one long-lived agent, but a task may carry
//! a model override (see `serve::CreateTaskRequest`). When it does, the worker
//! builds a throwaway agent bound to that model through an [`AgentBuilder`],
//! runs the single turn, and drops it. The trait lives here so `serve` (this
//! library) can depend on it while the concrete builder — which owns the full
//! set of startup inputs (config, store, sandbox, skills, rules) — is assembled
//! in the binary alongside `build_agent`.

use async_trait::async_trait;
use neuromance_agent::Agent;
use neuromance_client::LLMClient;

use crate::{RuntimeError, SessionReset};

/// Builds an [`Agent`] on demand, optionally overriding the configured model.
///
/// `model_override` is a raw `provider:model` string (e.g.
/// `anthropic:claude-opus-4-8`); `None` resolves the runtime's configured
/// model. The provider's credential and endpoint are reused either way — only
/// the model string changes.
#[async_trait]
pub trait AgentBuilder: Send + Sync {
    /// Construct a fresh agent, returning it alongside the in-process Python
    /// interpreter reset handle (`None` when no local interpreter exists).
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when the model string is malformed, a provider
    /// credential is missing, or the toolset cannot be assembled.
    async fn build(
        &self,
        model_override: Option<&str>,
    ) -> Result<(Agent<Box<dyn LLMClient>>, Option<SessionReset>), RuntimeError>;
}
