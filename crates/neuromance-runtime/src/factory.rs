//! Per-task agent construction.
//!
//! The serve worker normally reuses one long-lived agent, but a task may carry
//! a provider and/or model override (see `serve::CreateTaskRequest`). When it
//! does, the worker builds a throwaway agent bound to that provider/model
//! through an [`AgentBuilder`], runs the single turn, and drops it. The trait
//! lives here so `serve` (this library) can depend on it while the concrete
//! builder — which owns the full set of startup inputs (config, store, sandbox,
//! skills, rules) — is assembled in the binary alongside `build_agent`.

use async_trait::async_trait;
use neuromance_agent::Agent;
use neuromance_client::LLMClient;

use crate::{RuntimeError, SessionReset};

/// Builds an [`Agent`] on demand, optionally overriding the configured provider
/// and/or model.
///
/// `provider_override` names a configured `[[providers]]` entry whose credential
/// and endpoint the agent uses; `None` keeps the configured `agent.provider`.
/// `model_override` is a raw `provider:model` string (e.g.
/// `anthropic:claude-opus-4-8`); `None` resolves the selected provider's default
/// (then the agent's effective model). The selected provider always supplies the
/// credential — so a model override must name a model that credential covers,
/// unless paired with a provider override that does.
#[async_trait]
pub trait AgentBuilder: Send + Sync {
    /// Construct a fresh agent, returning it alongside the in-process Python
    /// interpreter reset handle (`None` when no local interpreter exists).
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when the provider names no configured entry, the
    /// model string is malformed, a provider credential is missing, or the
    /// toolset cannot be assembled.
    async fn build(
        &self,
        provider_override: Option<&str>,
        model_override: Option<&str>,
    ) -> Result<(Agent<Box<dyn LLMClient>>, Option<SessionReset>), RuntimeError>;
}
