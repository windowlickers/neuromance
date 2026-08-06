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
use neuromance_common::client::OutputSchema;

use crate::{RuntimeError, SessionReset};

/// Per-task overrides applied to a freshly built [`Agent`].
///
/// Every field is validated at enqueue time, so `build` can trust that the provider names a
/// configured entry and the model string parses.
#[derive(Debug, Clone, Default)]
pub struct AgentOverrides {
    /// Names a configured `[[providers]]` entry whose credential and endpoint the agent uses;
    /// `None` keeps the configured `agent.provider`.
    pub provider: Option<String>,
    /// A raw `provider:model` string (e.g. `anthropic:claude-opus-4-8`); `None` resolves the
    /// selected provider's default. The selected provider always supplies the credential, so a
    /// model override must name a model that credential covers.
    pub model: Option<String>,
    /// JSON Schema the agent's responses must satisfy; `None` leaves them unconstrained.
    pub output_schema: Option<OutputSchema>,
}

impl AgentOverrides {
    /// Whether any override is set, i.e. whether the task needs its own agent.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.provider.is_none() && self.model.is_none() && self.output_schema.is_none()
    }
}

/// Builds an [`Agent`] on demand, applying per-task [`AgentOverrides`].
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
        overrides: &AgentOverrides,
    ) -> Result<(Agent<Box<dyn LLMClient>>, Option<SessionReset>), RuntimeError>;
}
