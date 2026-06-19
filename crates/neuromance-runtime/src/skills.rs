//! Runtime assembly of the skill catalog into a [`SkillsHook`].
//!
//! [`build`] turns the `[skills]` config section into a [`SkillRuntime`]: a
//! built [`SkillCatalog`] plus the resolved budgets and invocation modes.
//! `build_agent` turns it into a [`SkillsHook`] (which injects the menu and
//! expands `$mention`s from inside the conversation loop) and registers the
//! `load_skill` tool from the same catalog.

use std::sync::Arc;

use tracing::{info, warn};

use neuromance_context::skills::{
    HttpSkillSource, LocalSkillSource, SkillCatalog, SkillSource, SkillsHook,
};

use crate::config::{Invocation, SkillsSettings};

/// A built skill catalog plus the budgets and invocation modes it is used with.
pub struct SkillRuntime {
    /// The aggregated, deduplicated catalog of skills.
    pub catalog: Arc<SkillCatalog>,
    /// Byte budget for the injected menu.
    pub menu_budget: usize,
    /// Byte budget for each loaded skill body.
    pub body_budget: usize,
    /// Which invocation mechanisms are enabled.
    pub invocation: Invocation,
}

impl SkillRuntime {
    /// Build the [`SkillsHook`] that injects the menu and expands `$mention`s
    /// from inside the conversation loop.
    #[must_use]
    pub fn hook(&self) -> Arc<SkillsHook> {
        Arc::new(SkillsHook::new(
            Arc::clone(&self.catalog),
            self.menu_budget,
            self.body_budget,
            self.invocation.mention(),
        ))
    }
}

/// Build a [`SkillRuntime`] from the `[skills]` config section.
///
/// Returns `None` when skills are not configured or no source is usable, so the
/// rest of the runtime can treat skills as entirely optional. On-host roots are
/// registered ahead of the remote endpoint, giving local skills precedence.
pub async fn build(settings: Option<&SkillsSettings>) -> Option<Arc<SkillRuntime>> {
    let settings = settings?;

    let mut sources: Vec<Box<dyn SkillSource>> = Vec::new();
    if !settings.roots.is_empty() {
        sources.push(Box::new(LocalSkillSource::new(settings.roots.clone())));
    }
    if let Some(endpoint) = &settings.endpoint {
        let mut http = HttpSkillSource::new(endpoint.clone());
        if let Some(var) = &settings.endpoint_token_env {
            if let Ok(token) = std::env::var(var) {
                http = http.with_auth_header("Authorization", format!("Bearer {token}"));
            } else {
                warn!(
                    "skills.endpoint_token_env `{var}` is unset; querying `{endpoint}` \
                     unauthenticated"
                );
            }
        }
        sources.push(Box::new(http));
    }

    if sources.is_empty() {
        warn!("[skills] is configured with no roots or endpoint; skills disabled");
        return None;
    }

    let catalog = SkillCatalog::build(sources).await;
    info!(skills = catalog.len(), "skill catalog built");
    Some(Arc::new(SkillRuntime {
        catalog: Arc::new(catalog),
        menu_budget: settings.menu_budget_bytes,
        body_budget: settings.body_budget_bytes,
        invocation: settings.invocation,
    }))
}
