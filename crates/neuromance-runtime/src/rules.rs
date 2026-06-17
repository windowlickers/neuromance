//! Runtime assembly of the rule catalog into a [`RulesHook`].
//!
//! [`build`] turns the `[rules]` config section into a [`RulesHook`] that
//! `build_agent` registers on the agent's `Core`. Unlike skills, rules need no
//! per-call-site wiring: the hook injects `always_apply` rules at conversation
//! start and glob-matched rules after a tool touches a matching path, all inside
//! the conversation loop.

use std::sync::Arc;

use tracing::{info, warn};

use neuromance_context::rules::{
    HttpRuleSource, LocalRuleSource, RuleCatalog, RuleSource, RulesHook,
};

use crate::config::RulesSettings;

/// Build a [`RulesHook`] from the `[rules]` config section.
///
/// Returns `None` when rules are not configured or no source is usable. On-host
/// roots are registered ahead of the remote endpoint, giving local rules
/// precedence.
pub async fn build(settings: Option<&RulesSettings>) -> Option<Arc<RulesHook>> {
    let settings = settings?;

    let mut sources: Vec<Box<dyn RuleSource>> = Vec::new();
    if !settings.roots.is_empty() {
        sources.push(Box::new(LocalRuleSource::new(settings.roots.clone())));
    }
    if let Some(endpoint) = &settings.endpoint {
        let mut http = HttpRuleSource::new(endpoint.clone());
        if let Some(var) = &settings.endpoint_token_env {
            if let Ok(token) = std::env::var(var) {
                http = http.with_auth_header("Authorization", format!("Bearer {token}"));
            } else {
                warn!(
                    "rules.endpoint_token_env `{var}` is unset; querying `{endpoint}` \
                     unauthenticated"
                );
            }
        }
        sources.push(Box::new(http));
    }

    if sources.is_empty() {
        warn!("[rules] is configured with no roots or endpoint; rules disabled");
        return None;
    }

    let catalog = RuleCatalog::build(sources).await;
    info!(rules = catalog.len(), "rule catalog built");
    Some(Arc::new(RulesHook::new(
        Arc::new(catalog),
        settings.body_budget_bytes,
    )))
}
