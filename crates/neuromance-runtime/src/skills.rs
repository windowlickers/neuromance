//! Runtime assembly of the skill catalog: a menu folded into the system
//! prompt, the `load_skill` tool, and a `$mention`-only [`SkillsHook`].
//!
//! [`build`] discovers skills from the configured sources. The menu tells the
//! model to call `load_skill`, which fetches a skill's body from its source on
//! demand. `build_agent` folds the menu into the system prompt, registers
//! `load_skill` on every agent in the tower, and registers the hook for
//! `$mention` expansion only.

use std::sync::Arc;

use tracing::{info, warn};

use neuromance_context::skills::{
    HttpSkillSource, LocalSkillSource, SkillCatalog, SkillSource, SkillsHook,
};
use neuromance_tools::{SkillTool, ToolImplementation};

use crate::config::SkillsSettings;

/// A skill catalog plus the budgets and mention mode.
pub struct SkillRuntime {
    /// The aggregated catalog over every configured source.
    pub catalog: Arc<SkillCatalog>,
    /// Byte budget for the menu folded into the system prompt.
    pub menu_budget: usize,
    /// Byte budget for each skill body, loaded by `load_skill` or a `$mention`.
    pub body_budget: usize,
    /// Whether `$mention`s in user input expand a skill body inline.
    pub mention: bool,
}

/// Append a skills menu to a system prompt, or return the prompt unchanged when
/// there is no menu.
///
/// The separator lives here alone. The self-delegation clone's prompt is built
/// from `agent.system_prompt` down a different path than the main agent's seed,
/// yet the two must come out byte-identical, so every caller — `oneshot::run`,
/// `serve::seed_new_conversation`, and `subagents::tower_config` — joins through
/// this one function.
#[must_use]
pub fn fold_menu(prompt: &str, menu: Option<&str>) -> String {
    menu.map_or_else(|| prompt.to_string(), |menu| format!("{prompt}\n\n{menu}"))
}

impl SkillRuntime {
    /// The menu to fold into the system prompt, if non-empty.
    #[must_use]
    pub fn menu(&self) -> Option<String> {
        self.catalog.menu(self.menu_budget)
    }

    /// Build the `load_skill` tool over this catalog. It returns harness-held
    /// text only, so it is auto-approved.
    #[must_use]
    pub fn tool(&self) -> Arc<dyn ToolImplementation> {
        Arc::new(SkillTool::new(Arc::clone(&self.catalog), self.body_budget))
    }

    /// Build the mention-only [`SkillsHook`], shared by the main agent and every
    /// subagent (its state is keyed by conversation).
    ///
    /// The hook never injects the menu (`inject_menu = false`). Callers fold
    /// [`menu`](Self::menu) into the system prompt instead: the host does it for
    /// the main agent's seed, and `subagents::build_parent_toolset` does it for
    /// each subagent's configured prompt. The menu then persists with the seed
    /// message, and the self-delegation clone's prompt matches the parent's
    /// byte for byte.
    #[must_use]
    pub fn hook(&self) -> Arc<SkillsHook> {
        Arc::new(SkillsHook::new(
            Arc::clone(&self.catalog),
            self.menu_budget,
            self.body_budget,
            self.mention,
            false,
        ))
    }
}

/// Build a [`SkillRuntime`] from the `[skills]` config section.
///
/// Returns `None` when skills are not configured, no source is usable, or the
/// sources list no skills — so the rest of the runtime can treat skills as
/// entirely optional. On-host roots take precedence over the remote
/// endpoint when a skill name appears in both.
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
    if catalog.is_empty() {
        warn!("[skills] sources list no skills; skills disabled");
        return None;
    }
    info!(skills = catalog.len(), "skill catalog built");

    Some(Arc::new(SkillRuntime {
        catalog: Arc::new(catalog),
        menu_budget: settings.menu_budget_bytes,
        body_budget: settings.body_budget_bytes,
        mention: settings.mention,
    }))
}
