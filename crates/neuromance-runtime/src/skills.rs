//! Runtime assembly of the skill catalog and the messages it injects.
//!
//! [`build`] turns the `[skills]` config section into a [`SkillRuntime`]: a
//! built [`SkillCatalog`] plus the resolved budgets and invocation modes. The
//! oneshot and serve paths use it to inject the menu into a conversation seed
//! and to expand `$mention`s in user input, while `build_agent` registers the
//! `load_skill` tool from the same catalog.

use std::sync::Arc;

use tracing::{info, warn};
use uuid::Uuid;

use neuromance_common::chat::Message;
use neuromance_context::skills::{HttpSkillSource, LocalSkillSource, SkillCatalog, SkillSource};

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
    /// The menu message to seed a conversation with, if the catalog is non-empty.
    #[must_use]
    pub fn menu_message(&self, conversation_id: Uuid) -> Option<Message> {
        self.catalog
            .menu(self.menu_budget)
            .map(|menu| Message::system(conversation_id, menu))
    }

    /// The `<skill>` body messages for every skill `$mention`ed in `text`.
    ///
    /// Returns an empty vector when `$mention` invocation is disabled.
    pub async fn mention_messages(&self, conversation_id: Uuid, text: &str) -> Vec<Message> {
        if !self.invocation.mention() {
            return Vec::new();
        }
        self.catalog
            .mention_messages(conversation_id, text, self.body_budget)
            .await
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
