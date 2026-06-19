//! Runtime assembly of the skill catalog: materialize to disk, then serve a
//! file-oriented menu plus a `$mention`-only [`SkillsHook`].
//!
//! [`build`] discovers skills from the configured sources, **materializes** each
//! one to a local temp dir as a `SKILL.md`, and rebuilds the catalog over that
//! dir. The agent then reads a skill's body from a local file (the menu lists
//! the path) instead of fetching it from the corpus at reasoning time — so the
//! corpus is only a boot-time dependency. `build_agent` folds the menu into the
//! system prompt and registers the hook for `$mention` expansion only.

use std::path::Path;
use std::sync::Arc;

use tempfile::TempDir;
use tracing::{info, warn};

use neuromance_context::skills::{
    HttpSkillSource, LocalSkillSource, SkillCatalog, SkillSource, SkillsHook, synthesize_skill_md,
};

use crate::config::SkillsSettings;

/// The on-disk filename every materialized skill is written to.
const SKILL_FILE: &str = "SKILL.md";

/// A catalog materialized to local disk, plus the budgets and mention mode.
pub struct SkillRuntime {
    /// The aggregated catalog, rebuilt over the materialized temp dir.
    pub catalog: Arc<SkillCatalog>,
    /// Byte budget for the menu folded into the system prompt.
    pub menu_budget: usize,
    /// Byte budget for each `$mention`-expanded skill body.
    pub body_budget: usize,
    /// Whether `$mention`s in user input expand a skill body inline.
    pub mention: bool,
    /// Owns the materialized skills on disk for the runtime's lifetime; the
    /// menu hands the agent literal paths under here, so it must outlive the
    /// agent run. Held only for its `Drop` (cleanup), never read.
    #[allow(dead_code, reason = "retained so the temp dir is not cleaned mid-run")]
    tempdir: TempDir,
}

impl SkillRuntime {
    /// The file-oriented menu to fold into the system prompt, if non-empty.
    #[must_use]
    pub fn menu(&self) -> Option<String> {
        self.catalog.menu_filesystem(self.menu_budget)
    }

    /// Build the mention-only [`SkillsHook`]. The menu is folded into the system
    /// prompt by the caller, so the hook never injects it (`inject_menu = false`).
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
/// Returns `None` when skills are not configured, no source is usable, or
/// nothing could be materialized to disk — so the rest of the runtime can treat
/// skills as entirely optional. On-host roots take precedence over the remote
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

    let tempdir = match tempfile::Builder::new().prefix("nm-skills-").tempdir() {
        Ok(dir) => dir,
        Err(e) => {
            warn!("failed to create skills temp dir: {e}; skills disabled");
            return None;
        }
    };

    let written = materialize(&catalog, tempdir.path()).await;
    if written == 0 {
        warn!("no skills could be materialized to disk; skills disabled");
        return None;
    }

    // Rebuild over the materialized dir so the menu renders `file:` paths and
    // every body read is local.
    let catalog = SkillCatalog::build(vec![Box::new(LocalSkillSource::new(vec![
        tempdir.path().to_path_buf(),
    ]))])
    .await;
    info!(
        skills = catalog.len(),
        dir = %tempdir.path().display(),
        "skill catalog materialized to disk"
    );

    Some(Arc::new(SkillRuntime {
        catalog: Arc::new(catalog),
        menu_budget: settings.menu_budget_bytes,
        body_budget: settings.body_budget_bytes,
        mention: settings.mention,
        tempdir,
    }))
}

/// Write each skill in `catalog` to `<root>/<slug>/SKILL.md`, returning the
/// number written. A skill whose body fails to load, whose id is not
/// filesystem-safe, or that cannot be written is logged and skipped.
async fn materialize(catalog: &SkillCatalog, root: &Path) -> usize {
    let metas: Vec<_> = catalog.metadata().cloned().collect();
    let mut written = 0usize;
    for meta in &metas {
        // Full body on disk (no budget truncation) — the agent's read tool sees
        // the whole skill; `body_budget` only bounds `$mention` injections.
        let body = match catalog.load(&meta.name, usize::MAX).await {
            Ok(body) => body,
            Err(e) => {
                warn!(skill = %meta.name, "failed to load skill body to materialize: {e}");
                continue;
            }
        };
        let Some(slug) = safe_slug(meta.id.as_str()) else {
            warn!(id = %meta.id, "skill id is not filesystem-safe; skipping");
            continue;
        };
        let contents = match synthesize_skill_md(meta, &body) {
            Ok(contents) => contents,
            Err(e) => {
                warn!(skill = %meta.name, "failed to render SKILL.md: {e}");
                continue;
            }
        };
        let dir = root.join(&slug);
        if let Err(e) = tokio::fs::create_dir_all(&dir).await {
            warn!(skill = %meta.name, "failed to create skill dir: {e}");
            continue;
        }
        if let Err(e) = tokio::fs::write(dir.join(SKILL_FILE), contents).await {
            warn!(skill = %meta.name, "failed to write SKILL.md: {e}");
            continue;
        }
        written += 1;
    }
    written
}

/// Map a skill id to a filesystem-safe single path segment, or `None` if it
/// degenerates to nothing usable. Non-`[A-Za-z0-9._-]` characters (including
/// path separators) become `_`, so traversal sequences like `../` cannot
/// escape the temp dir; an all-dots result (`.`, `..`, …) is rejected.
fn safe_slug(id: &str) -> Option<String> {
    let slug: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if slug.is_empty() || slug.chars().all(|c| c == '.') {
        return None;
    }
    Some(slug)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_slug_passes_clean_ids() {
        assert_eq!(safe_slug("forgejo-cli").as_deref(), Some("forgejo-cli"));
        assert_eq!(safe_slug("a.b_c-1").as_deref(), Some("a.b_c-1"));
    }

    #[test]
    fn test_safe_slug_neutralizes_separators() {
        assert_eq!(safe_slug("../escape").as_deref(), Some(".._escape"));
        assert_eq!(safe_slug("a/b").as_deref(), Some("a_b"));
    }

    #[test]
    fn test_safe_slug_rejects_dot_only_ids() {
        assert!(safe_slug("").is_none());
        assert!(safe_slug(".").is_none());
        assert!(safe_slug("..").is_none());
    }
}
