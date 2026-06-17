//! The aggregated, deduplicated view over every configured skill source.

use std::collections::{HashMap, HashSet};

use neuromance_common::chat::Message;
use tracing::warn;
use uuid::Uuid;

use super::error::SkillError;
use super::mention;
use super::model::{SkillLocator, SkillMetadata};
use super::source::SkillSource;

/// Default byte budget for the menu and for each loaded skill body.
pub const DEFAULT_BUDGET_BYTES: usize = 8192;

/// One catalog entry: the skill's metadata plus the index of the source that
/// owns it, so [`SkillCatalog::load`] can fetch its body.
struct Entry {
    meta: SkillMetadata,
    source: usize,
}

/// A read-only catalog of skills aggregated from one or more [`SkillSource`]s.
///
/// Sources are listed in precedence order: when two sources expose the same
/// skill `name`, the earlier source wins and the later one is dropped. Built
/// once (asynchronously) up front; menu rendering and mention resolution are
/// then synchronous, while [`SkillCatalog::load`] fetches a body on demand.
pub struct SkillCatalog {
    sources: Vec<Box<dyn SkillSource>>,
    entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
}

impl SkillCatalog {
    /// Build a catalog by listing every source in precedence order.
    ///
    /// A source whose `list` fails is logged and skipped rather than aborting
    /// the build, so one unreachable endpoint does not disable on-host skills.
    pub async fn build(sources: Vec<Box<dyn SkillSource>>) -> Self {
        let mut entries: Vec<Entry> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for (idx, source) in sources.iter().enumerate() {
            match source.list().await {
                Ok(metas) => {
                    for meta in metas {
                        if seen.insert(meta.name.clone()) {
                            entries.push(Entry { meta, source: idx });
                        }
                    }
                }
                Err(e) => warn!("skill source #{idx} failed to list, skipping: {e}"),
            }
        }
        entries.sort_by(|a, b| a.meta.id.cmp(&b.meta.id));
        let by_name = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.meta.name.clone(), i))
            .collect();
        Self {
            sources,
            entries,
            by_name,
        }
    }

    /// Whether the catalog holds no skills.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of skills in the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Iterate the metadata of every skill, sorted by id.
    pub fn metadata(&self) -> impl Iterator<Item = &SkillMetadata> {
        self.entries.iter().map(|e| &e.meta)
    }

    /// Look up a skill's metadata by its exact `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&SkillMetadata> {
        self.by_name.get(name).map(|&i| &self.entries[i].meta)
    }

    /// Render the progressive-disclosure menu within `budget_bytes`.
    ///
    /// Returns `None` when the catalog is empty. Lines that would exceed the
    /// budget are dropped and disclosed with a trailing "N omitted" line rather
    /// than silently truncated.
    #[must_use]
    pub fn menu(&self, budget_bytes: usize) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let mut lines: Vec<String> = Vec::new();
        let mut used = 0usize;
        let mut omitted = 0usize;
        for entry in &self.entries {
            let line = format!(
                "- {}: {} ({})",
                entry.meta.name,
                entry.meta.description,
                entry.meta.locator.render()
            );
            if used + line.len() + 1 > budget_bytes {
                omitted += 1;
                continue;
            }
            used += line.len() + 1;
            lines.push(line);
        }
        if omitted > 0 {
            lines.push(format!(
                "- {omitted} additional skill(s) omitted from this bounded skills list."
            ));
        }
        Some(format!(
            "<skills_instructions>\n\
             The skills below are reusable instructions for specific tasks. To load a skill's \
             full instructions, call the `load_skill` tool with its name, or write `$name` in \
             your request. Only load a skill when its description matches the task at hand.\n\n\
             {}\n\
             </skills_instructions>",
            lines.join("\n")
        ))
    }

    /// Resolve every skill mentioned in `text`, in catalog order, deduplicated.
    ///
    /// `skill://` URIs and Markdown links resolve by id, name, or path; bare
    /// `$name` mentions resolve by exact name (with env-var names filtered out).
    #[must_use]
    pub fn resolve_mentions(&self, text: &str) -> Vec<&SkillMetadata> {
        let mut out: Vec<&SkillMetadata> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for target in mention::scan_uri_targets(text) {
            if let Some(meta) = self.resolve_locator(&target)
                && seen.insert(meta.name.as_str())
            {
                out.push(meta);
            }
        }
        for name in mention::scan_sigil_names(text) {
            if let Some(meta) = self.get(&name)
                && seen.insert(meta.name.as_str())
            {
                out.push(meta);
            }
        }
        out
    }

    /// Match a `skill://` target against a skill by id, name, or local path tail.
    #[must_use]
    pub fn resolve_locator(&self, target: &str) -> Option<&SkillMetadata> {
        let target = target.trim_end_matches('/');
        self.entries.iter().map(|e| &e.meta).find(|meta| {
            meta.id.as_str() == target
                || meta.name == target
                || matches!(&meta.locator, SkillLocator::Local(path)
                    if path.to_string_lossy().ends_with(target))
        })
    }

    /// Load a skill's body by `name`, truncated to `budget_bytes`.
    ///
    /// # Errors
    /// Returns [`SkillError::NotFound`] if no skill has that name, or an I/O /
    /// HTTP error if the owning source cannot produce the body.
    pub async fn load(&self, name: &str, budget_bytes: usize) -> Result<String, SkillError> {
        let idx = *self
            .by_name
            .get(name)
            .ok_or_else(|| SkillError::NotFound(name.to_string()))?;
        let entry = &self.entries[idx];
        let body = self.sources[entry.source].load_body(&entry.meta.id).await?;
        Ok(truncate_on_char_boundary(&body, budget_bytes))
    }

    /// Build the per-turn body messages for every skill mentioned in `text`.
    ///
    /// Each resolved skill becomes one `User` message wrapped in `<skill>` tags.
    /// A skill whose body fails to load is logged and omitted, never aborting.
    pub async fn mention_messages(
        &self,
        conversation_id: Uuid,
        text: &str,
        budget_bytes: usize,
    ) -> Vec<Message> {
        let names: Vec<String> = self
            .resolve_mentions(text)
            .into_iter()
            .map(|m| m.name.clone())
            .collect();
        let mut messages = Vec::new();
        for name in names {
            match self.load(&name, budget_bytes).await {
                Ok(body) => {
                    let locator = self
                        .get(&name)
                        .map(|m| m.locator.render())
                        .unwrap_or_default();
                    messages.push(Message::user(
                        conversation_id,
                        format!("<skill>\n<name>{name}</name>\n<source>{locator}</source>\n{body}\n</skill>"),
                    ));
                }
                Err(e) => warn!("failed to load mentioned skill `{name}`: {e}"),
            }
        }
        messages
    }
}

/// Truncate `s` to at most `max_bytes`, never splitting a UTF-8 code point, and
/// append a disclosure line when truncation occurred.
fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    warn!("skill body truncated from {} to {} bytes", s.len(), end);
    format!("{}\n[skill body truncated to {max_bytes} bytes]", &s[..end])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use async_trait::async_trait;

    use super::*;
    use crate::skills::model::SkillId;

    /// In-memory source backed by `(name, body)` pairs for deterministic tests.
    struct MemSource {
        skills: Vec<(String, String)>,
    }

    impl MemSource {
        fn new(skills: &[(&str, &str)]) -> Self {
            Self {
                skills: skills
                    .iter()
                    .map(|(n, b)| ((*n).to_string(), (*b).to_string()))
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl SkillSource for MemSource {
        async fn list(&self) -> Result<Vec<SkillMetadata>, SkillError> {
            Ok(self
                .skills
                .iter()
                .map(|(name, _)| SkillMetadata {
                    id: SkillId::new(name),
                    name: name.clone(),
                    description: format!("desc of {name}"),
                    locator: SkillLocator::Remote {
                        endpoint: "mem://skills".to_string(),
                        id: name.clone(),
                    },
                    extra: serde_yaml::Mapping::new(),
                })
                .collect())
        }

        async fn load_body(&self, id: &SkillId) -> Result<String, SkillError> {
            self.skills
                .iter()
                .find(|(name, _)| name == id.as_str())
                .map(|(_, body)| body.clone())
                .ok_or_else(|| SkillError::NotFound(id.to_string()))
        }
    }

    async fn catalog(skills: &[(&str, &str)]) -> SkillCatalog {
        SkillCatalog::build(vec![Box::new(MemSource::new(skills))]).await
    }

    #[tokio::test]
    async fn test_empty_catalog_has_no_menu() {
        let cat = catalog(&[]).await;
        assert!(cat.is_empty());
        assert!(cat.menu(DEFAULT_BUDGET_BYTES).is_none());
    }

    #[tokio::test]
    async fn test_menu_lists_every_skill() {
        let cat = catalog(&[("alpha", "a"), ("beta", "b")]).await;
        let menu = cat.menu(DEFAULT_BUDGET_BYTES).unwrap();
        assert!(menu.contains("- alpha: desc of alpha"));
        assert!(menu.contains("- beta: desc of beta"));
        assert!(menu.contains("<skills_instructions>"));
    }

    #[tokio::test]
    async fn test_menu_budget_omits_and_discloses() {
        let cat = catalog(&[("alpha", "a"), ("beta", "b"), ("gamma", "c")]).await;
        // Budget large enough for one line only.
        let menu = cat.menu(40).unwrap();
        assert!(menu.contains("omitted from this bounded skills list"));
    }

    #[tokio::test]
    async fn test_higher_precedence_source_wins() {
        let high = MemSource::new(&[("dup", "high body")]);
        let low = MemSource::new(&[("dup", "low body")]);
        let cat = SkillCatalog::build(vec![Box::new(high), Box::new(low)]).await;
        assert_eq!(cat.len(), 1);
        assert_eq!(
            cat.load("dup", DEFAULT_BUDGET_BYTES).await.unwrap(),
            "high body"
        );
    }

    #[tokio::test]
    async fn test_load_unknown_skill_is_not_found() {
        let cat = catalog(&[("alpha", "a")]).await;
        let err = cat.load("ghost", DEFAULT_BUDGET_BYTES).await.unwrap_err();
        assert!(matches!(err, SkillError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_load_truncates_long_body_on_char_boundary() {
        let cat = catalog(&[("alpha", "héllo wörld this is quite long")]).await;
        let body = cat.load("alpha", 6).await.unwrap();
        assert!(body.starts_with("héll"));
        assert!(body.contains("truncated"));
    }

    #[tokio::test]
    async fn test_resolve_mentions_dedups_across_syntaxes() {
        let cat = catalog(&[("deploy", "x"), ("build", "y")]).await;
        let resolved = cat.resolve_mentions("use skill://deploy and also $deploy and $build");
        let names: Vec<&str> = resolved.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["deploy", "build"]);
    }

    #[tokio::test]
    async fn test_mention_messages_inject_bodies() {
        let cat = catalog(&[("deploy", "deploy instructions")]).await;
        let conv = Uuid::new_v4();
        let msgs = cat
            .mention_messages(conv, "please $deploy", DEFAULT_BUDGET_BYTES)
            .await;
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].content.contains("<name>deploy</name>"));
        assert!(msgs[0].content.contains("deploy instructions"));
    }
}
