//! The aggregated, deduplicated view over every configured rule source.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use tracing::warn;

use super::error::RuleError;
use super::model::{RuleId, RuleMetadata};
use super::source::RuleSource;

/// Default byte budget for each loaded rule body.
pub const DEFAULT_BUDGET_BYTES: usize = 8192;

/// One catalog entry: the rule's metadata, the source that owns it, and its
/// compiled glob set (empty when the rule has no globs, so it never path-matches).
struct Entry {
    meta: RuleMetadata,
    source: usize,
    set: GlobSet,
}

/// A read-only catalog of rules aggregated from one or more [`RuleSource`]s.
///
/// Sources are listed in precedence order: when two sources expose the same rule
/// id, the earlier source wins. Built once asynchronously up front; glob
/// matching is then synchronous, while [`RuleCatalog::load`] fetches a body on
/// demand.
pub struct RuleCatalog {
    sources: Vec<Box<dyn RuleSource>>,
    entries: Vec<Entry>,
    by_id: HashMap<RuleId, usize>,
}

impl RuleCatalog {
    /// Build a catalog by listing every source in precedence order.
    ///
    /// A source whose `list` fails, or a rule whose glob fails to compile, is
    /// logged and skipped rather than aborting the build.
    pub async fn build(sources: Vec<Box<dyn RuleSource>>) -> Self {
        let mut entries: Vec<Entry> = Vec::new();
        let mut seen: HashSet<RuleId> = HashSet::new();
        let mut any_listed = false;
        for (idx, source) in sources.iter().enumerate() {
            match source.list().await {
                Ok(metas) => {
                    any_listed = true;
                    for meta in metas {
                        if !seen.insert(meta.id.clone()) {
                            continue;
                        }
                        match compile_globs(&meta) {
                            Ok(set) => entries.push(Entry {
                                meta,
                                source: idx,
                                set,
                            }),
                            Err(e) => warn!("skipping rule with invalid glob: {e}"),
                        }
                    }
                }
                Err(e) => warn!("rule source #{idx} failed to list, skipping: {e}"),
            }
        }
        if !sources.is_empty() && !any_listed {
            warn!(
                "all {} rule source(s) failed to list; catalog is empty",
                sources.len()
            );
        }
        entries.sort_by(|a, b| a.meta.id.cmp(&b.meta.id));
        let by_id = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.meta.id.clone(), i))
            .collect();
        Self {
            sources,
            entries,
            by_id,
        }
    }

    /// Whether the catalog holds no rules.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of rules in the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The rules that apply on every conversation, sorted by id.
    pub fn always_apply(&self) -> impl Iterator<Item = &RuleMetadata> {
        self.entries
            .iter()
            .filter(|e| e.meta.always_apply)
            .map(|e| &e.meta)
    }

    /// The glob-triggered rules whose patterns match `path`, sorted by id.
    ///
    /// Always-apply rules are excluded — they are injected at conversation start,
    /// not per touched path.
    #[must_use]
    pub fn match_path(&self, path: &str) -> Vec<&RuleMetadata> {
        let basename = Path::new(path).file_name().and_then(|n| n.to_str());
        self.entries
            .iter()
            .filter(|e| !e.meta.always_apply)
            .filter(|e| e.set.is_match(path) || basename.is_some_and(|name| e.set.is_match(name)))
            .map(|e| &e.meta)
            .collect()
    }

    /// Load a rule's body by `id`, truncated to `budget_bytes`.
    ///
    /// # Errors
    /// Returns [`RuleError::NotFound`] if no rule has that id, or an I/O / HTTP
    /// error if the owning source cannot produce the body.
    pub async fn load(&self, id: &RuleId, budget_bytes: usize) -> Result<String, RuleError> {
        let idx = *self
            .by_id
            .get(id)
            .ok_or_else(|| RuleError::NotFound(id.to_string()))?;
        let entry = &self.entries[idx];
        let body = self.sources[entry.source].load_body(&entry.meta.id).await?;
        Ok(truncate_on_char_boundary(&body, budget_bytes))
    }
}

/// Compile a rule's globs into a [`GlobSet`].
///
/// Each pattern is anchored with a leading `**/` (unless already absolute or
/// `**`-prefixed) so a workspace-relative pattern like `src/**/*.rs` still
/// matches an absolute tool path, and `*.ts` matches a file in any directory.
fn compile_globs(meta: &RuleMetadata) -> Result<GlobSet, RuleError> {
    let mut builder = GlobSetBuilder::new();
    for pattern in &meta.globs {
        let glob = GlobBuilder::new(&normalize_glob(pattern))
            .literal_separator(true)
            .build()
            .map_err(|source| RuleError::Glob {
                rule: meta.id.to_string(),
                source,
            })?;
        builder.add(glob);
    }
    builder.build().map_err(|source| RuleError::Glob {
        rule: meta.id.to_string(),
        source,
    })
}

/// Anchor `pattern` so a relative glob matches anywhere in an absolute path.
fn normalize_glob(pattern: &str) -> String {
    if pattern.starts_with('/') || pattern.starts_with("**") {
        pattern.to_string()
    } else {
        format!("**/{pattern}")
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
    warn!("rule body truncated from {} to {} bytes", s.len(), end);
    format!("{}\n[rule body truncated to {max_bytes} bytes]", &s[..end])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use async_trait::async_trait;

    use super::*;
    use crate::rules::model::RuleLocator;

    /// In-memory source backed by `(id, globs, always_apply, body)` tuples.
    struct MemSource {
        rules: Vec<(String, Vec<String>, bool, String)>,
    }

    impl MemSource {
        fn new(rules: &[(&str, &[&str], bool, &str)]) -> Self {
            Self {
                rules: rules
                    .iter()
                    .map(|(id, globs, aa, body)| {
                        (
                            (*id).to_string(),
                            globs.iter().map(|g| (*g).to_string()).collect(),
                            *aa,
                            (*body).to_string(),
                        )
                    })
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl RuleSource for MemSource {
        async fn list(&self) -> Result<Vec<RuleMetadata>, RuleError> {
            Ok(self
                .rules
                .iter()
                .map(|(id, globs, aa, _)| RuleMetadata {
                    id: RuleId::new(id),
                    globs: globs.clone(),
                    always_apply: *aa,
                    description: None,
                    locator: RuleLocator::Remote {
                        endpoint: "mem://rules".to_string(),
                        id: id.clone(),
                    },
                    extra: serde_yaml::Mapping::new(),
                })
                .collect())
        }

        async fn load_body(&self, id: &RuleId) -> Result<String, RuleError> {
            self.rules
                .iter()
                .find(|(rid, ..)| rid == id.as_str())
                .map(|(.., body)| body.clone())
                .ok_or_else(|| RuleError::NotFound(id.to_string()))
        }
    }

    async fn catalog(rules: &[(&str, &[&str], bool, &str)]) -> RuleCatalog {
        RuleCatalog::build(vec![Box::new(MemSource::new(rules))]).await
    }

    #[tokio::test]
    async fn test_extension_glob_matches_nested_path() {
        let cat = catalog(&[("ts", &["*.ts"], false, "ts rules")]).await;
        let matched = cat.match_path("/a/b/c.ts");
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].id.as_str(), "ts");
    }

    #[tokio::test]
    async fn test_path_glob_matches_under_dir_not_elsewhere() {
        let cat = catalog(&[("rs", &["src/**/*.rs"], false, "rs rules")]).await;
        assert_eq!(cat.match_path("/proj/src/a/b.rs").len(), 1);
        assert!(cat.match_path("/proj/tests/a.rs").is_empty());
    }

    #[tokio::test]
    async fn test_non_matching_path_yields_nothing() {
        let cat = catalog(&[("ts", &["*.ts"], false, "x")]).await;
        assert!(cat.match_path("/a/b/c.rs").is_empty());
    }

    #[tokio::test]
    async fn test_always_apply_excluded_from_path_match() {
        let cat = catalog(&[("global", &[], true, "always")]).await;
        assert!(cat.match_path("/a/b.rs").is_empty());
        assert_eq!(cat.always_apply().count(), 1);
    }

    #[tokio::test]
    async fn test_malformed_glob_skipped() {
        let cat = catalog(&[
            ("bad", &["[unclosed"], false, "x"),
            ("ok", &["*.rs"], false, "y"),
        ])
        .await;
        assert_eq!(cat.len(), 1);
        assert_eq!(cat.match_path("/a.rs").len(), 1);
    }

    #[tokio::test]
    async fn test_load_truncates_on_char_boundary() {
        let cat = catalog(&[("r", &["*.rs"], false, "héllo wörld this is quite long")]).await;
        let body = cat.load(&RuleId::new("r"), 6).await.unwrap();
        assert!(body.starts_with("héll"));
        assert!(body.contains("truncated"));
    }
}
