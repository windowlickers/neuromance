//! Core data types describing a discovered rule file.

use std::fmt;
use std::path::PathBuf;

/// Identifier for a rule: its path relative to the discovery root on disk, or
/// the `id` a remote endpoint reports.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuleId(String);

impl RuleId {
    /// Wrap a raw id string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for RuleId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for RuleId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Where a rule file lives, retained for display and body loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleLocator {
    /// An on-host rule file at this path.
    Local(PathBuf),
    /// A rule served by a remote endpoint, addressed as `{endpoint}/{id}`.
    Remote {
        /// The base rules endpoint (e.g. `https://corpus/api/v1/rules`).
        endpoint: String,
        /// The id segment appended to the endpoint to fetch this rule.
        id: String,
    },
}

impl RuleLocator {
    /// Render the locator for display, e.g. `file: /rules/rust.md`.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Self::Local(path) => format!("file: {}", path.display()),
            Self::Remote { endpoint, id } => format!("{endpoint}/{id}"),
        }
    }
}

/// The discovery-tier description of a rule: everything except its body.
///
/// A rule activates either because it is `always_apply` (injected once at
/// conversation start) or because one of its `globs` matches a file path a tool
/// touches. `extra` retains any frontmatter fields beyond the recognized ones.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleMetadata {
    /// Stable identifier (relative path / remote id).
    pub id: RuleId,
    /// Glob patterns that trigger this rule when a touched path matches.
    pub globs: Vec<String>,
    /// Whether this rule is injected at conversation start regardless of path.
    pub always_apply: bool,
    /// Optional human-facing description from frontmatter.
    pub description: Option<String>,
    /// Where the body can be loaded from.
    pub locator: RuleLocator,
    /// Frontmatter fields other than the recognized ones, retained verbatim.
    pub extra: serde_yaml::Mapping,
}
