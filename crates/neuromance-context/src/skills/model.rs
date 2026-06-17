//! Core data types describing a discovered skill.

use std::fmt;
use std::path::PathBuf;

/// Identifier for a skill: its directory name on disk, or the `id` a remote
/// endpoint reports. Distinct from the human-facing [`SkillMetadata::name`],
/// though the two usually match.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SkillId(String);

impl SkillId {
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

impl fmt::Display for SkillId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for SkillId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SkillId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Where a skill's `SKILL.md` lives, retained for menu display and body loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillLocator {
    /// An on-host `SKILL.md` at this absolute or relative path.
    Local(PathBuf),
    /// A skill served by a remote endpoint, addressed as `{endpoint}/{id}`.
    Remote {
        /// The base skills endpoint (e.g. `https://corpus/api/v1/skills`).
        endpoint: String,
        /// The id segment appended to the endpoint to fetch this skill.
        id: String,
    },
}

impl SkillLocator {
    /// Render the locator for a menu line, e.g. `file: /skills/x/SKILL.md`.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Self::Local(path) => format!("file: {}", path.display()),
            Self::Remote { endpoint, id } => format!("{endpoint}/{id}"),
        }
    }
}

/// The menu-tier description of a skill: everything except the `SKILL.md` body.
///
/// `extra` retains any frontmatter fields beyond `name`/`description` (for
/// example the agentskills.io `license`, `allowed-tools`, or `metadata` keys)
/// so they are available without rejecting the skill.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillMetadata {
    /// Stable identifier (directory name / remote id).
    pub id: SkillId,
    /// Human-facing name from frontmatter; used for menu lines and `$mentions`.
    pub name: String,
    /// One-line "when to use" summary from frontmatter.
    pub description: String,
    /// Where the body can be loaded from.
    pub locator: SkillLocator,
    /// Frontmatter fields other than `name`/`description`, retained verbatim.
    pub extra: serde_yaml::Mapping,
}
