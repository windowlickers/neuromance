//! Errors raised while discovering, parsing, and loading skills.

use std::path::PathBuf;

use thiserror::Error;

/// A failure encountered by a [`SkillSource`](crate::skills::SkillSource) or the
/// [`SkillCatalog`](crate::skills::SkillCatalog).
#[derive(Debug, Error)]
pub enum SkillError {
    /// The `SKILL.md` did not begin with a `---` fenced YAML frontmatter block.
    #[error("SKILL.md must begin with a `---` fenced YAML frontmatter block")]
    MissingFrontmatter,

    /// The frontmatter YAML could not be parsed.
    #[error("invalid SKILL.md frontmatter YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// A required frontmatter field was absent or an empty string.
    #[error("SKILL.md frontmatter field `{0}` is required and must be a non-empty string")]
    MissingField(&'static str),

    /// No skill with the requested name or locator exists in the catalog.
    #[error("skill `{0}` not found")]
    NotFound(String),

    /// Reading a local `SKILL.md` from disk failed.
    #[error("failed to read skill file `{path}`: {source}")]
    Io {
        /// The path that could not be read.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// An HTTP request to a remote skills endpoint failed at the transport layer.
    #[error("skills endpoint request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// A remote skills endpoint returned a non-success HTTP status.
    #[error("skills endpoint `{url}` returned HTTP {status}")]
    HttpStatus {
        /// The requested URL.
        url: String,
        /// The status code returned.
        status: u16,
    },
}
