//! Errors raised while discovering, parsing, and loading rule files.

use std::path::PathBuf;

use thiserror::Error;

/// A failure encountered by a [`RuleSource`](crate::rules::RuleSource) or the
/// [`RuleCatalog`](crate::rules::RuleCatalog).
#[derive(Debug, Error)]
pub enum RuleError {
    /// A rule file's frontmatter YAML could not be parsed.
    #[error("invalid rule frontmatter YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// No rule with the requested id exists in the catalog.
    #[error("rule `{0}` not found")]
    NotFound(String),

    /// Reading a local rule file from disk failed.
    #[error("failed to read rule file `{path}`: {source}")]
    Io {
        /// The path that could not be read.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// An HTTP request to a remote rules endpoint failed at the transport layer.
    #[error("rules endpoint request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// A remote rules endpoint returned a non-success HTTP status.
    #[error("rules endpoint `{url}` returned HTTP {status}")]
    HttpStatus {
        /// The requested URL.
        url: String,
        /// The status code returned.
        status: u16,
    },

    /// A glob pattern in a rule's frontmatter could not be compiled.
    #[error("invalid glob pattern in rule `{rule}`: {source}")]
    Glob {
        /// The rule whose glob failed to compile.
        rule: String,
        /// The underlying globset error.
        #[source]
        source: globset::Error,
    },
}
