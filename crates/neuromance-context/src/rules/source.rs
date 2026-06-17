//! Rule sources: where rule files are discovered and bodies are loaded from.
//!
//! A [`RuleSource`] lists rule [`RuleMetadata`] (globs + flags) cheaply and
//! loads a full body on demand. [`LocalRuleSource`] walks on-host directory
//! roots recursively for `.md`/`.mdc` files; [`HttpRuleSource`] consumes a
//! corpus-shaped HTTP API.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use tracing::warn;

use super::error::RuleError;
use super::frontmatter::parse_rule;
use super::model::{RuleId, RuleLocator, RuleMetadata};

/// A provider of rules: a cheap metadata listing plus on-demand body loads.
#[async_trait]
pub trait RuleSource: Send + Sync {
    /// List every rule this source can provide, metadata only (no body).
    ///
    /// # Errors
    /// Returns a [`RuleError`] if the source cannot be enumerated at all;
    /// individual malformed files are skipped rather than failing the listing.
    async fn list(&self) -> Result<Vec<RuleMetadata>, RuleError>;

    /// Load the full body for `id`.
    ///
    /// # Errors
    /// Returns [`RuleError::NotFound`] if no such rule exists, or an I/O / HTTP
    /// error if the body cannot be fetched.
    async fn load_body(&self, id: &RuleId) -> Result<String, RuleError>;
}

/// Discovers rule files in on-host directory roots.
///
/// Each root is walked recursively for `.md` / `.mdc` files; a rule's id is its
/// path relative to the root. Roots are searched in order: when two roots define
/// the same id, the earlier root wins (higher precedence).
pub struct LocalRuleSource {
    roots: Vec<PathBuf>,
}

impl LocalRuleSource {
    /// Create a source over `roots`, highest-precedence root first.
    #[must_use]
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }

    /// Whether `path` is a rule file by extension.
    fn is_rule_file(path: &Path) -> bool {
        matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("md" | "mdc")
        )
    }

    /// Recursively collect rule files under `dir` into `out`.
    fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::collect(&path, out);
            } else if Self::is_rule_file(&path) {
                out.push(path);
            }
        }
    }

    /// The id for `path` relative to `root` (path separators normalized to `/`).
    fn id_for(root: &Path, path: &Path) -> String {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }
}

#[async_trait]
impl RuleSource for LocalRuleSource {
    async fn list(&self) -> Result<Vec<RuleMetadata>, RuleError> {
        let mut out: Vec<RuleMetadata> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for root in &self.roots {
            let mut files = Vec::new();
            Self::collect(root, &mut files);
            for path in files {
                let id = Self::id_for(root, &path);
                if seen.contains(&id) {
                    continue;
                }
                match std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|content| parse_rule(&content).map_err(|e| e.to_string()))
                {
                    Ok(parsed) => {
                        seen.insert(id.clone());
                        out.push(RuleMetadata {
                            id: RuleId::new(id),
                            globs: parsed.globs,
                            always_apply: parsed.always_apply,
                            description: parsed.description,
                            locator: RuleLocator::Local(path),
                            extra: parsed.extra,
                        });
                    }
                    Err(e) => warn!(rule = %id, "skipping malformed rule file: {e}"),
                }
            }
        }
        Ok(out)
    }

    async fn load_body(&self, id: &RuleId) -> Result<String, RuleError> {
        for root in &self.roots {
            let path = root.join(id.as_str());
            if path.is_file() {
                let content = std::fs::read_to_string(&path).map_err(|source| RuleError::Io {
                    path: path.clone(),
                    source,
                })?;
                return Ok(parse_rule(&content)?.body);
            }
        }
        Err(RuleError::NotFound(id.to_string()))
    }
}

/// One rule in a corpus-shaped list response (extra JSON fields are ignored).
#[derive(Debug, Deserialize)]
struct RuleSummaryDto {
    id: String,
    #[serde(default)]
    globs: Vec<String>,
    #[serde(default, alias = "alwaysApply")]
    always_apply: bool,
    #[serde(default)]
    description: Option<String>,
}

/// The `GET /rules` envelope returned by a corpus-shaped endpoint.
#[derive(Debug, Deserialize)]
struct RuleListDto {
    rules: Vec<RuleSummaryDto>,
}

/// A single rule's detail; only `body` is consumed.
#[derive(Debug, Deserialize)]
struct RuleDetailDto {
    body: String,
}

/// Fetches rules from a corpus-shaped HTTP endpoint.
///
/// `list` GETs the base endpoint and reads `{ "rules": [{ id, globs,
/// always_apply, description }] }`; `load_body` GETs `{endpoint}/{id}` and reads
/// `body`. Unknown JSON fields are ignored.
pub struct HttpRuleSource {
    endpoint: String,
    client: reqwest::Client,
    auth: Option<(String, String)>,
}

impl HttpRuleSource {
    /// Create a source against `endpoint` (e.g. `https://corpus/api/v1/rules`).
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            client: reqwest::Client::new(),
            auth: None,
        }
    }

    /// Attach a request header (name, value) to every request — typically
    /// `("Authorization", "Bearer …")` for an authenticated endpoint.
    #[must_use]
    pub fn with_auth_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = Some((name.into(), value.into()));
        self
    }

    /// Build a GET request to `url` with the optional auth header applied.
    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.get(url);
        if let Some((name, value)) = &self.auth {
            req = req.header(name, value);
        }
        req
    }

    /// URL for a single rule's detail.
    fn detail_url(&self, id: &str) -> String {
        format!("{}/{}", self.endpoint.trim_end_matches('/'), id)
    }
}

#[async_trait]
impl RuleSource for HttpRuleSource {
    async fn list(&self) -> Result<Vec<RuleMetadata>, RuleError> {
        let response = self.get(&self.endpoint).send().await?;
        if !response.status().is_success() {
            return Err(RuleError::HttpStatus {
                url: self.endpoint.clone(),
                status: response.status().as_u16(),
            });
        }
        let listing: RuleListDto = response.json().await?;
        let rules = listing
            .rules
            .into_iter()
            .map(|r| RuleMetadata {
                id: RuleId::new(&r.id),
                globs: r.globs,
                always_apply: r.always_apply,
                description: r.description,
                locator: RuleLocator::Remote {
                    endpoint: self.endpoint.clone(),
                    id: r.id,
                },
                extra: serde_yaml::Mapping::new(),
            })
            .collect();
        Ok(rules)
    }

    async fn load_body(&self, id: &RuleId) -> Result<String, RuleError> {
        let url = self.detail_url(id.as_str());
        let response = self.get(&url).send().await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(RuleError::NotFound(id.to_string()));
        }
        if !response.status().is_success() {
            return Err(RuleError::HttpStatus {
                url,
                status: response.status().as_u16(),
            });
        }
        Ok(response.json::<RuleDetailDto>().await?.body)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::fs;

    use tempfile::tempdir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[tokio::test]
    async fn test_local_lists_recursively_and_loads() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("rust");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            nested.join("conventions.md"),
            "---\nglobs: \"*.rs\"\n---\nrust body",
        )
        .unwrap();
        fs::write(
            dir.path().join("top.mdc"),
            "---\nalwaysApply: true\n---\ntop",
        )
        .unwrap();

        let source = LocalRuleSource::new(vec![dir.path().to_path_buf()]);
        let mut listed = source.list().await.unwrap();
        listed.sort_by(|a, b| a.id.cmp(&b.id));

        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].id.as_str(), "rust/conventions.md");
        assert_eq!(listed[0].globs, vec!["*.rs"]);
        assert!(listed[1].always_apply);

        let body = source
            .load_body(&RuleId::new("rust/conventions.md"))
            .await
            .unwrap();
        assert_eq!(body, "rust body");
    }

    #[tokio::test]
    async fn test_local_missing_rule_is_not_found() {
        let dir = tempdir().unwrap();
        let source = LocalRuleSource::new(vec![dir.path().to_path_buf()]);
        let err = source.load_body(&RuleId::new("nope.md")).await.unwrap_err();
        assert!(matches!(err, RuleError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_http_lists_and_loads() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/rules"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "revision": "deadbeef",
                "rules": [
                    { "id": "rust.md", "globs": ["*.rs"], "alwaysApply": false,
                      "description": "rust", "extra": "ignored" }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rules/rust.md"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "rust.md", "body": "remote rule body"
            })))
            .mount(&server)
            .await;

        let source = HttpRuleSource::new(format!("{}/rules", server.uri()));
        let listed = source.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].globs, vec!["*.rs"]);

        let body = source.load_body(&RuleId::new("rust.md")).await.unwrap();
        assert_eq!(body, "remote rule body");
    }
}
