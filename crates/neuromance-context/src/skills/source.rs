//! Skill sources: where skills are discovered and bodies are loaded from.
//!
//! A [`SkillSource`] lists menu-tier [`SkillMetadata`] cheaply and loads a full
//! `SKILL.md` body on demand. [`LocalSkillSource`] walks on-host directory
//! roots; [`HttpSkillSource`] consumes a corpus-shaped HTTP API.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use tracing::warn;

use super::error::SkillError;
use super::frontmatter::parse_skill;
use super::model::{SkillId, SkillLocator, SkillMetadata};

/// The conventional frontmatter file inside each skill directory.
const SKILL_FILE: &str = "SKILL.md";

/// A provider of skills: a cheap metadata listing plus on-demand body loads.
#[async_trait]
pub trait SkillSource: Send + Sync {
    /// List every skill this source can provide, metadata only (no body).
    ///
    /// # Errors
    /// Returns a [`SkillError`] if the source cannot be enumerated at all;
    /// individual malformed skills are skipped rather than failing the listing.
    async fn list(&self) -> Result<Vec<SkillMetadata>, SkillError>;

    /// Load the full `SKILL.md` body for `id`.
    ///
    /// # Errors
    /// Returns [`SkillError::NotFound`] if no such skill exists, or an I/O /
    /// HTTP error if the body cannot be fetched.
    async fn load_body(&self, id: &SkillId) -> Result<String, SkillError>;
}

/// Discovers skills in on-host directory roots.
///
/// Each root's immediate subdirectories that contain a `SKILL.md` are skills,
/// keyed by directory name. Roots are searched in order: when two roots define
/// the same id, the earlier root wins (higher precedence).
pub struct LocalSkillSource {
    roots: Vec<PathBuf>,
}

impl LocalSkillSource {
    /// Create a source over `roots`, highest-precedence root first.
    #[must_use]
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }

    /// Path to a candidate `SKILL.md` for `id` under `root`.
    fn skill_path(root: &Path, id: &str) -> PathBuf {
        root.join(id).join(SKILL_FILE)
    }
}

#[async_trait]
impl SkillSource for LocalSkillSource {
    async fn list(&self) -> Result<Vec<SkillMetadata>, SkillError> {
        let mut out: Vec<SkillMetadata> = Vec::new();
        let mut seen: Vec<String> = Vec::new();
        for root in &self.roots {
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            for entry in entries.flatten() {
                let id = entry.file_name().to_string_lossy().into_owned();
                let path = Self::skill_path(root, &id);
                if !path.is_file() || seen.contains(&id) {
                    continue;
                }
                let parsed = std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|content| parse_skill(&content).map_err(|e| e.to_string()));
                match parsed {
                    Ok(parsed) => {
                        seen.push(id.clone());
                        out.push(SkillMetadata {
                            id: SkillId::new(id),
                            name: parsed.name,
                            description: parsed.description,
                            locator: SkillLocator::Local(path),
                            extra: parsed.extra,
                        });
                    }
                    Err(e) => warn!(skill = %id, "skipping malformed SKILL.md: {e}"),
                }
            }
        }
        Ok(out)
    }

    async fn load_body(&self, id: &SkillId) -> Result<String, SkillError> {
        for root in &self.roots {
            let path = Self::skill_path(root, id.as_str());
            if path.is_file() {
                let content = std::fs::read_to_string(&path).map_err(|source| SkillError::Io {
                    path: path.clone(),
                    source,
                })?;
                return Ok(parse_skill(&content)?.body);
            }
        }
        Err(SkillError::NotFound(id.to_string()))
    }
}

/// One skill in a corpus-shaped list response (extra JSON fields are ignored).
#[derive(Debug, Deserialize)]
struct SkillSummaryDto {
    id: String,
    name: String,
    description: String,
}

/// The `GET /skills` envelope returned by a corpus-shaped endpoint.
#[derive(Debug, Deserialize)]
struct SkillListDto {
    skills: Vec<SkillSummaryDto>,
}

/// A single skill's detail; only `body` is consumed.
#[derive(Debug, Deserialize)]
struct SkillDetailDto {
    body: String,
}

/// Fetches skills from a corpus-shaped HTTP endpoint.
///
/// `list` GETs the base endpoint and reads `{ "skills": [{ id, name,
/// description }] }`; `load_body` GETs `{endpoint}/{id}` and reads `body`.
/// Unknown JSON fields (`provenance`, `url`, `revision`, …) are ignored, so the
/// source tolerates a richer server response.
pub struct HttpSkillSource {
    endpoint: String,
    client: reqwest::Client,
    auth: Option<(String, String)>,
}

impl HttpSkillSource {
    /// Create a source against `endpoint` (e.g. `https://corpus/api/v1/skills`).
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

    /// URL for a single skill's detail.
    fn detail_url(&self, id: &str) -> String {
        format!("{}/{}", self.endpoint.trim_end_matches('/'), id)
    }
}

#[async_trait]
impl SkillSource for HttpSkillSource {
    async fn list(&self) -> Result<Vec<SkillMetadata>, SkillError> {
        let response = self.get(&self.endpoint).send().await?;
        if !response.status().is_success() {
            return Err(SkillError::HttpStatus {
                url: self.endpoint.clone(),
                status: response.status().as_u16(),
            });
        }
        let listing: SkillListDto = response.json().await?;
        let skills = listing
            .skills
            .into_iter()
            .map(|s| SkillMetadata {
                id: SkillId::new(&s.id),
                name: s.name,
                description: s.description,
                locator: SkillLocator::Remote {
                    endpoint: self.endpoint.clone(),
                    id: s.id,
                },
                extra: serde_yaml::Mapping::new(),
            })
            .collect();
        Ok(skills)
    }

    async fn load_body(&self, id: &SkillId) -> Result<String, SkillError> {
        let url = self.detail_url(id.as_str());
        let response = self.get(&url).send().await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(SkillError::NotFound(id.to_string()));
        }
        if !response.status().is_success() {
            return Err(SkillError::HttpStatus {
                url,
                status: response.status().as_u16(),
            });
        }
        Ok(response.json::<SkillDetailDto>().await?.body)
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

    fn write_skill(root: &Path, id: &str, body: &str) {
        let dir = root.join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(SKILL_FILE), body).unwrap();
    }

    #[tokio::test]
    async fn test_local_lists_and_loads() {
        let dir = tempdir().unwrap();
        write_skill(
            dir.path(),
            "alpha",
            "---\nname: alpha\ndescription: first\n---\nthe alpha body",
        );
        let source = LocalSkillSource::new(vec![dir.path().to_path_buf()]);

        let listed = source.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "alpha");

        let body = source.load_body(&SkillId::new("alpha")).await.unwrap();
        assert_eq!(body, "the alpha body");
    }

    #[tokio::test]
    async fn test_local_skips_malformed_skill() {
        let dir = tempdir().unwrap();
        write_skill(
            dir.path(),
            "good",
            "---\nname: good\ndescription: ok\n---\nbody",
        );
        write_skill(dir.path(), "bad", "no frontmatter at all");
        let source = LocalSkillSource::new(vec![dir.path().to_path_buf()]);

        let listed = source.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "good");
    }

    #[tokio::test]
    async fn test_local_earlier_root_wins() {
        let high = tempdir().unwrap();
        let low = tempdir().unwrap();
        write_skill(
            high.path(),
            "dup",
            "---\nname: dup\ndescription: high\n---\nhigh",
        );
        write_skill(
            low.path(),
            "dup",
            "---\nname: dup\ndescription: low\n---\nlow",
        );
        let source =
            LocalSkillSource::new(vec![high.path().to_path_buf(), low.path().to_path_buf()]);

        let listed = source.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].description, "high");
        assert_eq!(
            source.load_body(&SkillId::new("dup")).await.unwrap(),
            "high"
        );
    }

    #[tokio::test]
    async fn test_local_missing_skill_is_not_found() {
        let dir = tempdir().unwrap();
        let source = LocalSkillSource::new(vec![dir.path().to_path_buf()]);
        let err = source.load_body(&SkillId::new("nope")).await.unwrap_err();
        assert!(matches!(err, SkillError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_http_lists_and_loads_ignoring_extra_fields() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/skills"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "url": "https://example/corpus.git",
                "revision": "deadbeef",
                "skills": [
                    { "id": "alpha", "name": "alpha", "description": "first",
                      "provenance": { "revision": "abc" } }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/skills/alpha"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "alpha", "name": "alpha", "description": "first",
                "body": "remote body", "url": null, "revision": null
            })))
            .mount(&server)
            .await;

        let source = HttpSkillSource::new(format!("{}/skills", server.uri()));
        let listed = source.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "alpha");

        let body = source.load_body(&SkillId::new("alpha")).await.unwrap();
        assert_eq!(body, "remote body");
    }

    #[tokio::test]
    async fn test_http_missing_skill_maps_to_not_found() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/skills/ghost"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let source = HttpSkillSource::new(format!("{}/skills", server.uri()));
        let err = source.load_body(&SkillId::new("ghost")).await.unwrap_err();
        assert!(matches!(err, SkillError::NotFound(_)));
    }
}
