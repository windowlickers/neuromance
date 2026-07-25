//! Library-level git operations that keep remote credentials out of the pod.
//!
//! Authenticated HTTP operations route through the tokenizer proxy: a clone
//! attaches the proxy as an HTTP proxy ([`git2::ProxyOptions`]) and carries a
//! sealed token in a custom header. The proxy reads the sealed token, injects
//! the real git credential server-side, and forwards upstream. The pod never
//! holds the plaintext credential, matching the model the LLM client uses
//! (see `neuromance-runtime/src/proxy.rs`).
//!
//! For the proxy to read the sealed header it must see the request in the
//! clear, so remotes are addressed with an `http://` scheme; the proxy
//! upgrades to TLS upstream. The proxy must understand git smart-HTTP
//! (`/info/refs?service=git-upload-pack`, `git-receive-pack`) and inject git
//! auth — that is a proxy-side capability, configured out of band.
//!
//! The credentials callback is anonymous; this module never calls
//! `Cred::userpass_plaintext`.

use std::fmt;
use std::path::{Component, Path, PathBuf};

use git2::build::RepoBuilder;
use git2::{FetchOptions, ProxyOptions, Repository};
use secrecy::{ExposeSecret, SecretString};

/// Tokenizer-proxy auth for remote git operations.
///
/// Carries the sealed token, never the plaintext credential. The proxy
/// unseals it and injects the real credential upstream.
#[derive(Clone)]
pub struct GitProxyAuth {
    /// The tokenizer proxy endpoint attached as an HTTP proxy.
    pub proxy_url: String,
    /// Header name carrying the sealed token (e.g. `X-Tokenizer-Token`).
    pub token_header: String,
    /// The sealed token the proxy unseals server-side.
    pub sealed_token: SecretString,
}

impl GitProxyAuth {
    /// The custom HTTP header line (`<header>: <sealed-token>`) attached to
    /// fetch requests for the proxy to consume.
    fn header_line(&self) -> String {
        format!(
            "{}: {}",
            self.token_header,
            self.sealed_token.expose_secret()
        )
    }
}

impl fmt::Debug for GitProxyAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GitProxyAuth")
            .field("proxy_url", &self.proxy_url)
            .field("token_header", &self.token_header)
            .field("sealed_token", &"[REDACTED]")
            .finish()
    }
}

/// Errors from library-level git operations.
#[derive(Debug, thiserror::Error)]
pub enum GitError {
    /// Cloning the remote failed (network, auth, or missing repository).
    #[error("cloning {url}: {source}")]
    Clone {
        /// The remote URL passed to the clone.
        url: String,
        #[source]
        source: git2::Error,
    },
    /// The requested reference could not be resolved or checked out.
    #[error("checking out '{reference}': {source}")]
    Checkout {
        /// The reference passed to [`clone_repository`].
        reference: String,
        #[source]
        source: git2::Error,
    },
    /// The sealed token file could not be read.
    #[error("reading sealed token from {}: {source}", path.display())]
    TokenRead {
        /// The token file path.
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The sealed token file exists but holds only whitespace.
    #[error("sealed token file '{}' is empty", path.display())]
    TokenEmpty {
        /// The token file path.
        path: PathBuf,
    },
    /// A destination path was absolute where a relative one is required.
    #[error("destination must be relative, got: {path}")]
    AbsoluteDest {
        /// The offending path.
        path: String,
    },
    /// A destination path attempted to escape its root with `..`.
    #[error("destination must not contain '..': {path}")]
    DestTraversal {
        /// The offending path.
        path: String,
    },
}

/// Clone `url` into `dest`, optionally checking out `reference`.
///
/// With `auth`, the fetch routes through the tokenizer proxy with the sealed
/// token header; without it the clone is anonymous (public remotes,
/// `file://` fixtures in tests).
///
/// `reference` accepts a branch name, tag, or commit SHA. Branches become
/// local branches with HEAD attached (like `git clone -b`); tags and SHAs
/// leave HEAD detached at the resolved commit.
///
/// Blocking — callers in async context wrap this in `spawn_blocking`.
///
/// # Errors
///
/// [`GitError::Clone`] when the remote cannot be cloned;
/// [`GitError::Checkout`] when `reference` does not resolve in the clone.
pub fn clone_repository(
    url: &str,
    dest: &Path,
    reference: Option<&str>,
    auth: Option<&GitProxyAuth>,
) -> Result<(), GitError> {
    let mut builder = RepoBuilder::new();
    if let Some(auth) = auth {
        builder.fetch_options(build_fetch_options(&auth.proxy_url, &auth.header_line()));
    }
    let repo = builder.clone(url, dest).map_err(|source| GitError::Clone {
        url: url.to_string(),
        source,
    })?;
    if let Some(reference) = reference {
        checkout_reference(&repo, reference)?;
    }
    Ok(())
}

/// Fetch options that route through the proxy with the sealed-token header.
///
/// No certificate-check override is installed: the pod→proxy hop is plaintext
/// `http://` by design (the proxy needs to read the sealed header in the
/// clear), so there is no certificate to verify there. Should any hop actually
/// use TLS — an `https://` proxy or remote — git2's normal verification
/// applies rather than being silently disabled.
fn build_fetch_options(proxy_url: &str, header: &str) -> FetchOptions<'static> {
    let mut proxy = ProxyOptions::new();
    proxy.url(proxy_url);
    let mut opts = FetchOptions::new();
    opts.proxy_options(proxy);
    opts.custom_headers(&[header]);
    opts
}

/// Check out `reference` in a fresh clone.
///
/// Resolution order: direct revparse (local branch, tag, SHA, full ref),
/// then `refs/remotes/origin/<reference>` — in which case a local branch of
/// the same name is created and HEAD attached to it.
fn checkout_reference(repo: &Repository, reference: &str) -> Result<(), GitError> {
    let err = |source| GitError::Checkout {
        reference: reference.to_string(),
        source,
    };

    if let Ok((object, resolved)) = repo.revparse_ext(reference) {
        repo.checkout_tree(&object, None).map_err(err)?;
        return resolved
            .and_then(|r| r.name().ok().map(str::to_string))
            .map_or_else(
                || repo.set_head_detached(object.id()),
                |name| repo.set_head(&name),
            )
            .map_err(err);
    }

    let (object, _) = repo
        .revparse_ext(&format!("refs/remotes/origin/{reference}"))
        .map_err(err)?;
    let commit = object.peel_to_commit().map_err(err)?;
    repo.branch(reference, &commit, false).map_err(err)?;
    repo.checkout_tree(&object, None).map_err(err)?;
    repo.set_head(&format!("refs/heads/{reference}"))
        .map_err(err)
}

/// Resolve `rel` against `root`, rejecting absolute paths and `..` traversal.
///
/// `None` resolves to `root` itself.
///
/// # Errors
///
/// [`GitError::AbsoluteDest`] and [`GitError::DestTraversal`] on paths that
/// would escape `root`.
pub fn resolve_within(root: &Path, rel: Option<&str>) -> Result<PathBuf, GitError> {
    let Some(rel) = rel else {
        return Ok(root.to_path_buf());
    };
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return Err(GitError::AbsoluteDest {
            path: rel.to_string(),
        });
    }
    if rel_path
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(GitError::DestTraversal {
            path: rel.to_string(),
        });
    }
    Ok(root.join(rel_path))
}

/// Read and validate a sealed token file, mirroring the runtime's proxy
/// loader.
///
/// # Errors
///
/// [`GitError::TokenRead`] when the file cannot be read;
/// [`GitError::TokenEmpty`] when it holds only whitespace.
pub fn read_token_file(path: &Path) -> Result<SecretString, GitError> {
    let raw = std::fs::read_to_string(path).map_err(|source| GitError::TokenRead {
        path: path.to_path_buf(),
        source,
    })?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(GitError::TokenEmpty {
            path: path.to_path_buf(),
        });
    }
    Ok(SecretString::from(trimmed.to_owned()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use git2::Signature;
    use std::io::Write as _;
    use tempfile::TempDir;

    fn test_auth() -> GitProxyAuth {
        GitProxyAuth {
            proxy_url: "http://proxy.local:8080".to_string(),
            token_header: "X-Tokenizer-Token".to_string(),
            sealed_token: SecretString::from("sealed.secret-value".to_string()),
        }
    }

    fn commit_file(repo: &Repository, name: &str, content: &str, message: &str) -> git2::Oid {
        let workdir = repo.workdir().unwrap();
        std::fs::write(workdir.join(name), content).unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new(name)).unwrap();
        index.write().unwrap();
        let tree = repo.find_tree(index.write_tree().unwrap()).unwrap();
        let sig = Signature::now("Tester", "test@example.com").unwrap();
        let parent = repo.head().ok().and_then(|h| h.peel_to_commit().ok());
        let parents: Vec<&git2::Commit> = parent.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)
            .unwrap()
    }

    /// A source repo with two commits, a `v1` tag on the first, and a
    /// `feature` branch carrying an extra file.
    fn source_repo(dir: &Path) {
        let repo = Repository::init(dir).unwrap();
        let first = commit_file(&repo, "a.txt", "one\n", "first");
        let commit = repo.find_commit(first).unwrap();
        repo.tag_lightweight("v1", commit.as_object(), false)
            .unwrap();
        repo.branch("feature", &commit, false).unwrap();
        commit_file(&repo, "a.txt", "two\n", "second");
        {
            let feature = repo
                .find_branch("feature", git2::BranchType::Local)
                .unwrap();
            let tree = feature.get().peel_to_tree().unwrap();
            let sig = Signature::now("Tester", "test@example.com").unwrap();
            let parent = feature.get().peel_to_commit().unwrap();
            let mut index = repo.index().unwrap();
            let workdir = repo.workdir().unwrap();
            std::fs::write(workdir.join("feature.txt"), "feat\n").unwrap();
            index.add_path(Path::new("feature.txt")).unwrap();
            let tree_id = index.write_tree().unwrap();
            drop(tree);
            let tree = repo.find_tree(tree_id).unwrap();
            repo.commit(
                Some("refs/heads/feature"),
                &sig,
                &sig,
                "feature work",
                &tree,
                &[&parent],
            )
            .unwrap();
        }
    }

    fn file_url(dir: &Path) -> String {
        format!("file://{}", dir.display())
    }

    #[test]
    fn test_clone_default_branch() {
        let src = TempDir::new().unwrap();
        source_repo(src.path());
        let dest = TempDir::new().unwrap();
        let dest = dest.path().join("clone");

        clone_repository(&file_url(src.path()), &dest, None, None).unwrap();
        assert_eq!(
            std::fs::read_to_string(dest.join("a.txt")).unwrap(),
            "two\n"
        );
    }

    #[test]
    fn test_clone_checks_out_tag_detached() {
        let src = TempDir::new().unwrap();
        source_repo(src.path());
        let dest = TempDir::new().unwrap();
        let dest = dest.path().join("clone");

        clone_repository(&file_url(src.path()), &dest, Some("v1"), None).unwrap();
        assert_eq!(
            std::fs::read_to_string(dest.join("a.txt")).unwrap(),
            "one\n"
        );
        let repo = Repository::open(&dest).unwrap();
        assert!(repo.head_detached().unwrap());
    }

    #[test]
    fn test_clone_checks_out_remote_branch_as_local() {
        let src = TempDir::new().unwrap();
        source_repo(src.path());
        let dest = TempDir::new().unwrap();
        let dest = dest.path().join("clone");

        clone_repository(&file_url(src.path()), &dest, Some("feature"), None).unwrap();
        assert!(dest.join("feature.txt").exists());
        let repo = Repository::open(&dest).unwrap();
        let head = repo.head().unwrap();
        assert_eq!(head.name().unwrap(), "refs/heads/feature");
    }

    #[test]
    fn test_clone_missing_remote_errors() {
        let dest = TempDir::new().unwrap();
        let err = clone_repository(
            "file:///nonexistent/nowhere",
            &dest.path().join("clone"),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, GitError::Clone { .. }), "got: {err}");
    }

    #[test]
    fn test_clone_unknown_reference_errors() {
        let src = TempDir::new().unwrap();
        source_repo(src.path());
        let dest = TempDir::new().unwrap();
        let err = clone_repository(
            &file_url(src.path()),
            &dest.path().join("clone"),
            Some("no-such-ref"),
            None,
        )
        .unwrap_err();
        assert!(matches!(err, GitError::Checkout { .. }), "got: {err}");
    }

    #[test]
    fn test_resolve_within_rejects_absolute_and_traversal() {
        let root = Path::new("/work");
        assert!(matches!(
            resolve_within(root, Some("/etc/passwd")),
            Err(GitError::AbsoluteDest { .. })
        ));
        assert!(matches!(
            resolve_within(root, Some("../escape")),
            Err(GitError::DestTraversal { .. })
        ));
        assert_eq!(
            resolve_within(root, Some("repo")).unwrap(),
            PathBuf::from("/work/repo")
        );
        assert_eq!(resolve_within(root, None).unwrap(), PathBuf::from("/work"));
    }

    #[test]
    fn test_header_line_carries_sealed_token() {
        let line = test_auth().header_line();
        assert_eq!(line, "X-Tokenizer-Token: sealed.secret-value");
    }

    #[test]
    fn test_debug_redacts_sealed_token() {
        let debug = format!("{:?}", test_auth());
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("sealed.secret-value"));
    }

    #[test]
    fn test_read_token_file_rejects_empty() {
        let mut token = tempfile::NamedTempFile::new().unwrap();
        token.write_all(b"   \n").unwrap();
        let err = read_token_file(token.path()).unwrap_err();
        assert!(matches!(err, GitError::TokenEmpty { .. }), "got: {err}");
    }

    #[test]
    fn test_read_token_file_trims() {
        let mut token = tempfile::NamedTempFile::new().unwrap();
        token.write_all(b"sealed-blob\n").unwrap();
        let secret = read_token_file(token.path()).unwrap();
        assert_eq!(secret.expose_secret(), "sealed-blob");
    }

    #[test]
    fn test_read_token_file_missing_errors() {
        let err = read_token_file(Path::new("/nonexistent/token")).unwrap_err();
        assert!(matches!(err, GitError::TokenRead { .. }), "got: {err}");
    }
}
