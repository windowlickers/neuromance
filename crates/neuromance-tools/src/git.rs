//! Library-level git operations that keep remote credentials out of the pod.
//!
//! Authenticated HTTP operations route through the tokenizer proxy: a clone
//! sends the request to the proxy and carries a sealed token in a custom
//! header. The proxy reads the sealed token, injects the real git credential
//! server-side, and forwards upstream. The pod never holds the plaintext
//! credential, matching the model the LLM client uses (see
//! `neuromance-runtime/src/proxy.rs`).
//!
//! For the proxy to read the sealed header it must see the request in the
//! clear, so remotes are addressed with an `http://` scheme; the proxy
//! upgrades to TLS upstream. The proxy must understand git smart-HTTP
//! (`/info/refs?service=git-upload-pack`, `git-receive-pack`) and inject git
//! auth — that is a proxy-side capability, configured out of band.
//!
//! Proxied clones shell out to the `git` CLI — so they need a `git` binary on
//! `PATH`, which the toolkit runtime image carries and the minimal one does
//! not. Anonymous clones stay in process with libgit2 and need no binary.
//! libgit2 cannot forward-proxy an `http://` remote: it
//! writes the absolute-form request line but still opens the socket to the
//! origin host (`use_connect_proxy()` in libgit2's `httpclient.c` gates the
//! proxy connection on an `https` scheme), so the sealed header never reaches
//! the proxy and the origin answers 401. The CLI honours
//! `http.<url>.proxy` for every scheme, so the proxied path uses it.
//!
//! No credential ever reaches the git process: the proxy config and the
//! sealed-token header travel in `GIT_CONFIG_*` environment variables, not in
//! argv, and this module never supplies a git credential. A proxied clone also
//! runs with the system and global config files switched off, so no ambient
//! `credential.helper` can inject one and no `url.<base>.insteadOf` can rewrite
//! the remote out from under the `http://` check.

use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

use git2::Repository;
use git2::build::RepoBuilder;
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
    /// A proxied clone found no `git` binary on `PATH`.
    #[error(
        "cloning {url} through the tokenizer proxy needs a `git` binary on PATH \
         and found none; run the toolkit runtime image (`neuromance-toolkit`), \
         which carries git — the minimal image does not"
    )]
    GitMissing {
        /// The remote URL passed to the clone.
        url: String,
    },
    /// A proxied clone could not run the `git` CLI.
    #[error("running `git clone` for {url}: {source}")]
    CloneSpawn {
        /// The remote URL passed to the clone.
        url: String,
        #[source]
        source: std::io::Error,
    },
    /// A proxied clone was asked for a scheme the proxy cannot read.
    #[error("proxied clone needs an http:// remote, got: {url}")]
    ProxyScheme {
        /// The remote URL passed to the clone.
        url: String,
    },
    /// A proxied clone ran but `git` exited non-zero.
    #[error("cloning {url} through the tokenizer proxy: {stderr}")]
    ProxyClone {
        /// The remote URL passed to the clone.
        url: String,
        /// `git`'s stderr, with the sealed token redacted.
        stderr: String,
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
/// With `auth`, the clone runs through the `git` CLI and routes via the
/// tokenizer proxy with the sealed-token header; without it the clone is
/// anonymous and in process (public remotes, `file://` fixtures in tests).
///
/// `reference` accepts a branch name, tag, or commit SHA. Branches become
/// local branches with HEAD attached (like `git clone -b`); tags and SHAs
/// leave HEAD detached at the resolved commit.
///
/// Blocking — callers in async context wrap this in `spawn_blocking`.
///
/// # Errors
///
/// [`GitError::Clone`], [`GitError::CloneSpawn`], [`GitError::GitMissing`], or
/// [`GitError::ProxyClone`] when the remote cannot be cloned;
/// [`GitError::ProxyScheme`] when a proxied clone is given a non-`http://`
/// remote; [`GitError::Checkout`] when `reference` does not resolve in the
/// clone.
pub fn clone_repository(
    url: &str,
    dest: &Path,
    reference: Option<&str>,
    auth: Option<&GitProxyAuth>,
) -> Result<(), GitError> {
    let repo = match auth {
        Some(auth) => clone_via_cli(url, dest, auth)?,
        None => RepoBuilder::new()
            .clone(url, dest)
            .map_err(|source| GitError::Clone {
                url: url.to_string(),
                source,
            })?,
    };
    if let Some(reference) = reference {
        checkout_reference(&repo, reference)?;
    }
    Ok(())
}

/// Clone through the tokenizer proxy with the `git` CLI, then reopen the
/// result so the caller checks out `reference` the same way either path does.
///
/// The proxy URL and the sealed-token header are passed as `GIT_CONFIG_*`
/// environment variables — never argv, which any process on the host can read.
/// `GIT_CONFIG_COUNT` replaces only the *environment* half of git's config;
/// the system and global files still apply, so they are switched off
/// explicitly. Without that, an ambient `url.<base>.insteadOf` would rewrite
/// `http://` to `https://` after the scheme check — tunnelling the request
/// through CONNECT, where the proxy cannot read the sealed header — and an
/// ambient `credential.helper` could put a real credential on the cleartext
/// pod→proxy hop.
///
/// `no_proxy`/`NO_PROXY` are cleared for the same reason. Either one matching
/// the remote host makes curl skip the proxy, and `http.<scope>.extraHeader`
/// rides along regardless — so the sealed token would reach an origin server
/// that was never meant to see it.
///
/// Dropping the pod's ambient `GIT_CONFIG_*` block also drops its
/// `http.sslCAInfo`. That is fine here: this hop is plaintext `http://` by
/// construction, so there is no certificate to verify.
///
/// `GIT_HTTP_LOW_SPEED_*` bounds a transfer that stalls mid-stream. It does
/// not cover connect or a slow-but-steady trickle; curl's own connect timeout
/// bounds the former.
fn clone_via_cli(url: &str, dest: &Path, auth: &GitProxyAuth) -> Result<Repository, GitError> {
    if !url.starts_with("http://") {
        return Err(GitError::ProxyScheme {
            url: url.to_string(),
        });
    }
    let output = clone_command(url, dest, auth).output().map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            GitError::GitMissing {
                url: url.to_string(),
            }
        } else {
            GitError::CloneSpawn {
                url: url.to_string(),
                source,
            }
        }
    })?;

    if !output.status.success() {
        return Err(GitError::ProxyClone {
            url: url.to_string(),
            stderr: redact(&String::from_utf8_lossy(&output.stderr), auth),
        });
    }

    Repository::open(dest).map_err(|source| GitError::Clone {
        url: url.to_string(),
        source,
    })
}

/// The `git clone` invocation for a proxied clone, environment and all.
///
/// Split out from [`clone_via_cli`] so a test can assert the environment
/// without a network: the config-file switches are the security boundary, not
/// an optimisation.
fn clone_command(url: &str, dest: &Path, auth: &GitProxyAuth) -> Command {
    let mut command = Command::new("git");
    command
        .arg("clone")
        .arg("--")
        .arg(url)
        .arg(dest)
        .stdin(Stdio::null())
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_HTTP_LOW_SPEED_LIMIT", "1000")
        .env("GIT_HTTP_LOW_SPEED_TIME", "60")
        .env_remove("no_proxy")
        .env_remove("NO_PROXY");
    let pairs = proxy_config_pairs(url, auth);
    command.env("GIT_CONFIG_COUNT", pairs.len().to_string());
    for (i, (key, value)) in pairs.into_iter().enumerate() {
        command.env(format!("GIT_CONFIG_KEY_{i}"), key);
        command.env(format!("GIT_CONFIG_VALUE_{i}"), value);
    }
    command
}

/// The `git config` entries that route a clone of `url` through the proxy,
/// in `GIT_CONFIG_KEY_<n>` / `GIT_CONFIG_VALUE_<n>` order.
fn proxy_config_pairs(url: &str, auth: &GitProxyAuth) -> Vec<(String, String)> {
    let scope = config_scope(url);
    vec![
        (format!("http.{scope}.proxy"), auth.proxy_url.clone()),
        (format!("http.{scope}.extraHeader"), auth.header_line()),
    ]
}

/// The `http.<url>` config scope the proxy settings attach to: the remote's
/// origin (`scheme://host[:port]/`), so every smart-HTTP sub-path of the
/// clone matches. Falls back to the whole URL when it has no origin to
/// derive, which still matches itself.
fn config_scope(url: &str) -> String {
    let origin = url::Url::parse(url)
        .map(|parsed| parsed.origin().ascii_serialization())
        .unwrap_or_default();
    if origin.is_empty() || origin == "null" {
        return url.to_string();
    }
    format!("{origin}/")
}

/// Strip the sealed token from `git`'s stderr before it reaches a log or an
/// error message.
fn redact(stderr: &str, auth: &GitProxyAuth) -> String {
    stderr
        .replace(auth.sealed_token.expose_secret(), "[REDACTED]")
        .trim()
        .to_string()
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

    /// A proxied clone must open its connection to the *proxy*, carrying the
    /// sealed header — never to the origin. libgit2's `ProxyOptions` silently
    /// connected to the origin for `http://` remotes, which the origin
    /// answered with 401.
    #[test]
    fn test_proxied_clone_reaches_the_proxy_with_the_sealed_header() {
        use std::io::Read as _;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let proxy_url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, seen) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            // Read until the header block ends: a single read() is not
            // guaranteed to return the whole request.
            let mut request = Vec::new();
            let mut chunk = [0_u8; 512];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => request.extend_from_slice(&chunk[..n]),
                }
            }
            let _ = tx.send(String::from_utf8_lossy(&request).to_string());
        });

        let auth = GitProxyAuth {
            proxy_url,
            ..test_auth()
        };
        let dest = TempDir::new().unwrap();
        // Port 1 refuses connections, so reaching the origin cannot look like
        // a success or a hang — only the proxy can serve this clone.
        let err = clone_repository(
            "http://127.0.0.1:1/darkhorse/kb.git",
            &dest.path().join("clone"),
            None,
            Some(&auth),
        )
        .unwrap_err();
        assert!(matches!(err, GitError::ProxyClone { .. }), "got: {err}");

        let request = seen
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("clone never connected to the proxy");
        assert!(
            request.contains("X-Tokenizer-Token: sealed.secret-value"),
            "proxy saw no sealed header: {request}"
        );
        assert!(
            request.starts_with("GET http://127.0.0.1:1/darkhorse/kb.git/info/refs"),
            "proxy saw no absolute-form request: {request}"
        );
    }

    #[test]
    fn test_proxy_clone_error_redacts_sealed_token() {
        let auth = test_auth();
        let redacted = redact(
            "fatal: unable to access sealed.secret-value: refused",
            &auth,
        );
        assert!(!redacted.contains("sealed.secret-value"));
        assert!(redacted.contains("[REDACTED]"));
    }

    /// A proxied clone must not inherit config from the pod's filesystem: an
    /// ambient `url.<base>.insteadOf` would rewrite `http://` to `https://`
    /// after the scheme check (hiding the sealed header inside a CONNECT
    /// tunnel), and an ambient `credential.helper` would put a real credential
    /// on the cleartext pod→proxy hop.
    #[test]
    fn test_proxied_clone_ignores_ambient_git_config_files() {
        let command = clone_command(
            "http://git.local/a/b.git",
            Path::new("/tmp/clone"),
            &test_auth(),
        );
        let env: std::collections::HashMap<_, _> = command
            .get_envs()
            .filter_map(|(k, v)| {
                let value = match v {
                    Some(v) => Some(v.to_str()?),
                    None => None,
                };
                Some((k.to_str()?, value))
            })
            .collect();
        assert_eq!(env.get("GIT_CONFIG_NOSYSTEM"), Some(&Some("1")));
        assert_eq!(env.get("GIT_CONFIG_GLOBAL"), Some(&Some("/dev/null")));
        assert_eq!(env.get("GIT_TERMINAL_PROMPT"), Some(&Some("0")));
        // `None` is a removal: curl must not be told to bypass the proxy for
        // the remote host, or the sealed header reaches the origin.
        assert_eq!(env.get("no_proxy"), Some(&None));
        assert_eq!(env.get("NO_PROXY"), Some(&None));
    }

    /// The proxy settings must reach git as numbered `GIT_CONFIG_*` entries —
    /// the count agreeing with the keys — and never as argv, which is
    /// world-readable through `/proc`.
    #[test]
    fn test_proxied_clone_passes_config_in_env_not_argv() {
        let url = "http://git.local/a/b.git";
        let auth = test_auth();
        assert_eq!(
            proxy_config_pairs(url, &auth),
            vec![
                (
                    "http.http://git.local/.proxy".to_string(),
                    "http://proxy.local:8080".to_string()
                ),
                (
                    "http.http://git.local/.extraHeader".to_string(),
                    "X-Tokenizer-Token: sealed.secret-value".to_string()
                ),
            ]
        );

        let command = clone_command(url, Path::new("/tmp/clone"), &auth);
        let env: std::collections::HashMap<_, _> = command
            .get_envs()
            .filter_map(|(k, v)| Some((k.to_str()?, v?.to_str()?)))
            .collect();
        assert_eq!(env.get("GIT_CONFIG_COUNT"), Some(&"2"));
        assert_eq!(
            env.get("GIT_CONFIG_VALUE_1"),
            Some(&"X-Tokenizer-Token: sealed.secret-value")
        );
        let argv: Vec<_> = command.get_args().filter_map(|a| a.to_str()).collect();
        assert!(
            !argv.iter().any(|a| a.contains("sealed.secret-value")),
            "sealed token leaked into argv: {argv:?}"
        );
    }

    #[test]
    fn test_config_scope_is_the_remote_origin() {
        assert_eq!(
            config_scope("http://git.windowlicke.rs/darkhorse/kb.git"),
            "http://git.windowlicke.rs/"
        );
        assert_eq!(
            config_scope("http://git.local:3000/a/b.git"),
            "http://git.local:3000/"
        );
        assert_eq!(config_scope("not a url"), "not a url");
    }

    #[test]
    fn test_proxied_clone_rejects_a_non_http_remote() {
        let dest = TempDir::new().unwrap();
        for url in [
            "https://git.windowlicke.rs/darkhorse/kb.git",
            "ssh://git@git.windowlicke.rs:2222/darkhorse/kb.git",
        ] {
            let err = clone_repository(url, &dest.path().join("clone"), None, Some(&test_auth()))
                .unwrap_err();
            assert!(matches!(err, GitError::ProxyScheme { .. }), "got: {err}");
        }
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
