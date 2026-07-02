//! Workspace seeding: git clones and object-archive unpacks into a fresh
//! workspace directory.

use std::path::Path;
use std::sync::Arc;

use object_store::ObjectStore;
use tracing::info;

use neuromance_tools::git::{GitProxyAuth, clone_repository, resolve_within};

use crate::config::WorkspaceDefinition;

use super::{WorkspaceError, archive};

/// Run every seed in `def` against `dir` (the not-yet-published workspace).
///
/// Git seeds clone through the tokenizer proxy when `git_proxy` is set,
/// anonymously otherwise. Object seeds fetch tar.zst archives from `storage`
/// (validated present at config load when object seeds exist).
pub(super) async fn seed(
    def: &WorkspaceDefinition,
    dir: &Path,
    git_proxy: Option<&GitProxyAuth>,
    storage: Option<&Arc<dyn ObjectStore>>,
) -> Result<(), WorkspaceError> {
    for git in &def.git {
        let dest_rel = git_dest(git)?;
        let dest =
            resolve_within(dir, Some(dest_rel)).map_err(|source| WorkspaceError::SeedDest {
                dest: dest_rel.to_string(),
                source,
            })?;
        let url = git.url.clone();
        let reference = git.reference.clone();
        let auth = git_proxy.cloned();
        info!(url = %url, dest = %dest.display(), "seeding workspace: git clone");
        tokio::task::spawn_blocking(move || {
            clone_repository(&url, &dest, reference.as_deref(), auth.as_ref())
        })
        .await
        .map_err(|e| WorkspaceError::SeedTask(e.to_string()))?
        .map_err(|source| WorkspaceError::Git {
            url: git.url.clone(),
            source,
        })?;
    }

    for object in &def.objects {
        let storage = storage.ok_or_else(|| {
            WorkspaceError::SeedTask(format!(
                "object seed '{}' requires [workspace.storage], which is not configured",
                object.key
            ))
        })?;
        let dest = resolve_within(dir, object.dest.as_deref()).map_err(|source| {
            WorkspaceError::SeedDest {
                dest: object.dest.clone().unwrap_or_default(),
                source,
            }
        })?;
        info!(key = %object.key, dest = %dest.display(), "seeding workspace: object archive");
        let bytes = storage
            .get(&object_store::path::Path::from(object.key.as_str()))
            .await
            .map_err(|source| WorkspaceError::Object {
                key: object.key.clone(),
                source,
            })?
            .bytes()
            .await
            .map_err(|source| WorkspaceError::Object {
                key: object.key.clone(),
                source,
            })?;
        tokio::fs::create_dir_all(&dest)
            .await
            .map_err(|source| WorkspaceError::Create {
                path: dest.clone(),
                source,
            })?;
        tokio::task::spawn_blocking(move || archive::unpack(bytes.as_ref(), &dest))
            .await
            .map_err(|e| WorkspaceError::SeedTask(e.to_string()))??;
    }

    Ok(())
}

/// A git seed's destination: the configured `dest`, defaulting to the
/// repository basename with any `.git` suffix stripped.
fn git_dest(seed: &crate::config::GitSeed) -> Result<&str, WorkspaceError> {
    if let Some(dest) = seed.dest.as_deref() {
        return Ok(dest);
    }
    let basename = seed
        .url
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .map(|name| name.trim_end_matches(".git"))
        .filter(|name| !name.is_empty());
    basename.ok_or_else(|| {
        WorkspaceError::SeedTask(format!(
            "git seed url '{}' has no path basename to derive a dest from; set dest explicitly",
            seed.url
        ))
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::config::GitSeed;

    fn seed_for(url: &str, dest: Option<&str>) -> GitSeed {
        GitSeed {
            url: url.to_string(),
            reference: None,
            dest: dest.map(str::to_string),
        }
    }

    #[test]
    fn test_git_dest_prefers_explicit_dest() {
        let seed = seed_for("http://git.example/org/repo.git", Some("checkout"));
        assert_eq!(git_dest(&seed).unwrap(), "checkout");
    }

    #[test]
    fn test_git_dest_derives_basename_stripping_git_suffix() {
        let seed = seed_for("http://git.example/org/repo.git", None);
        assert_eq!(git_dest(&seed).unwrap(), "repo");
        let seed = seed_for("http://git.example/org/tools", None);
        assert_eq!(git_dest(&seed).unwrap(), "tools");
    }

    #[test]
    fn test_git_dest_rejects_url_without_basename() {
        let seed = seed_for("http://git.example", None);
        // "git.example" is the last segment of a bare host URL — it derives,
        // which is odd but harmless; a truly empty path must error.
        assert!(git_dest(&seed).is_ok());
        let seed = seed_for("/", None);
        assert!(git_dest(&seed).is_err());
    }
}
