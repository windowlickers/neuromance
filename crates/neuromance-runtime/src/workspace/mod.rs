//! Per-conversation workspaces.
//!
//! A workspace is a directory at `<root>/<conversation_id>` where a task's
//! tools do their file work. The root is a volume shared with the sandbox
//! container (a k8s `emptyDir`), so the orchestrator — which holds the
//! credentials — can seed and snapshot what the sandbox executes against.
//!
//! [`WorkspaceManager`] is the mechanism only: it creates, seeds, and hands
//! out directories. Policy — which definition applies to a task — lives in
//! [`crate::config::RuntimeConfig::select_workspace`] and the request-level
//! override, resolved by the caller.

pub mod archive;
mod seed;
pub mod tool;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use tokio::sync::Mutex;
use tracing::warn;
use uuid::Uuid;

use neuromance_tools::git::{GitError, GitProxyAuth, read_token_file};

use crate::config::{WorkspaceDefinition, WorkspaceSettings, WorkspaceStorageSettings};
use crate::error::RuntimeError;

pub use archive::ArchiveError;

/// Name of the workspace-internal metadata directory. Holds the seed marker
/// (and, with persistence, the snapshot marker); always excluded from
/// snapshots.
const META_DIR: &str = ".neuromance";
/// Marker file recording which definition seeded this workspace.
const DEFINITION_MARKER: &str = "workspace";

#[derive(Debug, thiserror::Error)]
pub enum WorkspaceError {
    #[error("creating workspace dir {}: {source}", path.display())]
    Create {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("publishing workspace dir {}: {source}", path.display())]
    Publish {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("git seed {url}: {source}")]
    Git {
        url: String,
        #[source]
        source: GitError,
    },
    #[error("seed dest '{dest}': {source}")]
    SeedDest {
        dest: String,
        #[source]
        source: GitError,
    },
    #[error("object seed '{key}': {source}")]
    Object {
        key: String,
        #[source]
        source: object_store::Error,
    },
    #[error(transparent)]
    Archive(#[from] ArchiveError),
    /// A blocking seed task failed to run (join error) or hit an invariant
    /// the config layer should have rejected.
    #[error("workspace seeding: {0}")]
    SeedTask(String),
}

/// Creates and seeds per-conversation workspace directories.
pub struct WorkspaceManager {
    root: PathBuf,
    git_proxy: Option<GitProxyAuth>,
    storage: Option<Arc<dyn ObjectStore>>,
    /// Per-conversation guard so concurrent prepares of the same workspace
    /// serialize. The worker is serial today; this is cheap insurance for
    /// when it stops being so.
    locks: DashMap<Uuid, Arc<Mutex<()>>>,
}

impl WorkspaceManager {
    /// Build a manager from resolved parts. Prefer [`Self::from_config`]
    /// outside tests.
    #[must_use]
    pub fn new(
        root: PathBuf,
        git_proxy: Option<GitProxyAuth>,
        storage: Option<Arc<dyn ObjectStore>>,
    ) -> Self {
        Self {
            root,
            git_proxy,
            storage,
            locks: DashMap::new(),
        }
    }

    /// Build from the `[workspace]` config section, resolving credentials
    /// once, at startup — a missing env var or unreadable token file fails
    /// the boot rather than the first task.
    ///
    /// # Errors
    /// Returns [`RuntimeError::MissingEnv`] for absent credential env vars
    /// and [`RuntimeError::Config`] for an unusable storage or git-proxy
    /// configuration.
    pub fn from_config(settings: &WorkspaceSettings) -> Result<Self, RuntimeError> {
        let storage = settings
            .storage
            .as_ref()
            .map(build_object_store)
            .transpose()?;
        let git_proxy = settings
            .git_proxy
            .as_ref()
            .map(|proxy| {
                let sealed_token = read_token_file(&proxy.token_file)
                    .map_err(|e| RuntimeError::Config(format!("workspace.git_proxy: {e}")))?;
                Ok::<_, RuntimeError>(GitProxyAuth {
                    proxy_url: proxy.base_url.clone(),
                    token_header: proxy.token_header.clone(),
                    sealed_token,
                })
            })
            .transpose()?;
        Ok(Self::new(settings.root.clone(), git_proxy, storage))
    }

    /// The workspace directory for a conversation.
    #[must_use]
    pub fn dir_for(&self, conversation_id: Uuid) -> PathBuf {
        self.root.join(conversation_id.to_string())
    }

    /// Ensure the conversation's workspace exists, seeding it from
    /// `definition` when it is first created.
    ///
    /// An existing directory is returned as-is — seeding runs exactly once
    /// per conversation. Creation is crash-safe: the tree is built in a
    /// sibling temp dir and renamed into place, so a torn earlier attempt is
    /// discarded rather than half-reused.
    ///
    /// # Errors
    /// Returns a [`WorkspaceError`] when the directory cannot be created or
    /// a seed fails; the workspace is not published in that case.
    pub async fn prepare(
        &self,
        conversation_id: Uuid,
        definition: Option<&WorkspaceDefinition>,
    ) -> Result<PathBuf, WorkspaceError> {
        let lock = self.locks.entry(conversation_id).or_default().clone();
        let _guard = lock.lock().await;

        let dir = self.dir_for(conversation_id);
        if tokio::fs::try_exists(&dir).await.unwrap_or(false) {
            self.warn_on_definition_mismatch(&dir, definition).await;
            return Ok(dir);
        }

        let staging = self.root.join(format!(".tmp-{conversation_id}"));
        if tokio::fs::try_exists(&staging).await.unwrap_or(false) {
            tokio::fs::remove_dir_all(&staging)
                .await
                .map_err(|source| WorkspaceError::Create {
                    path: staging.clone(),
                    source,
                })?;
        }
        tokio::fs::create_dir_all(staging.join(META_DIR))
            .await
            .map_err(|source| WorkspaceError::Create {
                path: staging.clone(),
                source,
            })?;

        if let Some(def) = definition {
            seed::seed(
                def,
                &staging,
                self.git_proxy.as_ref(),
                self.storage.as_ref(),
            )
            .await?;
            tokio::fs::write(staging.join(META_DIR).join(DEFINITION_MARKER), &def.name)
                .await
                .map_err(|source| WorkspaceError::Create {
                    path: staging.clone(),
                    source,
                })?;
        }

        tokio::fs::rename(&staging, &dir)
            .await
            .map_err(|source| WorkspaceError::Publish {
                path: dir.clone(),
                source,
            })?;
        Ok(dir)
    }

    /// Log when a continuation asks for a different definition than the one
    /// that seeded the workspace. Seeding is first-creation-only by design,
    /// so the request is ignored — but silently would be confusing.
    async fn warn_on_definition_mismatch(
        &self,
        dir: &Path,
        definition: Option<&WorkspaceDefinition>,
    ) {
        let Some(requested) = definition else {
            return;
        };
        let seeded = tokio::fs::read_to_string(dir.join(META_DIR).join(DEFINITION_MARKER))
            .await
            .ok();
        if seeded.as_deref() != Some(requested.name.as_str()) {
            warn!(
                workspace = %dir.display(),
                seeded = seeded.as_deref().unwrap_or("<none>"),
                requested = %requested.name,
                "workspace already exists; seed definition ignored (seeding is first-creation-only)"
            );
        }
    }
}

/// Build the garage/S3 client, reading credentials from the environment.
fn build_object_store(
    storage: &WorkspaceStorageSettings,
) -> Result<Arc<dyn ObjectStore>, RuntimeError> {
    let access_key = std::env::var(&storage.access_key_id_env)
        .map_err(|_| RuntimeError::MissingEnv(storage.access_key_id_env.clone()))?;
    let secret_key = std::env::var(&storage.secret_access_key_env)
        .map_err(|_| RuntimeError::MissingEnv(storage.secret_access_key_env.clone()))?;
    let store = AmazonS3Builder::new()
        .with_endpoint(&storage.endpoint)
        .with_bucket_name(&storage.bucket)
        .with_region(&storage.region)
        .with_access_key_id(access_key)
        .with_secret_access_key(secret_key)
        // Garage speaks path-style; virtual-hosted style would mangle the
        // bucket into the hostname.
        .with_virtual_hosted_style_request(false)
        .with_allow_http(storage.endpoint.starts_with("http://"))
        .build()
        .map_err(|e| {
            RuntimeError::Config(format!(
                "workspace.storage endpoint '{}': {e}",
                storage.endpoint
            ))
        })?;
    Ok(Arc::new(store))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::config::{GitSeed, ObjectSeed};
    use globset::GlobSet;
    use object_store::memory::InMemory;
    use tempfile::TempDir;

    fn definition(name: &str) -> WorkspaceDefinition {
        WorkspaceDefinition {
            name: name.to_string(),
            providers: Vec::new(),
            models: Vec::new(),
            git: Vec::new(),
            objects: Vec::new(),
        }
    }

    fn manager(root: &Path) -> WorkspaceManager {
        WorkspaceManager::new(root.to_path_buf(), None, None)
    }

    #[tokio::test]
    async fn test_prepare_creates_dir_and_meta() {
        let root = TempDir::new().unwrap();
        let mgr = manager(root.path());
        let id = Uuid::new_v4();

        let dir = mgr.prepare(id, None).await.unwrap();
        assert_eq!(dir, root.path().join(id.to_string()));
        assert!(dir.join(META_DIR).is_dir());
        assert!(!dir.join(META_DIR).join(DEFINITION_MARKER).exists());
    }

    #[tokio::test]
    async fn test_prepare_seeds_once_and_preserves_mutations() {
        let root = TempDir::new().unwrap();
        // Local git fixture to seed from.
        let src = TempDir::new().unwrap();
        init_repo_with_file(src.path(), "seed.txt", "seeded");

        let mut def = definition("dev");
        def.git.push(GitSeed {
            url: format!("file://{}", src.path().display()),
            reference: None,
            dest: Some("repo".to_string()),
        });

        let mgr = manager(root.path());
        let id = Uuid::new_v4();
        let dir = mgr.prepare(id, Some(&def)).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("repo/seed.txt")).unwrap(),
            "seeded"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join(META_DIR).join(DEFINITION_MARKER)).unwrap(),
            "dev"
        );

        // Mutate, then prepare again: the mutation must survive (no re-seed).
        std::fs::write(dir.join("repo/seed.txt"), "mutated").unwrap();
        let again = mgr.prepare(id, Some(&def)).await.unwrap();
        assert_eq!(again, dir);
        assert_eq!(
            std::fs::read_to_string(dir.join("repo/seed.txt")).unwrap(),
            "mutated"
        );
    }

    #[tokio::test]
    async fn test_prepare_discards_stale_staging_dir() {
        let root = TempDir::new().unwrap();
        let mgr = manager(root.path());
        let id = Uuid::new_v4();

        // A torn earlier attempt left junk in the staging dir.
        let staging = root.path().join(format!(".tmp-{id}"));
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("junk.txt"), "torn").unwrap();

        let dir = mgr.prepare(id, None).await.unwrap();
        assert!(!dir.join("junk.txt").exists());
        assert!(!staging.exists());
    }

    #[tokio::test]
    async fn test_prepare_unpacks_object_seed_from_storage() {
        // Pack a fixture archive and place it in the in-memory store.
        let fixture = TempDir::new().unwrap();
        std::fs::write(fixture.path().join("goodies.txt"), "prefilled").unwrap();
        let mut archive_bytes = Vec::new();
        archive::pack(fixture.path(), &mut archive_bytes, &GlobSet::empty()).unwrap();

        let storage: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        futures_put(&storage, "seeds/scratch.tar.zst", archive_bytes).await;

        let mut def = definition("dev");
        def.objects.push(ObjectSeed {
            key: "seeds/scratch.tar.zst".to_string(),
            dest: Some("scratch".to_string()),
        });

        let root = TempDir::new().unwrap();
        let mgr = WorkspaceManager::new(root.path().to_path_buf(), None, Some(storage));
        let dir = mgr.prepare(Uuid::new_v4(), Some(&def)).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("scratch/goodies.txt")).unwrap(),
            "prefilled"
        );
    }

    #[tokio::test]
    async fn test_prepare_failed_seed_publishes_nothing() {
        let root = TempDir::new().unwrap();
        let mut def = definition("dev");
        def.git.push(GitSeed {
            url: "file:///nonexistent/nowhere".to_string(),
            reference: None,
            dest: Some("repo".to_string()),
        });

        let mgr = manager(root.path());
        let id = Uuid::new_v4();
        let err = mgr.prepare(id, Some(&def)).await.unwrap_err();
        assert!(matches!(err, WorkspaceError::Git { .. }), "got: {err}");
        assert!(
            !root.path().join(id.to_string()).exists(),
            "failed seed must not publish the workspace"
        );
    }

    async fn futures_put(storage: &Arc<dyn ObjectStore>, key: &str, bytes: Vec<u8>) {
        storage
            .put(&object_store::path::Path::from(key), bytes.into())
            .await
            .unwrap();
    }

    fn init_repo_with_file(dir: &Path, name: &str, content: &str) {
        let repo = git2::Repository::init(dir).unwrap();
        std::fs::write(dir.join(name), content).unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new(name)).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Tester", "test@example.com").unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[])
            .unwrap();
    }
}
