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
mod store;
pub mod tool;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use globset::{Glob, GlobSet, GlobSetBuilder};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

use neuromance_db::PgConversationStore;
use neuromance_tools::git::{GitError, GitProxyAuth, read_token_file};

use crate::config::{WorkspaceDefinition, WorkspaceSettings, WorkspaceStorageSettings};
use crate::error::RuntimeError;

pub use archive::ArchiveError;
pub use store::{InMemorySnapshotRefs, PgSnapshotRefs, SnapshotRef, SnapshotRefStore};

/// Name of the workspace-internal metadata directory. Holds the seed marker
/// (and, with persistence, the snapshot marker); always excluded from
/// snapshots.
const META_DIR: &str = ".neuromance";
/// Marker file recording which definition seeded this workspace.
const DEFINITION_MARKER: &str = "workspace";
/// Marker file holding the `ETag` of the snapshot this directory was
/// restored-from or last uploaded-as. Compared against the remote object's
/// `ETag` to skip a redundant restore when a continuation lands on the replica
/// that produced the snapshot. Lives inside the dir, so a torn dir has no
/// marker and always restores.
const SNAPSHOT_MARKER: &str = "snapshot-etag";

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

/// Snapshot/restore machinery, present when `[workspace.persistence]` is
/// configured.
struct PersistenceRuntime {
    /// Object key prefix; a conversation's snapshot lives at
    /// `<prefix><conversation_id>.tar.zst`.
    prefix: String,
    /// Paths excluded from snapshots (user globs plus the metadata dir).
    exclude: GlobSet,
    /// Latest-snapshot bookkeeping (postgres write-through when a database
    /// is configured; in-memory otherwise).
    refs: Arc<dyn SnapshotRefStore>,
}

impl PersistenceRuntime {
    fn key_for(&self, conversation_id: Uuid) -> String {
        format!("{}{conversation_id}.tar.zst", self.prefix)
    }
}

/// Creates, seeds, and (optionally) snapshots per-conversation workspace
/// directories.
pub struct WorkspaceManager {
    root: PathBuf,
    git_proxy: Option<GitProxyAuth>,
    storage: Option<Arc<dyn ObjectStore>>,
    persistence: Option<PersistenceRuntime>,
    /// Striped guards so concurrent prepares/snapshots of the same workspace
    /// serialize. The worker is serial today; this is cheap insurance for when
    /// it stops being so. A conversation always maps to the same stripe, so
    /// same-workspace operations serialize; distinct conversations may share a
    /// stripe and serialize needlessly (harmless), which keeps the pool a fixed
    /// size instead of growing one entry per conversation forever.
    locks: [Mutex<()>; LOCK_STRIPES],
}

/// Number of stripes in [`WorkspaceManager`]'s lock pool.
const LOCK_STRIPES: usize = 64;

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
            persistence: None,
            locks: std::array::from_fn(|_| Mutex::new(())),
        }
    }

    /// The stripe guarding a conversation's prepare/snapshot.
    const fn lock_for(&self, conversation_id: Uuid) -> &Mutex<()> {
        let idx = (conversation_id.as_u128() % LOCK_STRIPES as u128) as usize;
        &self.locks[idx]
    }

    /// Enable snapshot/restore against the manager's storage. `exclude` is
    /// applied on top of the always-excluded workspace metadata dir.
    #[must_use]
    pub fn with_persistence(
        mut self,
        prefix: String,
        exclude: GlobSet,
        refs: Arc<dyn SnapshotRefStore>,
    ) -> Self {
        self.persistence = Some(PersistenceRuntime {
            prefix,
            exclude,
            refs,
        });
        self
    }

    /// Whether snapshot/restore is configured.
    #[must_use]
    pub const fn persistence_enabled(&self) -> bool {
        self.persistence.is_some()
    }

    /// Build from the `[workspace]` config section, resolving credentials
    /// once, at startup — a missing env var or unreadable token file fails
    /// the boot rather than the first task. `store`, when present, makes
    /// snapshot refs durable across replicas.
    ///
    /// # Errors
    /// Returns [`RuntimeError::MissingEnv`] for absent credential env vars
    /// and [`RuntimeError::Config`] for an unusable storage or git-proxy
    /// configuration.
    pub fn from_config(
        settings: &WorkspaceSettings,
        store: Option<&Arc<PgConversationStore>>,
    ) -> Result<Self, RuntimeError> {
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
        let mut manager = Self::new(settings.root.clone(), git_proxy, storage);
        if let Some(persistence) = &settings.persistence {
            let exclude = build_exclusions(&persistence.exclude)?;
            let refs: Arc<dyn SnapshotRefStore> = match store {
                Some(store) => Arc::new(PgSnapshotRefs::new(Arc::clone(store))),
                None => Arc::new(InMemorySnapshotRefs::default()),
            };
            manager = manager.with_persistence(persistence.prefix.clone(), exclude, refs);
        }
        Ok(manager)
    }

    /// The workspace directory for a conversation.
    #[must_use]
    pub fn dir_for(&self, conversation_id: Uuid) -> PathBuf {
        self.root.join(conversation_id.to_string())
    }

    /// Ensure the conversation's workspace exists and is current, seeding it
    /// from `definition` when it is first created.
    ///
    /// With persistence configured, a continuation restores the latest
    /// snapshot from storage — unless the local dir's marker matches the
    /// remote `ETag`, meaning this replica produced (or already restored) that
    /// snapshot and the dir is intact. A restored workspace is never
    /// re-seeded. Without persistence an existing directory is returned
    /// as-is.
    ///
    /// Creation and restore are crash-safe: the tree is built in a sibling
    /// temp dir and renamed into place, so a torn earlier attempt is
    /// discarded rather than half-reused.
    ///
    /// # Errors
    /// Returns a [`WorkspaceError`] when the directory cannot be created, a
    /// seed fails, or the snapshot cannot be fetched; the workspace is not
    /// published in that case.
    pub async fn prepare(
        &self,
        conversation_id: Uuid,
        definition: Option<&WorkspaceDefinition>,
    ) -> Result<PathBuf, WorkspaceError> {
        let _guard = self.lock_for(conversation_id).lock().await;

        let dir = self.dir_for(conversation_id);
        let exists = tokio::fs::try_exists(&dir).await.unwrap_or(false);

        let remote_etag = match &self.persistence {
            Some(persistence) => {
                self.head_snapshot(&persistence.key_for(conversation_id))
                    .await?
            }
            None => None,
        };

        // A snapshot exists: restore unless this dir already carries it.
        if let Some(etag) = remote_etag {
            if exists && read_marker(&dir, SNAPSHOT_MARKER).await.as_deref() == Some(&*etag) {
                return Ok(dir);
            }
            if exists {
                info!(
                    conversation_id = %conversation_id,
                    "local workspace is stale; restoring from snapshot"
                );
            }
            self.restore(conversation_id, &dir, &etag).await?;
            return Ok(dir);
        }

        // No snapshot (or no persistence): keep an existing dir, else create
        // fresh and seed.
        if exists {
            self.warn_on_definition_mismatch(&dir, definition).await;
            return Ok(dir);
        }
        self.create_fresh(conversation_id, &dir, definition).await?;
        Ok(dir)
    }

    /// Snapshot the conversation's workspace to storage: tar.zst with the
    /// configured exclusions, uploaded to the snapshot key, marker and ref
    /// updated. No-op when persistence is off or the dir does not exist.
    ///
    /// # Errors
    /// Returns a [`WorkspaceError`] when packing or the upload fails.
    pub async fn snapshot(&self, conversation_id: Uuid) -> Result<(), WorkspaceError> {
        let Some(persistence) = &self.persistence else {
            return Ok(());
        };
        let storage = self.storage()?;
        let _guard = self.lock_for(conversation_id).lock().await;

        let dir = self.dir_for(conversation_id);
        if !tokio::fs::try_exists(&dir).await.unwrap_or(false) {
            return Ok(());
        }

        let pack_dir = dir.clone();
        let exclude = persistence.exclude.clone();
        let bytes = tokio::task::spawn_blocking(move || {
            let mut buf = Vec::new();
            archive::pack(&pack_dir, &mut buf, &exclude)?;
            Ok::<_, ArchiveError>(buf)
        })
        .await
        .map_err(|e| WorkspaceError::SeedTask(e.to_string()))??;

        let key = persistence.key_for(conversation_id);
        let size_bytes = i64::try_from(bytes.len()).unwrap_or(i64::MAX);
        let result = storage
            .put(&object_store::path::Path::from(key.as_str()), bytes.into())
            .await
            .map_err(|source| WorkspaceError::Object {
                key: key.clone(),
                source,
            })?;

        write_marker(
            &dir,
            SNAPSHOT_MARKER,
            result.e_tag.as_deref().unwrap_or_default(),
        )
        .await;
        persistence
            .refs
            .record_snapshot(
                conversation_id,
                &SnapshotRef {
                    object_key: key.clone(),
                    etag: result.e_tag,
                    size_bytes,
                    snapshotted_at: Utc::now(),
                },
            )
            .await;
        info!(%conversation_id, key, size_bytes, "workspace snapshot uploaded");
        Ok(())
    }

    /// The `ETag` of the conversation's remote snapshot, or `None` when no
    /// object exists. A backend that reports no `ETag` maps to an empty string,
    /// which never matches a real marker — conservatively forcing a restore.
    async fn head_snapshot(&self, key: &str) -> Result<Option<String>, WorkspaceError> {
        let storage = self.storage()?;
        match storage.head(&object_store::path::Path::from(key)).await {
            Ok(meta) => Ok(Some(meta.e_tag.unwrap_or_default())),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(source) => Err(WorkspaceError::Object {
                key: key.to_string(),
                source,
            }),
        }
    }

    /// Fetch the snapshot and swap it into place: unpack into a staging dir,
    /// stamp the marker, then replace any existing dir atomically.
    async fn restore(
        &self,
        conversation_id: Uuid,
        dir: &Path,
        etag: &str,
    ) -> Result<(), WorkspaceError> {
        let Some(persistence) = &self.persistence else {
            return Ok(());
        };
        let storage = self.storage()?;
        let key = persistence.key_for(conversation_id);
        let bytes = storage
            .get(&object_store::path::Path::from(key.as_str()))
            .await
            .map_err(|source| WorkspaceError::Object {
                key: key.clone(),
                source,
            })?
            .bytes()
            .await
            .map_err(|source| WorkspaceError::Object {
                key: key.clone(),
                source,
            })?;

        let staging = self.fresh_staging(conversation_id).await?;
        let unpack_dir = staging.clone();
        tokio::task::spawn_blocking(move || archive::unpack(bytes.as_ref(), &unpack_dir))
            .await
            .map_err(|e| WorkspaceError::SeedTask(e.to_string()))??;
        // The metadata dir is excluded from snapshots, so recreate it.
        tokio::fs::create_dir_all(staging.join(META_DIR))
            .await
            .map_err(|source| WorkspaceError::Create {
                path: staging.clone(),
                source,
            })?;
        write_marker(&staging, SNAPSHOT_MARKER, etag).await;

        if tokio::fs::try_exists(dir).await.unwrap_or(false) {
            tokio::fs::remove_dir_all(dir)
                .await
                .map_err(|source| WorkspaceError::Publish {
                    path: dir.to_path_buf(),
                    source,
                })?;
        }
        tokio::fs::rename(&staging, dir)
            .await
            .map_err(|source| WorkspaceError::Publish {
                path: dir.to_path_buf(),
                source,
            })?;
        Ok(())
    }

    /// Create a fresh workspace in staging, run seeds, and publish it.
    async fn create_fresh(
        &self,
        conversation_id: Uuid,
        dir: &Path,
        definition: Option<&WorkspaceDefinition>,
    ) -> Result<(), WorkspaceError> {
        let staging = self.fresh_staging(conversation_id).await?;
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

        tokio::fs::rename(&staging, dir)
            .await
            .map_err(|source| WorkspaceError::Publish {
                path: dir.to_path_buf(),
                source,
            })?;
        Ok(())
    }

    /// An empty staging dir for this conversation, discarding any torn
    /// leftover from an earlier attempt.
    async fn fresh_staging(&self, conversation_id: Uuid) -> Result<PathBuf, WorkspaceError> {
        let staging = self.root.join(format!(".tmp-{conversation_id}"));
        if tokio::fs::try_exists(&staging).await.unwrap_or(false) {
            tokio::fs::remove_dir_all(&staging)
                .await
                .map_err(|source| WorkspaceError::Create {
                    path: staging.clone(),
                    source,
                })?;
        }
        tokio::fs::create_dir_all(&staging)
            .await
            .map_err(|source| WorkspaceError::Create {
                path: staging.clone(),
                source,
            })?;
        Ok(staging)
    }

    /// The object store, which persistence and object seeds require. Config
    /// validation enforces the pairing; this is the typed backstop.
    fn storage(&self) -> Result<&Arc<dyn ObjectStore>, WorkspaceError> {
        self.storage.as_ref().ok_or_else(|| {
            WorkspaceError::SeedTask(
                "[workspace.persistence] requires [workspace.storage]".to_string(),
            )
        })
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

/// Read a marker file from the workspace metadata dir. Missing or empty
/// reads as `None`.
async fn read_marker(dir: &Path, name: &str) -> Option<String> {
    let value = tokio::fs::read_to_string(dir.join(META_DIR).join(name))
        .await
        .ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Write a marker file into the workspace metadata dir. Best-effort: a marker
/// that fails to write only costs a redundant restore later.
async fn write_marker(dir: &Path, name: &str, value: &str) {
    let path = dir.join(META_DIR).join(name);
    if let Err(e) = tokio::fs::write(&path, value).await {
        warn!(path = %path.display(), error = %e, "failed to write workspace marker");
    }
}

/// Compile snapshot exclusion globs: the user's patterns plus the workspace
/// metadata dir, which never belongs in a snapshot.
fn build_exclusions(patterns: &[String]) -> Result<GlobSet, RuntimeError> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns
        .iter()
        .map(String::as_str)
        .chain([META_DIR, ".neuromance/**"])
    {
        builder.add(Glob::new(pattern).map_err(|e| {
            RuntimeError::Config(format!(
                "workspace.persistence.exclude pattern '{pattern}' is not a valid glob: {e}"
            ))
        })?);
    }
    builder
        .build()
        .map_err(|e| RuntimeError::Config(format!("workspace.persistence.exclude: {e}")))
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

    /// A persistence-enabled manager over `storage` and `refs`, rooted at
    /// `root`, excluding `target/**` from snapshots.
    fn persistent_manager(
        root: &Path,
        storage: &Arc<dyn ObjectStore>,
        refs: &Arc<InMemorySnapshotRefs>,
    ) -> WorkspaceManager {
        WorkspaceManager::new(root.to_path_buf(), None, Some(Arc::clone(storage))).with_persistence(
            "snapshots/".to_string(),
            build_exclusions(&["target/**".to_string()]).unwrap(),
            Arc::clone(refs) as Arc<dyn SnapshotRefStore>,
        )
    }

    #[tokio::test]
    async fn test_snapshot_restores_on_another_replica() {
        let storage: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let refs = Arc::new(InMemorySnapshotRefs::default());
        let id = Uuid::new_v4();

        // Replica A: create, work, snapshot (with an excluded cache dir).
        let root_a = TempDir::new().unwrap();
        let mgr_a = persistent_manager(root_a.path(), &storage, &refs);
        let dir_a = mgr_a.prepare(id, None).await.unwrap();
        std::fs::write(dir_a.join("work.txt"), "progress").unwrap();
        std::fs::create_dir_all(dir_a.join("target/debug")).unwrap();
        std::fs::write(dir_a.join("target/debug/huge.o"), "x").unwrap();
        mgr_a.snapshot(id).await.unwrap();

        let recorded = refs.get_snapshot(id).await.expect("ref recorded");
        assert_eq!(recorded.object_key, format!("snapshots/{id}.tar.zst"));
        assert!(recorded.size_bytes > 0);

        // Replica B: same storage, different pod-local root.
        let root_b = TempDir::new().unwrap();
        let mgr_b = persistent_manager(root_b.path(), &storage, &refs);
        let dir_b = mgr_b.prepare(id, None).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir_b.join("work.txt")).unwrap(),
            "progress"
        );
        assert!(
            !dir_b.join("target").exists(),
            "excluded dirs must not travel in snapshots"
        );
    }

    #[tokio::test]
    async fn test_prepare_skips_restore_when_marker_current() {
        let storage: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let refs = Arc::new(InMemorySnapshotRefs::default());
        let root = TempDir::new().unwrap();
        let mgr = persistent_manager(root.path(), &storage, &refs);
        let id = Uuid::new_v4();

        let dir = mgr.prepare(id, None).await.unwrap();
        std::fs::write(dir.join("work.txt"), "v1").unwrap();
        mgr.snapshot(id).await.unwrap();

        // Same replica continues: post-snapshot local mutations must survive
        // because the marker matches the remote `ETag` (no restore).
        std::fs::write(dir.join("work.txt"), "v2-local").unwrap();
        mgr.prepare(id, None).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("work.txt")).unwrap(),
            "v2-local"
        );
    }

    #[tokio::test]
    async fn test_prepare_restores_when_marker_stale_or_missing() {
        let storage: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let refs = Arc::new(InMemorySnapshotRefs::default());
        let root = TempDir::new().unwrap();
        let mgr = persistent_manager(root.path(), &storage, &refs);
        let id = Uuid::new_v4();

        let dir = mgr.prepare(id, None).await.unwrap();
        std::fs::write(dir.join("work.txt"), "snapshotted").unwrap();
        mgr.snapshot(id).await.unwrap();

        // A stale marker (another replica produced a newer snapshot, or the
        // dir is torn and the marker vanished) forces a wipe-and-restore.
        std::fs::write(dir.join("work.txt"), "diverged").unwrap();
        std::fs::write(dir.join(META_DIR).join(SNAPSHOT_MARKER), "stale-etag").unwrap();
        mgr.prepare(id, None).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("work.txt")).unwrap(),
            "snapshotted"
        );

        // Torn dir: no marker at all.
        std::fs::write(dir.join("work.txt"), "diverged-again").unwrap();
        std::fs::remove_file(dir.join(META_DIR).join(SNAPSHOT_MARKER)).unwrap();
        mgr.prepare(id, None).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("work.txt")).unwrap(),
            "snapshotted"
        );
    }

    #[tokio::test]
    async fn test_restored_workspace_is_not_reseeded() {
        let storage: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let refs = Arc::new(InMemorySnapshotRefs::default());
        let id = Uuid::new_v4();

        // Seed from a local git fixture, mutate, snapshot.
        let src = TempDir::new().unwrap();
        init_repo_with_file(src.path(), "seed.txt", "seeded");
        let mut def = definition("dev");
        def.git.push(GitSeed {
            url: format!("file://{}", src.path().display()),
            reference: None,
            dest: Some("repo".to_string()),
        });

        let root_a = TempDir::new().unwrap();
        let mgr_a = persistent_manager(root_a.path(), &storage, &refs);
        let dir_a = mgr_a.prepare(id, Some(&def)).await.unwrap();
        std::fs::write(dir_a.join("repo/seed.txt"), "mutated-after-seed").unwrap();
        mgr_a.snapshot(id).await.unwrap();

        // A continuation elsewhere restores the snapshot; the definition must
        // not re-clone over the mutation.
        let root_b = TempDir::new().unwrap();
        let mgr_b = persistent_manager(root_b.path(), &storage, &refs);
        let dir_b = mgr_b.prepare(id, Some(&def)).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir_b.join("repo/seed.txt")).unwrap(),
            "mutated-after-seed"
        );
    }

    /// End-to-end against a real garage/S3 endpoint. Guards the builder
    /// config (path-style, allow-http) that `InMemory` cannot exercise.
    ///
    /// ```bash
    /// NEUROMANCE_TEST_S3_ENDPOINT=http://localhost:3900 \
    /// NEUROMANCE_TEST_S3_BUCKET=workspaces \
    /// NEUROMANCE_TEST_S3_KEY_ID=... NEUROMANCE_TEST_S3_SECRET=... \
    ///     cargo test -p neuromance-runtime -- --ignored garage
    /// ```
    #[tokio::test]
    #[ignore = "requires garage via NEUROMANCE_TEST_S3_* env vars"]
    async fn test_snapshot_round_trip_against_live_garage() {
        let settings = crate::config::WorkspaceStorageSettings {
            endpoint: std::env::var("NEUROMANCE_TEST_S3_ENDPOINT").unwrap(),
            bucket: std::env::var("NEUROMANCE_TEST_S3_BUCKET").unwrap(),
            region: std::env::var("NEUROMANCE_TEST_S3_REGION")
                .unwrap_or_else(|_| "garage".to_string()),
            access_key_id_env: "NEUROMANCE_TEST_S3_KEY_ID".to_string(),
            secret_access_key_env: "NEUROMANCE_TEST_S3_SECRET".to_string(),
        };
        let storage = build_object_store(&settings).unwrap();
        let refs = Arc::new(InMemorySnapshotRefs::default());
        let id = Uuid::new_v4();

        let root_a = TempDir::new().unwrap();
        let mgr_a = persistent_manager(root_a.path(), &storage, &refs);
        let dir_a = mgr_a.prepare(id, None).await.unwrap();
        std::fs::write(dir_a.join("work.txt"), "via-garage").unwrap();
        mgr_a.snapshot(id).await.unwrap();

        let root_b = TempDir::new().unwrap();
        let mgr_b = persistent_manager(root_b.path(), &storage, &refs);
        let dir_b = mgr_b.prepare(id, None).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir_b.join("work.txt")).unwrap(),
            "via-garage"
        );
    }

    #[tokio::test]
    async fn test_snapshot_without_persistence_is_noop() {
        let root = TempDir::new().unwrap();
        let mgr = manager(root.path());
        let id = Uuid::new_v4();
        mgr.prepare(id, None).await.unwrap();
        mgr.snapshot(id).await.unwrap();
        assert!(!mgr.persistence_enabled());
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
