//! Snapshot-ref bookkeeping: which object holds a conversation's latest
//! workspace snapshot.
//!
//! Mirrors `ConversationSink`'s shape: a narrow, single-purpose async trait
//! with an in-memory working-set impl and a postgres write-through impl.
//! Writes are best-effort (log-and-continue), matching the runtime's
//! mid-run persistence policy — the object store stays authoritative for
//! restore via `ETag` comparison, so a lost ref never corrupts a workspace.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use tracing::warn;
use uuid::Uuid;

use neuromance_db::{PgConversationStore, WorkspaceSnapshotRecord};

/// A recorded workspace snapshot: where it lives and what the object store
/// said about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotRef {
    /// Object key of the snapshot archive within the workspace bucket.
    pub object_key: String,
    /// `ETag` returned by the object store for the upload, when it gave one.
    pub etag: Option<String>,
    /// Archive size in bytes.
    pub size_bytes: i64,
    /// When the snapshot completed.
    pub snapshotted_at: DateTime<Utc>,
}

/// Records and recalls the latest snapshot ref per conversation.
#[async_trait]
pub trait SnapshotRefStore: Send + Sync {
    /// Record `snapshot` as the conversation's latest. Best-effort: failures
    /// are logged, never surfaced — see the module docs.
    async fn record_snapshot(&self, conversation_id: Uuid, snapshot: &SnapshotRef);

    /// The latest recorded snapshot for a conversation, if any.
    async fn get_snapshot(&self, conversation_id: Uuid) -> Option<SnapshotRef>;
}

/// Working-set-only refs; lost on restart, like the in-memory task store.
#[derive(Default)]
pub struct InMemorySnapshotRefs {
    refs: DashMap<Uuid, SnapshotRef>,
}

#[async_trait]
impl SnapshotRefStore for InMemorySnapshotRefs {
    async fn record_snapshot(&self, conversation_id: Uuid, snapshot: &SnapshotRef) {
        self.refs.insert(conversation_id, snapshot.clone());
    }

    async fn get_snapshot(&self, conversation_id: Uuid) -> Option<SnapshotRef> {
        self.refs.get(&conversation_id).map(|r| r.clone())
    }
}

/// Postgres write-through refs, durable across replicas.
pub struct PgSnapshotRefs {
    store: Arc<PgConversationStore>,
}

impl PgSnapshotRefs {
    #[must_use]
    pub const fn new(store: Arc<PgConversationStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl SnapshotRefStore for PgSnapshotRefs {
    async fn record_snapshot(&self, conversation_id: Uuid, snapshot: &SnapshotRef) {
        let record = WorkspaceSnapshotRecord {
            conversation_id,
            object_key: snapshot.object_key.clone(),
            etag: snapshot.etag.clone(),
            size_bytes: snapshot.size_bytes,
            snapshotted_at: snapshot.snapshotted_at,
        };
        if let Err(e) = self.store.upsert_workspace_snapshot(&record).await {
            warn!(%conversation_id, error = %e, "failed to record workspace snapshot ref");
        }
    }

    async fn get_snapshot(&self, conversation_id: Uuid) -> Option<SnapshotRef> {
        match self.store.get_workspace_snapshot(conversation_id).await {
            Ok(record) => record.map(|r| SnapshotRef {
                object_key: r.object_key,
                etag: r.etag,
                size_bytes: r.size_bytes,
                snapshotted_at: r.snapshotted_at,
            }),
            Err(e) => {
                warn!(%conversation_id, error = %e, "failed to read workspace snapshot ref");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[tokio::test]
    async fn test_in_memory_refs_round_trip_and_overwrite() {
        let refs = InMemorySnapshotRefs::default();
        let id = Uuid::new_v4();
        assert!(refs.get_snapshot(id).await.is_none());

        let first = SnapshotRef {
            object_key: "snapshots/x.tar.zst".to_string(),
            etag: Some("v1".to_string()),
            size_bytes: 10,
            snapshotted_at: Utc::now(),
        };
        refs.record_snapshot(id, &first).await;
        assert_eq!(refs.get_snapshot(id).await, Some(first.clone()));

        let second = SnapshotRef {
            etag: Some("v2".to_string()),
            ..first
        };
        refs.record_snapshot(id, &second).await;
        assert_eq!(
            refs.get_snapshot(id).await.unwrap().etag.as_deref(),
            Some("v2")
        );
    }
}
