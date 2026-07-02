-- Workspace snapshot refs: which object in the workspace bucket holds the
-- latest snapshot of a conversation's working directory. One row per
-- conversation, updated in place on every snapshot. Advisory for restore
-- (the object store is authoritative via ETag comparison); load-bearing for
-- audit and external retention sweeps over `snapshotted_at`.
CREATE TABLE workspace_snapshots (
    conversation_id UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    object_key      TEXT        NOT NULL,
    etag            TEXT,
    size_bytes      BIGINT      NOT NULL,
    snapshotted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
