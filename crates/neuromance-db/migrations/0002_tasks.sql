-- Provenance: which task contributed which messages. The runtime brackets each
-- task run by the per-conversation `seq` high-water mark before and after the
-- run, so a task maps to the `[start_seq, end_seq]` range in `messages`. Only
-- runs that persisted at least one message are recorded (start_seq <= end_seq).
CREATE TABLE tasks (
    id              UUID PRIMARY KEY,
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    start_seq       BIGINT      NOT NULL,
    end_seq         BIGINT      NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX tasks_conversation_id_idx ON tasks (conversation_id);
