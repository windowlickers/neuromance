-- Parent/child lineage between conversations. A subagent runs in its own
-- conversation; this links it back to the conversation that spawned it so a
-- delegation tree can be reconstructed. `parent_task_id` records the runtime
-- task the whole tree belongs to, when known. Both are NULL for a root
-- conversation (one not spawned by another agent).
ALTER TABLE conversations
    ADD COLUMN parent_conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    ADD COLUMN parent_task_id         UUID;

CREATE INDEX conversations_parent_id_idx ON conversations (parent_conversation_id);
