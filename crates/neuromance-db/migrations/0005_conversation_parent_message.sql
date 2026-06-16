-- Pin a child conversation to the exact launch site within its parent: the
-- assistant message that emitted the spawning tool call, and the tool call id
-- within that message. Complements the conversation-level parent link from
-- 0003. ON DELETE SET NULL so pruning a parent message does not cascade away
-- the child conversation.
ALTER TABLE conversations
    ADD COLUMN parent_message_id   UUID REFERENCES messages(id) ON DELETE SET NULL,
    ADD COLUMN parent_tool_call_id TEXT;
