CREATE TABLE conversations (
    id          UUID PRIMARY KEY,
    title       TEXT,
    description TEXT,
    status      TEXT        NOT NULL DEFAULT 'active',
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY,
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq             BIGINT      NOT NULL,
    role            TEXT        NOT NULL,
    content         TEXT        NOT NULL,
    tool_calls      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    tool_call_id    TEXT,
    name            TEXT,
    reasoning       JSONB,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    timestamp       TIMESTAMPTZ NOT NULL,
    UNIQUE (conversation_id, seq)
);

CREATE INDEX conversations_updated_at_idx ON conversations (updated_at DESC);
