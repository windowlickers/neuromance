-- Task status is now durable so any replica behind the k8s Service can answer
-- GET /tasks/{id} and GET /tasks. The status row is created at enqueue, before
-- any message is persisted, so start_seq/end_seq -- known only after the run --
-- become nullable. The provenance upsert (record_task) and the status upsert
-- (record_task_status) both key on id and each touch only their own columns, so
-- neither clobbers the other's writes regardless of arrival order.
ALTER TABLE tasks
    ALTER COLUMN start_seq DROP NOT NULL,
    ALTER COLUMN end_seq   DROP NOT NULL,
    ADD COLUMN status                 TEXT        NOT NULL DEFAULT 'pending',
    ADD COLUMN output                 TEXT,
    ADD COLUMN error                  TEXT,
    ADD COLUMN queue_depth_at_enqueue BIGINT      NOT NULL DEFAULT 0,
    ADD COLUMN updated_at             TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE INDEX tasks_status_idx ON tasks (status);
