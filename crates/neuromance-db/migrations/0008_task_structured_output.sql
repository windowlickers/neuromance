-- Parsed structured output for tasks submitted with an `output_schema`.
--
-- Kept separate from `output` (the prose the agent returned) so a caller can
-- read the machine-readable result without re-parsing. NULL for every task that
-- carried no schema.
ALTER TABLE tasks ADD COLUMN structured JSONB;
