//! Sandboxed tool-execution boundary.
//!
//! Tool execution is split out of the orchestrator process into a separate
//! sandbox process that runs as a restricted service account with no database
//! credentials and no network route to Postgres. The orchestrator advertises
//! the sandbox's tools to the LLM, applies approval, and dispatches each
//! approved call over gRPC; the sandbox executes capability tools (`bash`,
//! `read`/`write`/`edit`, `grep`/`find`/`ls`, `execute_python`) against the
//! filesystem and network.
//!
//! Deployment: the two processes run as separate containers in one pod. The
//! workspace volume mounts into the *sandbox* container only; the orchestrator
//! holds the DB and LLM credentials. The gRPC channel is loopback-only within
//! the pod.

pub mod proto;
