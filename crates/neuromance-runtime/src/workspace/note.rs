//! The workspace instruction handed to an agent, and the subagent decorator
//! that attaches it.
//!
//! The main agent gets [`note`] folded into its seed system message, where it
//! persists with the conversation (`oneshot::run`, `serve::seed_new_conversation`).
//! A subagent has no seed to fold into — its system prompt is fixed when the
//! delegation tower is built, long before any conversation has a workspace — so
//! [`WorkspaceNoteSubagent`] appends the note to each delegated task instead,
//! reading the directory from the ambient [`DelegationContext`] at run time.
//! Same pattern as [`WorkspaceCwdTool`](super::tool::WorkspaceCwdTool).

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use neuromance_common::delegation;
use neuromance_common::subagent::{Subagent, SubagentError};
use neuromance_common::task::{Outcome, Task};

/// The instruction telling an agent where its file work belongs.
#[must_use]
pub fn note(dir: &Path) -> String {
    format!(
        "Your working directory is {}. Do all file work there; use absolute paths beneath it.",
        dir.display()
    )
}

/// Wraps a subagent so every delegated task carries the ambient workspace
/// [`note`]. Outside any workspace scope the wrapper is a pass-through.
pub struct WorkspaceNoteSubagent {
    inner: Arc<dyn Subagent>,
}

impl WorkspaceNoteSubagent {
    #[must_use]
    pub fn new(inner: Arc<dyn Subagent>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Subagent for WorkspaceNoteSubagent {
    fn id(&self) -> &str {
        self.inner.id()
    }

    async fn run(&self, task: Task, cancel: CancellationToken) -> Result<Outcome, SubagentError> {
        let Some(dir) = delegation::current().workspace_dir else {
            return self.inner.run(task, cancel).await;
        };
        // Append rather than replace: the delegating agent's context is the
        // substance of the task, and `LocalSubagent::run` folds the whole field
        // into the user message.
        let note = note(&dir);
        let task = Task {
            context: Some(match task.context {
                Some(context) => format!("{context}\n\n{note}"),
                None => note,
            }),
            ..task
        };
        self.inner.run(task, cancel).await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_common::delegation::DelegationContext;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Records the task it was handed so a test can inspect what the decorator
    /// passed through.
    #[derive(Default)]
    struct RecordingSubagent {
        seen: Mutex<Option<Task>>,
    }

    #[async_trait]
    impl Subagent for RecordingSubagent {
        fn id(&self) -> &'static str {
            "recorder"
        }

        async fn run(
            &self,
            task: Task,
            _cancel: CancellationToken,
        ) -> Result<Outcome, SubagentError> {
            let id = task.id;
            *self.seen.lock().unwrap() = Some(task);
            Ok(Outcome::new(id, "ok"))
        }
    }

    fn scoped(dir: &str) -> DelegationContext {
        DelegationContext {
            workspace_dir: Some(PathBuf::from(dir)),
            ..DelegationContext::default()
        }
    }

    /// Inside a workspace scope the note is appended to the delegator's own
    /// context, not substituted for it — losing the caller's context would
    /// strip the task of the detail it was delegated with.
    #[tokio::test]
    async fn test_note_appends_to_existing_context() {
        let inner = Arc::new(RecordingSubagent::default());
        let wrapper = WorkspaceNoteSubagent::new(inner.clone());
        let task = Task::new("do the thing").with_context("prior context");

        delegation::scope(
            scoped("/workspace/abc"),
            wrapper.run(task, CancellationToken::new()),
        )
        .await
        .unwrap();

        let seen = inner.seen.lock().unwrap().clone().unwrap();
        let context = seen.context.expect("context must be set");
        assert!(context.starts_with("prior context"), "got: {context}");
        assert!(context.contains("/workspace/abc"), "got: {context}");
    }

    /// With no caller context the note stands alone rather than being appended
    /// to an empty string.
    #[tokio::test]
    async fn test_note_is_whole_context_when_task_has_none() {
        let inner = Arc::new(RecordingSubagent::default());
        let wrapper = WorkspaceNoteSubagent::new(inner.clone());

        delegation::scope(
            scoped("/workspace/xyz"),
            wrapper.run(Task::new("do the thing"), CancellationToken::new()),
        )
        .await
        .unwrap();

        let seen = inner.seen.lock().unwrap().clone().unwrap();
        assert_eq!(
            seen.context.as_deref(),
            Some(note(Path::new("/workspace/xyz")).as_str())
        );
    }

    /// Outside a workspace scope the task is forwarded untouched: the wrapper
    /// is registered once at startup and must not invent a directory for a
    /// run that has none.
    #[tokio::test]
    async fn test_note_is_pass_through_outside_workspace_scope() {
        let inner = Arc::new(RecordingSubagent::default());
        let wrapper = WorkspaceNoteSubagent::new(inner.clone());
        let task = Task::new("do the thing").with_context("prior context");

        wrapper.run(task, CancellationToken::new()).await.unwrap();

        let seen = inner.seen.lock().unwrap().clone().unwrap();
        assert_eq!(seen.context.as_deref(), Some("prior context"));
    }
}
