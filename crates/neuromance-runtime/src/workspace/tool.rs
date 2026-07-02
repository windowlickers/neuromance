//! Ambient-workspace default cwd for locally-executed tools.
//!
//! The sandbox path injects the workspace as bash's default cwd server-side
//! (see `sandbox/server.rs`); this wrapper is the in-process counterpart,
//! reading the workspace from the ambient [`DelegationContext`] at execution
//! time so one wrapped instance serves every task the worker runs.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use neuromance_common::delegation;
use neuromance_common::tools::Tool;
use neuromance_tools::{ToolError, ToolImplementation, with_default_cwd};

/// Wraps a tool so calls default their `cwd` to the ambient workspace
/// directory. An explicit caller-supplied `cwd` always wins; outside any
/// workspace scope the wrapper is a pass-through.
pub struct WorkspaceCwdTool {
    inner: Arc<dyn ToolImplementation>,
}

impl WorkspaceCwdTool {
    #[must_use]
    pub fn new(inner: Arc<dyn ToolImplementation>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl ToolImplementation for WorkspaceCwdTool {
    fn get_definition(&self) -> Tool {
        self.inner.get_definition()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        match delegation::current().workspace_dir {
            Some(dir) => {
                let name = self.inner.get_definition().function.name;
                let args = with_default_cwd(&name, args.clone(), &dir);
                self.inner.execute(&args).await
            }
            None => self.inner.execute(args).await,
        }
    }

    fn is_auto_approved(&self) -> bool {
        self.inner.is_auto_approved()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_common::delegation::DelegationContext;
    use neuromance_tools::{BashTool, with_default_cwd};
    use serde_json::json;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn test_with_default_cwd_only_touches_bash_without_explicit_cwd() {
        let root = Path::new("/workspace/abc");
        let injected = with_default_cwd("bash", json!({"command": "pwd"}), root);
        assert_eq!(injected["cwd"], "/workspace/abc");

        let explicit = with_default_cwd("bash", json!({"command": "pwd", "cwd": "/tmp"}), root);
        assert_eq!(explicit["cwd"], "/tmp", "explicit cwd must win");

        let other = with_default_cwd("read", json!({"path": "/x"}), root);
        assert!(other.get("cwd").is_none(), "non-bash tools untouched");
    }

    #[tokio::test]
    async fn test_wrapped_bash_runs_in_ambient_workspace() {
        let dir = TempDir::new().unwrap();
        let tool = WorkspaceCwdTool::new(Arc::new(BashTool::new(Vec::new())));
        let ctx = DelegationContext {
            workspace_dir: Some(dir.path().to_path_buf()),
            ..DelegationContext::default()
        };

        let out = delegation::scope(ctx, tool.execute(&json!({"command": "pwd -P"})))
            .await
            .unwrap();
        let expected = dir.path().canonicalize().unwrap();
        assert!(
            out.contains(&expected.display().to_string()),
            "bash should run in the ambient workspace dir; got: {out}"
        );
    }

    #[tokio::test]
    async fn test_wrapped_bash_passes_through_without_scope() {
        let tool = WorkspaceCwdTool::new(Arc::new(BashTool::new(Vec::new())));
        let out = tool.execute(&json!({"command": "echo ok"})).await.unwrap();
        assert!(out.contains("ok"));
    }
}
