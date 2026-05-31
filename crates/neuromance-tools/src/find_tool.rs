use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use globset::{GlobBuilder, GlobMatcher};
use ignore::WalkBuilder;
use serde_json::Value;

use crate::factory::ToolFactory;
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head};
use crate::{ToolError, ToolImplementation, ToolRegistry, opt_u64, resolve_search_path};
use neuromance_common::tools::{Function, Parameters, Property, Tool};

/// Default cap on the number of paths returned.
const DEFAULT_LIMIT: u64 = 1000;

/// Finds files and directories by glob pattern, respecting `.gitignore`.
///
/// Backed by the same `ignore` walker as [`crate::GrepTool`]: git-ignored and
/// hidden entries are skipped. Auto-approved: read-only.
pub struct FindTool;

#[async_trait]
impl ToolImplementation for FindTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "pattern".to_string(),
            Property::string(
                "Glob pattern to match, e.g. '*.rs', '**/*.toml', or 'src/**/mod.rs'.",
            ),
        );
        properties.insert(
            "path".to_string(),
            Property::string(
                "Optional absolute directory to search. Defaults to the current directory.",
            ),
        );
        properties.insert(
            "limit".to_string(),
            Property::number(format!(
                "Maximum number of paths to return. Defaults to {DEFAULT_LIMIT}."
            )),
        );

        Tool::builder()
            .function(Function {
                name: "find".to_string(),
                description: "Find files and directories by glob pattern, respecting .gitignore. \
                              Returns paths relative to the search directory; directories end \
                              with '/'."
                    .to_string(),
                parameters: Parameters::new(properties, vec!["pattern".into()]).into(),
            })
            .build()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        let obj = args
            .as_object()
            .ok_or_else(|| ToolError::InvalidArguments("expected object arguments".into()))?;

        let pattern = obj
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments("missing 'pattern' parameter".into()))?;

        let root = resolve_search_path(obj.get("path").and_then(Value::as_str))?;
        if !root.is_dir() {
            return Err(ToolError::InvalidArguments(format!(
                "path is not a directory: {}",
                root.display()
            )));
        }

        let limit = usize::try_from(opt_u64(obj.get("limit"), "limit", DEFAULT_LIMIT)?.max(1))
            .unwrap_or(usize::MAX);
        let glob = GlobBuilder::new(pattern)
            .literal_separator(false)
            .build()
            .map(|g| g.compile_matcher())
            .map_err(|e| ToolError::InvalidArguments(format!("invalid glob '{pattern}': {e}")))?;

        tokio::task::spawn_blocking(move || run_find(&root, &glob, limit))
            .await
            .map_err(|e| ToolError::execution(format!("find task failed: {e}")))
    }

    fn is_auto_approved(&self) -> bool {
        true
    }
}

fn run_find(root: &Path, glob: &GlobMatcher, limit: usize) -> String {
    let mut hits: Vec<String> = Vec::new();
    let mut limit_reached = false;

    for entry in WalkBuilder::new(root).require_git(false).build() {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if rel.is_empty() {
            continue; // the search root itself
        }
        if !glob.is_match(&rel) {
            continue;
        }
        if entry.file_type().is_some_and(|t| t.is_dir()) {
            hits.push(format!("{rel}/"));
        } else {
            hits.push(rel);
        }
        if hits.len() >= limit {
            limit_reached = true;
            break;
        }
    }

    hits.sort();
    finalize(&hits, limit, limit_reached)
}

fn finalize(hits: &[String], limit: usize, limit_reached: bool) -> String {
    if hits.is_empty() {
        return "No files found.".to_string();
    }
    let body = hits.join("\n");
    let capped = truncate_head(&body, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let truncated = capped.is_truncated();
    let (shown, total) = (capped.shown_lines, capped.total_lines);
    let mut out = capped.content;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    if truncated {
        let _ = writeln!(out, "[output truncated: showing {shown} of {total} paths]");
    }
    if limit_reached {
        let _ = writeln!(
            out,
            "[result limit {limit} reached; refine the pattern or raise 'limit' for more]"
        );
    }
    out
}

/// Factory that registers [`FindTool`] under the name `find`.
/// Takes no configuration.
pub struct FindToolFactory;

impl ToolFactory for FindToolFactory {
    fn name(&self) -> &'static str {
        "find"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<(), ToolError> {
        registry.register(Arc::new(FindTool));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    async fn find(args: Value) -> String {
        FindTool.execute(&args).await.unwrap()
    }

    #[tokio::test]
    async fn test_find_by_extension() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/c.rs"), "").unwrap();

        let out = find(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() })).await;
        assert!(out.contains("a.rs"), "got:\n{out}");
        assert!(out.contains("sub/c.rs"), "nested match missing:\n{out}");
        assert!(!out.contains("b.txt"));
    }

    #[tokio::test]
    async fn test_find_marks_directories() {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join("mydir")).unwrap();

        let out = find(json!({ "pattern": "mydir", "path": dir.path().to_str().unwrap() })).await;
        assert!(out.contains("mydir/"), "got:\n{out}");
    }

    #[tokio::test]
    async fn test_find_respects_gitignore() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.rs\n").unwrap();
        std::fs::write(dir.path().join("ignored.rs"), "").unwrap();
        std::fs::write(dir.path().join("kept.rs"), "").unwrap();

        let out = find(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() })).await;
        assert!(out.contains("kept.rs"));
        assert!(
            !out.contains("ignored.rs"),
            "git-ignored file listed:\n{out}"
        );
    }

    #[tokio::test]
    async fn test_find_no_matches() {
        let dir = tempdir().unwrap();
        let out = find(json!({ "pattern": "*.zzz", "path": dir.path().to_str().unwrap() })).await;
        assert_eq!(out, "No files found.");
    }

    #[tokio::test]
    async fn test_find_invalid_glob_errors() {
        let dir = tempdir().unwrap();
        let err = FindTool
            .execute(&json!({ "pattern": "[unclosed", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid glob"));
    }

    #[test]
    fn test_is_auto_approved() {
        assert!(FindTool.is_auto_approved());
    }
}
