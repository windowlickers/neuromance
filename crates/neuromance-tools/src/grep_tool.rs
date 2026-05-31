use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use globset::{GlobBuilder, GlobMatcher};
use ignore::WalkBuilder;
use regex::{Regex, RegexBuilder};
use serde_json::Value;

use crate::factory::ToolFactory;
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head};
use crate::{ToolError, ToolImplementation, ToolRegistry, opt_u64, resolve_search_path};
use neuromance_common::tools::{Function, Parameters, Property, Tool};

/// Default cap on the number of matching lines returned.
const DEFAULT_LIMIT: u64 = 100;
/// Per-line character cap so one long line can't dominate the output.
const MAX_LINE_CHARS: usize = 500;

/// Searches file contents by regular expression, respecting `.gitignore`.
///
/// Backed by the same `ignore` walker that powers ripgrep: it skips
/// git-ignored and hidden files and never descends into ignored directories.
/// Auto-approved: read-only.
pub struct GrepTool;

#[async_trait]
impl ToolImplementation for GrepTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "pattern".to_string(),
            Property::string("Regular expression to search for (Rust regex syntax)."),
        );
        properties.insert(
            "path".to_string(),
            Property::string(
                "Optional absolute file or directory to search. Defaults to the current directory.",
            ),
        );
        properties.insert(
            "glob".to_string(),
            Property::string("Optional glob filtering which files are searched, e.g. '*.rs'."),
        );
        properties.insert(
            "ignore_case".to_string(),
            Property::boolean("Case-insensitive search. Defaults to false."),
        );
        properties.insert(
            "literal".to_string(),
            Property::boolean(
                "Treat the pattern as a literal string, not a regex. Defaults to false.",
            ),
        );
        properties.insert(
            "context".to_string(),
            Property::number(
                "Lines of context to show before and after each match. Defaults to 0.",
            ),
        );
        properties.insert(
            "limit".to_string(),
            Property::number(format!(
                "Maximum number of matching lines to return. Defaults to {DEFAULT_LIMIT}."
            )),
        );

        Tool::builder()
            .function(Function {
                name: "grep".to_string(),
                description: "Search file contents by regular expression, respecting .gitignore. \
                              Returns `path:line:text` for matches. Hidden and git-ignored files \
                              are skipped."
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
        if !root.exists() {
            return Err(ToolError::InvalidArguments(format!(
                "path does not exist: {}",
                root.display()
            )));
        }

        let ignore_case = obj
            .get("ignore_case")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let literal = obj.get("literal").and_then(Value::as_bool).unwrap_or(false);
        let context = usize::try_from(opt_u64(obj.get("context"), "context", 0)?).unwrap_or(0);
        let limit = usize::try_from(opt_u64(obj.get("limit"), "limit", DEFAULT_LIMIT)?.max(1))
            .unwrap_or(usize::MAX);

        let query = GrepQuery {
            regex: build_regex(pattern, literal, ignore_case)?,
            glob: build_glob(obj.get("glob").and_then(Value::as_str))?,
            context,
            limit,
        };

        tokio::task::spawn_blocking(move || query.run(&root))
            .await
            .map_err(|e| ToolError::execution(format!("grep task failed: {e}")))
    }

    fn is_auto_approved(&self) -> bool {
        true
    }
}

/// A compiled grep request: the matcher plus the output budget.
struct GrepQuery {
    regex: Regex,
    glob: Option<GlobMatcher>,
    context: usize,
    limit: usize,
}

fn build_regex(pattern: &str, literal: bool, ignore_case: bool) -> Result<Regex, ToolError> {
    let source = if literal {
        regex::escape(pattern)
    } else {
        pattern.to_string()
    };
    RegexBuilder::new(&source)
        .case_insensitive(ignore_case)
        .build()
        .map_err(|e| ToolError::InvalidArguments(format!("invalid regex '{pattern}': {e}")))
}

fn build_glob(glob: Option<&str>) -> Result<Option<GlobMatcher>, ToolError> {
    glob.map(|g| {
        GlobBuilder::new(g)
            .literal_separator(false)
            .build()
            .map(|glob| glob.compile_matcher())
            .map_err(|e| ToolError::InvalidArguments(format!("invalid glob '{g}': {e}")))
    })
    .transpose()
}

impl GrepQuery {
    fn run(&self, root: &Path) -> String {
        let root_is_file = root.is_file();
        let mut body = String::new();
        let mut matches = 0usize;

        for entry in WalkBuilder::new(root).require_git(false).build() {
            if matches >= self.limit {
                break;
            }
            let Ok(entry) = entry else { continue };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let display = display_path(entry.path(), root, root_is_file);
            if self.glob.as_ref().is_some_and(|g| !g.is_match(&display)) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(entry.path()) else {
                continue; // binary or unreadable file
            };
            matches += self.search_file(&display, &text, self.limit - matches, &mut body);
        }

        self.finalize(&body, matches)
    }

    fn search_file(&self, display: &str, text: &str, remaining: usize, out: &mut String) -> usize {
        let lines: Vec<&str> = text.lines().collect();
        let mut emitted = 0usize;

        for (idx, line) in lines.iter().enumerate() {
            if emitted >= remaining {
                break;
            }
            if !self.regex.is_match(line) {
                continue;
            }
            if self.context > 0 && !out.is_empty() {
                out.push_str("--\n");
            }
            let start = idx.saturating_sub(self.context);
            let end = (idx + self.context).min(lines.len().saturating_sub(1));
            for (j, ctx_line) in lines.iter().enumerate().take(end + 1).skip(start) {
                let line_no = j + 1;
                let cell = truncate_cell(ctx_line);
                let sep = if j == idx { ':' } else { '-' };
                let _ = writeln!(out, "{display}{sep}{line_no}{sep}{cell}");
            }
            emitted += 1;
        }

        emitted
    }

    fn finalize(&self, body: &str, matches: usize) -> String {
        if matches == 0 {
            return "No matches found.".to_string();
        }
        let capped = truncate_head(body, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        let truncated = capped.is_truncated();
        let (shown, total) = (capped.shown_lines, capped.total_lines);
        let mut out = capped.content;
        if !out.ends_with('\n') {
            out.push('\n');
        }
        if truncated {
            let _ = writeln!(out, "[output truncated: showing {shown} of {total} lines]");
        }
        if matches >= self.limit {
            let _ = writeln!(
                out,
                "[match limit {} reached; refine the pattern or raise 'limit' for more]",
                self.limit
            );
        }
        out
    }
}

fn display_path(path: &Path, root: &Path, root_is_file: bool) -> String {
    if root_is_file {
        return path.file_name().map_or_else(
            || path.to_string_lossy().into_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
    }
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn truncate_cell(line: &str) -> String {
    if line.chars().count() <= MAX_LINE_CHARS {
        return line.to_string();
    }
    let head: String = line.chars().take(MAX_LINE_CHARS).collect();
    format!("{head} [line truncated]")
}

/// Factory that registers [`GrepTool`] under the name `grep`.
/// Takes no configuration.
pub struct GrepToolFactory;

impl ToolFactory for GrepToolFactory {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<(), ToolError> {
        registry.register(Arc::new(GrepTool));
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

    async fn grep(args: Value) -> String {
        GrepTool.execute(&args).await.unwrap()
    }

    #[tokio::test]
    async fn test_grep_basic_match() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha\nbeta\ngamma\n").unwrap();

        let out = grep(json!({ "pattern": "beta", "path": dir.path().to_str().unwrap() })).await;
        assert!(out.contains("a.txt:2:beta"), "got:\n{out}");
        assert!(!out.contains("alpha"));
    }

    #[tokio::test]
    async fn test_grep_no_matches() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha\n").unwrap();

        let out = grep(json!({ "pattern": "zzz", "path": dir.path().to_str().unwrap() })).await;
        assert_eq!(out, "No matches found.");
    }

    #[tokio::test]
    async fn test_grep_respects_gitignore() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();
        std::fs::write(dir.path().join("secret.txt"), "needle\n").unwrap();
        std::fs::write(dir.path().join("public.txt"), "needle\n").unwrap();

        let out = grep(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() })).await;
        assert!(out.contains("public.txt"), "got:\n{out}");
        assert!(
            !out.contains("secret.txt"),
            "git-ignored file searched:\n{out}"
        );
    }

    #[tokio::test]
    async fn test_grep_glob_filter() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "needle\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "needle\n").unwrap();

        let out = grep(json!({
            "pattern": "needle",
            "path": dir.path().to_str().unwrap(),
            "glob": "*.rs",
        }))
        .await;
        assert!(out.contains("a.rs"));
        assert!(!out.contains("b.txt"));
    }

    #[tokio::test]
    async fn test_grep_ignore_case() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "Hello World\n").unwrap();

        let out = grep(json!({
            "pattern": "hello",
            "path": dir.path().to_str().unwrap(),
            "ignore_case": true,
        }))
        .await;
        assert!(out.contains("a.txt:1:Hello World"));
    }

    #[tokio::test]
    async fn test_grep_literal_escapes_regex() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a.b\naxb\n").unwrap();

        let out = grep(json!({
            "pattern": "a.b",
            "path": dir.path().to_str().unwrap(),
            "literal": true,
        }))
        .await;
        assert!(out.contains("a.txt:1:a.b"));
        assert!(!out.contains("axb"));
    }

    #[tokio::test]
    async fn test_grep_context_lines() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\nthree\nfour\n").unwrap();

        let out = grep(json!({
            "pattern": "three",
            "path": dir.path().to_str().unwrap(),
            "context": 1,
        }))
        .await;
        assert!(out.contains("a.txt-2-two"));
        assert!(out.contains("a.txt:3:three"));
        assert!(out.contains("a.txt-4-four"));
    }

    #[tokio::test]
    async fn test_grep_limit_footer() {
        let dir = tempdir().unwrap();
        let body = "match\n".repeat(10);
        std::fs::write(dir.path().join("a.txt"), body).unwrap();

        let out = grep(json!({
            "pattern": "match",
            "path": dir.path().to_str().unwrap(),
            "limit": 3,
        }))
        .await;
        assert!(out.contains("match limit 3 reached"), "got:\n{out}");
    }

    #[tokio::test]
    async fn test_grep_invalid_regex_errors() {
        let dir = tempdir().unwrap();
        let err = GrepTool
            .execute(&json!({ "pattern": "(", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid regex"));
    }

    #[tokio::test]
    async fn test_grep_relative_path_rejected() {
        let err = GrepTool
            .execute(&json!({ "pattern": "x", "path": "rel/dir" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must be absolute"));
    }

    #[test]
    fn test_is_auto_approved() {
        assert!(GrepTool.is_auto_approved());
    }
}
