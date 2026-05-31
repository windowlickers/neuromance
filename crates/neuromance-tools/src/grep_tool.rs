use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use globset::{GlobBuilder, GlobMatcher};
use grep::printer::StandardBuilder;
use grep::regex::{RegexMatcher, RegexMatcherBuilder};
use grep::searcher::{BinaryDetection, Searcher, SearcherBuilder};
use ignore::WalkBuilder;
use serde_json::Value;

use crate::factory::ToolFactory;
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head};
use crate::{ToolError, ToolImplementation, ToolRegistry, opt_u64, resolve_search_path};
use neuromance_common::tools::{Function, Parameters, Property, Tool};

/// Default cap on the number of matching lines returned.
const DEFAULT_LIMIT: u64 = 100;
/// Per-line byte cap so one long line can't dominate the output; longer lines
/// are shown as a truncated preview.
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
            matcher: build_matcher(pattern, literal, ignore_case)?,
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
    matcher: RegexMatcher,
    glob: Option<GlobMatcher>,
    context: usize,
    limit: usize,
}

fn build_matcher(
    pattern: &str,
    literal: bool,
    ignore_case: bool,
) -> Result<RegexMatcher, ToolError> {
    RegexMatcherBuilder::new()
        .case_insensitive(ignore_case)
        .fixed_strings(literal)
        .build(pattern)
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

        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .before_context(self.context)
            .after_context(self.context)
            .binary_detection(BinaryDetection::quit(0))
            .build();

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
            matches += self.search_file(&mut searcher, &display, entry.path(), &mut body);
        }

        self.finalize(&body, matches)
    }

    /// Searches one file with ripgrep's engine, appending `path:line:text` lines
    /// to `out`, and returns the number of matching lines emitted.
    fn search_file(
        &self,
        searcher: &mut Searcher,
        display: &str,
        path: &Path,
        out: &mut String,
    ) -> usize {
        let mut printer = StandardBuilder::new()
            .heading(false)
            .max_columns(Some(MAX_LINE_CHARS as u64))
            .max_columns_preview(true)
            .build_no_color(Vec::new());

        let mut sink = printer.sink_with_path(&self.matcher, display);
        if searcher
            .search_path(&self.matcher, path, &mut sink)
            .is_err()
        {
            return 0; // unreadable file
        }
        let count = usize::try_from(sink.match_count()).unwrap_or(usize::MAX);
        drop(sink);

        out.push_str(&String::from_utf8_lossy(&printer.into_inner().into_inner()));
        count
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
    async fn test_grep_skips_binary_file() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("data.bin"), b"needle\x00\x01\x02needle\n").unwrap();
        std::fs::write(dir.path().join("text.txt"), "needle\n").unwrap();

        let out = grep(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() })).await;
        assert!(out.contains("text.txt"), "got:\n{out}");
        assert!(!out.contains("data.bin"), "binary file searched:\n{out}");
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
