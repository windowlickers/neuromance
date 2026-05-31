use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;

use crate::factory::ToolFactory;
use crate::{ToolError, ToolImplementation, ToolRegistry};
use neuromance_common::tools::{Function, ObjectSchema, Parameters, Property, Tool};

/// Performs string replacement on a file's contents.
///
/// Accepts either a single `old_string`/`new_string` replacement or a batch
/// of `edits` applied sequentially. Each replacement is matched exactly first;
/// if that fails, a fuzzy match tolerant of line-ending and common Unicode
/// punctuation differences is attempted. The file's existing line-ending
/// convention (LF vs CRLF) and a leading byte-order mark are preserved.
///
/// Not auto-approved: this tool mutates the filesystem.
pub struct EditTool;

/// One replacement within an edit request.
struct Edit {
    old: String,
    new: String,
    replace_all: bool,
}

/// Why a single replacement could not be applied.
enum EditFail {
    Empty,
    Noop,
    NotFound,
    Ambiguous(usize),
}

/// The line-ending convention detected in a file.
#[derive(Clone, Copy)]
enum Eol {
    Lf,
    Crlf,
}

#[async_trait]
impl ToolImplementation for EditTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            Property::string("Absolute path to the file to edit."),
        );
        properties.insert(
            "old_string".to_string(),
            Property::string(
                "Text to find. Must be unique in the file unless replace_all is true. \
                 Ignored when 'edits' is provided.",
            ),
        );
        properties.insert(
            "new_string".to_string(),
            Property::string("Replacement text. Ignored when 'edits' is provided."),
        );
        properties.insert(
            "replace_all".to_string(),
            Property::boolean("If true, replace every occurrence. Defaults to false."),
        );

        let mut item_props = HashMap::new();
        item_props.insert(
            "old_string".to_string(),
            Property::string("Text to find for this replacement."),
        );
        item_props.insert(
            "new_string".to_string(),
            Property::string("Replacement text for this replacement."),
        );
        item_props.insert(
            "replace_all".to_string(),
            Property::boolean("Replace every occurrence. Defaults to false."),
        );
        let item_schema =
            ObjectSchema::new(item_props, vec!["old_string".into(), "new_string".into()]);
        properties.insert(
            "edits".to_string(),
            Property::array(
                "Optional batch of replacements applied sequentially, each matched against \
                 the result of the previous. Use instead of old_string/new_string.",
                item_schema,
            ),
        );

        Tool::builder()
            .function(Function {
                name: "edit".to_string(),
                description: "Replace text in a file. Provide a single old_string/new_string, \
                              or a batch via 'edits' applied in order. Each match must be unique \
                              unless replace_all is set. Matching falls back to a fuzzy match \
                              tolerant of line-ending and Unicode punctuation differences."
                    .to_string(),
                parameters: Parameters::new(properties, vec!["path".into()]).into(),
            })
            .build()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        let obj = args
            .as_object()
            .ok_or_else(|| ToolError::InvalidArguments("expected object arguments".into()))?;

        let path_str = obj
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' parameter".into()))?;
        let path = PathBuf::from(path_str);
        if !path.is_absolute() {
            return Err(ToolError::InvalidArguments(format!(
                "'path' must be absolute, got: {}",
                path.display()
            )));
        }

        let edits = parse_edits(obj)?;

        let raw = fs::read_to_string(&path).await.map_err(|e| {
            ToolError::execution(format!("failed to read file '{}': {e}", path.display()))
        })?;
        let had_bom = raw.starts_with('\u{feff}');
        let content = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
        let eol = detect_eol(content);

        let mut working = content.to_string();
        let mut total = 0usize;
        for (idx, edit) in edits.iter().enumerate() {
            match apply_edit(&working, edit, eol) {
                Ok((next, n)) => {
                    working = next;
                    total += n;
                }
                Err(fail) => return Err(edit_error(&fail, idx, edits.len(), &path)),
            }
        }

        let final_content = if had_bom {
            format!("\u{feff}{working}")
        } else {
            working
        };
        fs::write(&path, &final_content).await.map_err(|e| {
            ToolError::execution(format!("failed to write file '{}': {e}", path.display()))
        })?;

        Ok(success_message(edits.len(), total, &path))
    }
}

fn parse_edits(obj: &serde_json::Map<String, Value>) -> Result<Vec<Edit>, ToolError> {
    if let Some(value) = obj.get("edits").filter(|v| !v.is_null()) {
        let arr = value
            .as_array()
            .ok_or_else(|| ToolError::InvalidArguments("'edits' must be an array".into()))?;
        if arr.is_empty() {
            return Err(ToolError::InvalidArguments(
                "'edits' must not be empty".into(),
            ));
        }
        return arr.iter().map(parse_one_edit).collect();
    }

    let old = obj
        .get("old_string")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ToolError::InvalidArguments(
                "provide either 'edits' or both 'old_string' and 'new_string'".into(),
            )
        })?;
    let new = obj
        .get("new_string")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidArguments("missing 'new_string' parameter".into()))?;
    let replace_all = obj
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(vec![Edit {
        old: old.to_string(),
        new: new.to_string(),
        replace_all,
    }])
}

fn parse_one_edit(value: &Value) -> Result<Edit, ToolError> {
    let item = value
        .as_object()
        .ok_or_else(|| ToolError::InvalidArguments("each edit must be an object".into()))?;
    let old = item
        .get("old_string")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidArguments("edit is missing 'old_string'".into()))?;
    let new = item
        .get("new_string")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidArguments("edit is missing 'new_string'".into()))?;
    let replace_all = item
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(Edit {
        old: old.to_string(),
        new: new.to_string(),
        replace_all,
    })
}

fn apply_edit(content: &str, edit: &Edit, eol: Eol) -> Result<(String, usize), EditFail> {
    if edit.old.is_empty() {
        return Err(EditFail::Empty);
    }
    if edit.old == edit.new {
        return Err(EditFail::Noop);
    }
    let new = normalize_eol(&edit.new, eol);

    let exact = content.matches(edit.old.as_str()).count();
    if exact > 0 {
        if exact > 1 && !edit.replace_all {
            return Err(EditFail::Ambiguous(exact));
        }
        return Ok(if edit.replace_all {
            (content.replace(edit.old.as_str(), &new), exact)
        } else {
            (content.replacen(edit.old.as_str(), &new, 1), 1)
        });
    }

    let spans = fuzzy_find(content, &edit.old);
    if spans.is_empty() {
        return Err(EditFail::NotFound);
    }
    if spans.len() > 1 && !edit.replace_all {
        return Err(EditFail::Ambiguous(spans.len()));
    }

    let chosen = if edit.replace_all {
        spans.as_slice()
    } else {
        &spans[..1]
    };
    let mut out = content.to_string();
    for (start, end) in chosen.iter().rev() {
        out.replace_range(*start..*end, &new);
    }
    Ok((out, chosen.len()))
}

/// Finds non-overlapping occurrences of `old` in `content` after normalizing
/// both, returning the original byte spans `[start, end)` to replace.
fn fuzzy_find(content: &str, old: &str) -> Vec<(usize, usize)> {
    let (chars, offsets) = normalize_with_offsets(content);
    let needle: Vec<char> = old.chars().filter_map(normalize_char).collect();
    let len = needle.len();
    if len == 0 || len > chars.len() {
        return Vec::new();
    }

    let mut spans = Vec::new();
    let mut i = 0;
    while i + len <= chars.len() {
        if chars[i..i + len] == needle[..] {
            let start = offsets[i];
            let last = chars[i + len - 1];
            let end = offsets[i + len - 1] + last.len_utf8();
            spans.push((start, end));
            i += len;
        } else {
            i += 1;
        }
    }
    spans
}

/// Normalizes `text`, returning the normalized characters alongside the
/// original byte offset each one came from. Dropped characters (e.g. `\r`)
/// leave no entry, so a normalized span maps cleanly back to original bytes.
fn normalize_with_offsets(text: &str) -> (Vec<char>, Vec<usize>) {
    let mut chars = Vec::new();
    let mut offsets = Vec::new();
    for (byte, ch) in text.char_indices() {
        if let Some(norm) = normalize_char(ch) {
            chars.push(norm);
            offsets.push(byte);
        }
    }
    (chars, offsets)
}

/// Maps a character to its fuzzy-match-normalized form, or `None` to drop it.
const fn normalize_char(ch: char) -> Option<char> {
    match ch {
        '\r' => None,
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => Some('\''),
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => Some('"'),
        '\u{2010}'..='\u{2015}' | '\u{2212}' => Some('-'),
        '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => Some(' '),
        other => Some(other),
    }
}

fn detect_eol(content: &str) -> Eol {
    match content.find('\n') {
        Some(i) if i > 0 && content.as_bytes()[i - 1] == b'\r' => Eol::Crlf,
        _ => Eol::Lf,
    }
}

fn normalize_eol(text: &str, eol: Eol) -> String {
    let lf = text.replace("\r\n", "\n").replace('\r', "\n");
    match eol {
        Eol::Lf => lf,
        Eol::Crlf => lf.replace('\n', "\r\n"),
    }
}

fn edit_error(fail: &EditFail, idx: usize, total_edits: usize, path: &Path) -> ToolError {
    let at = if total_edits > 1 {
        format!(" (edit {})", idx + 1)
    } else {
        String::new()
    };
    let msg = match fail {
        EditFail::Empty => format!("'old_string' must not be empty{at}"),
        EditFail::Noop => format!("'old_string' and 'new_string' are identical (no-op){at}"),
        EditFail::NotFound => format!("'old_string' not found in {}{at}", path.display()),
        EditFail::Ambiguous(n) => format!(
            "'old_string' found {n} times in {}{at} but replace_all is false; \
             either provide more context or set replace_all=true",
            path.display()
        ),
    };
    ToolError::InvalidArguments(msg)
}

fn success_message(edit_count: usize, replacements: usize, path: &Path) -> String {
    if edit_count == 1 {
        let plural = if replacements == 1 { "" } else { "s" };
        return format!(
            "replaced {replacements} occurrence{plural} in {}",
            path.display()
        );
    }
    let plural = if replacements == 1 { "" } else { "s" };
    format!(
        "applied {edit_count} edits ({replacements} replacement{plural}) in {}",
        path.display()
    )
}

/// Factory that registers [`EditTool`] under the name `edit`.
/// Takes no configuration.
pub struct EditToolFactory;

impl ToolFactory for EditToolFactory {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<(), ToolError> {
        registry.register(Arc::new(EditTool));
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
    use tokio::fs as tokio_fs;

    async fn write_file(contents: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.txt");
        tokio_fs::write(&path, contents).await.unwrap();
        (dir, path)
    }

    #[tokio::test]
    async fn test_edit_single_match() {
        let (_dir, path) = write_file("hello world").await;
        let result = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "world",
                "new_string": "rust",
            }))
            .await
            .unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(tokio_fs::read_to_string(&path).await.unwrap(), "hello rust");
    }

    #[tokio::test]
    async fn test_edit_replace_all() {
        let (_dir, path) = write_file("foo bar foo baz foo").await;
        let result = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "foo",
                "new_string": "FOO",
                "replace_all": true,
            }))
            .await
            .unwrap();
        assert!(result.contains("replaced 3 occurrences"));
        assert_eq!(
            tokio_fs::read_to_string(&path).await.unwrap(),
            "FOO bar FOO baz FOO"
        );
    }

    #[tokio::test]
    async fn test_edit_ambiguous_without_replace_all() {
        let (_dir, path) = write_file("x x x").await;
        let err = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "x",
                "new_string": "y",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("found 3 times"));
    }

    #[tokio::test]
    async fn test_edit_not_found() {
        let (_dir, path) = write_file("hello").await;
        let err = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "absent",
                "new_string": "x",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_edit_noop_rejected() {
        let (_dir, path) = write_file("abc").await;
        let err = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "same",
                "new_string": "same",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("identical"));
    }

    #[tokio::test]
    async fn test_edit_empty_old_string_rejected() {
        let (_dir, path) = write_file("abc").await;
        let err = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "",
                "new_string": "x",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[tokio::test]
    async fn test_edit_multi_sequential() {
        let (_dir, path) = write_file("a b c").await;
        let result = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "edits": [
                    { "old_string": "a", "new_string": "X" },
                    { "old_string": "c", "new_string": "Z" },
                ],
            }))
            .await
            .unwrap();
        assert!(result.contains("applied 2 edits"));
        assert_eq!(tokio_fs::read_to_string(&path).await.unwrap(), "X b Z");
    }

    #[tokio::test]
    async fn test_edit_multi_is_atomic_on_failure() {
        let (_dir, path) = write_file("a b c").await;
        let err = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "edits": [
                    { "old_string": "a", "new_string": "X" },
                    { "old_string": "zzz", "new_string": "Z" },
                ],
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("(edit 2)"));
        // First edit must not have been persisted.
        assert_eq!(tokio_fs::read_to_string(&path).await.unwrap(), "a b c");
    }

    #[tokio::test]
    async fn test_edit_empty_edits_rejected() {
        let (_dir, path) = write_file("abc").await;
        let err = EditTool
            .execute(&json!({ "path": path.to_str().unwrap(), "edits": [] }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[tokio::test]
    async fn test_edit_fuzzy_smart_quotes() {
        // File uses curly quotes; the model sends straight quotes.
        let (_dir, path) = write_file("let s = \u{201c}hi\u{201d};\n").await;
        let result = EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "let s = \"hi\";",
                "new_string": "let s = \"bye\";",
            }))
            .await
            .unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(
            tokio_fs::read_to_string(&path).await.unwrap(),
            "let s = \"bye\";\n"
        );
    }

    #[tokio::test]
    async fn test_edit_preserves_crlf() {
        let (_dir, path) = write_file("one\r\ntwo\r\nthree\r\n").await;
        EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "two",
                "new_string": "A\nB",
            }))
            .await
            .unwrap();
        let result = tokio_fs::read_to_string(&path).await.unwrap();
        assert_eq!(result, "one\r\nA\r\nB\r\nthree\r\n");
        assert!(!result.contains("A\nB"));
    }

    #[tokio::test]
    async fn test_edit_crlf_old_string_with_lf() {
        // File is CRLF; the model sends a multi-line old_string using LF.
        let (_dir, path) = write_file("alpha\r\nbeta\r\ngamma\r\n").await;
        EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "alpha\nbeta",
                "new_string": "ALPHA\nBETA",
            }))
            .await
            .unwrap();
        assert_eq!(
            tokio_fs::read_to_string(&path).await.unwrap(),
            "ALPHA\r\nBETA\r\ngamma\r\n"
        );
    }

    #[tokio::test]
    async fn test_edit_preserves_bom() {
        let (_dir, path) = write_file("\u{feff}hello world").await;
        EditTool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "world",
                "new_string": "rust",
            }))
            .await
            .unwrap();
        let result = tokio_fs::read_to_string(&path).await.unwrap();
        assert!(result.starts_with('\u{feff}'));
        assert_eq!(result, "\u{feff}hello rust");
    }

    #[test]
    fn test_is_not_auto_approved() {
        assert!(!EditTool.is_auto_approved());
    }
}
