//! Lenient rule-file frontmatter parsing.
//!
//! A rule file is Markdown with an optional `---` fenced YAML frontmatter
//! block. All frontmatter is optional: a plain `.md` with no fence parses to a
//! body-only rule with no globs (which therefore never path-matches). Recognized
//! keys are `globs` (a YAML sequence or a comma-separated scalar; `paths` is
//! accepted as an alias), `always_apply` (accepting `alwaysApply`), and
//! `description`. Any other keys are retained in [`ParsedRule::extra`].

use serde_yaml::{Mapping, Value};

use super::error::RuleError;

/// The outcome of parsing a rule file.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedRule {
    /// Glob patterns that trigger the rule, if any.
    pub globs: Vec<String>,
    /// Whether the rule applies on every conversation.
    pub always_apply: bool,
    /// Optional human-facing description.
    pub description: Option<String>,
    /// Frontmatter keys other than the recognized ones.
    pub extra: Mapping,
    /// The Markdown body following the frontmatter (or the whole file).
    pub body: String,
}

/// Parse a rule file's text into its frontmatter and body.
///
/// # Errors
/// Returns [`RuleError::Yaml`] only if a frontmatter fence is present but does
/// not contain valid YAML. A file with no fence is parsed as body-only.
pub fn parse_rule(content: &str) -> Result<ParsedRule, RuleError> {
    let Some((yaml, body)) = split_frontmatter(content) else {
        return Ok(ParsedRule {
            globs: Vec::new(),
            always_apply: false,
            description: None,
            extra: Mapping::new(),
            body: content.to_string(),
        });
    };
    let mut map: Mapping = serde_yaml::from_str(&yaml)?;
    let globs = take_globs(&mut map);
    let always_apply = take_always_apply(&mut map);
    let description = take_description(&mut map);
    Ok(ParsedRule {
        globs,
        always_apply,
        description,
        extra: map,
        body,
    })
}

/// Split `content` into its YAML frontmatter and body, or `None` when there is
/// no leading `---` fence.
fn split_frontmatter(content: &str) -> Option<(String, String)> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let trimmed = content.trim_start();
    if !(trimmed.starts_with("---\n") || trimmed.starts_with("---\r\n")) {
        return None;
    }
    let after_open = trimmed.find('\n').map_or(trimmed.len(), |nl| nl + 1);
    let rest = &trimmed[after_open..];

    let mut offset = 0usize;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            let yaml = rest[..offset].to_string();
            let body = rest[offset + line.len()..].to_string();
            return Some((yaml, body));
        }
        offset += line.len();
    }
    None
}

/// Remove and parse the `globs` (or `paths`) key as a list of patterns.
///
/// Accepts a YAML sequence of strings or a single comma-separated scalar.
fn take_globs(map: &mut Mapping) -> Vec<String> {
    let value = map
        .remove(Value::from("globs"))
        .or_else(|| map.remove(Value::from("paths")));
    let Some(value) = value else {
        return Vec::new();
    };
    match value {
        Value::Sequence(seq) => seq
            .iter()
            .filter_map(|v| v.as_str().map(str::trim))
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect(),
        Value::String(s) => s
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

/// Remove and parse the `always_apply` (or `alwaysApply`) boolean key.
fn take_always_apply(map: &mut Mapping) -> bool {
    map.remove(Value::from("always_apply"))
        .or_else(|| map.remove(Value::from("alwaysApply")))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Remove and parse the optional `description` key.
fn take_description(map: &mut Mapping) -> Option<String> {
    map.remove(Value::from("description"))
        .and_then(|v| v.as_str().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_globs_as_yaml_sequence() {
        let parsed = parse_rule("---\nglobs:\n  - \"*.ts\"\n  - \"*.tsx\"\n---\nbody").unwrap();
        assert_eq!(parsed.globs, vec!["*.ts", "*.tsx"]);
        assert_eq!(parsed.body, "body");
    }

    #[test]
    fn test_globs_as_csv_scalar() {
        let parsed = parse_rule("---\nglobs: \"*.ts, *.tsx\"\n---\nbody").unwrap();
        assert_eq!(parsed.globs, vec!["*.ts", "*.tsx"]);
    }

    #[test]
    fn test_paths_alias_for_globs() {
        let parsed = parse_rule("---\npaths:\n  - \"*.rs\"\n---\nbody").unwrap();
        assert_eq!(parsed.globs, vec!["*.rs"]);
    }

    #[test]
    fn test_always_apply_camel_and_snake() {
        let camel = parse_rule("---\nalwaysApply: true\n---\nb").unwrap();
        assert!(camel.always_apply);
        let snake = parse_rule("---\nalways_apply: true\n---\nb").unwrap();
        assert!(snake.always_apply);
    }

    #[test]
    fn test_plain_markdown_has_no_globs() {
        let parsed = parse_rule("# Just a doc\nno frontmatter").unwrap();
        assert!(parsed.globs.is_empty());
        assert!(!parsed.always_apply);
        assert_eq!(parsed.body, "# Just a doc\nno frontmatter");
    }

    #[test]
    fn test_description_and_extra_retained() {
        let parsed =
            parse_rule("---\ndescription: Rust rules\nglobs: \"*.rs\"\nauthor: me\n---\nb")
                .unwrap();
        assert_eq!(parsed.description.as_deref(), Some("Rust rules"));
        assert!(parsed.extra.contains_key(Value::from("author")));
    }

    #[test]
    fn test_closing_fence_at_eof_yields_empty_body() {
        let parsed = parse_rule("---\nglobs: \"*.rs\"\n---").unwrap();
        assert_eq!(parsed.body, "");
    }

    #[test]
    fn test_leading_blank_lines_before_fence() {
        let parsed = parse_rule("\n\n---\nalwaysApply: true\n---\nbody").unwrap();
        assert!(parsed.always_apply);
        assert_eq!(parsed.body, "body");
    }
}
