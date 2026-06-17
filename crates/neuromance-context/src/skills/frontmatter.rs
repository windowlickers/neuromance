//! Lenient `SKILL.md` frontmatter parsing.
//!
//! A `SKILL.md` is a `---` fenced YAML block followed by a Markdown body. Per
//! the agentskills.io standard, `name` and `description` are required; any
//! other frontmatter keys are retained in [`ParsedSkill::extra`] rather than
//! rejected.

use serde_yaml::{Mapping, Value};

use super::error::SkillError;

/// The outcome of parsing a `SKILL.md`: required fields, retained extras, body.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedSkill {
    /// The `name` frontmatter field.
    pub name: String,
    /// The `description` frontmatter field.
    pub description: String,
    /// Frontmatter keys other than `name`/`description`.
    pub extra: Mapping,
    /// The Markdown body following the closing `---` fence.
    pub body: String,
}

/// Parse a `SKILL.md`'s text into its frontmatter and body.
///
/// # Errors
/// Returns [`SkillError::MissingFrontmatter`] if no fenced block is present,
/// [`SkillError::Yaml`] if the block is not valid YAML, or
/// [`SkillError::MissingField`] if `name`/`description` are absent or empty.
pub fn parse_skill(content: &str) -> Result<ParsedSkill, SkillError> {
    let (yaml, body) = split_frontmatter(content)?;
    let mut map: Mapping = serde_yaml::from_str(&yaml)?;
    let name = take_required_string(&mut map, "name")?;
    let description = take_required_string(&mut map, "description")?;
    Ok(ParsedSkill {
        name,
        description,
        extra: map,
        body,
    })
}

/// Split `content` into its YAML frontmatter and the body after the fence.
fn split_frontmatter(content: &str) -> Result<(String, String), SkillError> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let trimmed = content.trim_start();
    let opens = trimmed.starts_with("---\n") || trimmed.starts_with("---\r\n");
    if !opens {
        return Err(SkillError::MissingFrontmatter);
    }
    // Skip past the opening fence line.
    let after_open = trimmed.find('\n').map_or(trimmed.len(), |nl| nl + 1);
    let rest = &trimmed[after_open..];

    let mut offset = 0usize;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            let yaml = rest[..offset].to_string();
            let body = rest[offset + line.len()..].to_string();
            return Ok((yaml, body));
        }
        offset += line.len();
    }
    Err(SkillError::MissingFrontmatter)
}

/// Remove `key` from the mapping and return it as a trimmed, non-empty string.
fn take_required_string(map: &mut Mapping, key: &'static str) -> Result<String, SkillError> {
    let value = map
        .remove(Value::from(key))
        .ok_or(SkillError::MissingField(key))?;
    let text = value.as_str().ok_or(SkillError::MissingField(key))?.trim();
    if text.is_empty() {
        return Err(SkillError::MissingField(key));
    }
    Ok(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parses_name_description_and_body() {
        let content =
            "---\nname: forgejo-cli\ndescription: Use the fj CLI.\n---\n# Body\nDo the thing.\n";
        let parsed = parse_skill(content).expect("parse");
        assert_eq!(parsed.name, "forgejo-cli");
        assert_eq!(parsed.description, "Use the fj CLI.");
        assert_eq!(parsed.body, "# Body\nDo the thing.\n");
        assert!(parsed.extra.is_empty());
    }

    #[test]
    fn test_retains_extra_frontmatter_fields() {
        let content =
            "---\nname: x\ndescription: y\nlicense: MIT\nmetadata:\n  short: z\n---\nbody";
        let parsed = parse_skill(content).expect("parse");
        assert_eq!(
            parsed
                .extra
                .get(Value::from("license"))
                .and_then(Value::as_str),
            Some("MIT")
        );
        assert!(parsed.extra.contains_key(Value::from("metadata")));
    }

    #[test]
    fn test_missing_frontmatter_fence_errors() {
        let err = parse_skill("no frontmatter here").unwrap_err();
        assert!(matches!(err, SkillError::MissingFrontmatter));
    }

    #[test]
    fn test_missing_name_errors_naming_the_field() {
        let err = parse_skill("---\ndescription: y\n---\nbody").unwrap_err();
        assert!(matches!(err, SkillError::MissingField("name")));
    }

    #[test]
    fn test_empty_description_errors() {
        let err = parse_skill("---\nname: x\ndescription: \"\"\n---\nbody").unwrap_err();
        assert!(matches!(err, SkillError::MissingField("description")));
    }

    #[test]
    fn test_closing_fence_at_eof_yields_empty_body() {
        let parsed = parse_skill("---\nname: x\ndescription: y\n---").expect("parse");
        assert_eq!(parsed.body, "");
    }

    #[test]
    fn test_leading_blank_lines_before_fence() {
        let parsed = parse_skill("\n\n---\nname: x\ndescription: y\n---\nbody").expect("parse");
        assert_eq!(parsed.name, "x");
        assert_eq!(parsed.body, "body");
    }
}
