//! Customizable CLI display theming.
//!
//! Provides a PS1-like format string system for styling CLI output.
//! Users define themes in `~/.config/neuromance/theme.toml`.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

const RESET: &str = "\x1b[0m";

fn color_code(name: &str) -> Option<u8> {
    match name {
        "black" => Some(30),
        "red" => Some(31),
        "green" => Some(32),
        "yellow" => Some(33),
        "blue" => Some(34),
        "magenta" => Some(35),
        "cyan" => Some(36),
        "white" => Some(37),
        "bright_black" => Some(90),
        "bright_red" => Some(91),
        "bright_green" => Some(92),
        "bright_yellow" => Some(93),
        "bright_blue" => Some(94),
        "bright_magenta" => Some(95),
        "bright_cyan" => Some(96),
        "bright_white" => Some(97),
        _ => None,
    }
}

fn modifier_code(name: &str) -> Option<u8> {
    match name {
        "bold" => Some(1),
        "dim" => Some(2),
        "italic" => Some(3),
        "underline" => Some(4),
        _ => None,
    }
}

/// Converts a space-separated style spec (e.g. "bold `bright_yellow`")
/// into an ANSI escape sequence (e.g. `\x1b[1;93m]`).
fn style_to_ansi(spec: &str) -> Result<String> {
    let mut codes: Vec<u8> = Vec::new();

    for token in spec.split_whitespace() {
        if let Some(c) = modifier_code(token) {
            codes.push(c);
        } else if let Some(c) = color_code(token) {
            codes.push(c);
        } else {
            return Err(anyhow::anyhow!("unknown style token: '{token}'"));
        }
    }

    if codes.is_empty() {
        return Ok(String::new());
    }

    let mut out = String::from("\x1b[");
    for (i, code) in codes.iter().enumerate() {
        if i > 0 {
            out.push(';');
        }
        let _ = write!(out, "{code}");
    }
    out.push('m');
    Ok(out)
}

#[derive(Debug, Clone)]
enum Segment {
    Literal(String),
    Variable(String),
}

/// A parsed template string with pre-compiled ANSI escape
/// sequences and variable placeholders.
#[derive(Debug, Clone)]
pub struct Template {
    segments: Vec<Segment>,
}

impl Template {
    /// Parses a format string containing `<style>text</>` style
    /// tags and `{variable}` placeholders.
    pub fn parse(input: &str) -> Result<Self> {
        let no_color = std::env::var_os("NO_COLOR").is_some();
        let mut segments = Vec::new();
        let mut buf = String::new();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                '<' => {
                    chars.next();
                    // Collect tag content up to '>'
                    let mut tag = String::new();
                    let mut found_close = false;
                    for c in chars.by_ref() {
                        if c == '>' {
                            found_close = true;
                            break;
                        }
                        tag.push(c);
                    }
                    if !found_close {
                        return Err(anyhow::anyhow!("unclosed '<' in template"));
                    }

                    if !buf.is_empty() {
                        segments.push(Segment::Literal(std::mem::take(&mut buf)));
                    }

                    if tag == "/" {
                        if !no_color {
                            segments.push(Segment::Literal(RESET.to_string()));
                        }
                    } else if !no_color {
                        let ansi = style_to_ansi(&tag)?;
                        segments.push(Segment::Literal(ansi));
                    }
                }
                '{' => {
                    chars.next();
                    let mut var_name = String::new();
                    let mut found_close = false;
                    for c in chars.by_ref() {
                        if c == '}' {
                            found_close = true;
                            break;
                        }
                        var_name.push(c);
                    }
                    if !found_close {
                        return Err(anyhow::anyhow!("unclosed '{{' in template"));
                    }

                    if !buf.is_empty() {
                        segments.push(Segment::Literal(std::mem::take(&mut buf)));
                    }
                    segments.push(Segment::Variable(var_name));
                }
                '\\' => {
                    chars.next();
                    if let Some(&next) = chars.peek() {
                        match next {
                            'n' => {
                                chars.next();
                                buf.push('\n');
                            }
                            't' => {
                                chars.next();
                                buf.push('\t');
                            }
                            _ => buf.push('\\'),
                        }
                    } else {
                        buf.push('\\');
                    }
                }
                _ => {
                    chars.next();
                    buf.push(ch);
                }
            }
        }

        if !buf.is_empty() {
            segments.push(Segment::Literal(buf));
        }

        Ok(Self { segments })
    }

    /// Renders the template by substituting variables.
    /// Unknown variables are replaced with empty strings.
    pub fn render(&self, vars: &[(&str, &str)]) -> String {
        let mut out = String::new();
        for seg in &self.segments {
            match seg {
                Segment::Literal(s) => out.push_str(s),
                Segment::Variable(name) => {
                    if let Some((_, val)) = vars.iter().find(|(k, _)| k == name) {
                        out.push_str(val);
                    }
                }
            }
        }
        out
    }
}

/// All themeable display elements.
pub struct Theme {
    pub prompt_user: Template,
    pub repl_title: Template,
    pub repl_subtitle: Template,
    pub assistant_header: Template,
    pub assistant_footer: Template,
    pub tool_call: Template,
    pub tool_arg: Template,
    pub tool_result_ok: Template,
    pub tool_result_err: Template,
    pub usage_tokens: Template,
}

/// Raw TOML representation for deserialization.
#[derive(Deserialize, Default)]
struct ThemeFile {
    prompt: Option<PromptSection>,
    repl: Option<ReplSection>,
    assistant: Option<AssistantSection>,
    tool: Option<ToolSection>,
    usage: Option<UsageSection>,
}

#[derive(Deserialize, Default)]
struct PromptSection {
    user: Option<String>,
}

#[derive(Deserialize, Default)]
struct ReplSection {
    title: Option<String>,
    subtitle: Option<String>,
}

#[derive(Deserialize, Default)]
struct AssistantSection {
    header: Option<String>,
    footer: Option<String>,
}

#[derive(Deserialize, Default)]
struct ToolSection {
    call: Option<String>,
    arg: Option<String>,
    result_ok: Option<String>,
    result_err: Option<String>,
}

#[derive(Deserialize, Default)]
struct UsageSection {
    tokens: Option<String>,
}

fn parse_or_default(custom: Option<&str>, default: &str) -> Template {
    custom
        .and_then(|s| match Template::parse(s) {
            Ok(t) => Some(t),
            Err(e) => {
                log::warn!("theme: failed to parse template: {e}");
                None
            }
        })
        .unwrap_or_else(|| {
            // Defaults are known-good — infallible
            Template::parse(default).unwrap_or_else(|_| Template { segments: vec![] })
        })
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            prompt_user: parse_or_default(None, "<bold bright_green>></> "),
            repl_title: parse_or_default(None, "<bold bright_magenta>Neuromance REPL</>"),
            repl_subtitle: parse_or_default(None, "<dim>Ctrl-D to exit</>"),
            assistant_header: parse_or_default(
                None,
                "\n╭─● <bold bright_magenta>Assistant</>\n╰───────────○",
            ),
            assistant_footer: parse_or_default(None, ""),
            tool_call: parse_or_default(
                None,
                "\n○ <bright_yellow>Tool call:</> \
                 <bold bright_green>{name}</>",
            ),
            tool_arg: parse_or_default(None, "  <cyan>{key}</>: {value}"),
            tool_result_ok: parse_or_default(
                None,
                "○ <bright_green>Result:</> \
                 <bright_cyan>{name}</>",
            ),
            tool_result_err: parse_or_default(
                None,
                "○ <bright_red>Failed:</> \
                 <bright_cyan>{name}</>",
            ),
            usage_tokens: parse_or_default(
                None,
                "\n<bright_blue>○</> {total} tokens \
                 (in: {input}, out: {output})",
            ),
        }
    }
}

impl Theme {
    /// Loads theme from `~/.config/neuromance/theme.toml`.
    /// Falls back to defaults on missing file or parse errors.
    pub fn load() -> Self {
        let Some(path) = theme_path() else {
            return Self::default();
        };

        let contents = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Self::default();
            }
            Err(e) => {
                log::warn!("theme: failed to read {}: {e}", path.display());
                return Self::default();
            }
        };

        let file: ThemeFile = match toml::from_str(&contents) {
            Ok(f) => f,
            Err(e) => {
                log::warn!("theme: failed to parse TOML: {e}");
                return Self::default();
            }
        };

        Self::from_file(file)
    }

    fn from_file(file: ThemeFile) -> Self {
        let defaults = Self::default();

        let prompt = file.prompt.unwrap_or_default();
        let repl = file.repl.unwrap_or_default();
        let assistant = file.assistant.unwrap_or_default();
        let tool = file.tool.unwrap_or_default();
        let usage = file.usage.unwrap_or_default();

        Self {
            prompt_user: prompt.user.as_deref().map_or(defaults.prompt_user, |s| {
                parse_or_default(Some(s), "<bold bright_green>></> ")
            }),
            repl_title: repl.title.as_deref().map_or(defaults.repl_title, |s| {
                parse_or_default(Some(s), "<bold bright_magenta>Neuromance REPL</>")
            }),
            repl_subtitle: repl
                .subtitle
                .as_deref()
                .map_or(defaults.repl_subtitle, |s| {
                    parse_or_default(Some(s), "<dim>Ctrl-D to exit</>")
                }),
            assistant_header: assistant
                .header
                .as_deref()
                .map_or(defaults.assistant_header, |s| {
                    parse_or_default(
                        Some(s),
                        "\n╭─● <bold bright_magenta>Assistant</>\n╰───────────○",
                    )
                }),
            assistant_footer: assistant
                .footer
                .as_deref()
                .map_or(defaults.assistant_footer, |s| parse_or_default(Some(s), "")),
            tool_call: tool.call.as_deref().map_or(defaults.tool_call, |s| {
                parse_or_default(
                    Some(s),
                    "\n○ <bright_yellow>Tool call:</> \
                         <bold bright_green>{name}</>",
                )
            }),
            tool_arg: tool.arg.as_deref().map_or(defaults.tool_arg, |s| {
                parse_or_default(Some(s), "  <cyan>{key}</>: {value}")
            }),
            tool_result_ok: tool
                .result_ok
                .as_deref()
                .map_or(defaults.tool_result_ok, |s| {
                    parse_or_default(
                        Some(s),
                        "○ <bright_green>Result:</> \
                         <bright_cyan>{name}</>",
                    )
                }),
            tool_result_err: tool
                .result_err
                .as_deref()
                .map_or(defaults.tool_result_err, |s| {
                    parse_or_default(
                        Some(s),
                        "○ <bright_red>Failed:</> \
                         <bright_cyan>{name}</>",
                    )
                }),
            usage_tokens: usage.tokens.as_deref().map_or(defaults.usage_tokens, |s| {
                parse_or_default(
                    Some(s),
                    "\n<bright_blue>○</> {total} tokens \
                         (in: {input}, out: {output})",
                )
            }),
        }
    }
}

fn theme_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("neuromance").join("theme.toml"))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn parse_plain_text() {
        let t = Template::parse("hello world").unwrap();
        assert_eq!(t.render(&[]), "hello world");
    }

    #[test]
    fn parse_variable_substitution() {
        let t = Template::parse("hello {name}!").unwrap();
        assert_eq!(t.render(&[("name", "Alice")]), "hello Alice!");
    }

    #[test]
    fn parse_missing_variable_renders_empty() {
        let t = Template::parse("hello {name}!").unwrap();
        assert_eq!(t.render(&[]), "hello !");
    }

    #[test]
    fn parse_style_tags_produce_ansi() {
        // NO_COLOR might be set in CI, so test both paths
        let t = Template::parse("<bold>text</>").unwrap();
        let rendered = t.render(&[]);
        if std::env::var_os("NO_COLOR").is_some() {
            assert_eq!(rendered, "text");
        } else {
            assert_eq!(rendered, "\x1b[1mtext\x1b[0m");
        }
    }

    #[test]
    fn parse_compound_style() {
        let t = Template::parse("<bold bright_yellow>hi</>").unwrap();
        let rendered = t.render(&[]);
        if std::env::var_os("NO_COLOR").is_some() {
            assert_eq!(rendered, "hi");
        } else {
            assert_eq!(rendered, "\x1b[1;93mhi\x1b[0m");
        }
    }

    #[test]
    fn parse_mixed_styles_and_vars() {
        let t = Template::parse("<bold>{name}</> said <cyan>hello</>").unwrap();
        let rendered = t.render(&[("name", "Bob")]);
        if std::env::var_os("NO_COLOR").is_some() {
            assert_eq!(rendered, "Bob said hello");
        } else {
            assert_eq!(rendered, "\x1b[1mBob\x1b[0m said \x1b[36mhello\x1b[0m");
        }
    }

    #[test]
    fn parse_newline_escape() {
        let t = Template::parse("line1\\nline2").unwrap();
        assert_eq!(t.render(&[]), "line1\nline2");
    }

    #[test]
    fn parse_error_unclosed_tag() {
        assert!(Template::parse("<bold").is_err());
    }

    #[test]
    fn parse_error_unclosed_variable() {
        assert!(Template::parse("{name").is_err());
    }

    #[test]
    fn parse_error_unknown_style() {
        assert!(Template::parse("<foobar>text</>").is_err());
    }

    #[test]
    fn color_code_all_basic() {
        assert_eq!(color_code("black"), Some(30));
        assert_eq!(color_code("white"), Some(37));
        assert_eq!(color_code("bright_black"), Some(90));
        assert_eq!(color_code("bright_white"), Some(97));
        assert_eq!(color_code("nope"), None);
    }

    #[test]
    fn modifier_code_all() {
        assert_eq!(modifier_code("bold"), Some(1));
        assert_eq!(modifier_code("dim"), Some(2));
        assert_eq!(modifier_code("italic"), Some(3));
        assert_eq!(modifier_code("underline"), Some(4));
        assert_eq!(modifier_code("nope"), None);
    }

    #[test]
    fn style_to_ansi_single() {
        let result = style_to_ansi("bold").unwrap();
        assert_eq!(result, "\x1b[1m");
    }

    #[test]
    fn style_to_ansi_compound() {
        let result = style_to_ansi("bold bright_magenta").unwrap();
        assert_eq!(result, "\x1b[1;95m");
    }

    #[test]
    fn style_to_ansi_unknown_token() {
        assert!(style_to_ansi("sparkle").is_err());
    }

    #[test]
    fn theme_default_loads() {
        let theme = Theme::default();
        // Should produce non-empty renders
        let header = theme.assistant_header.render(&[]);
        assert!(header.contains("Assistant"));
    }

    #[test]
    fn theme_from_partial_toml() {
        let toml_str = r#"
[prompt]
user = "<bold>> </>"
"#;
        let file: ThemeFile = toml::from_str(toml_str).unwrap();
        let theme = Theme::from_file(file);

        // Prompt should be overridden
        let prompt = theme.prompt_user.render(&[]);
        assert!(prompt.contains("> "));

        // Assistant header should still be default
        let header = theme.assistant_header.render(&[]);
        assert!(header.contains("Assistant"));
    }

    #[test]
    fn theme_from_empty_toml() {
        let file: ThemeFile = toml::from_str("").unwrap();
        let theme = Theme::from_file(file);
        let header = theme.assistant_header.render(&[]);
        assert!(header.contains("Assistant"));
    }

    #[test]
    fn theme_full_toml() {
        let toml_str = r#"
[prompt]
user = "<bold bright_yellow>┃</> "

[repl]
title = "<bold bright_magenta>My REPL</>"
subtitle = "<dim>quit with Ctrl-D</>"

[assistant]
header = "\n<bold>┏━</> <bold bright_magenta>Assistant</>"
footer = "<bold>┗━</>"

[tool]
call = "\n<bright_yellow>Tool:</> <bold bright_green>{name}</>"
arg = "  <cyan>{key}</>: {value}"
result_ok = "<bright_green>✓</> <bright_cyan>{name}</>"
result_err = "<bright_red>✗</> <bright_cyan>{name}</>"

[usage]
tokens = "<bright_blue>●</> {total} tok"
"#;
        let file: ThemeFile = toml::from_str(toml_str).unwrap();
        let theme = Theme::from_file(file);

        let header = theme.assistant_header.render(&[]);
        assert!(header.contains("Assistant"));

        let call = theme.tool_call.render(&[("name", "search")]);
        assert!(call.contains("search"));

        let usage =
            theme
                .usage_tokens
                .render(&[("total", "100"), ("input", "80"), ("output", "20")]);
        assert!(usage.contains("100"));
    }

    #[test]
    fn template_multiple_variables() {
        let t = Template::parse("{a} and {b} and {c}").unwrap();
        assert_eq!(
            t.render(&[("a", "1"), ("b", "2"), ("c", "3")]),
            "1 and 2 and 3"
        );
    }

    #[test]
    fn template_repeated_variable() {
        let t = Template::parse("{x} {x} {x}").unwrap();
        assert_eq!(t.render(&[("x", "ha")]), "ha ha ha");
    }

    #[test]
    fn template_empty_input() {
        let t = Template::parse("").unwrap();
        assert_eq!(t.render(&[]), "");
    }
}
