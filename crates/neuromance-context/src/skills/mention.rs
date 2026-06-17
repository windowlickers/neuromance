//! Detecting skill mentions in free user text.
//!
//! Three syntaxes summon a skill, mirroring codex's grammar:
//! - a bare sigil `$skill-name`,
//! - a `skill://<id-or-path>` URI, and
//! - a Markdown link `[text](skill://<id-or-path>)` (captured by the same URI scan).
//!
//! Bare names are filtered against a blocklist of common shell environment
//! variables so that shell-flavored prose (`$PATH`, `$HOME`, …) never misfires.

/// Shell environment variables that must never be read as skill mentions.
const COMMON_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "USER", "SHELL", "PWD", "OLDPWD", "LANG", "LC_ALL", "TERM", "EDITOR", "VISUAL",
    "PAGER", "LOGNAME", "MAIL", "TMPDIR", "DISPLAY", "HOSTNAME", "UID", "GID",
];

/// Whether `name` is a common environment variable and should be ignored.
#[must_use]
pub(super) fn is_common_env_var(name: &str) -> bool {
    COMMON_ENV_VARS.contains(&name)
}

/// Characters permitted in a bare `$name` mention.
fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ':')
}

/// Characters permitted in a `skill://` URI target.
fn is_uri_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ':' | '.' | '/')
}

/// Bare `$name` mentions, in order, with common env-var names removed.
#[must_use]
pub(super) fn scan_sigil_names(text: &str) -> Vec<String> {
    scan_after(text, "$", is_name_char)
        .into_iter()
        .filter(|name| !is_common_env_var(name))
        .collect()
}

/// `skill://<target>` URI targets, in order (also matches Markdown link URLs).
#[must_use]
pub(super) fn scan_uri_targets(text: &str) -> Vec<String> {
    scan_after(text, "skill://", is_uri_char)
}

/// Collect each maximal run of `allowed` characters immediately following an
/// occurrence of `prefix`. Empty runs are skipped.
fn scan_after(text: &str, prefix: &str, allowed: fn(char) -> bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find(prefix) {
        let after = &rest[pos + prefix.len()..];
        let token: String = after.chars().take_while(|&c| allowed(c)).collect();
        if !token.is_empty() {
            out.push(token);
        }
        rest = after;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scans_bare_sigil_names() {
        let names = scan_sigil_names("please run $deploy then $build-app now");
        assert_eq!(names, vec!["deploy", "build-app"]);
    }

    #[test]
    fn test_filters_common_env_vars() {
        let names = scan_sigil_names("echo $PATH and $HOME but use $deploy");
        assert_eq!(names, vec!["deploy"]);
    }

    #[test]
    fn test_scans_skill_uris_including_markdown_links() {
        let targets =
            scan_uri_targets("see skill://forgejo-cli and [here](skill://other/SKILL.md).");
        assert_eq!(targets, vec!["forgejo-cli", "other/SKILL.md"]);
    }

    #[test]
    fn test_double_sigil_does_not_loop() {
        let names = scan_sigil_names("$$deploy");
        assert_eq!(names, vec!["deploy"]);
    }

    #[test]
    fn test_no_mentions_yields_empty() {
        assert!(scan_sigil_names("nothing to summon here").is_empty());
        assert!(scan_uri_targets("nothing to summon here").is_empty());
    }
}
