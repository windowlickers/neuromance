//! The gate controlling whether message content reaches telemetry.
//!
//! Spans carry token counts and model parameters unconditionally. Prompts,
//! completions, tool arguments and tool results are different: they hold user
//! data, so they ship only when an operator opts in with
//! `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`, the environment
//! variable the GenAI semantic conventions define for exactly this purpose.
//!
//! Default off.

use std::sync::OnceLock;

use tracing::warn;

const CAPTURE_ENV: &str = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

/// The spellings that turn capture on.
const TRUTHY: [&str; 4] = ["true", "1", "on", "yes"];

/// The spellings that turn it off deliberately, as opposed to by typo.
const FALSY: [&str; 4] = ["false", "0", "off", "no"];

/// True when the operator has opted in to capturing message content.
///
/// Read once. This is checked on every chat turn and every tool call, and the
/// environment cannot change under a running process in any way this code
/// should honour.
#[must_use]
pub fn capture_message_content() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var(CAPTURE_ENV).ok();
        warn_on_unrecognized(raw.as_deref());
        parse_capture_flag(raw.as_deref())
    })
}

/// Interpret the raw environment value.
///
/// Accepts the usual truthy spellings, case-insensitively and ignoring
/// surrounding whitespace. Anything else — including an unset or empty value,
/// or a typo — is off, because the failure mode of guessing wrong is leaking
/// user prompts to a trace backend.
fn parse_capture_flag(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        .is_some_and(|value| TRUTHY.iter().any(|ok| value.eq_ignore_ascii_case(ok)))
}

/// The value worth complaining about, if there is one.
///
/// `None` for an unset, empty, or understood value. A deliberate `false` is not
/// a mistake, so warning about it would train operators to ignore the warning.
fn unrecognized_value(raw: Option<&str>) -> Option<&str> {
    let value = raw.map(str::trim).filter(|value| !value.is_empty())?;
    let recognized = TRUTHY
        .iter()
        .chain(FALSY.iter())
        .any(|known| value.eq_ignore_ascii_case(known));
    (!recognized).then_some(value)
}

/// Say so once when the variable is set to something meaningless.
///
/// Failing closed is right, but silence is not: an operator who writes `ture`
/// sees no prompts in traces and debugs the collector instead of the typo.
/// Logged once, from the `OnceLock` initializer.
fn warn_on_unrecognized(raw: Option<&str>) {
    if let Some(value) = unrecognized_value(raw) {
        warn!(
            env = CAPTURE_ENV,
            value,
            "unrecognized value; message content capture stays off. \
             Accepted: true/1/on/yes to enable, false/0/off/no to disable",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_capture_flag_defaults_off() {
        assert!(!parse_capture_flag(None));
        assert!(!parse_capture_flag(Some("")));
        assert!(!parse_capture_flag(Some("   ")));
    }

    #[test]
    fn test_parse_capture_flag_accepts_truthy_variants() {
        for raw in ["true", "TRUE", "True", "1", "on", "ON", "yes", "  true  "] {
            assert!(parse_capture_flag(Some(raw)), "{raw:?} should enable");
        }
    }

    /// A typo must not enable capture. Failing closed keeps prompts out of the
    /// trace backend when someone writes `OTEL_..._CONTENT=ture`.
    #[test]
    fn test_parse_capture_flag_rejects_anything_else() {
        for raw in ["false", "0", "off", "no", "ture", "enabled", "y"] {
            assert!(!parse_capture_flag(Some(raw)), "{raw:?} should not enable");
        }
    }

    /// The warning separates a typo from a deliberate disable. Warning on
    /// `false` would train operators to ignore it, and warning on an unset
    /// variable would fire for every process that never opted in.
    #[test]
    fn test_a_deliberate_value_is_not_worth_warning_about() {
        for quiet in [None, Some(""), Some("   "), Some("true"), Some("FALSE")] {
            assert_eq!(unrecognized_value(quiet), None, "{quiet:?} must stay quiet");
        }
    }

    /// The typo the whole warning exists for. The value comes back so the log
    /// can name it — an operator needs to see what they actually set.
    #[test]
    fn test_a_typo_is_reported_with_the_value_that_caused_it() {
        assert_eq!(unrecognized_value(Some("ture")), Some("ture"));
        assert_eq!(unrecognized_value(Some("enabled")), Some("enabled"));
        assert_eq!(unrecognized_value(Some("  y  ")), Some("y"));
    }
}
