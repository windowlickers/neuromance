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

const CAPTURE_ENV: &str = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

/// True when the operator has opted in to capturing message content.
///
/// Read once. This is checked on every chat turn and every tool call, and the
/// environment cannot change under a running process in any way this code
/// should honour.
#[must_use]
pub fn capture_message_content() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_capture_flag(std::env::var(CAPTURE_ENV).ok().as_deref()))
}

/// Interpret the raw environment value.
///
/// Accepts the usual truthy spellings, case-insensitively and ignoring
/// surrounding whitespace. Anything else — including an unset or empty value,
/// or a typo — is off, because the failure mode of guessing wrong is leaking
/// user prompts to a trace backend.
fn parse_capture_flag(raw: Option<&str>) -> bool {
    raw.map(str::trim).is_some_and(|value| {
        value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("1")
            || value.eq_ignore_ascii_case("on")
            || value.eq_ignore_ascii_case("yes")
    })
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
}
