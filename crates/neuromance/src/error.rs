use neuromance_client::ClientError;
use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreError {
    #[error(transparent)]
    Client(#[from] ClientError),

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Maximum turns exceeded: {0}")]
    MaxTurnsExceeded(String),

    #[error("User quit: {0}")]
    UserQuit(String),

    #[error("Cancelled: {0}")]
    Cancelled(String),

    #[error("No response: {0}")]
    NoResponse(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Hook {hook} failed: {source}")]
    Hook {
        /// Name of the hook that failed.
        hook: String,
        /// The underlying error returned by the hook.
        #[source]
        source: anyhow::Error,
    },

    #[error("Context compaction error: {0}")]
    CompactionError(String),
}

impl CoreError {
    /// A stable, low-cardinality slug naming why the run failed.
    ///
    /// Intended for the `reason` label on `neuromance_tasks_total`, so the value
    /// set is fixed and small. Never derive this from [`Display`](std::fmt::Display)
    /// output — the inner strings carry provider text, file paths, and tool
    /// names, any of which would blow up label cardinality.
    ///
    /// [`Self::Client`] delegates to [`ClientError::reason`] rather than
    /// collapsing into one bucket: provider failures are the common case, so a
    /// single `client_error` slug would answer "the client failed" and nothing
    /// more.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance::CoreError;
    ///
    /// let err = CoreError::MaxTurnsExceeded("20 turns".to_string());
    /// assert_eq!(err.reason(), "max_turns_exceeded");
    /// ```
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Client(e) => e.reason(),
            Self::ToolError(_) => "tool_error",
            Self::MaxTurnsExceeded(_) => "max_turns_exceeded",
            Self::UserQuit(_) => "user_quit",
            Self::Cancelled(_) => "cancelled",
            Self::NoResponse(_) => "no_response",
            Self::InvalidInput(_) => "invalid_input",
            Self::Serialization(_) => "serialization",
            Self::Hook { .. } => "hook",
            Self::CompactionError(_) => "compaction",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One arm per variant, so adding a variant without a slug fails to compile.
    /// `#[non_exhaustive]` does not apply inside the defining crate, so this
    /// match really is exhaustive.
    #[test]
    fn test_every_variant_reports_a_slug() {
        let table = [
            (CoreError::ToolError("boom".into()), "tool_error"),
            (
                CoreError::MaxTurnsExceeded("20".into()),
                "max_turns_exceeded",
            ),
            (CoreError::UserQuit("bye".into()), "user_quit"),
            (CoreError::Cancelled("sigterm".into()), "cancelled"),
            (CoreError::NoResponse("empty".into()), "no_response"),
            (
                CoreError::InvalidInput("no messages".into()),
                "invalid_input",
            ),
            (
                CoreError::Hook {
                    hook: "persistence".into(),
                    source: anyhow::anyhow!("db down"),
                },
                "hook",
            ),
            (CoreError::CompactionError("too big".into()), "compaction"),
        ];

        for (error, expected) in table {
            assert_eq!(error.reason(), expected, "wrong slug for {error:?}");
        }
    }

    /// The delegation is the point: a provider timeout and a rate limit must not
    /// land in the same bucket, because that is the distinction the 21% task
    /// failure rate needs explained.
    #[test]
    fn test_client_errors_keep_their_own_slug() {
        let timeout = CoreError::Client(ClientError::TimeoutError);
        let auth = CoreError::Client(ClientError::AuthenticationError("bad key".into()));

        assert_eq!(timeout.reason(), "timeout");
        assert_eq!(auth.reason(), "auth");
        assert_ne!(timeout.reason(), auth.reason());
    }
}
