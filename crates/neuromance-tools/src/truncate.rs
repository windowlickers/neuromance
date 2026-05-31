//! Shared output-truncation helpers for the file and search tools.
//!
//! Tool output sent to a model has to be bounded. Reads, directory listings,
//! and search results want the *start* of the output ([`truncate_head`]);
//! shell output wants the *end*, since errors and final results live there
//! ([`truncate_tail`]). Both cap on whichever of the line or byte budget is
//! hit first and report what was dropped so the caller can render an
//! actionable footer.

/// Default cap on bytes emitted by a single tool invocation.
pub const DEFAULT_MAX_BYTES: usize = 64 * 1024;
/// Default cap on lines emitted by a single tool invocation.
pub const DEFAULT_MAX_LINES: usize = 2000;

/// Which budget caused truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
    /// The line budget was reached first.
    Lines,
    /// The byte budget was reached first.
    Bytes,
}

/// Result of capping a block of output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncation {
    /// The retained text.
    pub content: String,
    /// `Some` if anything was dropped, naming the budget that was hit.
    pub truncated_by: Option<TruncatedBy>,
    /// Number of lines retained in [`content`](Self::content).
    pub shown_lines: usize,
    /// Number of lines in the original input.
    pub total_lines: usize,
}

impl Truncation {
    /// Whether any input was dropped.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.truncated_by.is_some()
    }
}

/// Keep the first lines/bytes of `text`, dropping the tail.
///
/// At least the first line is always retained, even if it alone exceeds
/// `max_bytes`, so output is never empty for non-empty input.
#[must_use]
pub fn truncate_head(text: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let segments: Vec<&str> = text.split_inclusive('\n').collect();
    let total_lines = segments.len();
    let mut content = String::new();
    let mut shown_lines = 0;
    let mut truncated_by = None;

    for seg in &segments {
        if shown_lines >= max_lines {
            truncated_by = Some(TruncatedBy::Lines);
            break;
        }
        if !content.is_empty() && content.len() + seg.len() > max_bytes {
            truncated_by = Some(TruncatedBy::Bytes);
            break;
        }
        content.push_str(seg);
        shown_lines += 1;
    }

    Truncation {
        content,
        truncated_by,
        shown_lines,
        total_lines,
    }
}

/// Keep the last lines/bytes of `text`, dropping the head.
///
/// At least the final line is always retained, even if it alone exceeds
/// `max_bytes`.
#[must_use]
pub fn truncate_tail(text: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let segments: Vec<&str> = text.split_inclusive('\n').collect();
    let total_lines = segments.len();
    let mut chosen: Vec<&str> = Vec::new();
    let mut bytes = 0;
    let mut truncated_by = None;

    for seg in segments.iter().rev() {
        if chosen.len() >= max_lines {
            truncated_by = Some(TruncatedBy::Lines);
            break;
        }
        if !chosen.is_empty() && bytes + seg.len() > max_bytes {
            truncated_by = Some(TruncatedBy::Bytes);
            break;
        }
        bytes += seg.len();
        chosen.push(seg);
    }
    chosen.reverse();

    Truncation {
        shown_lines: chosen.len(),
        content: chosen.concat(),
        truncated_by,
        total_lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_head_keeps_first_lines_when_over_line_budget() {
        let t = truncate_head("a\nb\nc\nd\n", 2, DEFAULT_MAX_BYTES);
        assert_eq!(t.content, "a\nb\n");
        assert_eq!(t.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(t.shown_lines, 2);
        assert_eq!(t.total_lines, 4);
    }

    #[test]
    fn test_tail_keeps_last_lines_when_over_line_budget() {
        let t = truncate_tail("a\nb\nc\nd\n", 2, DEFAULT_MAX_BYTES);
        assert_eq!(t.content, "c\nd\n");
        assert_eq!(t.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(t.shown_lines, 2);
        assert_eq!(t.total_lines, 4);
    }

    #[test]
    fn test_head_under_budget_is_untruncated() {
        let t = truncate_head("a\nb\n", DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        assert_eq!(t.content, "a\nb\n");
        assert!(!t.is_truncated());
    }

    #[test]
    fn test_head_byte_budget() {
        // Each line is "xxxx\n" = 5 bytes; budget of 7 fits one line only.
        let t = truncate_head("xxxx\nyyyy\nzzzz\n", DEFAULT_MAX_LINES, 7);
        assert_eq!(t.content, "xxxx\n");
        assert_eq!(t.truncated_by, Some(TruncatedBy::Bytes));
    }

    #[test]
    fn test_first_line_over_byte_budget_is_still_kept() {
        let t = truncate_head("aaaaaaaaaa\nb\n", DEFAULT_MAX_LINES, 4);
        assert_eq!(t.content, "aaaaaaaaaa\n");
        assert_eq!(t.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(t.shown_lines, 1);
    }

    #[test]
    fn test_no_trailing_newline_counts_last_line() {
        let t = truncate_head("a\nb\nc", DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        assert_eq!(t.total_lines, 3);
        assert_eq!(t.content, "a\nb\nc");
    }

    #[test]
    fn test_empty_input() {
        let t = truncate_head("", DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        assert_eq!(t.content, "");
        assert!(!t.is_truncated());
        assert_eq!(t.total_lines, 0);
    }
}
