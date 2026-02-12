//! Per-run aggregated statistics for [`Core`] orchestration.
//!
//! [`RunStats`] folds the stats-bearing variants of [`CoreEvent`] into a
//! single record. [`Core::chat_with_tool_loop`] returns one of these alongside
//! the message history; consumers of [`Core::run`] directly can build the
//! same aggregate by calling [`RunStats::observe`] for each event they pull
//! off the stream.
//!
//! [`Core`]: crate::Core
//! [`Core::run`]: crate::Core::run
//! [`Core::chat_with_tool_loop`]: crate::Core::chat_with_tool_loop

use neuromance_common::CacheMetrics;

use crate::events::CoreEvent;

/// Aggregated statistics for a single run of [`Core::chat_with_tool_loop`]
/// or a manual drain of [`Core::run`].
///
/// [`Core::run`]: crate::Core::run
/// [`Core::chat_with_tool_loop`]: crate::Core::chat_with_tool_loop
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    /// Cache and token usage aggregated across every turn in the run.
    pub cache_metrics: CacheMetrics,
    /// Number of tool calls that executed successfully.
    pub successful_tool_calls: u64,
    /// Number of tool calls that failed during execution.
    pub failed_tool_calls: u64,
}

impl RunStats {
    /// Update stats from a single [`CoreEvent`]. Non-stats events are ignored.
    pub fn observe(&mut self, event: &CoreEvent) {
        match event {
            CoreEvent::Usage(usage) => self.cache_metrics.record(usage),
            CoreEvent::ToolResult { success: true, .. } => self.successful_tool_calls += 1,
            CoreEvent::ToolResult { success: false, .. } => self.failed_tool_calls += 1,
            CoreEvent::Delta(_)
            | CoreEvent::ApprovalRequest { .. }
            | CoreEvent::Completed(_)
            | CoreEvent::Compaction { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use neuromance_common::client::Usage;
    use neuromance_common::tools::{FunctionCall, ToolCall};
    use tokio::sync::oneshot;

    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        }
    }

    #[test]
    fn observe_records_usage() {
        let mut stats = RunStats::default();
        stats.observe(&CoreEvent::Usage(usage(50, 30)));
        stats.observe(&CoreEvent::Usage(usage(20, 10)));

        assert_eq!(stats.cache_metrics.total_input_tokens, 70);
        assert_eq!(stats.cache_metrics.total_output_tokens, 40);
        assert_eq!(stats.cache_metrics.total_requests, 2);
    }

    #[test]
    fn observe_counts_tool_results() {
        let mut stats = RunStats::default();
        stats.observe(&CoreEvent::ToolResult {
            name: "ok".into(),
            result: "yes".into(),
            success: true,
        });
        stats.observe(&CoreEvent::ToolResult {
            name: "ok2".into(),
            result: "yes".into(),
            success: true,
        });
        stats.observe(&CoreEvent::ToolResult {
            name: "bad".into(),
            result: "no".into(),
            success: false,
        });

        assert_eq!(stats.successful_tool_calls, 2);
        assert_eq!(stats.failed_tool_calls, 1);
    }

    #[test]
    fn observe_ignores_non_stats_events() {
        let mut stats = RunStats::default();
        stats.observe(&CoreEvent::Delta("hi".into()));
        stats.observe(&CoreEvent::Completed(Vec::new()));

        let (tx, _rx) = oneshot::channel();
        stats.observe(&CoreEvent::ApprovalRequest {
            tool_call: ToolCall {
                id: "id".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "n".into(),
                    arguments: String::new(),
                },
                index: None,
            },
            responder: tx,
        });

        assert_eq!(stats.cache_metrics.total_requests, 0);
        assert_eq!(stats.successful_tool_calls, 0);
        assert_eq!(stats.failed_tool_calls, 0);
    }
}
