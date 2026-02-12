//! Event types for Core orchestration.
//!
//! [`Core::run`](crate::Core::run) returns a [`Stream`](futures::Stream) of these events.
//! Most are observational — deltas, tool results, usage. [`CoreEvent::ApprovalRequest`]
//! is bi-directional: Core pauses until the consumer answers via the attached
//! `oneshot::Sender`.
//!
//! ## Approval: stream event or stored callback
//!
//! Two ways to answer tool-approval requests:
//!
//! 1. **Stream event** (default) — match [`CoreEvent::ApprovalRequest`] and
//!    `responder.send(ToolApproval::...)`.
//! 2. **Stored callback** — set [`Core::with_tool_approval_callback`] and Core
//!    answers internally; [`CoreEvent::ApprovalRequest`] is never yielded.
//!
//! [`Core::with_tool_approval_callback`]: crate::Core::with_tool_approval_callback

use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use neuromance_common::chat::Message;
use neuromance_common::client::Usage;
use neuromance_common::tools::{ToolApproval, ToolCall};
use tokio::sync::oneshot;

/// Events emitted by [`Core::run`](crate::Core::run).
///
/// Terminal event is always [`CoreEvent::Completed`]; drain the stream until
/// you see it to collect the final message history.
#[derive(Debug)]
pub enum CoreEvent {
    /// Streaming content chunk received from the LLM.
    Delta(String),

    /// A tool finished executing.
    ToolResult {
        /// Name of the tool that was executed.
        name: String,
        /// Stringified result or error message.
        result: String,
        /// Whether execution succeeded.
        success: bool,
    },

    /// Token usage reported by the LLM for a single turn.
    Usage(Usage),

    /// Context compaction was performed on the conversation history.
    Compaction {
        /// Token count before compaction.
        original_tokens: usize,
        /// Token count after compaction.
        compacted_tokens: usize,
        /// Number of messages that were summarized.
        messages_summarized: usize,
        /// Whether compaction was actually performed.
        was_compacted: bool,
    },

    /// The model requested a tool call that is not auto-approved.
    ///
    /// Answer by sending a [`ToolApproval`] on `responder`. If the sender is
    /// dropped, Core treats it as a denial.
    ///
    /// Only yielded when no [`Core::with_tool_approval_callback`] is set —
    /// otherwise Core answers internally.
    ///
    /// [`Core::with_tool_approval_callback`]: crate::Core::with_tool_approval_callback
    ApprovalRequest {
        /// The tool call awaiting approval.
        tool_call: ToolCall,
        /// Send a [`ToolApproval`] here to unblock execution.
        responder: oneshot::Sender<ToolApproval>,
    },

    /// Terminal event. The conversation loop finished; payload is the full
    /// message history including assistant and tool messages produced this run.
    Completed(Vec<Message>),
}

/// Async callback for approving tool calls.
///
/// Escape hatch for consumers who prefer stored callbacks over reacting to
/// [`CoreEvent::ApprovalRequest`] in the stream. See module docs.
pub type ToolApprovalCallback =
    Box<dyn Fn(&ToolCall) -> Pin<Box<dyn Future<Output = ToolApproval> + Send>> + Send + Sync>;

/// Async callback for transforming messages between turns.
///
/// Receives the full message history after tool execution and returns a
/// (potentially modified) version. Primary use case: context compaction.
pub type TurnCallback = Box<
    dyn Fn(Vec<Message>) -> Pin<Box<dyn Future<Output = Result<Vec<Message>>> + Send>>
        + Send
        + Sync,
>;
