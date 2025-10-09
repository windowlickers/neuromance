//! Event types for Core orchestration
//!
//! This module defines events that Core emits during conversation execution,
//! enabling flexible handling of streaming content, tool execution, and usage tracking.
//!
//! ## Events vs. Tool Approval
//!
//! Events are **observability** - fire-and-forget notifications about what's happening.
//! Tool approval is **control flow** - the system waits for a decision before proceeding.
//!
//! This is why `ToolApprovalCallback` is separate from `EventCallback`:
//! - Events return `()` and are non-blocking notifications
//! - Tool approval returns `ToolApproval` and blocks execution until resolved
//!
//! Both are async to allow I/O operations (e.g., updating shared state or prompting users).

use std::future::Future;
use std::pin::Pin;

use neuromance_common::client::Usage;
use neuromance_common::tools::{ToolApproval, ToolCall};

/// Type alias for async tool approval callback functions
pub type ToolApprovalCallback =
    Box<dyn Fn(&ToolCall) -> Pin<Box<dyn Future<Output = ToolApproval> + Send>> + Send + Sync>;

/// Events emitted by Core during conversation execution
///
/// These are one-way notifications for observability. Core does not wait for
/// or react to the callback's completion beyond awaiting it.
#[derive(Debug, Clone)]
pub enum CoreEvent {
    /// Streaming content chunk received from LLM
    Streaming(String),

    /// Tool execution completed with result
    ToolResult {
        /// Name of the tool that was executed
        name: String,
        /// Result or error message from execution
        result: String,
        /// Whether execution succeeded
        success: bool,
    },

    /// Token usage information from LLM response
    Usage(Usage),
}

/// Async callback for receiving Core events
///
/// Returns `()` because events are notifications, not control flow decisions.
/// The callback is awaited to allow async I/O operations.
pub type EventCallback =
    Box<dyn Fn(CoreEvent) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;
