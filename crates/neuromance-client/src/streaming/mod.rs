//! Shared streaming infrastructure for SSE-based provider clients.
//!
//! Each provider's wire protocol differs (event types, sentinels, accumulator
//! state) but the SSE plumbing — connection setup, retry-policy disablement,
//! `Event::Open` skipping, JSON parsing, stream-end detection, HTTP-status
//! error extraction — is identical. This module factors that plumbing out
//! behind the [`StreamingProvider`] trait.

mod sse;

pub use sse::{StreamingProvider, run_sse_stream};
