//! Shared streaming infrastructure for SSE-based provider clients.
//!
//! Each provider's wire protocol differs (event types, sentinels, accumulator
//! state) but the SSE plumbing — connection setup, retry-policy disablement,
//! `Event::Open` skipping, JSON parsing, stream-end detection, HTTP-status
//! error extraction — is identical. This module factors that plumbing out
//! behind the [`StreamingProvider`] trait.

// Items here are currently exercised only by `#[cfg(test)]` tests in
// `streaming::sse`; the allows come off once provider clients adopt the driver.
#![allow(dead_code, unused_imports)]

mod sse;

pub use sse::{ChatChunkStream, StreamingProvider, run_sse_stream};
