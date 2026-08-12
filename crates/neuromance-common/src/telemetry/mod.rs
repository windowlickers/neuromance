//! Shared telemetry vocabulary.
//!
//! Attribute keys and enum values for the OpenTelemetry GenAI semantic
//! conventions live in [`genai`]; the gate that decides whether prompt and
//! completion text is attached to spans lives in [`capture`].
//!
//! This module holds strings, not instruments. It carries no OpenTelemetry
//! dependency, so every crate in the workspace can name a `gen_ai.*` attribute
//! without pulling in the SDK. The crates that actually emit telemetry
//! (`neuromance-client` for spans and metrics, `neuromance-runtime` for the
//! exporters) own that dependency.

pub mod capture;
pub mod genai;

pub use capture::capture_message_content;
