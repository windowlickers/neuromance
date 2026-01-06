//! State types for the context state machine.
//!
//! These types are used as type parameters in `Context<S>` to enforce
//! valid state transitions at compile time using the typestate pattern.

use serde::{Deserialize, Serialize};

/// Marker trait for valid context states.
///
/// All state types must implement this trait to be used with `Context<S>`.
/// This trait is sealed to prevent external implementations.
pub trait ContextState: private::Sealed {}

/// Raw state - initial state with unprocessed conversation.
///
/// This is the entry point for all contexts. No transformations have been applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Raw;

/// Filtered state - after applying filter criteria.
///
/// Messages may have been filtered by role, time range, or other criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Filtered;

/// Transformed state - after applying transformations.
///
/// Messages may have been summarized, merged, or otherwise modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transformed;

/// Validated state - after validation for LLM submission.
///
/// The conversation has been validated and is ready for final preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Validated;

/// Ready state - final state, ready for LLM execution.
///
/// All transformations and validations are complete. The conversation
/// is ready to be sent to an LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ready;

// Implement ContextState for all state types
impl ContextState for Raw {}
impl ContextState for Filtered {}
impl ContextState for Transformed {}
impl ContextState for Validated {}
impl ContextState for Ready {}

// Sealed trait pattern to prevent external implementations
mod private {
    use super::*;

    pub trait Sealed {}

    impl Sealed for Raw {}
    impl Sealed for Filtered {}
    impl Sealed for Transformed {}
    impl Sealed for Validated {}
    impl Sealed for Ready {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_types_exist() {
        // Just verify that all state types can be constructed
        let _raw = Raw;
        let _filtered = Filtered;
        let _transformed = Transformed;
        let _validated = Validated;
        let _ready = Ready;
    }
}
