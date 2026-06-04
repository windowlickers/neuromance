//! Task and outcome types for subagent delegation.
//!
//! A [`Task`] is a unit of work handed to a subagent; an [`Outcome`] is what the
//! subagent produces. They are intentionally provider- and transport-agnostic so
//! the same pair describes both in-process and (eventually) remote subagents.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A unit of work delegated to a subagent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// Stable identifier, generated at construction.
    pub id: Uuid,
    /// What the subagent should do, phrased as the user-facing instruction.
    pub instructions: String,
    /// Optional supporting context appended to the instructions.
    pub context: Option<String>,
}

impl Task {
    /// Create a task with the given instructions and a fresh id.
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            instructions: instructions.into(),
            context: None,
        }
    }

    /// Attach supporting context to the task.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// The result of running a [`Task`] through a subagent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    /// The task this outcome answers.
    pub task_id: Uuid,
    /// The subagent's answer.
    pub content: String,
    /// Optional reasoning trace, when the subagent exposes one.
    pub reasoning: Option<String>,
}

impl Outcome {
    /// Create an outcome for `task_id` carrying `content` and no reasoning.
    #[must_use]
    pub fn new(task_id: Uuid, content: impl Into<String>) -> Self {
        Self {
            task_id,
            content: content.into(),
            reasoning: None,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_new_sets_fresh_id_and_no_context() {
        let task = Task::new("do the thing");
        assert_eq!(task.instructions, "do the thing");
        assert!(task.context.is_none());

        let other = Task::new("do the thing");
        assert_ne!(task.id, other.id, "each task gets a distinct id");
    }

    #[test]
    fn test_with_context_attaches_context() {
        let task = Task::new("summarize").with_context("the input is long");
        assert_eq!(task.context.as_deref(), Some("the input is long"));
    }

    #[test]
    fn test_outcome_round_trips_through_serde() {
        let id = Uuid::new_v4();
        let outcome = Outcome {
            task_id: id,
            content: "answer".to_string(),
            reasoning: Some("because".to_string()),
        };

        let json = serde_json::to_string(&outcome).expect("serialize");
        let back: Outcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(outcome, back);
    }
}
