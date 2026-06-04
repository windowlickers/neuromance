//! Subagent combinators: subagents built from other subagents.
//!
//! [`FanoutVote`] is the worked example that sets the combinator style — a
//! combinator is itself a [`Subagent`], so combinators nest and compose with
//! leaves uniformly.

use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use tokio_util::sync::CancellationToken;

use neuromance_common::task::{Outcome, Task};

use super::{Subagent, SubagentError};

/// Runs a task across several member subagents in parallel, then asks a judge
/// subagent to pick or synthesize the best answer.
///
/// Member failures are dropped; if every member fails, [`FanoutVote::run`]
/// returns [`SubagentError::NoOutcomes`] without invoking the judge.
pub struct FanoutVote {
    id: String,
    members: Vec<Arc<dyn Subagent>>,
    judge: Arc<dyn Subagent>,
}

impl FanoutVote {
    /// Build a fanout-vote over `members`, adjudicated by `judge`.
    ///
    /// # Errors
    /// Returns [`SubagentError::NoOutcomes`] if `members` is empty — a fanout
    /// with no members can never produce an outcome.
    pub fn new(
        id: impl Into<String>,
        members: Vec<Arc<dyn Subagent>>,
        judge: Arc<dyn Subagent>,
    ) -> Result<Self, SubagentError> {
        if members.is_empty() {
            return Err(SubagentError::NoOutcomes);
        }
        Ok(Self {
            id: id.into(),
            members,
            judge,
        })
    }
}

/// Build the judge's task: the original instructions plus every candidate answer,
/// asking the judge to choose or synthesize the best one.
fn build_judge_task(original: &Task, candidates: &[Outcome]) -> Task {
    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "Several agents independently answered the following task:\n\n{}\n",
        original.instructions
    );
    for (i, candidate) in candidates.iter().enumerate() {
        let _ = writeln!(
            prompt,
            "--- Candidate {} ---\n{}\n",
            i + 1,
            candidate.content
        );
    }
    prompt.push_str(
        "Choose the single best answer or synthesize one from the candidates. \
         Respond with only the final answer.",
    );

    let task = Task::new(prompt);
    match &original.context {
        Some(ctx) => task.with_context(ctx.clone()),
        None => task,
    }
}

#[async_trait]
impl Subagent for FanoutVote {
    fn id(&self) -> &str {
        &self.id
    }

    async fn run(&self, task: Task, cancel: CancellationToken) -> Result<Outcome, SubagentError> {
        let runs = self
            .members
            .iter()
            .map(|m| m.run(task.clone(), cancel.clone()));
        let candidates: Vec<Outcome> = join_all(runs)
            .await
            .into_iter()
            .filter_map(Result::ok)
            .collect();

        if candidates.is_empty() {
            return Err(SubagentError::NoOutcomes);
        }

        let judge_task = build_judge_task(&task, &candidates);
        let mut verdict = self.judge.run(judge_task, cancel).await?;
        // The verdict answers the original task, not the synthetic judge task.
        verdict.task_id = task.id;
        Ok(verdict)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use std::sync::Mutex;

    use uuid::Uuid;

    use super::*;

    /// Returns a canned answer; records the last task it was asked to run.
    struct MockSubagent {
        id: String,
        answer: String,
        last_instructions: Mutex<Option<String>>,
    }

    impl MockSubagent {
        fn new(id: &str, answer: &str) -> Arc<Self> {
            Arc::new(Self {
                id: id.to_string(),
                answer: answer.to_string(),
                last_instructions: Mutex::new(None),
            })
        }
    }

    #[async_trait]
    impl Subagent for MockSubagent {
        fn id(&self) -> &str {
            &self.id
        }

        async fn run(
            &self,
            task: Task,
            _cancel: CancellationToken,
        ) -> Result<Outcome, SubagentError> {
            *self.last_instructions.lock().expect("lock") = Some(task.instructions.clone());
            Ok(Outcome::new(task.id, self.answer.clone()))
        }
    }

    /// Always fails — used to exercise the all-members-fail path.
    struct FailingSubagent;

    #[async_trait]
    impl Subagent for FailingSubagent {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "failing"
        }

        async fn run(
            &self,
            _task: Task,
            _cancel: CancellationToken,
        ) -> Result<Outcome, SubagentError> {
            Err(SubagentError::NoOutcomes)
        }
    }

    #[tokio::test]
    async fn test_judge_sees_member_answers_and_returns_verdict() {
        let member_a = MockSubagent::new("a", "answer A");
        let member_b = MockSubagent::new("b", "answer B");
        let judge = MockSubagent::new("judge", "final verdict");

        let fanout = FanoutVote::new(
            "vote",
            vec![member_a, member_b],
            Arc::clone(&judge) as Arc<dyn Subagent>,
        )
        .expect("non-empty members");

        let task = Task::new("what is the answer?");
        let outcome = fanout
            .run(task.clone(), CancellationToken::new())
            .await
            .expect("run succeeds");

        assert_eq!(outcome.content, "final verdict");
        assert_eq!(
            outcome.task_id, task.id,
            "verdict answers the original task"
        );

        let judge_prompt = judge
            .last_instructions
            .lock()
            .expect("lock")
            .clone()
            .expect("judge ran");
        assert!(judge_prompt.contains("answer A"));
        assert!(judge_prompt.contains("answer B"));
        assert!(judge_prompt.contains("what is the answer?"));
    }

    #[tokio::test]
    async fn test_all_members_fail_returns_no_outcomes_without_judging() {
        let judge = MockSubagent::new("judge", "final verdict");
        let fanout = FanoutVote::new(
            "vote",
            vec![Arc::new(FailingSubagent), Arc::new(FailingSubagent)],
            Arc::clone(&judge) as Arc<dyn Subagent>,
        )
        .expect("non-empty members");

        let err = fanout
            .run(Task::new("q"), CancellationToken::new())
            .await
            .expect_err("all members failed");
        assert!(matches!(err, SubagentError::NoOutcomes));
        assert!(
            judge.last_instructions.lock().expect("lock").is_none(),
            "judge must not run when there are no candidates",
        );
    }

    #[test]
    fn test_new_rejects_empty_members() {
        let judge = MockSubagent::new("judge", "v");
        let result = FanoutVote::new("vote", vec![], judge as Arc<dyn Subagent>);
        assert!(matches!(result, Err(SubagentError::NoOutcomes)));
    }

    #[test]
    fn test_build_judge_task_enumerates_candidates() {
        let original = Task::new("solve");
        let candidates = vec![
            Outcome::new(Uuid::new_v4(), "first"),
            Outcome::new(Uuid::new_v4(), "second"),
        ];
        let judge_task = build_judge_task(&original, &candidates);
        assert!(judge_task.instructions.contains("Candidate 1"));
        assert!(judge_task.instructions.contains("first"));
        assert!(judge_task.instructions.contains("Candidate 2"));
        assert!(judge_task.instructions.contains("second"));
    }
}
