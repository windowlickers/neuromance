//! Bridge that exposes [`Subagent`]s to Python code running in the REPL.
//!
//! [`SubagentRepl`] injects a single `run_subagent` callable into the namespace,
//! backed by a registry of named subagents. Because the registry holds
//! `Arc<dyn Subagent>`, entries can be leaf agents or combinators alike — the
//! Python side only sees a uniform primitive. Agents can then *write their own*
//! orchestration techniques in Python instead of relying on fixed Rust combinators:
//!
//! ```python
//! # fanout + vote, defined entirely in Python by the agent
//! answers = [run_subagent('worker', question) for _ in range(3)]
//! verdict = run_subagent('judge', 'Pick the best:\n' + '\n'.join(answers))
//! ```
//!
//! The bridge depends only on the `neuromance-common` subagent contract, never on
//! the agent crate, so `agent -> repl` (repl-as-a-tool) stays cycle-free.

use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use neuromance_common::subagent::Subagent;
use neuromance_common::task::Task;

use super::{PythonRepl, PythonReplTool};
use crate::ReplError;

/// A [`PythonRepl`] with named subagents injected as a `run_subagent` callable.
pub struct SubagentRepl {
    repl: Arc<PythonRepl>,
    subagents: Arc<HashMap<String, Arc<dyn Subagent>>>,
    cancel: CancellationToken,
}

impl SubagentRepl {
    /// Wrap `repl`, register `subagents` by name, and inject `run_subagent`.
    ///
    /// The injected Python signature is
    /// `run_subagent(name, instructions, context=None) -> str`, returning the
    /// subagent's [`Outcome::content`](neuromance_common::task::Outcome::content).
    /// `cancel` is passed to every subagent run, so cancelling it aborts in-flight
    /// delegations.
    ///
    /// # Errors
    /// Returns [`ReplError`] if injecting the callback into the REPL fails.
    pub fn new(
        repl: Arc<PythonRepl>,
        subagents: HashMap<String, Arc<dyn Subagent>>,
        cancel: CancellationToken,
    ) -> Result<Self, ReplError> {
        let bridge = Self {
            repl,
            subagents: Arc::new(subagents),
            cancel,
        };
        bridge.inject_run_subagent()?;
        Ok(bridge)
    }

    /// The prepared REPL, with `run_subagent` available in its namespace.
    #[must_use]
    pub fn repl(&self) -> Arc<PythonRepl> {
        Arc::clone(&self.repl)
    }

    /// Consume the bridge, exposing the prepared REPL as an `execute_python` tool.
    #[must_use]
    pub fn into_tool(self) -> PythonReplTool {
        PythonReplTool::new(self.repl)
    }

    fn inject_run_subagent(&self) -> Result<(), ReplError> {
        let subagents = Arc::clone(&self.subagents);
        let cancel = self.cancel.clone();

        self.repl
            .inject_function("run_subagent", move |args, kwargs| {
                let subagents = Arc::clone(&subagents);
                let cancel = cancel.clone();
                Box::pin(async move {
                    let name = args
                        .first()
                        .ok_or_else(|| "run_subagent: missing 'name' argument".to_string())?;
                    let instructions = args.get(1).ok_or_else(|| {
                        "run_subagent: missing 'instructions' argument".to_string()
                    })?;
                    let subagent = subagents
                        .get(name)
                        .ok_or_else(|| format!("run_subagent: unknown subagent '{name}'"))?;

                    let mut task = Task::new(instructions.clone());
                    if let Some(context) = kwargs.get("context") {
                        task = task.with_context(context.clone());
                    }

                    let outcome = subagent
                        .run(task, cancel)
                        .await
                        .map_err(|e| format!("subagent '{name}' failed: {e}"))?;
                    Ok(outcome.content)
                })
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use neuromance_common::subagent::SubagentError;
    use neuromance_common::task::Outcome;
    use serial_test::serial;

    use super::*;

    /// Echoes its instructions; records the context it last received.
    struct MockSubagent {
        id: String,
        last_context: Mutex<Option<String>>,
    }

    impl MockSubagent {
        fn new(id: &str) -> Arc<Self> {
            Arc::new(Self {
                id: id.to_string(),
                last_context: Mutex::new(None),
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
            *self.last_context.lock().unwrap() = task.context.clone();
            Ok(Outcome::new(task.id, format!("ran: {}", task.instructions)))
        }
    }

    fn registry(entries: Vec<(&str, Arc<dyn Subagent>)>) -> HashMap<String, Arc<dyn Subagent>> {
        entries
            .into_iter()
            .map(|(name, sub)| (name.to_string(), sub))
            .collect()
    }

    #[tokio::test]
    #[serial]
    async fn test_run_subagent_returns_outcome_content() {
        let repl = Arc::new(PythonRepl::new().unwrap());
        let worker = MockSubagent::new("worker");
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![("worker", worker as Arc<dyn Subagent>)]),
            CancellationToken::new(),
        )
        .unwrap();

        let result = bridge
            .repl()
            .execute("result = run_subagent('worker', 'do x')")
            .await
            .unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(
            repl.get_variable("result").await.unwrap().as_deref(),
            Some("ran: do x")
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_context_kwarg_threads_through() {
        let repl = Arc::new(PythonRepl::new().unwrap());
        let worker = MockSubagent::new("worker");
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![("worker", Arc::clone(&worker) as Arc<dyn Subagent>)]),
            CancellationToken::new(),
        )
        .unwrap();

        let result = bridge
            .repl()
            .execute("result = run_subagent('worker', 'do x', context='ctx')")
            .await
            .unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(
            worker.last_context.lock().unwrap().as_deref(),
            Some("ctx"),
            "context kwarg should reach the subagent",
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_unknown_subagent_raises_python_error() {
        let repl = Arc::new(PythonRepl::new().unwrap());
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![]),
            CancellationToken::new(),
        )
        .unwrap();

        let result = bridge
            .repl()
            .execute("run_subagent('nope', 'x')")
            .await
            .unwrap();
        assert!(!result.success);
        assert!(
            result.stderr.contains("unknown subagent 'nope'"),
            "stderr: {}",
            result.stderr
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_python_defined_fanout_vote_technique() {
        let repl = Arc::new(PythonRepl::new().unwrap());
        let worker = MockSubagent::new("worker");
        let judge = MockSubagent::new("judge");
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![
                ("worker", worker as Arc<dyn Subagent>),
                ("judge", judge as Arc<dyn Subagent>),
            ]),
            CancellationToken::new(),
        )
        .unwrap();

        // The "technique" is pure Python: fan out to the worker, then judge.
        let code = r"
answers = [run_subagent('worker', 'q%d' % i) for i in range(3)]
verdict = run_subagent('judge', 'pick:\n' + '\n'.join(answers))
";
        let result = bridge.repl().execute(code).await.unwrap();
        assert!(result.success, "stderr: {}", result.stderr);

        let verdict = repl.get_variable("verdict").await.unwrap().unwrap();
        // judge echoes its instructions, which embed every worker answer.
        assert!(verdict.starts_with("ran: pick:"));
        assert!(verdict.contains("ran: q0"));
        assert!(verdict.contains("ran: q1"));
        assert!(verdict.contains("ran: q2"));
    }
}
