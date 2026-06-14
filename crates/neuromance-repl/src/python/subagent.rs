//! Bridge that exposes [`Subagent`]s to Python code running in the REPL.
//!
//! [`SubagentRepl`] injects two primitives into the namespace, backed by a
//! registry of named subagents. Because the registry holds `Arc<dyn Subagent>`,
//! entries can be leaf agents or combinators alike — the Python side only sees
//! uniform primitives, and agents can *write their own* orchestration techniques
//! in Python instead of relying on fixed Rust combinators:
//!
//! - `run_subagent(name, instructions, context=None) -> str` runs one subagent
//!   and blocks for its result. Use it for **dependent** chains where each step
//!   needs the previous step's output.
//! - `spawn_agents([Agent(...), ...]) -> list[str]` runs a batch of independent
//!   subagents **concurrently** and returns their results in input order. Use it
//!   for fan-out, where serial `run_subagent` calls would needlessly run one at a
//!   time.
//!
//! ```python
//! # fan out concurrently, then judge — the fan-out runs in parallel
//! answers = spawn_agents([Agent('worker', question) for _ in range(3)])
//! verdict = run_subagent('judge', 'Pick the best:\n' + '\n'.join(answers))
//! ```
//!
//! The bridge depends only on the `neuromance-common` subagent contract, never on
//! the agent crate, so `agent -> repl` (repl-as-a-tool) stays cycle-free.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use neuromance_common::subagent::Subagent;
use neuromance_common::task::Task;

use super::{PythonRepl, PythonReplTool};
use crate::ReplError;

/// One entry of a `spawn_agents` batch, deserialized from the JSON the Python
/// `spawn_agents` wrapper produces from `Agent(...)` specs.
#[derive(Debug, Deserialize)]
struct AgentSpec {
    /// Registry name of the subagent to run.
    agent: String,
    /// Task instructions for the subagent.
    instructions: String,
    /// Optional context threaded onto the task.
    context: Option<String>,
}

/// Python prelude defining the ergonomic `Agent`/`spawn_agents` helpers.
///
/// Defined in globals (see [`PythonRepl::define_prelude`]) so the helpers
/// survive a REPL `reset`. `spawn_agents` marshals the batch as JSON across the
/// string-coercing callback boundary and decodes the JSON array of results;
/// both `json` and `__spawn_agents_json` are resolved from globals at call time.
const PRELUDE: &str = r#"
def Agent(agent, instructions, context=None):
    return {"agent": agent, "instructions": instructions, "context": context}

def spawn_agents(agents):
    return json.loads(__spawn_agents_json(json.dumps(list(agents))))
"#;

/// A [`PythonRepl`] with named subagents injected as a `run_subagent` callable.
pub struct SubagentRepl {
    repl: Arc<PythonRepl>,
    subagents: Arc<HashMap<String, Arc<dyn Subagent>>>,
    cancel: CancellationToken,
}

impl SubagentRepl {
    /// Wrap `repl`, register `subagents` by name, and inject the delegation
    /// primitives `run_subagent` and `spawn_agents` plus their `Agent` helper.
    ///
    /// The injected Python signatures are:
    /// - `run_subagent(name, instructions, context=None) -> str` — runs one
    ///   subagent and returns its
    ///   [`Outcome::content`](neuromance_common::task::Outcome::content).
    /// - `spawn_agents([Agent(name, instructions, context=None), ...]) -> list[str]`
    ///   — runs the batch concurrently and returns each subagent's content in
    ///   input order.
    ///
    /// `cancel` is passed to every subagent run, so cancelling it aborts
    /// in-flight delegations.
    ///
    /// `spawn_agents` requires the `json` module, which is present in the default
    /// [`PythonReplConfig`](super::PythonReplConfig). A config that omits `json`
    /// makes `spawn_agents` raise `NameError` at call time.
    ///
    /// # Errors
    /// Returns [`ReplError`] if injecting a callback or the prelude into the REPL
    /// fails.
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
        bridge.inject_spawn_agents()?;
        bridge.repl.define_prelude(PRELUDE)?;
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

    fn inject_spawn_agents(&self) -> Result<(), ReplError> {
        let subagents = Arc::clone(&self.subagents);
        let cancel = self.cancel.clone();

        self.repl
            .inject_function("__spawn_agents_json", move |args, _kwargs| {
                let subagents = Arc::clone(&subagents);
                let cancel = cancel.clone();
                Box::pin(async move {
                    let payload = args
                        .first()
                        .ok_or_else(|| "spawn_agents: missing JSON payload".to_string())?;
                    let specs: Vec<AgentSpec> = serde_json::from_str(payload)
                        .map_err(|e| format!("spawn_agents: invalid spec payload: {e}"))?;

                    let mut runs = Vec::with_capacity(specs.len());
                    let mut unknown = Vec::new();
                    for spec in specs {
                        let Some(subagent) = subagents.get(&spec.agent).map(Arc::clone) else {
                            unknown.push(spec.agent);
                            continue;
                        };
                        let cancel = cancel.clone();
                        let name = spec.agent;
                        let mut task = Task::new(spec.instructions);
                        if let Some(context) = spec.context {
                            task = task.with_context(context);
                        }
                        runs.push(async move {
                            subagent
                                .run(task, cancel)
                                .await
                                .map(|outcome| outcome.content)
                                .map_err(|e| format!("agent '{name}' failed: {e}"))
                        });
                    }
                    if !unknown.is_empty() {
                        return Err(format!(
                            "spawn_agents: unknown subagent(s): {}",
                            unknown.join(", ")
                        ));
                    }

                    let mut contents = Vec::with_capacity(runs.len());
                    let mut errors = Vec::new();
                    for result in join_all(runs).await {
                        match result {
                            Ok(content) => contents.push(content),
                            Err(e) => errors.push(e),
                        }
                    }
                    if !errors.is_empty() {
                        return Err(errors.join("; "));
                    }

                    serde_json::to_string(&contents)
                        .map_err(|e| format!("spawn_agents: failed to encode results: {e}"))
                })
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use async_trait::async_trait;
    use neuromance_common::subagent::SubagentError;
    use neuromance_common::task::Outcome;
    use serial_test::serial;
    use tokio::sync::Barrier;

    use super::*;

    /// Echoes its instructions; records the context it last received.
    ///
    /// Optional knobs drive the batch tests: `barrier` makes `run` block until
    /// every concurrent run has arrived (proving concurrency), `delay` staggers
    /// completion order, and `fail` makes the run error.
    struct MockSubagent {
        id: String,
        last_context: Mutex<Option<String>>,
        barrier: Option<Arc<Barrier>>,
        delay: Duration,
        fail: bool,
    }

    impl MockSubagent {
        fn new(id: &str) -> Arc<Self> {
            Self::build(id, None, Duration::ZERO, false)
        }

        fn with_barrier(id: &str, barrier: Arc<Barrier>) -> Arc<Self> {
            Self::build(id, Some(barrier), Duration::ZERO, false)
        }

        fn with_delay(id: &str, delay: Duration) -> Arc<Self> {
            Self::build(id, None, delay, false)
        }

        fn failing(id: &str) -> Arc<Self> {
            Self::build(id, None, Duration::ZERO, true)
        }

        fn build(
            id: &str,
            barrier: Option<Arc<Barrier>>,
            delay: Duration,
            fail: bool,
        ) -> Arc<Self> {
            Arc::new(Self {
                id: id.to_string(),
                last_context: Mutex::new(None),
                barrier,
                delay,
                fail,
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
            if let Some(barrier) = &self.barrier {
                barrier.wait().await;
            }
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            if self.fail {
                return Err(SubagentError::execution(format!("{} boom", self.id)));
            }
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

    #[tokio::test]
    #[serial]
    async fn test_spawn_agents_runs_concurrently() {
        // The worker blocks on a 3-way barrier, so the batch can only complete
        // if all three runs are in flight at once. A sequential implementation
        // would block on the first run forever and trip the timeout.
        let barrier = Arc::new(Barrier::new(3));
        let worker = MockSubagent::with_barrier("worker", barrier);
        let repl = Arc::new(PythonRepl::new().unwrap());
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![("worker", worker as Arc<dyn Subagent>)]),
            CancellationToken::new(),
        )
        .unwrap();

        let code = r"
results = spawn_agents([Agent('worker', 'a'), Agent('worker', 'b'), Agent('worker', 'c')])
";
        let result = tokio::time::timeout(Duration::from_secs(2), bridge.repl().execute(code))
            .await
            .expect("spawn_agents deadlocked — runs were not concurrent")
            .unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(
            repl.get_variable("results").await.unwrap().as_deref(),
            Some("['ran: a', 'ran: b', 'ran: c']"),
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_spawn_agents_preserves_input_order() {
        // Descending delays make completion order (c, b, a) differ from input
        // order (a, b, c); the result must follow input order regardless.
        let repl = Arc::new(PythonRepl::new().unwrap());
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![
                (
                    "a",
                    MockSubagent::with_delay("a", Duration::from_millis(60)) as Arc<dyn Subagent>,
                ),
                (
                    "b",
                    MockSubagent::with_delay("b", Duration::from_millis(30)) as Arc<dyn Subagent>,
                ),
                (
                    "c",
                    MockSubagent::with_delay("c", Duration::ZERO) as Arc<dyn Subagent>,
                ),
            ]),
            CancellationToken::new(),
        )
        .unwrap();

        let code = r"
results = spawn_agents([Agent('a', 'first'), Agent('b', 'second'), Agent('c', 'third')])
";
        let result = bridge.repl().execute(code).await.unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(
            repl.get_variable("results").await.unwrap().as_deref(),
            Some("['ran: first', 'ran: second', 'ran: third']"),
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_spawn_agents_threads_context_per_spec() {
        let worker = MockSubagent::new("worker");
        let repl = Arc::new(PythonRepl::new().unwrap());
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![("worker", Arc::clone(&worker) as Arc<dyn Subagent>)]),
            CancellationToken::new(),
        )
        .unwrap();

        let result = bridge
            .repl()
            .execute("spawn_agents([Agent('worker', 'do x', context='ctx')])")
            .await
            .unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(
            worker.last_context.lock().unwrap().as_deref(),
            Some("ctx"),
            "per-spec context should reach the subagent",
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_spawn_agents_unknown_agent_raises() {
        let repl = Arc::new(PythonRepl::new().unwrap());
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![(
                "worker",
                MockSubagent::new("worker") as Arc<dyn Subagent>,
            )]),
            CancellationToken::new(),
        )
        .unwrap();

        let result = bridge
            .repl()
            .execute("spawn_agents([Agent('worker', 'x'), Agent('nope', 'y')])")
            .await
            .unwrap();
        assert!(!result.success);
        assert!(
            result.stderr.contains("unknown subagent(s): nope"),
            "stderr: {}",
            result.stderr
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_spawn_agents_aggregates_failures() {
        // One member fails; the batch raises naming it, while the sibling still ran.
        let good = MockSubagent::new("good");
        let bad = MockSubagent::failing("bad");
        let repl = Arc::new(PythonRepl::new().unwrap());
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![
                ("good", Arc::clone(&good) as Arc<dyn Subagent>),
                ("bad", bad as Arc<dyn Subagent>),
            ]),
            CancellationToken::new(),
        )
        .unwrap();

        let result = bridge
            .repl()
            .execute("spawn_agents([Agent('good', 'ok', context='g-ctx'), Agent('bad', 'boom')])")
            .await
            .unwrap();
        assert!(!result.success);
        assert!(
            result.stderr.contains("agent 'bad' failed"),
            "stderr: {}",
            result.stderr
        );
        assert_eq!(
            good.last_context.lock().unwrap().as_deref(),
            Some("g-ctx"),
            "the sibling should still have run despite the batch failing",
        );
    }

    /// Records the delegation context observed when run, proving whether the
    /// ambient context survived the relay into the subagent.
    struct DelegationProbe {
        seen: Mutex<Option<neuromance_common::delegation::DelegationContext>>,
    }

    #[async_trait]
    impl Subagent for DelegationProbe {
        #[allow(clippy::unnecessary_literal_bound)] // trait fixes the return as `&str`
        fn id(&self) -> &str {
            "probe"
        }

        async fn run(
            &self,
            task: Task,
            _cancel: CancellationToken,
        ) -> Result<Outcome, SubagentError> {
            *self.seen.lock().unwrap() = Some(neuromance_common::delegation::current());
            Ok(Outcome::new(task.id, "ok".to_string()))
        }
    }

    /// A subagent runs from inside `execute`'s `spawn_blocking`/`block_on`, which
    /// a task-local does not cross. The relay must carry the ambient delegation
    /// context to the subagent run anyway; without it the probe would observe the
    /// default context.
    #[tokio::test]
    #[serial]
    async fn test_delegation_context_relays_across_spawn_blocking() {
        use neuromance_common::delegation::{self, DelegationContext};

        let probe = Arc::new(DelegationProbe {
            seen: Mutex::new(None),
        });
        let repl = Arc::new(PythonRepl::new().unwrap());
        let bridge = SubagentRepl::new(
            Arc::clone(&repl),
            registry(vec![("probe", Arc::clone(&probe) as Arc<dyn Subagent>)]),
            CancellationToken::new(),
        )
        .unwrap();

        let parent = DelegationContext {
            conversation_id: Some(uuid::Uuid::new_v4()),
            task_id: Some(uuid::Uuid::new_v4()),
        };

        let result =
            delegation::scope(parent, bridge.repl().execute("run_subagent('probe', 'go')"))
                .await
                .unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(
            *probe.seen.lock().unwrap(),
            Some(parent),
            "the subagent run should observe the parent delegation context relayed \
             across the spawn_blocking boundary",
        );
    }
}
