//! Python REPL using Python's `code.InteractiveConsole`.
//!
//! Uses Python's built-in REPL infrastructure for:
//! - Automatic multi-line statement handling
//! - Better error formatting with tracebacks
//! - More Python-native execution model
//!
//! Trade-offs vs Direct implementation:
//! - Better UX for interactive use
//! - Less control over execution environment
//! - No restricted builtins

use std::sync::{Arc, Mutex};
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{ReplError, ReplResult};

use super::PythonReplConfig;
use super::callback::{self, PythonCallback};
use super::capture::redirect_streams;
use super::state::{SharedState, WithShared};

/// Mutable state consolidated behind a single `Mutex`.
struct InteractivePythonState {
    console: Py<PyAny>,
    shared: SharedState,
}

impl WithShared for InteractivePythonState {
    fn shared(&self) -> &SharedState {
        &self.shared
    }

    fn shared_mut(&mut self) -> &mut SharedState {
        &mut self.shared
    }
}

/// Python REPL using `InteractiveConsole` for multi-line support.
///
/// # Thread Safety
///
/// `Py<T>` is `Send + Sync` in `PyO3` — it is a GIL-independent
/// reference-counted pointer. Actual Python object access requires
/// `Python::attach()` + `.bind(py)` to re-acquire the GIL. All
/// mutable state is consolidated in a single `std::sync::Mutex`
/// (not `tokio::sync::Mutex`) because all Python work runs inside
/// `spawn_blocking` and the GIL already serializes Python access.
pub struct InteractivePythonRepl {
    config: PythonReplConfig,
    state: Arc<Mutex<InteractivePythonState>>,
}

impl std::fmt::Debug for InteractivePythonRepl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractivePythonRepl")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl InteractivePythonRepl {
    /// Create a new interactive Python REPL with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if Python initialization fails.
    pub fn new() -> Result<Self, ReplError> {
        Self::with_config(PythonReplConfig::default())
    }

    /// Create a new interactive Python REPL with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if Python initialization fails.
    pub fn with_config(config: PythonReplConfig) -> Result<Self, ReplError> {
        Python::attach(|py| {
            // Import code module
            let code_module = py.import("code")?;

            // Create local namespace
            let locals = PyDict::new(py);

            // Create InteractiveConsole instance
            let interactive_console = code_module.getattr("InteractiveConsole")?;

            let console = interactive_console.call1((locals.clone(),))?;

            Ok(Self {
                config,
                state: Arc::new(Mutex::new(InteractivePythonState {
                    console: console.unbind(),
                    shared: SharedState::new(locals.unbind()),
                })),
            })
        })
    }

    /// Push a line of code to the console.
    ///
    /// Returns `true` if more input is needed (incomplete statement).
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if execution fails.
    pub async fn push(&self, line: &str) -> Result<bool, ReplError> {
        let state = Arc::clone(&self.state);
        let line = line.to_string();

        tokio::task::spawn_blocking(move || {
            let guard = state.lock().map_err(|_| ReplError::StatePoisoned)?;
            Python::attach(|py| {
                let console_ref = guard.console.bind(py);

                Ok(console_ref
                    .call_method1("push", (line,))?
                    .extract::<bool>()?)
            })
        })
        .await?
    }
}

impl InteractivePythonRepl {
    /// Execute code in the interactive REPL. State persists between calls.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if execution fails or times out.
    #[allow(clippy::cast_possible_truncation)] // Execution time won't exceed u64::MAX ms
    pub async fn execute(&self, code: &str) -> Result<ReplResult, ReplError> {
        let code = code.to_string();
        let state = Arc::clone(&self.state);
        let timeout = self.config.timeout;

        let result = tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || {
                let start = Instant::now();

                Python::attach(|py| {
                    let s = &mut *state.lock().map_err(|_| ReplError::StatePoisoned)?;

                    let locals_ref = s.shared.locals.bind(py);

                    callback::inject_callbacks_if_needed(
                        py,
                        locals_ref,
                        &s.shared.callbacks,
                        &mut s.shared.injected_callbacks,
                    )?;

                    let streams = redirect_streams(py)?;

                    let console_ref = s.console.bind(py);

                    let mut exec_error = None;
                    for line in code.lines() {
                        if let Err(e) = console_ref.call_method1("push", (line,)) {
                            exec_error = Some(e);
                            break;
                        }
                    }

                    if exec_error.is_none()
                        && let Err(e) = console_ref.call_method1("push", ("",))
                    {
                        exec_error = Some(e);
                    }

                    let (stdout, stderr) = streams.restore(py)?;

                    if let Some(e) = exec_error {
                        return Err(e.into());
                    }

                    let success = stderr.is_empty();
                    Ok(ReplResult {
                        stdout,
                        stderr,
                        success,
                        return_value: None,
                        execution_time_ms: start.elapsed().as_millis() as u64,
                    })
                })
            }),
        )
        .await;

        match result {
            Ok(Ok(repl_result)) => repl_result,
            Ok(Err(e)) => Err(e.into()),
            Err(_) => Err(ReplError::Timeout(timeout)),
        }
    }

    /// Reset the REPL, clearing all state.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if reset fails.
    pub async fn reset(&self) -> Result<(), ReplError> {
        let state = Arc::clone(&self.state);

        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let mut guard = state.lock().map_err(|_| ReplError::StatePoisoned)?;

                let code_module = py.import("code")?;
                let locals = PyDict::new(py);
                let interactive_console = code_module.getattr("InteractiveConsole")?;
                let console = interactive_console.call1((locals.clone(),))?;

                guard.console = console.unbind();
                guard.shared.locals = locals.unbind();
                guard.shared.injected_callbacks.clear();
                drop(guard);

                Ok(())
            })
        })
        .await?
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &PythonReplConfig {
        &self.config
    }

    /// Inject a callable function into the REPL environment.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if the state mutex is poisoned.
    pub fn inject_function(&self, name: &str, callback: PythonCallback) -> Result<(), ReplError> {
        super::inject_function(&self.state, name, callback)
    }

    /// Get a variable's string representation from the REPL.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if the variable can't be serialized.
    pub async fn get_variable(&self, name: &str) -> Result<Option<String>, ReplError> {
        super::get_variable(&self.state, name).await
    }

    /// Set a variable in the REPL environment.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if the variable can't be set.
    pub async fn set_variable(&self, name: &str, value: &str) -> Result<(), ReplError> {
        super::set_variable(&self.state, name, value).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_interactive_basic_execution() {
        let repl = InteractivePythonRepl::new().unwrap();
        let result = repl.execute("x = 10 + 5").await.unwrap();
        assert!(result.success);

        let x = repl.get_variable("x").await.unwrap();
        assert_eq!(x, Some("15".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_interactive_multiline_function() {
        let repl = InteractivePythonRepl::new().unwrap();

        let code = r"
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
";

        let _ = repl.execute(code).await.unwrap();
        let result = repl.get_variable("result").await.unwrap();
        assert_eq!(result, Some("120".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_interactive_state_persistence() {
        let repl = InteractivePythonRepl::new().unwrap();

        let _ = repl.execute("x = 10").await.unwrap();
        let _ = repl.execute("y = x + 5").await.unwrap();

        let y = repl.get_variable("y").await.unwrap();
        assert_eq!(y, Some("15".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_interactive_push() {
        let repl = InteractivePythonRepl::new().unwrap();

        // Complete statement
        let needs_more = repl.push("x = 42").await.unwrap();
        assert!(!needs_more);

        // Incomplete statement
        let needs_more = repl.push("def foo():").await.unwrap();
        assert!(needs_more);

        // Complete the function
        let needs_more = repl.push("    return 10").await.unwrap();
        assert!(needs_more);

        // Empty line finalizes
        let needs_more = repl.push("").await.unwrap();
        assert!(!needs_more);
    }

    #[tokio::test]
    #[serial]
    async fn test_interactive_stdout_capture() {
        let repl = InteractivePythonRepl::new().unwrap();
        let result = repl.execute("print('hello world')").await.unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "hello world");
    }

    #[tokio::test]
    #[serial]
    async fn test_interactive_stderr_capture() {
        let repl = InteractivePythonRepl::new().unwrap();
        let result = repl.execute("1 / 0").await.unwrap();
        assert!(!result.success);
        assert!(result.stderr.contains("ZeroDivisionError"));
    }

    #[tokio::test]
    #[serial]
    async fn test_interactive_multiple_prints() {
        let repl = InteractivePythonRepl::new().unwrap();
        let result = repl
            .execute("print('line1')\nprint('line2')\nprint('line3')")
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.stdout.contains("line1"));
        assert!(result.stdout.contains("line2"));
        assert!(result.stdout.contains("line3"));
    }

    #[tokio::test]
    #[serial]
    async fn test_interactive_reset() {
        let repl = InteractivePythonRepl::new().unwrap();

        let _ = repl.execute("x = 42").await.unwrap();
        assert_eq!(
            repl.get_variable("x").await.unwrap(),
            Some("42".to_string())
        );

        repl.reset().await.unwrap();
        assert_eq!(repl.get_variable("x").await.unwrap(), None);
    }
}
