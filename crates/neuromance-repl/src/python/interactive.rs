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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::RwLock;

use crate::{ReplConfig, ReplEnvironment, ReplError, ReplResult};

use super::PythonCallback;

/// Python REPL using `InteractiveConsole` for multi-line support.
pub struct InteractivePythonRepl {
    /// Configuration
    config: ReplConfig,

    /// Python's code.InteractiveConsole instance
    console: Arc<RwLock<Py<PyAny>>>,

    /// Local namespace accessible to console
    locals: Arc<RwLock<Py<PyDict>>>,

    /// Injected callback functions
    callbacks: Arc<RwLock<HashMap<String, Arc<PythonCallback>>>>,
}

impl InteractivePythonRepl {
    /// Create a new interactive Python REPL with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if Python initialization fails.
    pub fn new() -> Result<Self, ReplError> {
        Self::with_config(ReplConfig::default())
    }

    /// Create a new interactive Python REPL with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if Python initialization fails.
    pub fn with_config(config: ReplConfig) -> Result<Self, ReplError> {
        Python::attach(|py| {
            // Import code module
            let code_module = py
                .import("code")
                .map_err(|e| ReplError::InitializationError(e.to_string()))?;

            // Create local namespace
            let locals = PyDict::new(py);

            // Create InteractiveConsole instance
            let interactive_console = code_module
                .getattr("InteractiveConsole")
                .map_err(|e| ReplError::InitializationError(e.to_string()))?;

            let console = interactive_console
                .call1((locals.clone(),))
                .map_err(|e| ReplError::InitializationError(e.to_string()))?;

            Ok(Self {
                config,
                console: Arc::new(RwLock::new(console.unbind())),
                locals: Arc::new(RwLock::new(locals.unbind())),
                callbacks: Arc::new(RwLock::new(std::collections::HashMap::new())),
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
    #[allow(clippy::significant_drop_tightening)] // Guard is used multiple times
    pub async fn push(&self, line: &str) -> Result<bool, ReplError> {
        let console_arc = Arc::clone(&self.console);
        let line = line.to_string();

        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let console = console_arc.blocking_read();
                let console_ref = console.bind(py);

                console_ref
                    .call_method1("push", (line,))
                    .map_err(|e| ReplError::ExecutionError(e.to_string()))?
                    .extract::<bool>()
                    .map_err(|e| ReplError::ExecutionError(e.to_string()))
            })
        })
        .await
        .map_err(|e| ReplError::ExecutionError(e.to_string()))?
    }
}

#[async_trait]
#[allow(clippy::significant_drop_tightening)] // Guards are used multiple times
impl ReplEnvironment for InteractivePythonRepl {
    #[allow(clippy::cast_possible_truncation)] // Execution time won't exceed u64::MAX ms
    async fn execute(&self, code: &str) -> Result<ReplResult, ReplError> {
        let start = Instant::now();

        let code = code.to_string();
        let console_arc = Arc::clone(&self.console);
        let callbacks_arc = Arc::clone(&self.callbacks);
        let locals_arc = Arc::clone(&self.locals);
        let timeout = self.config.timeout;

        // Execute in blocking task with timeout
        let result =
            tokio::time::timeout(
                timeout,
                tokio::task::spawn_blocking(move || {
                    Python::attach(|py| {
                        // Inject callbacks into locals
                        let callbacks = callbacks_arc.blocking_read();
                        let locals_guard = locals_arc.blocking_read();
                        let locals_ref = locals_guard.bind(py);

                        for (name, callback) in callbacks.iter() {
                            let cb = Arc::clone(callback);
                            let py_func = pyo3::types::PyCFunction::new_closure(
                            py,
                            None,
                            None,
                            move |args: &Bound<pyo3::types::PyTuple>,
                                  kwargs: Option<&Bound<PyDict>>| {
                                #[allow(clippy::unnecessary_debug_formatting)]
                                let args_vec: Vec<String> = args
                                    .iter()
                                    .map(|arg| {
                                        arg.extract::<String>()
                                            .unwrap_or_else(|_| format!("{arg:?}"))
                                    })
                                    .collect();

                                // Extract keyword arguments into a HashMap
                                #[allow(clippy::unnecessary_debug_formatting)]
                                let kwargs_map: HashMap<String, String> = kwargs
                                    .map(|kw| {
                                        kw.iter()
                                            .filter_map(|(k, v)| {
                                                let key = k.extract::<String>().ok()?;
                                                let value = v
                                                    .extract::<String>()
                                                    .unwrap_or_else(|_| format!("{v:?}"));
                                                Some((key, value))
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default();

                                match cb(args_vec, kwargs_map) {
                                    Ok(result) => Ok(result),
                                    Err(e) => {
                                        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
                                    }
                                }
                            },
                        )
                        .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

                            locals_ref
                                .set_item(name.as_str(), py_func)
                                .map_err(|e| ReplError::ExecutionError(e.to_string()))?;
                        }
                        drop(callbacks);
                        drop(locals_guard);

                        // Push each line to the console
                        let console = console_arc.blocking_read();
                        let console_ref = console.bind(py);

                        for line in code.lines() {
                            console_ref
                                .call_method1("push", (line,))
                                .map_err(|e| ReplError::ExecutionError(e.to_string()))?;
                        }

                        // Push empty line to finalize any incomplete statement
                        console_ref
                            .call_method1("push", ("",))
                            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

                        Ok(ReplResult::success(String::new())
                            .with_execution_time(start.elapsed().as_millis() as u64))
                    })
                }),
            )
            .await;

        match result {
            Ok(Ok(repl_result)) => repl_result,
            Ok(Err(e)) => Err(ReplError::ExecutionError(format!("{e:?}"))),
            Err(_) => Err(ReplError::Timeout(timeout)),
        }
    }

    async fn reset(&self) -> Result<(), ReplError> {
        let console_arc = Arc::clone(&self.console);
        let locals_arc = Arc::clone(&self.locals);

        // Reset by creating a new InteractiveConsole in blocking task
        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let code_module = py
                    .import("code")
                    .map_err(|e| ReplError::InitializationError(e.to_string()))?;

                let locals = PyDict::new(py);
                let interactive_console = code_module
                    .getattr("InteractiveConsole")
                    .map_err(|e| ReplError::InitializationError(e.to_string()))?;

                let console = interactive_console
                    .call1((locals.clone(),))
                    .map_err(|e| ReplError::InitializationError(e.to_string()))?;

                // Update both console and locals
                *console_arc.blocking_write() = console.unbind();
                *locals_arc.blocking_write() = locals.unbind();

                Ok(())
            })
        })
        .await
        .map_err(|e| ReplError::ExecutionError(e.to_string()))?
    }

    fn config(&self) -> &ReplConfig {
        &self.config
    }

    fn language_name(&self) -> &'static str {
        "python"
    }

    async fn inject_function(&self, name: &str, callback: PythonCallback) -> Result<(), ReplError> {
        self.callbacks
            .write()
            .await
            .insert(name.to_string(), Arc::new(callback));
        Ok(())
    }

    async fn get_variable(&self, name: &str) -> Result<Option<String>, ReplError> {
        let locals_arc = Arc::clone(&self.locals);
        let name = name.to_string();

        let result = tokio::task::spawn_blocking(move || {
            let locals = locals_arc.blocking_read();
            Python::attach(|py| {
                let locals_dict = locals.bind(py);
                match locals_dict.get_item(&name) {
                    Ok(Some(value)) => {
                        let str_repr = value.str().map_err(|e| e.to_string())?;
                        Ok(Some(
                            str_repr.extract::<String>().map_err(|e| e.to_string())?,
                        ))
                    }
                    Ok(None) => Ok(None),
                    Err(e) => Err(e.to_string()),
                }
            })
        })
        .await
        .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        result.map_err(ReplError::ExecutionError)
    }

    async fn set_variable(&self, name: &str, value: &str) -> Result<(), ReplError> {
        let locals_arc = Arc::clone(&self.locals);
        let name = name.to_string();
        let value = value.to_string();

        tokio::task::spawn_blocking(move || {
            let locals = locals_arc.blocking_write();
            Python::attach(|py| {
                let locals_dict = locals.bind(py);
                let py_value = pyo3::IntoPyObject::into_pyobject(value, py)
                    .map_err(|e| ReplError::ExecutionError(e.to_string()))?;
                locals_dict
                    .set_item(&name, py_value)
                    .map_err(|e| ReplError::ExecutionError(e.to_string()))
            })
        })
        .await
        .map_err(|e| ReplError::ExecutionError(e.to_string()))?
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_interactive_basic_execution() {
        let repl = InteractivePythonRepl::new().unwrap();
        let result = repl.execute("x = 10 + 5").await.unwrap();
        assert!(result.success);

        let x = repl.get_variable("x").await.unwrap();
        assert_eq!(x, Some("15".to_string()));
    }

    #[tokio::test]
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
    async fn test_interactive_state_persistence() {
        let repl = InteractivePythonRepl::new().unwrap();

        let _ = repl.execute("x = 10").await.unwrap();
        let _ = repl.execute("y = x + 5").await.unwrap();

        let y = repl.get_variable("y").await.unwrap();
        assert_eq!(y, Some("15".to_string()));
    }

    #[tokio::test]
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
