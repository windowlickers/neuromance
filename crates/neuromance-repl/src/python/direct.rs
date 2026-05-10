//! Python REPL implementation using `PyO3`.
//!
//! Provides Python execution with:
//! - Restricted builtins (safe allowlist only)
//! - Persistent state between executions
//! - Function injection for callbacks to Rust
//! - Output capture (stdout/stderr)
//! - Configurable module imports

use crate::error::PyResultExt;
use crate::{ReplError, ReplResult};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::borrow::Cow;
use std::ffi::CString;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::PythonReplConfig;
use super::builtins::create_restricted_builtins;
use super::callback::{self, PythonCallback};
use super::capture::redirect_streams;
use super::state::{SharedState, WithShared};

/// Mutable state consolidated behind a single `Mutex`.
struct PythonState {
    globals: Py<PyDict>,
    shared: SharedState,
}

impl WithShared for PythonState {
    fn shared(&self) -> &SharedState {
        &self.shared
    }

    fn shared_mut(&mut self) -> &mut SharedState {
        &mut self.shared
    }
}

/// Python REPL environment with restricted builtins and state persistence.
///
/// # Thread Safety
///
/// `Py<T>` is `Send + Sync` in `PyO3` — it is a GIL-independent
/// reference-counted pointer. Actual Python object access requires
/// `Python::attach()` + `.bind(py)` to re-acquire the GIL. All
/// mutable state is consolidated in a single `std::sync::Mutex`
/// (not `tokio::sync::Mutex`) because all Python work runs inside
/// `spawn_blocking` and the GIL already serializes Python access.
pub struct PythonRepl {
    config: PythonReplConfig,
    state: Arc<Mutex<PythonState>>,
}

impl std::fmt::Debug for PythonRepl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonRepl")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PythonRepl {
    /// Create a new Python REPL with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if Python initialization fails.
    pub fn new() -> Result<Self, ReplError> {
        Self::with_config(PythonReplConfig::default())
    }

    /// Create a new Python REPL with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns `ReplError` if Python initialization fails.
    pub fn with_config(config: PythonReplConfig) -> Result<Self, ReplError> {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            let locals = PyDict::new(py);

            // Setup restricted builtins
            let builtins = create_restricted_builtins(py, &config.python_modules)?;
            globals
                .set_item("__builtins__", builtins)
                .at("set_item __builtins__ in globals")?;

            // Add configured modules
            Self::add_modules(py, &globals, &config.python_modules)?;

            Ok(Self {
                config,
                state: Arc::new(Mutex::new(PythonState {
                    globals: globals.unbind(),
                    shared: SharedState::new(locals.unbind()),
                })),
            })
        })
    }

    /// Add modules from configuration.
    fn add_modules(
        py: Python,
        globals: &Bound<PyDict>,
        modules: &[Cow<'static, str>],
    ) -> Result<(), ReplError> {
        for module_name in modules {
            match py.import(module_name.as_ref()) {
                Ok(module) => {
                    globals
                        .set_item(module_name.as_ref(), module)
                        .at("set_item user-configured module in globals")?;
                }
                Err(e) => {
                    log::warn!("Failed to import Python module '{module_name}': {e}");
                }
            }
        }

        Ok(())
    }
}

impl PythonRepl {
    /// Execute code in the REPL. State persists between calls.
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
                    let s = &mut *state
                        .lock()
                        .map_err(|e| ReplError::StatePoisoned(e.to_string()))?;

                    let globals_ref = s.globals.bind(py);
                    let locals_ref = s.shared.locals.bind(py);

                    callback::inject_callbacks_if_needed(
                        py,
                        globals_ref,
                        &s.shared.callbacks,
                        &mut s.shared.injected_callbacks,
                    )?;

                    let c_code = CString::new(code.as_str()).map_err(|e| {
                        ReplError::InvalidInput(format!(
                            "code contains an interior NUL byte at position {}",
                            e.nul_position()
                        ))
                    })?;

                    let streams = redirect_streams(py)?;

                    let exec_result = py
                        .run(c_code.as_c_str(), Some(globals_ref), Some(locals_ref))
                        .map_err(ReplError::PythonExec);

                    let (stdout, stderr) = streams.restore(py)?;
                    let elapsed_ms = start.elapsed().as_millis() as u64;

                    match exec_result {
                        Ok(()) => Ok(ReplResult {
                            stdout,
                            stderr,
                            success: true,
                            return_value: None,
                            execution_time_ms: elapsed_ms,
                        }),
                        Err(e) => Ok(ReplResult {
                            stdout: String::new(),
                            stderr: e.to_string(),
                            success: false,
                            return_value: None,
                            execution_time_ms: elapsed_ms,
                        }),
                    }
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
                let mut guard = state
                    .lock()
                    .map_err(|e| ReplError::StatePoisoned(e.to_string()))?;
                guard.shared.locals = PyDict::new(py).unbind();
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::redundant_clone)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::collections::HashMap;
    use std::time::Duration;

    #[tokio::test]
    #[serial]
    async fn test_python_repl_basic_execution() {
        let repl = PythonRepl::new().unwrap();
        let result = repl.execute("x = 10 + 5").await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_state_persistence() {
        let repl = PythonRepl::new().unwrap();

        let _ = repl.execute("x = 10").await.unwrap();
        let _ = repl.execute("y = x + 5").await.unwrap();

        let y_value = repl.get_variable("y").await.unwrap();
        assert_eq!(y_value, Some("15".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_error_handling() {
        let repl = PythonRepl::new().unwrap();
        let result = repl.execute("1 / 0").await.unwrap();
        assert!(!result.success);
        assert!(result.stderr.contains("ZeroDivisionError"));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_reset() {
        let repl = PythonRepl::new().unwrap();

        let _ = repl.execute("x = 42").await.unwrap();
        assert_eq!(
            repl.get_variable("x").await.unwrap(),
            Some("42".to_string())
        );

        repl.reset().await.unwrap();
        assert_eq!(repl.get_variable("x").await.unwrap(), None);
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_callback_injection() {
        let repl = PythonRepl::new().unwrap();

        // Inject a callback that doubles the input number
        repl.inject_function(
            "double",
            Box::new(|args: Vec<String>, _kwargs: HashMap<String, String>| {
                Box::pin(async move {
                    if let Some(arg) = args.first()
                        && let Ok(num) = arg.parse::<i32>()
                    {
                        return Ok((num * 2).to_string());
                    }
                    Err("Invalid argument".to_string())
                })
            }),
        )
        .unwrap();

        let _ = repl.execute("result = double('21')").await.unwrap();
        let result = repl.get_variable("result").await.unwrap();
        assert_eq!(result, Some("42".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_callback_err_propagates_as_runtime_error() {
        let repl = PythonRepl::new().unwrap();
        repl.inject_function(
            "boom",
            Box::new(|_args: Vec<String>, _kwargs: HashMap<String, String>| {
                Box::pin(async move { Err("boom: callback rejected input".to_string()) })
            }),
        )
        .unwrap();

        let result = repl.execute("boom()").await.unwrap();
        assert!(!result.success);
        assert!(
            result.stderr.contains("RuntimeError"),
            "expected RuntimeError, got: {}",
            result.stderr,
        );
        assert!(
            result.stderr.contains("boom: callback rejected input"),
            "expected propagated message, got: {}",
            result.stderr,
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_reinject_callback_replaces_previous() {
        let repl = PythonRepl::new().unwrap();

        repl.inject_function(
            "f",
            Box::new(|_args: Vec<String>, _kwargs: HashMap<String, String>| {
                Box::pin(async move { Ok("A".to_string()) })
            }),
        )
        .unwrap();

        let r1 = repl.execute("x = f()").await.unwrap();
        assert!(r1.success, "stderr: {}", r1.stderr);
        assert_eq!(repl.get_variable("x").await.unwrap(), Some("A".into()));

        // Re-inject under the same name. mod.rs::inject_function removes "f"
        // from injected_callbacks so callback::inject_callbacks_if_needed
        // will rebind on the next execute().
        repl.inject_function(
            "f",
            Box::new(|_args: Vec<String>, _kwargs: HashMap<String, String>| {
                Box::pin(async move { Ok("B".to_string()) })
            }),
        )
        .unwrap();

        let r2 = repl.execute("y = f()").await.unwrap();
        assert!(r2.success, "stderr: {}", r2.stderr);
        assert_eq!(
            repl.get_variable("y").await.unwrap(),
            Some("B".into()),
            "second injection did not replace the first PyCFunction"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_stdout_capture() {
        let repl = PythonRepl::new().unwrap();
        let result = repl.execute("print('hi')").await.unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "hi");
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_multiple_prints() {
        let repl = PythonRepl::new().unwrap();
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
    async fn test_python_repl_restricted_builtins() {
        let repl = PythonRepl::new().unwrap();

        // Safe builtins work
        let result = repl.execute("x = len([1, 2, 3])").await.unwrap();
        assert!(result.success);

        // exec is blocked
        let result = repl.execute("exec('print(1)')").await.unwrap();
        assert!(!result.success);

        // __import__('os') is blocked (direct sandbox escape)
        let result = repl.execute("__import__('os')").await.unwrap();
        assert!(!result.success);
        assert!(result.stderr.contains("Import of 'os'"));

        // import os statement is blocked
        let result = repl.execute("import os").await.unwrap();
        assert!(!result.success);
        assert!(result.stderr.contains("Import of 'os'"));

        // Configured module via import statement works
        let result = repl.execute("import math\ny = math.pi").await.unwrap();
        assert!(result.success);

        // Submodule of configured module works
        let result = repl
            .execute("from collections import defaultdict")
            .await
            .unwrap();
        assert!(result.success);
    }

    /// Systematically attempts every known import sandbox escape vector.
    #[tokio::test]
    #[serial]
    async fn test_import_sandbox_escape_attempts() {
        let repl = PythonRepl::new().unwrap();

        let blocked = [
            // Direct __import__ call
            ("__import__('os')", "direct __import__ call"),
            // Import statements for disallowed modules
            ("import os", "import statement"),
            ("import subprocess", "import subprocess"),
            ("import shutil", "import shutil"),
            // from ... import
            ("from os import system", "from os import"),
            ("from os.path import join", "from os.path import"),
            // Dotted import (top-level is checked)
            ("import os.path", "dotted import"),
            // __builtins__ dict gives back our restricted import
            (
                "__builtins__['__import__']('os')",
                "__builtins__ dict access",
            ),
            // importlib is not in allowed modules
            ("import importlib", "import importlib"),
            (
                "__import__('importlib').import_module('os')",
                "importlib.import_module",
            ),
            // eval/exec/compile are not in safe builtins
            ("eval(\"__import__('os')\")", "eval"),
            ("exec('import os')", "exec"),
            ("compile('import os', '', 'exec')", "compile"),
            // Reach real __import__ through a pre-loaded module's globals
            (
                "math.__builtins__['__import__']('os')",
                "module __builtins__ dict",
            ),
            (
                "type(math).__dict__['__builtins__'].__getitem__('__import__')('os')",
                "type(module).__dict__ chain",
            ),
            // Walk MRO to find a class whose __init__.__globals__
            // has the real builtins
            (
                concat!(
                    "[c for c in ",
                    "().__class__.__bases__[0].__subclasses__() ",
                    "if c.__name__ == 'catch_warnings'",
                    "][0].__init__.__globals__['__import__']('os')",
                ),
                "subclass walk to catch_warnings",
            ),
            // Patch the allowlist through __globals__
            (
                concat!(
                    "__import__.__globals__['_allowed']",
                    ".add('os'); import os",
                ),
                "patch _allowed via __globals__",
            ),
            // Grab the real __import__ from the _b reference
            (
                concat!(
                    "__builtins__['__import__'] = ",
                    "__import__.__globals__['_b'].__import__; ",
                    "import os",
                ),
                "replace import with real via _b.__import__",
            ),
        ];

        for (code, label) in &blocked {
            let result = repl.execute(code).await.unwrap();
            assert!(!result.success, "ESCAPE: {label} succeeded — code: {code}");
        }
    }

    /// Property: any module name whose top-level segment is not in the
    /// configured allowlist must be rejected with an `Import of '{name}'`
    /// message, regardless of dotted suffix or letter case.
    #[tokio::test]
    #[serial]
    async fn prop_disallowed_imports_always_blocked() {
        use proptest::strategy::{Strategy, ValueTree};
        use proptest::test_runner::{Config, TestRunner};
        use std::collections::HashSet;

        let repl = PythonRepl::new().unwrap();
        let allowed: HashSet<String> = repl
            .config()
            .python_modules
            .iter()
            .map(ToString::to_string)
            .collect();

        let allowed_filter = allowed.clone();
        let strategy = (
            "[A-Za-z][A-Za-z0-9_]{0,15}",
            proptest::option::of("[A-Za-z][A-Za-z0-9_]{0,15}(\\.[A-Za-z][A-Za-z0-9_]{0,15}){0,3}"),
        )
            .prop_filter(
                "top-level segment must not be in allowlist",
                move |(base, _)| !allowed_filter.contains(base),
            );

        // Generate cases synchronously, then exercise the async REPL serially.
        // proptest's `proptest!` macro is sync-only and `repl.execute` is async,
        // so we drive the strategy by hand instead of nesting block_on.
        let mut runner = TestRunner::new(Config {
            cases: 64,
            ..Config::default()
        });
        let mut cases: Vec<(String, Option<String>)> = Vec::with_capacity(64);
        for _ in 0..64 {
            let tree = strategy.new_tree(&mut runner).unwrap();
            cases.push(tree.current());
        }

        for (base, suffix) in cases {
            let name = match suffix {
                Some(s) => format!("{base}.{s}"),
                None => base,
            };
            let code = format!("__import__({name:?})");
            let result = repl.execute(&code).await.unwrap();
            assert!(!result.success, "import of {name:?} unexpectedly succeeded");
            let needle = format!("Import of '{name}'");
            assert!(
                result.stderr.contains(&needle),
                "import of {name:?}: stderr={}",
                result.stderr
            );
        }
    }

    // Known limitation: Function definitions across multiple execute() calls
    // don't persist. Use InteractivePythonRepl for better multi-line support.
    #[tokio::test]
    #[serial]
    #[ignore = "Function definitions don't persist across execute() calls"]
    async fn test_python_repl_functions() {
        let repl = PythonRepl::new().unwrap();

        // Define function
        let _ = repl
            .execute(
                r"
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
",
            )
            .await
            .unwrap();

        // Call function in separate execution
        let _ = repl.execute("result = factorial(5)").await.unwrap();

        let result = repl.get_variable("result").await.unwrap();
        assert_eq!(result, Some("120".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_loops() {
        let repl = PythonRepl::new().unwrap();

        let _ = repl
            .execute(
                r"
total = 0
for i in range(1, 11):
    total += i
",
            )
            .await
            .unwrap();

        let total = repl.get_variable("total").await.unwrap();
        assert_eq!(total, Some("55".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_safe_modules() {
        let repl = PythonRepl::new().unwrap();

        // Math module - pre-loaded, can use directly
        let _ = repl.execute("result = math.sqrt(16)").await.unwrap();
        assert_eq!(
            repl.get_variable("result").await.unwrap(),
            Some("4.0".to_string())
        );

        // JSON module - pre-loaded, can use directly
        let _ = repl
            .execute("data = json.dumps({'key': 'value'})")
            .await
            .unwrap();
        let data = repl.get_variable("data").await.unwrap();
        assert!(data.unwrap().contains("key"));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_list_comprehensions() {
        let repl = PythonRepl::new().unwrap();

        let _ = repl
            .execute("squares = [x**2 for x in range(5)]")
            .await
            .unwrap();
        let _ = repl.execute("count = len(squares)").await.unwrap();

        let count = repl.get_variable("count").await.unwrap();
        assert_eq!(count, Some("5".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_dict_operations() {
        let repl = PythonRepl::new().unwrap();

        let _ = repl
            .execute("data = {'name': 'Alice', 'age': 30}")
            .await
            .unwrap();
        let _ = repl.execute("name = data['name']").await.unwrap();

        let name = repl.get_variable("name").await.unwrap();
        assert_eq!(name, Some("Alice".to_string()));
    }

    // Known limitation: Complex multi-line exception handling
    // Use InteractivePythonRepl for better Python-native execution.
    #[tokio::test]
    #[serial]
    #[ignore = "Complex multi-line exception handling not supported"]
    async fn test_python_repl_exception_handling() {
        let repl = PythonRepl::new().unwrap();

        // Python exception handling in multi-line blocks
        let code = "try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    x = -1";

        let result = repl.execute(code).await.unwrap();
        assert!(result.success);

        let x = repl.get_variable("x").await.unwrap();
        assert_eq!(x, Some("-1".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_set_variable() {
        let repl = PythonRepl::new().unwrap();

        repl.set_variable("injected_value", "42").await.unwrap();
        let _ = repl
            .execute("doubled = int(injected_value) * 2")
            .await
            .unwrap();

        let doubled = repl.get_variable("doubled").await.unwrap();
        assert_eq!(doubled, Some("84".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_lambda_functions() {
        let repl = PythonRepl::new().unwrap();

        let _ = repl.execute("double = lambda x: x * 2").await.unwrap();
        let _ = repl.execute("result = double(21)").await.unwrap();

        let result = repl.get_variable("result").await.unwrap();
        assert_eq!(result, Some("42".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_multiple_callbacks() {
        let repl = PythonRepl::new().unwrap();

        // Inject multiple callbacks
        repl.inject_function(
            "add",
            Box::new(|args: Vec<String>, _kwargs: HashMap<String, String>| {
                Box::pin(async move {
                    if args.len() < 2 {
                        return Err("Need 2 arguments".to_string());
                    }
                    let a: i32 = args[0].parse().map_err(|_| "Invalid number".to_string())?;
                    let b: i32 = args[1].parse().map_err(|_| "Invalid number".to_string())?;
                    Ok((a + b).to_string())
                })
            }),
        )
        .unwrap();

        repl.inject_function(
            "multiply",
            Box::new(|args: Vec<String>, _kwargs: HashMap<String, String>| {
                Box::pin(async move {
                    if args.len() < 2 {
                        return Err("Need 2 arguments".to_string());
                    }
                    let a: i32 = args[0].parse().map_err(|_| "Invalid number".to_string())?;
                    let b: i32 = args[1].parse().map_err(|_| "Invalid number".to_string())?;
                    Ok((a * b).to_string())
                })
            }),
        )
        .unwrap();

        let _ = repl.execute("result = add('10', '5')").await.unwrap();
        assert_eq!(
            repl.get_variable("result").await.unwrap(),
            Some("15".to_string())
        );

        let _ = repl.execute("result2 = multiply('3', '7')").await.unwrap();
        assert_eq!(
            repl.get_variable("result2").await.unwrap(),
            Some("21".to_string())
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_callback_with_kwargs() {
        let repl = PythonRepl::new().unwrap();

        // Inject a callback that uses keyword arguments
        repl.inject_function(
            "greet",
            Box::new(|args: Vec<String>, kwargs: HashMap<String, String>| {
                Box::pin(async move {
                    let name = args.first().cloned().unwrap_or_else(|| "World".to_string());
                    let greeting = kwargs
                        .get("greeting")
                        .cloned()
                        .unwrap_or_else(|| "Hello".to_string());
                    Ok(format!("{greeting}, {name}!"))
                })
            }),
        )
        .unwrap();

        // Test with positional arg only
        let _ = repl.execute("result1 = greet('Alice')").await.unwrap();
        assert_eq!(
            repl.get_variable("result1").await.unwrap(),
            Some("Hello, Alice!".to_string())
        );

        // Test with keyword argument
        let _ = repl
            .execute("result2 = greet('Bob', greeting='Hi')")
            .await
            .unwrap();
        assert_eq!(
            repl.get_variable("result2").await.unwrap(),
            Some("Hi, Bob!".to_string())
        );

        // Test with only keyword argument (no positional)
        let _ = repl
            .execute("result3 = greet(greeting='Hey')")
            .await
            .unwrap();
        assert_eq!(
            repl.get_variable("result3").await.unwrap(),
            Some("Hey, World!".to_string())
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_callback_str_coerces_non_string_args() {
        let repl = PythonRepl::new().unwrap();

        let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);

        repl.inject_function(
            "echo",
            Box::new(move |args: Vec<String>, kwargs: HashMap<String, String>| {
                let captured = Arc::clone(&captured_clone);
                Box::pin(async move {
                    let mut all = args.clone();
                    let mut keys: Vec<_> = kwargs.keys().collect();
                    keys.sort();
                    for k in keys {
                        all.push(format!("{k}={}", kwargs[k]));
                    }
                    captured.lock().unwrap().extend(all.iter().cloned());
                    Ok(all.join("|"))
                })
            }),
        )
        .unwrap();

        let result = repl.execute("r1 = echo(42)").await.unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(repl.get_variable("r1").await.unwrap(), Some("42".into()));

        let result = repl.execute("r2 = echo([1, 2, 3])").await.unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(
            repl.get_variable("r2").await.unwrap(),
            Some("[1, 2, 3]".into())
        );

        let result = repl.execute("r3 = echo(payload={'k': 1})").await.unwrap();
        assert!(result.success, "stderr: {}", result.stderr);
        assert_eq!(
            repl.get_variable("r3").await.unwrap(),
            Some("payload={'k': 1}".into())
        );

        let snapshot = captured.lock().unwrap().clone();
        assert_eq!(
            snapshot,
            vec!["42", "[1, 2, 3]", "payload={'k': 1}"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_config() {
        let config = PythonReplConfig {
            timeout: Duration::from_secs(10),
            python_modules: vec!["math".into(), "json".into()],
        };

        let repl = PythonRepl::with_config(config).unwrap();
        assert_eq!(repl.config().timeout, Duration::from_secs(10));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_callback_injection_efficiency() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let repl = PythonRepl::new().unwrap();

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        repl.inject_function(
            "track_calls",
            Box::new(move |args: Vec<String>, _kwargs: HashMap<String, String>| {
                let call_count = Arc::clone(&call_count_clone);
                Box::pin(async move {
                    call_count.fetch_add(1, Ordering::SeqCst);
                    Ok(format!("Called with: {args:?}"))
                })
            }),
        )
        .unwrap();

        // Capture id(track_calls) on every execute. If the gate in
        // inject_callbacks_if_needed regressed and rebuilt the
        // PyCFunction each call, these ids would diverge.
        for i in 0..5 {
            let code =
                format!("result_{i} = track_calls('test_{i}')\ntrack_id_{i} = id(track_calls)");
            let _ = repl.execute(&code).await.unwrap();
        }

        let id0 = repl.get_variable("track_id_0").await.unwrap();
        assert!(id0.is_some(), "track_id_0 should have been bound");
        for i in 1..5 {
            let id_i = repl.get_variable(&format!("track_id_{i}")).await.unwrap();
            assert_eq!(
                id_i, id0,
                "track_calls was re-injected on execute #{i}; \
                 expected the same PyCFunction across calls"
            );
        }

        for i in 0..5 {
            let var = repl.get_variable(&format!("result_{i}")).await.unwrap();
            assert!(var.is_some());
            assert!(var.unwrap().contains(&format!("test_{i}")));
        }
        assert_eq!(call_count.load(Ordering::SeqCst), 5);

        repl.reset().await.unwrap();

        let _ = repl
            .execute(
                "result_after_reset = track_calls('after_reset')\n\
                 track_id_after_reset = id(track_calls)",
            )
            .await
            .unwrap();

        let id_after_reset = repl.get_variable("track_id_after_reset").await.unwrap();
        assert!(id_after_reset.is_some());
        assert_ne!(
            id_after_reset, id0,
            "reset() did not force re-injection; PyCFunction id was unchanged"
        );

        let var = repl.get_variable("result_after_reset").await.unwrap();
        assert!(var.unwrap().contains("after_reset"));
        assert_eq!(call_count.load(Ordering::SeqCst), 6);
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_execution_time_accuracy() {
        let config = PythonReplConfig::default().with_modules(["time"]);
        let repl = PythonRepl::with_config(config).unwrap();

        // Execute code with a known sleep duration
        let result = repl
            .execute(
                r"
import time
time.sleep(0.05)  # Sleep for 50ms
x = 42
",
            )
            .await
            .unwrap();

        assert!(result.success);

        // Execution time should be roughly 50ms, not including any queue wait time
        // Allow 20ms margin for overhead (lock acquisition, Python startup, etc.)
        assert!(
            result.execution_time_ms >= 50,
            "Execution time should be at least 50ms, got {}ms",
            result.execution_time_ms
        );

        // But shouldn't be excessively high (would indicate queue wait time was included)
        assert!(
            result.execution_time_ms < 200,
            "Execution time suspiciously high ({}ms), may include queue wait time",
            result.execution_time_ms
        );

        // Verify the code actually executed
        let x = repl.get_variable("x").await.unwrap();
        assert_eq!(x, Some("42".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_fast_execution_timing() {
        let repl = PythonRepl::new().unwrap();

        // Execute very fast code
        let result = repl.execute("y = 1 + 1").await.unwrap();

        assert!(result.success);

        // Should complete in well under 100ms
        assert!(
            result.execution_time_ms < 100,
            "Simple arithmetic took {}ms, expected < 100ms",
            result.execution_time_ms
        );

        // Verify the code executed
        let y = repl.get_variable("y").await.unwrap();
        assert_eq!(y, Some("2".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_timeout_returns_timeout_error() {
        let config = PythonReplConfig {
            timeout: Duration::from_millis(100),
            python_modules: vec!["time".into()],
        };
        let repl = PythonRepl::with_config(config).unwrap();
        let err = repl
            .execute("import time; time.sleep(5)")
            .await
            .unwrap_err();
        assert!(matches!(err, ReplError::Timeout(d) if d == Duration::from_millis(100)));
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_rejects_nul_byte_in_code() {
        let repl = PythonRepl::new().unwrap();
        let err = repl.execute("x = 1\0y = 2").await.unwrap_err();
        assert!(matches!(err, ReplError::InvalidInput(ref msg) if msg.contains("NUL")));
    }
}
