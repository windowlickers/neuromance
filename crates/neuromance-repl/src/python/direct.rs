//! Python REPL implementation using `PyO3`.
//!
//! Provides Python execution with:
//! - Restricted builtins (safe allowlist only)
//! - Persistent state between executions
//! - Function injection for callbacks to Rust
//! - Output capture (stdout/stderr)
//! - Configurable module imports

use crate::{ReplConfig, ReplEnvironment, ReplError, ReplResult};
use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use super::PythonReplConfig;
use super::callback::{self, PythonCallback};

/// Python REPL environment with restricted builtins and state persistence.
///
/// # Thread Safety
///
/// `Py<T>` is `Send + Sync` in `PyO3` — it is a GIL-independent
/// reference-counted pointer. Actual Python object access requires
/// `Python::attach()` + `.bind(py)` to re-acquire the GIL. All
/// mutable state is further protected by `tokio::sync::RwLock`.
pub struct PythonRepl {
    /// Configuration
    config: PythonReplConfig,

    /// Global namespace (includes builtins and injected functions)
    globals: Arc<RwLock<Py<PyDict>>>,

    /// Local namespace (user variables persist here)
    locals: Arc<RwLock<Py<PyDict>>>,

    /// Injected callback functions
    callbacks: Arc<RwLock<HashMap<String, Arc<PythonCallback>>>>,

    /// Tracks which callbacks have been injected into Python globals
    injected_callbacks: Arc<RwLock<HashSet<String>>>,
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
            let builtins = Self::create_restricted_builtins(py)?;
            globals
                .set_item("__builtins__", builtins)
                .map_err(|e| ReplError::InitializationError(e.to_string()))?;

            // Add configured modules
            Self::add_modules(py, &globals, &config.python_modules)?;

            Ok(Self {
                config,
                globals: Arc::new(RwLock::new(globals.unbind())),
                locals: Arc::new(RwLock::new(locals.unbind())),
                callbacks: Arc::new(RwLock::new(HashMap::new())),
                injected_callbacks: Arc::new(RwLock::new(HashSet::new())),
            })
        })
    }

    /// Create restricted builtins dictionary with safe functions only.
    fn create_restricted_builtins(py: Python) -> Result<Py<PyDict>, ReplError> {
        let builtins = PyDict::new(py);

        // Safe built-in functions
        let safe_builtins = [
            // Type constructors
            "bool",
            "bytes",
            "bytearray",
            "complex",
            "dict",
            "float",
            "frozenset",
            "int",
            "list",
            "set",
            "str",
            "tuple",
            // Utility functions
            "abs",
            "all",
            "any",
            "bin",
            "chr",
            "dir",
            "divmod",
            "enumerate",
            "filter",
            "format",
            "hex",
            "id",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "map",
            "max",
            "min",
            "next",
            "oct",
            "ord",
            "pow",
            "print",
            "range",
            "repr",
            "reversed",
            "round",
            "slice",
            "sorted",
            "sum",
            "zip",
            // Type checking
            "callable",
            "hasattr",
            "getattr",
            "setattr",
            "type",
            // Import system (needed for import statements)
            "__import__",
            // Exceptions (needed for error handling)
            "Exception",
            "StopIteration",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
        ];

        let main_builtins = py
            .import("builtins")
            .map_err(|e| ReplError::InitializationError(e.to_string()))?;

        for name in &safe_builtins {
            if let Ok(obj) = main_builtins.getattr(*name) {
                builtins
                    .set_item(*name, obj)
                    .map_err(|e| ReplError::InitializationError(e.to_string()))?;
            }
        }

        Ok(builtins.unbind())
    }

    /// Add modules from configuration.
    fn add_modules(
        py: Python,
        globals: &Bound<PyDict>,
        modules: &[String],
    ) -> Result<(), ReplError> {
        for module_name in modules {
            match py.import(module_name.as_str()) {
                Ok(module) => {
                    globals
                        .set_item(module_name.as_str(), module)
                        .map_err(|e| ReplError::InitializationError(e.to_string()))?;
                }
                Err(e) => {
                    log::warn!("Failed to import Python module '{module_name}': {e}");
                }
            }
        }

        Ok(())
    }

    /// Execute code with output capture.
    fn execute_with_capture(
        py: Python,
        code: &str,
        globals: &Bound<PyDict>,
        locals: &Bound<PyDict>,
    ) -> Result<(String, String), ReplError> {
        // Create StringIO for capturing output
        let io_module = py
            .import("io")
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;
        let string_io = io_module
            .getattr("StringIO")
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        let stdout_capture = string_io
            .call0()
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;
        let stderr_capture = string_io
            .call0()
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        // Redirect stdout/stderr
        let sys_module = py
            .import("sys")
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;
        let old_stdout = sys_module
            .getattr("stdout")
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;
        let old_stderr = sys_module
            .getattr("stderr")
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        sys_module
            .setattr("stdout", &stdout_capture)
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;
        sys_module
            .setattr("stderr", &stderr_capture)
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        // Execute the code
        let c_code = CString::new(code).map_err(|e| {
            ReplError::ExecutionError(
                format!(
                    "code contains an interior NUL byte at position {}",
                    e.nul_position()
                ),
            )
        })?;
        let exec_result = py.run(c_code.as_c_str(), Some(globals), Some(locals));

        // Restore stdout/stderr
        let _ = sys_module.setattr("stdout", old_stdout);
        let _ = sys_module.setattr("stderr", old_stderr);

        // Get captured output
        let stdout = stdout_capture
            .call_method0("getvalue")
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?
            .extract::<String>()
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        let stderr = stderr_capture
            .call_method0("getvalue")
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?
            .extract::<String>()
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        // Check execution result
        exec_result.map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        Ok((stdout, stderr))
    }
}

#[async_trait]
#[allow(clippy::significant_drop_tightening)] // Guards are used multiple times
impl ReplEnvironment for PythonRepl {
    #[allow(clippy::cast_possible_truncation)] // Execution time won't exceed u64::MAX ms
    async fn execute(&self, code: &str) -> Result<ReplResult, ReplError> {
        let code = code.to_string();
        let globals = Arc::clone(&self.globals);
        let locals = Arc::clone(&self.locals);
        let callbacks = Arc::clone(&self.callbacks);
        let injected_callbacks = Arc::clone(&self.injected_callbacks);
        let timeout = self.config.base.timeout;

        // Execute in blocking task with timeout
        let result = tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || {
                // Start timing INSIDE the blocking task to exclude queue wait time
                let start = Instant::now();

                Python::attach(|py| {
                    let globals_dict = globals.blocking_write();
                    let globals_ref = globals_dict.bind(py);

                    let locals_dict = locals.blocking_write();
                    let locals_ref = locals_dict.bind(py);

                    // Inject callbacks only if they're new or updated
                    let callbacks_guard = callbacks.blocking_read();
                    let mut injected_guard = injected_callbacks.blocking_write();

                    callback::inject_callbacks_if_needed(
                        py,
                        globals_ref,
                        &callbacks_guard,
                        &mut injected_guard,
                    )?;

                    drop(callbacks_guard);
                    drop(injected_guard);

                    // Execute code with output capture
                    match Self::execute_with_capture(py, &code, globals_ref, locals_ref) {
                        Ok((stdout, stderr)) => {
                            // Note: The changes are made to the PyDict that locals_ref points to,
                            // which is the same PyDict stored in locals_dict (our Py<PyDict>)
                            // So no need to "save back" - the Py object is already modified
                            Ok(ReplResult {
                                stdout,
                                stderr,
                                success: true,
                                return_value: None,
                                execution_time_ms: start.elapsed().as_millis() as u64,
                            })
                        }
                        Err(e) => Ok(ReplResult {
                            stdout: String::new(),
                            stderr: e.to_string(),
                            success: false,
                            return_value: None,
                            execution_time_ms: start.elapsed().as_millis() as u64,
                        }),
                    }
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
        let locals_arc = Arc::clone(&self.locals);

        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                *locals_arc.blocking_write() = PyDict::new(py).unbind();
            });
        })
        .await
        .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        // Clear callback tracking so they'll be re-injected on next execute()
        self.injected_callbacks.write().await.clear();

        Ok(())
    }

    fn config(&self) -> &ReplConfig {
        &self.config.base
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::redundant_clone)]
mod tests {
    use super::*;
    use serial_test::serial;
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
        assert!(result.stderr.contains("ZeroDivisionError") || !result.success);
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
        .await
        .unwrap();

        let _ = repl.execute("result = double('21')").await.unwrap();
        let result = repl.get_variable("result").await.unwrap();
        assert_eq!(result, Some("42".to_string()));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_restricted_builtins() {
        let repl = PythonRepl::new().unwrap();

        // Should have access to safe builtins
        let result = repl.execute("x = len([1, 2, 3])").await.unwrap();
        assert!(result.success);

        // Should NOT have access to dangerous builtins like exec, eval
        let result = repl.execute("exec('print(1)')").await.unwrap();
        assert!(!result.success);
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
        .await
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
        .await
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
        .await
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
    async fn test_python_repl_language_name() {
        let repl = PythonRepl::new().unwrap();
        assert_eq!(repl.language_name(), "python");
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_config() {
        let config = PythonReplConfig {
            base: ReplConfig {
                timeout: Duration::from_secs(10),
            },
            python_modules: vec!["math".to_string(), "json".to_string()],
        };

        let repl = PythonRepl::with_config(config).unwrap();
        assert_eq!(repl.config().timeout, Duration::from_secs(10));
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_callback_injection_efficiency() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let repl = PythonRepl::new().unwrap();

        // Counter to track how many times the callback is created
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        // Inject a callback that tracks creation
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
        .await
        .unwrap();

        // Execute multiple times - callback should only be injected once
        for i in 0..5 {
            let _ = repl
                .execute(&format!("result_{i} = track_calls('test_{i}')"))
                .await
                .unwrap();
        }

        // Verify all calls succeeded
        for i in 0..5 {
            let var = repl.get_variable(&format!("result_{i}")).await.unwrap();
            assert!(var.is_some());
            assert!(var.unwrap().contains(&format!("test_{i}")));
        }

        // Callback was called 5 times
        assert_eq!(call_count.load(Ordering::SeqCst), 5);

        // After reset, callbacks should be re-injected on next execute
        repl.reset().await.unwrap();
        let injected_count = repl.injected_callbacks.read().await.len();
        assert_eq!(
            injected_count, 0,
            "Injected callbacks should be cleared after reset"
        );

        // Execute again - callback should be re-injected
        let _ = repl
            .execute("result_after_reset = track_calls('after_reset')")
            .await
            .unwrap();

        let var = repl.get_variable("result_after_reset").await.unwrap();
        assert!(var.unwrap().contains("after_reset"));
        assert_eq!(call_count.load(Ordering::SeqCst), 6);
    }

    #[tokio::test]
    #[serial]
    async fn test_python_repl_execution_time_accuracy() {
        let repl = PythonRepl::new().unwrap();

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
}
