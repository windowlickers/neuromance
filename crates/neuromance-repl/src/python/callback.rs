//! Shared callback creation and injection for Python REPLs.

use crate::ReplError;
use futures::future::BoxFuture;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Callback function type for injected functions.
///
/// Receives:
/// - `args`: Positional arguments as strings
/// - `kwargs`: Keyword arguments as a string-to-string map
///
/// Returns a `BoxFuture` to allow async operations within the callback.
/// Use `Box::pin(async move { ... })` when creating callbacks.
pub type PythonCallback = Box<
    dyn Fn(Vec<String>, HashMap<String, String>) -> BoxFuture<'static, Result<String, String>>
        + Send
        + Sync,
>;

/// Create a Python-callable closure from a Rust callback.
///
/// # Errors
///
/// Returns `ReplError::ExecutionError` if the `PyCFunction` closure cannot be created.
pub fn create_py_callback<'py>(
    py: Python<'py>,
    callback: &Arc<PythonCallback>,
) -> Result<Bound<'py, PyCFunction>, ReplError> {
    let cb = Arc::clone(callback);
    PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<PyTuple>,
              kwargs: Option<&Bound<PyDict>>| {
            #[allow(clippy::unnecessary_debug_formatting)]
            let args_vec: Vec<String> = args
                .iter()
                .map(|arg| {
                    arg.extract::<String>()
                        .unwrap_or_else(|_| format!("{arg:?}"))
                })
                .collect();

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

            // Release the GIL before calling the callback, as it may do
            // blocking operations (e.g., waiting for an agent loop response)
            let cb_clone = Arc::clone(&cb);
            let result = args.py().detach(move || {
                tokio::runtime::Handle::current()
                    .block_on(cb_clone(args_vec, kwargs_map))
            });

            match result {
                Ok(result) => Ok(result),
                Err(e) => Err(PyErr::new::<
                    pyo3::exceptions::PyRuntimeError,
                    _,
                >(e)),
            }
        },
    )
    .map_err(|e| ReplError::ExecutionError(e.to_string()))
}

/// Inject callbacks into a Python dict if not already injected.
///
/// Only injects callbacks that are new or have been updated,
/// avoiding redundant Python object creation on every `execute()`.
/// Stale entries in `injected` are removed automatically.
///
/// # Errors
///
/// Returns `ReplError::ExecutionError` if a callback cannot be
/// created or injected into the target dict.
#[allow(clippy::implicit_hasher)]
pub fn inject_callbacks_if_needed(
    py: Python,
    target: &Bound<PyDict>,
    callbacks: &HashMap<String, Arc<PythonCallback>>,
    injected: &mut HashSet<String>,
) -> Result<(), ReplError> {
    let current_names: HashSet<String> =
        callbacks.keys().cloned().collect();

    // Remove stale entries
    injected.retain(|name| current_names.contains(name));

    for (name, callback) in callbacks {
        if injected.contains(name) {
            continue;
        }

        let py_func = create_py_callback(py, callback)?;

        target
            .set_item(name.as_str(), py_func)
            .map_err(|e| ReplError::ExecutionError(e.to_string()))?;

        injected.insert(name.clone());
    }

    Ok(())
}
