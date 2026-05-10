//! Shared callback creation and injection for Python REPLs.

use crate::ReplError;
use crate::error::PyResultExt;
use futures::future::BoxFuture;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Callback function type for injected functions.
///
/// Receives:
/// - `args`: Positional arguments, each coerced via Python's `str()` before
///   being handed to the callback. A Python `int`, `list`, or arbitrary
///   object reaches the callback as its `str()` form (e.g. `"42"`,
///   `"[1, 2, 3]"`).
/// - `kwargs`: Keyword arguments as a string-to-string map. Keys must be
///   strings (Python guarantees this for `**kwargs`); values are coerced
///   via Python's `str()` like positional args.
///
/// If `str()` fails or the resulting object cannot be extracted as a
/// `String`, the failure is propagated to Python as an exception rather
/// than silently substituted.
///
/// Returns a `BoxFuture` to allow async operations within the callback.
/// Use `Box::pin(async move { ... })` when creating callbacks.
///
/// Most callers should pass an unboxed closure to
/// [`PythonRepl::inject_function`](super::PythonRepl::inject_function) or
/// [`InteractivePythonRepl::inject_function`](super::InteractivePythonRepl::inject_function)
/// — those entry points box internally. This alias is the storage form
/// held inside [`SharedState`](super::state::SharedState).
pub type PythonCallback = Box<
    dyn Fn(Vec<String>, HashMap<String, String>) -> BoxFuture<'static, Result<String, String>>
        + Send
        + Sync,
>;

/// Create a Python-callable closure from a Rust callback.
///
/// # Errors
///
/// Returns `ReplError::PythonInfra` if the `PyCFunction` closure cannot be created.
pub(crate) fn create_py_callback<'py>(
    py: Python<'py>,
    callback: &Arc<PythonCallback>,
) -> Result<Bound<'py, PyCFunction>, ReplError> {
    let cb = Arc::clone(callback);
    PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<PyTuple>, kwargs: Option<&Bound<PyDict>>| {
            let args_vec: Vec<String> = args
                .iter()
                .map(|arg| arg.str()?.extract::<String>())
                .collect::<PyResult<_>>()?;

            let kwargs_map: HashMap<String, String> = kwargs
                .map(|kw| {
                    kw.iter()
                        .map(|(k, v)| {
                            let key = k.extract::<String>()?;
                            let value = v.str()?.extract::<String>()?;
                            Ok((key, value))
                        })
                        .collect::<PyResult<_>>()
                })
                .transpose()?
                .unwrap_or_default();

            // Release the GIL before calling the callback, as it may do
            // blocking operations (e.g., waiting for an agent loop response)
            let cb_clone = Arc::clone(&cb);
            let result = args.py().detach(move || {
                tokio::runtime::Handle::current().block_on(cb_clone(args_vec, kwargs_map))
            });

            match result {
                Ok(result) => Ok(result),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
            }
        },
    )
    .at("create PyCFunction closure")
}

/// Inject callbacks into a Python dict if not already injected.
///
/// Only injects callbacks that are new or have been updated,
/// avoiding redundant Python object creation on every `execute()`.
///
/// # Errors
///
/// Returns `ReplError::PythonInfra` if a callback cannot be
/// created or injected into the target dict.
#[allow(clippy::implicit_hasher)]
pub(crate) fn inject_callbacks_if_needed(
    py: Python,
    target: &Bound<PyDict>,
    callbacks: &HashMap<String, Arc<PythonCallback>>,
    injected: &mut HashSet<String>,
) -> Result<(), ReplError> {
    for (name, callback) in callbacks {
        if injected.contains(name) {
            continue;
        }

        let py_func = create_py_callback(py, callback)?;

        target
            .set_item(name.as_str(), py_func)
            .at("set_item callback in target dict")?;

        injected.insert(name.clone());
    }

    Ok(())
}
