use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};
use std::borrow::Cow;
use std::collections::HashSet;

use crate::ReplError;
use crate::error::PyResultExt;

/// Create restricted builtins with a filtered `__import__`
/// that only allows modules from the configured allowlist.
pub(super) fn create_restricted_builtins(
    py: Python,
    allowed_modules: &[Cow<'static, str>],
) -> Result<Py<PyDict>, ReplError> {
    let builtins = PyDict::new(py);
    let main_builtins = py.import("builtins").at("import builtins")?;

    SAFE_PYTHON_BUILTINS
        .iter()
        .try_for_each(|name| -> Result<(), ReplError> {
            if let Ok(obj) = main_builtins.getattr(*name) {
                builtins
                    .set_item(*name, obj)
                    .at("set_item safe builtin into restricted dict")?;
            }
            Ok(())
        })?;

    let restricted_import = create_filtered_import(py, allowed_modules)?;
    builtins
        .set_item("__import__", restricted_import)
        .at("set_item __import__ into restricted builtins")?;

    Ok(builtins.unbind())
}

/// Build a Rust-backed `__import__` replacement that only allows
/// modules from the given allowlist.
///
/// The allowlist lives in Rust memory so Python code cannot
/// tamper with it via `__globals__`, `__closure__`, etc.
fn create_filtered_import(
    py: Python,
    allowed_modules: &[Cow<'static, str>],
) -> Result<Py<PyAny>, ReplError> {
    let allowed: HashSet<Cow<'static, str>> = allowed_modules.iter().cloned().collect();

    // Pre-format the sorted allowlist once so blocked-import errors can name
    // the actual permitted modules without paying the formatting cost on
    // every rejection.
    let allowed_list = {
        let mut sorted: Vec<&str> = allowed.iter().map(AsRef::as_ref).collect();
        sorted.sort_unstable();
        sorted.join(", ")
    };

    let real_import = py
        .import("builtins")
        .at("import builtins (filtered_import)")?
        .getattr("__import__")
        .at("getattr builtins.__import__")?
        .unbind();

    let func = PyCFunction::new_closure(
        py,
        Some(c"_restricted_import"),
        None,
        move |args: &Bound<PyTuple>, kwargs: Option<&Bound<PyDict>>| -> PyResult<Py<PyAny>> {
            let py = args.py();
            let name: String = args.get_item(0)?.extract()?;
            let top = name
                .split_once('.')
                .map_or(name.as_str(), |(prefix, _)| prefix);
            if !allowed.contains(top) {
                return Err(pyo3::exceptions::PyImportError::new_err(format!(
                    "Import of '{name}' is not allowed; allowed top-level modules: {allowed_list}"
                )));
            }
            Ok(real_import.bind(py).call(args, kwargs)?.unbind())
        },
    )
    .at("create _restricted_import PyCFunction")?;

    Ok(func.unbind().into())
}

/// Safe Python builtins allowed in the restricted environment.
pub const SAFE_PYTHON_BUILTINS: &[&str] = &[
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
    // Exceptions (needed for error handling)
    "Exception",
    "StopIteration",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
];
