//! Shared state for Python REPL implementations.
//!
//! Both [`PythonRepl`](super::PythonRepl) and
//! [`InteractivePythonRepl`](super::InteractivePythonRepl) share
//! locals, callbacks, and injection tracking.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::Py;
use pyo3::types::PyDict;

use super::callback::PythonCallback;

/// State shared by all Python REPL variants.
///
/// Visibility note: kept `pub` because the surrounding `state` module is
/// already declared `pub(crate)`, which limits reach to this crate.
/// Writing `pub(crate)` here would be redundant
/// (`clippy::redundant_pub_crate`).
pub struct SharedState {
    pub locals: Py<PyDict>,
    pub callbacks: HashMap<String, Arc<PythonCallback>>,
    pub injected_callbacks: HashSet<String>,
}

impl SharedState {
    pub fn new(locals: Py<PyDict>) -> Self {
        Self {
            locals,
            callbacks: HashMap::new(),
            injected_callbacks: HashSet::new(),
        }
    }
}

/// Accessor trait for state types that embed a [`SharedState`].
///
/// Each REPL variant has its own per-variant state (e.g. `globals`
/// for [`PythonRepl`](super::PythonRepl), `console` for
/// [`InteractivePythonRepl`](super::InteractivePythonRepl)) plus a
/// [`SharedState`] field. This trait lets the helpers in
/// [`super::inject_function`], [`super::get_variable`], and
/// [`super::set_variable`] reach the shared field generically over
/// both state structs without duplicating the helper bodies.
///
/// Static dispatch only — there are no `dyn WithShared` users.
///
/// Visibility note: kept `pub` for the same reason as
/// [`SharedState`] — the `state` module is already `pub(crate)`.
pub trait WithShared {
    fn shared(&self) -> &SharedState;
    fn shared_mut(&mut self) -> &mut SharedState;
}
