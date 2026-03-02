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

/// Implemented by state types that embed a [`SharedState`].
pub trait WithShared {
    fn shared(&self) -> &SharedState;
    fn shared_mut(&mut self) -> &mut SharedState;
}
