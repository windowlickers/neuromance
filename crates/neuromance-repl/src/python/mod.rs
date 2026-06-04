//! Python REPL implementations.
//!
//! - [`PythonRepl`]: Direct execution with restricted builtins and callback injection
//! - [`InteractivePythonRepl`]: Uses Python's `code.InteractiveConsole` for multi-line support
//!
//! # Example
//!
//! ```rust,no_run
//! use neuromance_repl::python::{PythonRepl, PythonReplConfig};
//! use std::collections::HashMap;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = PythonReplConfig::default().with_modules(["statistics"]);
//!
//! let repl = PythonRepl::with_config(config)?;
//!
//! // Inject async callback for LLM integration (receives args and kwargs)
//! repl.inject_function("llm_query", |args, _kwargs: HashMap<String, String>| {
//!     Box::pin(async move {
//!         Ok(format!("Response to: {}", args[0]))
//!     })
//! })?;
//!
//! repl.execute("result = llm_query('summarize')").await?;
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ReplError;
use crate::error::PyResultExt;

pub mod builtins;
pub use builtins::SAFE_PYTHON_BUILTINS;
pub mod callback;
mod capture;
pub mod direct;
pub mod interactive;
pub(crate) mod state;

#[cfg(feature = "tools")]
pub mod tool;

#[cfg(feature = "subagent")]
pub mod subagent;

// Re-export main types
pub use callback::PythonCallback;
pub use direct::PythonRepl;
pub use interactive::InteractivePythonRepl;
pub(crate) use state::WithShared;

// Re-export tool implementation
#[cfg(feature = "tools")]
pub use tool::{PythonReplTool, PythonReplToolFactory};

// Re-export subagent bridge
#[cfg(feature = "subagent")]
pub use subagent::SubagentRepl;

/// Register a callback, marking it for re-injection if it replaces
/// an existing one with the same name.
#[allow(clippy::significant_drop_tightening)]
pub(crate) fn inject_function<S: WithShared>(
    state: &Arc<Mutex<S>>,
    name: &str,
    callback: PythonCallback,
) -> Result<(), ReplError> {
    let mut guard = state
        .lock()
        .map_err(|e| ReplError::StatePoisoned(e.to_string()))?;
    let shared = guard.shared_mut();
    shared.injected_callbacks.remove(name);
    shared
        .callbacks
        .insert(name.to_string(), Arc::new(callback));
    Ok(())
}

/// Get a variable's string representation from the locals dict.
pub(crate) async fn get_variable<S: WithShared + Send + 'static>(
    state: &Arc<Mutex<S>>,
    name: &str,
) -> Result<Option<String>, ReplError> {
    let state = Arc::clone(state);
    let name = name.to_string();

    tokio::task::spawn_blocking(move || {
        let guard = state
            .lock()
            .map_err(|e| ReplError::StatePoisoned(e.to_string()))?;
        Python::attach(|py| {
            let locals_dict = guard.shared().locals.bind(py);
            match locals_dict.get_item(&name).at_var("get_item", &name)? {
                Some(value) => {
                    let str_repr = value.str().at_var("str() value", &name)?;
                    Ok(Some(
                        str_repr
                            .extract::<String>()
                            .at_var("extract -> String", &name)?,
                    ))
                }
                None => Ok(None),
            }
        })
    })
    .await?
}

/// Set a variable in the locals dict.
pub(crate) async fn set_variable<S: WithShared + Send + 'static>(
    state: &Arc<Mutex<S>>,
    name: &str,
    value: &str,
) -> Result<(), ReplError> {
    let state = Arc::clone(state);
    let name = name.to_string();
    let value = value.to_string();

    tokio::task::spawn_blocking(move || {
        let guard = state
            .lock()
            .map_err(|e| ReplError::StatePoisoned(e.to_string()))?;
        Python::attach(|py| {
            let locals_dict = guard.shared().locals.bind(py);
            let py_value = pyo3::types::PyString::new(py, &value);
            locals_dict
                .set_item(&name, py_value)
                .at_var("set_item", &name)?;
            Ok(())
        })
    })
    .await?
}

/// Python REPL configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonReplConfig {
    /// Maximum execution time per code block
    pub timeout: Duration,

    /// Python modules to import and make available globally.
    /// Standard library modules are imported if available.
    /// Third-party packages must be installed in the Python environment.
    ///
    /// Use [`PythonReplConfig::with_modules`] to extend the list. The
    /// `Cow<'static, str>` storage is an internal detail and not part of
    /// the public API.
    pub(crate) python_modules: Vec<Cow<'static, str>>,
}

impl Default for PythonReplConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            python_modules: vec![
                Cow::Borrowed("math"),
                Cow::Borrowed("random"),
                Cow::Borrowed("datetime"),
                Cow::Borrowed("json"),
                Cow::Borrowed("re"),
                Cow::Borrowed("itertools"),
                Cow::Borrowed("collections"),
                Cow::Borrowed("functools"),
            ],
        }
    }
}

impl PythonReplConfig {
    /// Append modules to [`Self::python_modules`], returning the updated config.
    ///
    /// Extends rather than replaces so chaining onto
    /// [`PythonReplConfig::default`] keeps the standard-library modules.
    #[must_use]
    pub fn with_modules<I, S>(mut self, modules: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<Cow<'static, str>>,
    {
        self.python_modules
            .extend(modules.into_iter().map(Into::into));
        self
    }
}
