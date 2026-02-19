//! Python REPL implementations.
//!
//! - [`PythonRepl`]: Direct execution with restricted builtins and callback injection
//! - [`InteractivePythonRepl`]: Uses Python's `code.InteractiveConsole` for multi-line support
//!
//! # Example
//!
//! ```rust,no_run
//! use neuromance_repl::python::{PythonRepl, PythonReplConfig};
//! use neuromance_repl::ReplEnvironment;
//! use std::collections::HashMap;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut config = PythonReplConfig::default();
//! config.python_modules.push("numpy".to_string());
//!
//! let repl = PythonRepl::with_config(config)?;
//!
//! // Inject async callback for LLM integration (receives args and kwargs)
//! repl.inject_function("llm_query", Box::new(|args, _kwargs: HashMap<String, String>| {
//!     Box::pin(async move {
//!         Ok(format!("Response to: {}", args[0]))
//!     })
//! })).await?;
//!
//! repl.execute("result = llm_query('summarize')").await?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::ReplConfig;

pub mod callback;
pub mod direct;
pub mod interactive;

#[cfg(feature = "tools")]
pub mod tool;

// Re-export main types
pub use callback::PythonCallback;
pub use direct::PythonRepl;
pub use interactive::InteractivePythonRepl;

// Re-export tool implementation
#[cfg(feature = "tools")]
pub use tool::PythonReplTool;

/// Python-specific REPL configuration.
///
/// Extends [`ReplConfig`] with Python-specific settings like
/// which modules to pre-import into the execution environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonReplConfig {
    /// Base REPL configuration (timeout, etc.)
    #[serde(flatten)]
    pub base: ReplConfig,

    /// Python modules to import and make available globally.
    /// Standard library modules are imported if available.
    /// Third-party packages must be installed in the Python environment.
    pub python_modules: Vec<String>,
}

impl Default for PythonReplConfig {
    fn default() -> Self {
        Self {
            base: ReplConfig::default(),
            python_modules: vec![
                "math".to_string(),
                "random".to_string(),
                "datetime".to_string(),
                "json".to_string(),
                "re".to_string(),
                "itertools".to_string(),
                "collections".to_string(),
                "functools".to_string(),
            ],
        }
    }
}
