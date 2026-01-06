//! Python REPL implementations.
//!
//! - [`PythonRepl`]: Direct execution with restricted builtins and callback injection
//! - [`InteractivePythonRepl`]: Uses Python's `code.InteractiveConsole` for multi-line support
//!
//! # Example
//!
//! ```rust,no_run
//! use neuromance_repl::python::PythonRepl;
//! use neuromance_repl::{ReplEnvironment, ReplConfig};
//! use std::collections::HashMap;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut config = ReplConfig::default();
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

pub mod direct;
pub mod interactive;

#[cfg(feature = "tools")]
pub mod tool;

// Re-export main types
pub use direct::PythonRepl;
pub use interactive::InteractivePythonRepl;

// Re-export callback type
pub use direct::PythonCallback;

// Re-export tool implementation
#[cfg(feature = "tools")]
pub use tool::PythonReplTool;
