//! Rule files: glob-triggered instruction injection.
//!
//! A *rule* is a Markdown file (`.md`/`.mdc`) with optional YAML frontmatter.
//! Recognized keys are `globs` (a sequence or comma-separated scalar; `paths` is
//! an alias), `always_apply` (`alwaysApply`), and `description`. A rule with
//! `always_apply` is injected once at conversation start; a rule with globs is
//! injected the first time a file-path tool call touches a matching path. Unlike
//! skills, rules are pushed in by location, not summoned by intent.
//!
//! ```no_run
//! use std::sync::Arc;
//! use neuromance_context::rules::{LocalRuleSource, RuleCatalog, RulesHook};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let catalog = RuleCatalog::build(vec![
//!     Box::new(LocalRuleSource::new(vec!["/etc/neuromance/rules".into()])),
//! ])
//! .await;
//! let hook = RulesHook::new(Arc::new(catalog), 8192);
//! # let _ = hook;
//! # Ok(())
//! # }
//! ```

mod catalog;
mod error;
mod frontmatter;
mod hook;
mod model;
mod source;

pub use catalog::{DEFAULT_BUDGET_BYTES, RuleCatalog};
pub use error::RuleError;
pub use frontmatter::{ParsedRule, parse_rule};
pub use hook::RulesHook;
pub use model::{RuleId, RuleLocator, RuleMetadata};
pub use source::{HttpRuleSource, LocalRuleSource, RuleSource};
