//! Agent skills: discovery, progressive-disclosure menus, and on-demand loading.
//!
//! A *skill* is a directory containing a `SKILL.md` — YAML frontmatter
//! (`name`, `description`, plus optional agentskills.io fields) and a Markdown
//! body of reusable instructions for a class of task. Skills are context: a
//! cheap menu of `name: description` is injected once, and a skill's full body
//! is loaded only when needed — summoned by the `load_skill` tool or a `$name`
//! mention, or, for a filesystem-equipped agent, read from disk after the
//! catalog is materialized with [`synthesize_skill_md`]. Either way the system
//! prompt stays stable and cache-friendly.
//!
//! ```no_run
//! use neuromance_context::skills::{HttpSkillSource, LocalSkillSource, SkillCatalog};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let catalog = SkillCatalog::build(vec![
//!     Box::new(LocalSkillSource::new(vec!["/etc/neuromance/skills".into()])),
//!     Box::new(HttpSkillSource::new("https://corpus/api/v1/skills")),
//! ])
//! .await;
//!
//! if let Some(menu) = catalog.menu(8192) {
//!     println!("{menu}");
//! }
//! # Ok(())
//! # }
//! ```

mod catalog;
mod error;
mod frontmatter;
mod hook;
mod mention;
mod model;
mod source;

pub use catalog::{DEFAULT_BUDGET_BYTES, SkillCatalog, synthesize_skill_md};
pub use error::SkillError;
pub use frontmatter::{ParsedSkill, parse_skill};
pub use hook::SkillsHook;
pub use model::{SkillId, SkillLocator, SkillMetadata};
pub use source::{HttpSkillSource, LocalSkillSource, SkillSource};
