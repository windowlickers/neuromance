//! Postgres persistence for neuromance conversations and messages.
//!
//! This crate provides a shared durable record that any neuromance agent can
//! write its conversations into. The store is deliberately write-oriented:
//! the contents of conversations — message order, tool calls, tool results —
//! are the primary artifact.
//!
//! # Semantics
//!
//! The `messages` table is an **append-only log**. [`ConversationSink::append_messages`]
//! is idempotent per [`Message::id`]: re-sending an already-persisted history
//! snapshot inserts nothing, so callers can safely retry after failures.
//! Ordering is recorded in a per-conversation `seq` column assigned at insert
//! time under a per-conversation row lock, which keeps concurrent writers from
//! colliding. Context compaction never rewrites the log — summary messages
//! (which carry fresh ids) append after the original history.
//!
//! Timestamps are stored as `TIMESTAMPTZ` and therefore truncated to
//! microsecond precision on round-trip.
//!
//! # Example
//!
//! ```no_run
//! use neuromance_db::{ConversationSink, PgConversationStore};
//! use neuromance_common::chat::Message;
//! use std::time::Duration;
//! use uuid::Uuid;
//!
//! # async fn example() -> Result<(), neuromance_db::DbError> {
//! let store =
//!     PgConversationStore::connect("postgres://localhost/neuromance", 5, Duration::from_secs(5))
//!         .await?;
//! store.migrate().await?;
//!
//! let conversation_id = Uuid::new_v4();
//! let messages = vec![Message::user(conversation_id, "hello")];
//! store.append_messages(conversation_id, &messages).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Developing against the schema
//!
//! Queries use sqlx's compile-time checked macros, backed by the committed
//! `.sqlx/` offline metadata — no database is needed to build. When changing
//! a query or the schema, regenerate the metadata from a live database:
//!
//! ```bash
//! docker run -d --name neuromance-pg -e POSTGRES_PASSWORD=pg -p 5432:5432 postgres:16
//! export DATABASE_URL=postgres://postgres:pg@localhost:5432/neuromance
//! cd crates/neuromance-db
//! cargo sqlx database setup   # create db + run ./migrations
//! cargo sqlx prepare          # regenerate .sqlx/ — commit it
//! ```
//!
//! [`Message::id`]: neuromance_common::chat::Message

mod error;
mod hook;
mod rows;
mod sink;
mod store;

pub use error::DbError;
pub use hook::PersistenceHook;
pub use sink::ConversationSink;
pub use store::{ConversationSummary, PgConversationStore};
