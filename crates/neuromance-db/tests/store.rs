//! Integration tests against a real postgres.
//!
//! These are `#[ignore]`d because CI has no postgres service. Run them
//! locally with a `DATABASE_URL` pointing at a postgres superuser
//! (`#[sqlx::test]` creates a throwaway database per test):
//!
//! ```bash
//! DATABASE_URL=postgres://postgres:pg@localhost:5432/neuromance \
//!     cargo test -p neuromance-db -- --ignored
//! ```

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use chrono::SubsecRound;
use neuromance_common::chat::{Conversation, ConversationStatus, Message, ReasoningContent};
use neuromance_common::client::{InputTokensDetails, Usage};
use neuromance_common::tools::ToolCall;
use neuromance_db::{ConversationSink, PgConversationStore};
use sqlx::PgPool;
use uuid::Uuid;

const fn sample_usage() -> Usage {
    Usage {
        prompt_tokens: 120,
        completion_tokens: 34,
        total_tokens: 154,
        cost: Some(0.0012),
        input_tokens_details: Some(InputTokensDetails {
            cached_tokens: 80,
            cache_creation_tokens: 0,
        }),
        output_tokens_details: None,
    }
}

fn sample_history(conversation_id: Uuid) -> Vec<Message> {
    let mut assistant = Message::assistant(conversation_id, "checking the weather")
        .with_tool_calls(vec![ToolCall::new("get_weather", r#"{"city":"Berlin"}"#)])
        .unwrap();
    assistant.model = Some("claude-sonnet-4-5-20250929".to_string());
    assistant.provider = Some("anthropic".to_string());
    assistant.usage = Some(sample_usage());
    let tool_call_id = assistant.tool_calls[0].id.clone();
    vec![
        Message::system(conversation_id, "you are helpful"),
        Message::user(conversation_id, "what's the weather in Berlin?"),
        assistant,
        Message::tool(
            conversation_id,
            "22C and sunny",
            tool_call_id,
            "get_weather".to_string(),
        )
        .unwrap(),
    ]
}

async fn stored_seqs(pool: &PgPool, conversation_id: Uuid) -> Vec<(i64, Uuid)> {
    sqlx::query_as("SELECT seq, id FROM messages WHERE conversation_id = $1 ORDER BY seq")
        .bind(conversation_id)
        .fetch_all(pool)
        .await
        .unwrap()
}

/// Message ids attributed to a task via the `[start_seq, end_seq]` join — the
/// reconstruction path callers use to ask "what did this task produce?".
async fn task_message_ids(pool: &PgPool, task_id: Uuid) -> Vec<Uuid> {
    sqlx::query_scalar(
        "SELECT m.id FROM messages m JOIN tasks t ON t.conversation_id = m.conversation_id \
         WHERE t.id = $1 AND m.seq BETWEEN t.start_seq AND t.end_seq ORDER BY m.seq",
    )
    .bind(task_id)
    .fetch_all(pool)
    .await
    .unwrap()
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_append_is_idempotent_and_seq_is_dense(pool: PgPool) {
    let store = PgConversationStore::new(pool.clone());
    let conversation_id = Uuid::new_v4();
    let history = sample_history(conversation_id);

    let first = store
        .append_messages(conversation_id, &history)
        .await
        .unwrap();
    assert_eq!(first, 4);

    let second = store
        .append_messages(conversation_id, &history)
        .await
        .unwrap();
    assert_eq!(second, 0, "re-sending the same history must insert nothing");

    let seqs = stored_seqs(&pool, conversation_id).await;
    let expected: Vec<(i64, Uuid)> = history
        .iter()
        .enumerate()
        .map(|(i, m)| (i64::try_from(i).unwrap(), m.id))
        .collect();
    assert_eq!(seqs, expected, "seq must be dense and follow caller order");
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_model_provider_usage_round_trip_through_store(pool: PgPool) {
    let store = PgConversationStore::new(pool.clone());
    let conversation_id = Uuid::new_v4();
    let history = sample_history(conversation_id);
    store
        .append_messages(conversation_id, &history)
        .await
        .unwrap();

    let conversation = store
        .get_conversation(conversation_id)
        .await
        .unwrap()
        .expect("conversation should exist after append");

    let assistant = conversation
        .messages
        .iter()
        .find(|m| m.role == neuromance_common::chat::MessageRole::Assistant)
        .expect("history has an assistant message");
    assert_eq!(
        assistant.model.as_deref(),
        Some("claude-sonnet-4-5-20250929")
    );
    assert_eq!(assistant.provider.as_deref(), Some("anthropic"));
    assert_eq!(assistant.usage, Some(sample_usage()));

    // Non-assistant rows carry no client metadata.
    let user = conversation
        .messages
        .iter()
        .find(|m| m.role == neuromance_common::chat::MessageRole::User)
        .expect("history has a user message");
    assert_eq!(user.model, None);
    assert_eq!(user.provider, None);
    assert_eq!(user.usage, None);
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_compacted_history_appends_without_rewriting(pool: PgPool) {
    let store = PgConversationStore::new(pool.clone());
    let conversation_id = Uuid::new_v4();
    let history = sample_history(conversation_id);
    store
        .append_messages(conversation_id, &history)
        .await
        .unwrap();

    // Compaction keeps the system message, replaces the middle with a summary,
    // and continues with fresh messages.
    let compacted = vec![
        history[0].clone(),
        Message::user(conversation_id, "[summary] asked about Berlin weather"),
        Message::assistant(conversation_id, "anything else?"),
    ];
    let inserted = store
        .append_messages(conversation_id, &compacted)
        .await
        .unwrap();
    assert_eq!(
        inserted, 2,
        "only the summary and new assistant message are new"
    );

    let seqs = stored_seqs(&pool, conversation_id).await;
    assert_eq!(seqs.len(), 6);
    // Original log is untouched...
    for (i, message) in history.iter().enumerate() {
        assert_eq!(seqs[i], (i64::try_from(i).unwrap(), message.id));
    }
    // ...and the new messages appended after it, preserving batch order.
    assert_eq!(seqs[4], (4, compacted[1].id));
    assert_eq!(seqs[5], (5, compacted[2].id));
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_get_conversation_round_trips_contents(pool: PgPool) {
    let store = PgConversationStore::new(pool);
    let mut conversation = Conversation::new().with_title("weather chat");
    conversation
        .metadata
        .insert("agent_id".to_string(), serde_json::json!("test-agent"));

    store.upsert_conversation(&conversation).await.unwrap();

    let mut history = sample_history(conversation.id);
    history[2].reasoning = Some(ReasoningContent::with_signature("hmm", "sig"));
    store
        .append_messages(conversation.id, &history)
        .await
        .unwrap();

    let loaded = store
        .get_conversation(conversation.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded.id, conversation.id);
    assert_eq!(loaded.title.as_deref(), Some("weather chat"));
    assert_eq!(loaded.status, ConversationStatus::Active);
    assert_eq!(loaded.metadata, conversation.metadata);

    let loaded_messages = loaded.get_messages();
    assert_eq!(loaded_messages.len(), history.len());
    for (loaded_message, original) in loaded_messages.iter().zip(&history) {
        assert_eq!(loaded_message.id, original.id);
        assert_eq!(loaded_message.role, original.role);
        assert_eq!(loaded_message.content, original.content);
        assert_eq!(loaded_message.tool_calls, original.tool_calls);
        assert_eq!(loaded_message.tool_call_id, original.tool_call_id);
        assert_eq!(loaded_message.name, original.name);
        assert_eq!(loaded_message.reasoning, original.reasoning);
        // TIMESTAMPTZ stores microseconds; chrono produces nanoseconds.
        assert_eq!(
            loaded_message.timestamp,
            original.timestamp.trunc_subsecs(6)
        );
    }
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_missing_conversation_is_none(pool: PgPool) {
    let store = PgConversationStore::new(pool);
    assert!(
        store
            .get_conversation(Uuid::new_v4())
            .await
            .unwrap()
            .is_none()
    );
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_status_update_and_listing(pool: PgPool) {
    let store = PgConversationStore::new(pool);
    let conversation_id = Uuid::new_v4();
    store
        .append_messages(conversation_id, &sample_history(conversation_id))
        .await
        .unwrap();

    store
        .set_conversation_status(conversation_id, ConversationStatus::Deleted)
        .await
        .unwrap();

    let summaries = store.list_conversations(10, 0).await.unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].id, conversation_id);
    assert_eq!(summaries[0].status, ConversationStatus::Deleted);
    assert_eq!(summaries[0].message_count, 4);
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_mismatched_conversation_id_is_skipped(pool: PgPool) {
    let store = PgConversationStore::new(pool);
    let conversation_id = Uuid::new_v4();
    let stray = Message::user(Uuid::new_v4(), "wrong conversation");
    let ok = Message::user(conversation_id, "right conversation");

    let inserted = store
        .append_messages(conversation_id, &[stray, ok])
        .await
        .unwrap();
    assert_eq!(inserted, 1);
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_all_mismatched_batch_touches_nothing(pool: PgPool) {
    let store = PgConversationStore::new(pool.clone());
    let conversation_id = Uuid::new_v4();
    let stray = Message::user(Uuid::new_v4(), "wrong conversation");

    let inserted = store
        .append_messages(conversation_id, &[stray])
        .await
        .unwrap();
    assert_eq!(
        inserted, 0,
        "a batch with no matching messages inserts nothing"
    );

    // The early return must short-circuit before opening a transaction, so the
    // conversation FK row is never created.
    let exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM conversations WHERE id = $1)")
            .bind(conversation_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert!(!exists, "early return must not create the conversation row");
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_concurrent_appends_do_not_collide(pool: PgPool) {
    let store = PgConversationStore::new(pool.clone());
    let conversation_id = Uuid::new_v4();
    let batch_a: Vec<Message> = (0..10)
        .map(|i| Message::user(conversation_id, format!("a{i}")))
        .collect();
    let batch_b: Vec<Message> = (0..10)
        .map(|i| Message::user(conversation_id, format!("b{i}")))
        .collect();

    let (a, b) = tokio::join!(
        store.append_messages(conversation_id, &batch_a),
        store.append_messages(conversation_id, &batch_b),
    );
    assert_eq!(a.unwrap() + b.unwrap(), 20);

    let seqs = stored_seqs(&pool, conversation_id).await;
    assert_eq!(seqs.len(), 20);
    let expected: Vec<i64> = (0..20).collect();
    let actual: Vec<i64> = seqs.iter().map(|(seq, _)| *seq).collect();
    assert_eq!(actual, expected, "seqs must be dense with no collisions");
}

#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_task_provenance_brackets_each_run_by_seq_range(pool: PgPool) {
    let store = PgConversationStore::new(pool.clone());
    let conversation_id = Uuid::new_v4();

    // An empty conversation has no high-water seq — the bracket starts at 0.
    assert_eq!(store.max_seq(conversation_id).await.unwrap(), None);

    // First task contributes the whole initial history (seq 0..=3).
    let history = sample_history(conversation_id);
    store
        .append_messages(conversation_id, &history)
        .await
        .unwrap();
    assert_eq!(store.max_seq(conversation_id).await.unwrap(), Some(3));
    let task_one = Uuid::new_v4();
    store
        .record_task(task_one, conversation_id, 0, 3)
        .await
        .unwrap();

    // Second task appends two more messages (seq 4..=5).
    let follow_up = vec![
        Message::user(conversation_id, "thanks"),
        Message::assistant(conversation_id, "you're welcome"),
    ];
    store
        .append_messages(conversation_id, &follow_up)
        .await
        .unwrap();
    assert_eq!(store.max_seq(conversation_id).await.unwrap(), Some(5));
    let task_two = Uuid::new_v4();
    store
        .record_task(task_two, conversation_id, 4, 5)
        .await
        .unwrap();

    // Each task maps to exactly the messages it produced, in order.
    assert_eq!(
        task_message_ids(&pool, task_one).await,
        history.iter().map(|m| m.id).collect::<Vec<_>>(),
    );
    assert_eq!(
        task_message_ids(&pool, task_two).await,
        follow_up.iter().map(|m| m.id).collect::<Vec<_>>(),
    );

    // Re-recording a task updates its range rather than duplicating the row.
    store
        .record_task(task_two, conversation_id, 4, 5)
        .await
        .unwrap();
    let task_two_rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tasks WHERE id = $1")
        .bind(task_two)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(task_two_rows, 1);
}

/// Reads the lineage columns for a conversation row.
async fn parent_link(pool: &PgPool, id: Uuid) -> (Option<Uuid>, Option<Uuid>) {
    sqlx::query_as("SELECT parent_conversation_id, parent_task_id FROM conversations WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .unwrap()
}

/// `set_conversation_parent` links a child to its spawning parent, works whether
/// or not the child row exists yet, survives later message appends, and updates
/// idempotently on re-link. Roots stay unlinked.
#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_set_conversation_parent_links_child(pool: PgPool) {
    let store = PgConversationStore::new(pool.clone());
    let parent = Uuid::new_v4();
    let child = Uuid::new_v4();
    let task = Uuid::new_v4();

    // Parent must exist for both self-FKs (conversation + message) to resolve.
    let parent_history = sample_history(parent);
    let launching = parent_history
        .iter()
        .find(|m| !m.tool_calls.is_empty())
        .expect("sample history has a tool-calling assistant message");
    let parent_message_id = launching.id;
    let parent_tool_call_id = launching.tool_calls[0].id.clone();
    store.append_messages(parent, &parent_history).await.unwrap();
    assert_eq!(
        parent_link(&pool, parent).await,
        (None, None),
        "root is unlinked"
    );

    // Link before the child has any messages (the upsert creates the row).
    store
        .set_conversation_parent(
            child,
            parent,
            Some(task),
            Some(parent_message_id),
            Some(&parent_tool_call_id),
        )
        .await
        .unwrap();
    assert_eq!(parent_link(&pool, child).await, (Some(parent), Some(task)));

    // Appending the child's messages does not clobber the link.
    store
        .append_messages(child, &sample_history(child))
        .await
        .unwrap();
    assert_eq!(parent_link(&pool, child).await, (Some(parent), Some(task)));

    // The message-level launch site round-trips through get_conversation.
    let loaded = store.get_conversation(child).await.unwrap().unwrap();
    assert_eq!(loaded.parent_conversation_id, Some(parent));
    assert_eq!(loaded.parent_message_id, Some(parent_message_id));
    assert_eq!(loaded.parent_tool_call_id, Some(parent_tool_call_id));

    // Re-linking (e.g. a retried run) updates in place rather than duplicating.
    let task_retry = Uuid::new_v4();
    store
        .set_conversation_parent(child, parent, Some(task_retry), None, None)
        .await
        .unwrap();
    assert_eq!(
        parent_link(&pool, child).await,
        (Some(parent), Some(task_retry))
    );
    let relinked = store.get_conversation(child).await.unwrap().unwrap();
    assert_eq!(relinked.parent_message_id, None, "re-link cleared message id");
    assert_eq!(relinked.parent_tool_call_id, None);
    let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM conversations WHERE id = $1")
        .bind(child)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(rows, 1);
}

/// `list_child_conversations` returns only the children of a given parent,
/// carrying their lineage columns, and excludes the parent itself.
#[sqlx::test(migrations = "./migrations")]
#[ignore = "requires postgres via DATABASE_URL"]
async fn test_list_child_conversations(pool: PgPool) {
    let store = PgConversationStore::new(pool.clone());
    let parent = Uuid::new_v4();
    let child_a = Uuid::new_v4();
    let child_b = Uuid::new_v4();
    let unrelated = Uuid::new_v4();

    for id in [parent, child_a, child_b, unrelated] {
        store.append_messages(id, &sample_history(id)).await.unwrap();
    }
    for child in [child_a, child_b] {
        store
            .set_conversation_parent(child, parent, None, None, None)
            .await
            .unwrap();
    }

    let children = store.list_child_conversations(parent, 10, 0).await.unwrap();
    let ids: std::collections::HashSet<Uuid> = children.iter().map(|c| c.id).collect();
    assert_eq!(ids, [child_a, child_b].into_iter().collect());
    assert!(
        children
            .iter()
            .all(|c| c.parent_conversation_id == Some(parent)),
        "each child reports its parent"
    );
    assert!(
        store
            .list_child_conversations(unrelated, 10, 0)
            .await
            .unwrap()
            .is_empty(),
        "a conversation with no children returns nothing"
    );
}
