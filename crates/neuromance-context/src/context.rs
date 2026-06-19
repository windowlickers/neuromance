//! Batch context operations over a [`ContextLedger`].
//!
//! These functions apply whole-history transformations — filtering,
//! transformation pipelines, and LLM-backed compaction — to a
//! [`ContextLedger`], recording each as a tracked
//! [`Operation::Replace`](neuromance_common::context::Operation::Replace) edit
//! attributed to [`EditSource::core`]. The ledger itself lives in
//! [`neuromance_common`]; these are the heavier operations that depend on the
//! tokenizer and LLM machinery in this crate.

use neuromance_client::LLMClient;
use neuromance_common::context::{ContextLedger, EditSource};
use serde_json::json;

use crate::compaction::{CompactionResult, Compactor};
use crate::error::TokenCounterError;
use crate::transforms::{self, FilterCriteria, TransformPipeline};

/// Drops messages that fail `criteria`, recording the rewrite in the ledger.
pub fn filter(ledger: &mut ContextLedger, criteria: &FilterCriteria) {
    let filtered = transforms::apply_filter(ledger.conversation().clone(), criteria.clone());
    let details = json!({ "op": "filter", "criteria": serde_json::to_value(criteria).ok() });
    ledger.replace(
        EditSource::core(),
        filtered.get_messages().to_vec(),
        Some(details),
    );
}

/// Applies the default transformation pipeline, recording the rewrite.
pub fn transform(ledger: &mut ContextLedger) {
    let transformed = transforms::apply_transform(ledger.conversation().clone());
    ledger.replace(
        EditSource::core(),
        transformed.get_messages().to_vec(),
        Some(json!({ "op": "transform" })),
    );
}

/// Applies `pipeline` to the history, recording the rewrite.
pub fn transform_with(ledger: &mut ContextLedger, pipeline: &TransformPipeline) {
    let transformed = transforms::apply_transform_pipeline(ledger.conversation().clone(), pipeline);
    ledger.replace(
        EditSource::core(),
        transformed.get_messages().to_vec(),
        Some(json!({ "op": "transform_with_pipeline" })),
    );
}

/// Summarizes older messages with `compactor`, recording the rewrite.
///
/// # Errors
///
/// Returns an error if the compactor's LLM call or token counting fails.
pub async fn compact<C: LLMClient>(
    ledger: &mut ContextLedger,
    compactor: &Compactor<C>,
) -> Result<CompactionResult, TokenCounterError> {
    let result = compactor.compact(ledger.conversation()).await?;
    let details = json!({
        "op": "compact",
        "strategy": format!("{:?}", compactor.config().strategy),
        "original_tokens": result.original_tokens,
        "compacted_tokens": result.compacted_tokens,
        "messages_summarized": result.messages_summarized,
        "was_compacted": result.was_compacted,
    });
    ledger.replace(
        EditSource::core(),
        result.conversation.get_messages().to_vec(),
        Some(details),
    );
    Ok(result)
}

/// Compacts only if the conversation exceeds the compactor's token threshold.
///
/// # Errors
///
/// Returns an error if token counting or the compactor's LLM call fails.
pub async fn compact_if_needed<C: LLMClient>(
    ledger: &mut ContextLedger,
    compactor: &Compactor<C>,
) -> Result<Option<CompactionResult>, TokenCounterError> {
    if compactor.needs_compaction(ledger.conversation())? {
        Ok(Some(compact(ledger, compactor).await?))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuromance_common::Conversation;
    use neuromance_common::chat::{Message, MessageRole};
    use neuromance_common::context::Operation;

    #[test]
    fn test_filter_keeps_matching_roles_and_records_replace() {
        let conv = Conversation::new();
        let id = conv.id;
        let mut ledger = ContextLedger::new(conv);
        ledger.append(
            EditSource::core(),
            [
                Message::user(id, "hi"),
                Message::system(id, "sys"),
                Message::assistant(id, "yo"),
            ],
        );

        filter(
            &mut ledger,
            &FilterCriteria::default().with_roles(vec![MessageRole::User]),
        );

        assert_eq!(ledger.messages().len(), 1);
        assert_eq!(ledger.messages()[0].role, MessageRole::User);
        assert!(ledger.metadata().has_operation(Operation::Replace));
    }

    #[test]
    fn test_transform_removes_empty_and_records() {
        let conv = Conversation::new();
        let id = conv.id;
        let mut ledger = ContextLedger::new(conv);
        ledger.append(
            EditSource::core(),
            [
                Message::user(id, "hello"),
                Message::assistant(id, ""),
                Message::user(id, "hello"),
            ],
        );

        transform(&mut ledger);

        // RemoveEmpty drops the blank assistant turn; Deduplicate drops the
        // repeated user turn, leaving one message.
        assert_eq!(ledger.messages().len(), 1);
        assert_eq!(ledger.metadata().records_for(Operation::Replace).len(), 1);
    }
}
