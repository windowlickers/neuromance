use chrono::{DateTime, Utc};
use neuromance_client::LLMClient;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use uuid::Uuid;

use neuromance_common::Conversation;

use crate::error::TokenCounterError;

pub use crate::compaction::{CompactionConfig, CompactionResult, Compactor};
pub use crate::metadata::ContextMetadata;
pub use crate::state::{ContextState, Filtered, Raw, Ready, Transformed};
pub use crate::transforms::{FilterCriteria, TransformPipeline};

/// Main context container that tracks a conversation through state transitions.
///
/// Uses the typestate pattern to ensure valid state transitions at compile time.
/// The `S` parameter represents the current state of the context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context<S: ContextState> {
    /// Unique identifier for this context
    pub id: Uuid,

    /// The conversation being managed
    conversation: Conversation,

    /// Metadata tracking all transformations applied
    metadata: ContextMetadata,

    /// When this context was created
    created_at: DateTime<Utc>,

    /// When this context was last modified
    updated_at: DateTime<Utc>,

    /// State marker (zero-sized type)
    #[serde(skip)]
    state: PhantomData<S>,
}

impl Context<Raw> {
    /// Creates a new context in the raw state with the given conversation.
    pub fn new(conversation: Conversation) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            conversation,
            metadata: ContextMetadata::new(),
            created_at: now,
            updated_at: now,
            state: PhantomData,
        }
    }

    /// Transitions to the filtered state by applying filter criteria.
    pub fn filter(mut self, criteria: FilterCriteria) -> Context<Filtered> {
        self.metadata.add_transformation("filter", &criteria);
        self.updated_at = Utc::now();

        // Apply filtering logic (to be implemented in transforms module)
        let filtered_conversation = crate::transforms::apply_filter(self.conversation, criteria);

        Context {
            id: self.id,
            conversation: filtered_conversation,
            metadata: self.metadata,
            created_at: self.created_at,
            updated_at: self.updated_at,
            state: PhantomData,
        }
    }

    /// Skips filtering and moves directly to the filtered state.
    pub fn skip_filter(mut self) -> Context<Filtered> {
        self.metadata.add_transformation("skip_filter", &());
        self.updated_at = Utc::now();

        Context {
            id: self.id,
            conversation: self.conversation,
            metadata: self.metadata,
            created_at: self.created_at,
            updated_at: self.updated_at,
            state: PhantomData,
        }
    }
}

impl Context<Filtered> {
    /// Transitions to the transformed state by applying default transformation operations.
    pub fn transform(mut self) -> Context<Transformed> {
        self.metadata.add_transformation("transform", &());
        self.updated_at = Utc::now();

        // Apply transformation logic (to be implemented in transforms module)
        let transformed_conversation = crate::transforms::apply_transform(self.conversation);

        Context {
            id: self.id,
            conversation: transformed_conversation,
            metadata: self.metadata,
            created_at: self.created_at,
            updated_at: self.updated_at,
            state: PhantomData,
        }
    }

    /// Transitions to the transformed state by applying a custom transform pipeline.
    pub fn transform_with(mut self, pipeline: &TransformPipeline) -> Context<Transformed> {
        self.metadata
            .add_transformation("transform_with_pipeline", &());
        self.updated_at = Utc::now();

        let transformed_conversation =
            crate::transforms::apply_transform_pipeline(self.conversation, pipeline);

        Context {
            id: self.id,
            conversation: transformed_conversation,
            metadata: self.metadata,
            created_at: self.created_at,
            updated_at: self.updated_at,
            state: PhantomData,
        }
    }

    /// Compacts the conversation using an LLM to summarize older messages.
    ///
    /// This is the primary method for context window management. It uses the provided
    /// compactor to intelligently summarize conversation history while preserving
    /// important context.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use neuromance_context::{Context, Compactor, CompactionConfig};
    ///
    /// let context = Context::new(conversation)
    ///     .filter(FilterCriteria::default());
    ///
    /// let compactor = Compactor::new(client, token_counter)
    ///     .with_config(CompactionConfig::new(4000));
    ///
    /// let compacted = context.compact(&compactor).await?;
    /// ```
    pub async fn compact<C: LLMClient>(
        mut self,
        compactor: &Compactor<C>,
    ) -> Result<(Context<Transformed>, CompactionResult), TokenCounterError> {
        let result = compactor.compact(&self.conversation).await?;

        // Record the compaction in metadata
        self.metadata.add_transformation(
            "compact",
            &serde_json::json!({
                "strategy": format!("{:?}", compactor.config().strategy),
                "original_tokens": result.original_tokens,
                "compacted_tokens": result.compacted_tokens,
                "messages_summarized": result.messages_summarized,
                "was_compacted": result.was_compacted,
            }),
        );
        self.updated_at = Utc::now();

        let context = Context {
            id: self.id,
            conversation: result.conversation.clone(),
            metadata: self.metadata,
            created_at: self.created_at,
            updated_at: self.updated_at,
            state: PhantomData,
        };

        Ok((context, result))
    }

    /// Compacts the conversation only if it exceeds the token threshold.
    ///
    /// This is a convenience method that checks whether compaction is needed
    /// before performing it.
    pub async fn compact_if_needed<C: LLMClient>(
        self,
        compactor: &Compactor<C>,
    ) -> Result<(Context<Transformed>, Option<CompactionResult>), TokenCounterError> {
        if compactor.needs_compaction(&self.conversation)? {
            let (context, result) = self.compact(compactor).await?;
            Ok((context, Some(result)))
        } else {
            Ok((self.skip_transform(), None))
        }
    }

    /// Skips transformation and moves directly to validation.
    pub fn skip_transform(mut self) -> Context<Transformed> {
        self.metadata.add_transformation("skip_transform", &());
        self.updated_at = Utc::now();

        Context {
            id: self.id,
            conversation: self.conversation,
            metadata: self.metadata,
            created_at: self.created_at,
            updated_at: self.updated_at,
            state: PhantomData,
        }
    }
}

impl Context<Transformed> {
    /// Transitions to the ready state, indicating the context is prepared for LLM execution.
    pub fn ready(mut self) -> Context<Ready> {
        self.metadata.add_transformation("ready", &());
        self.updated_at = Utc::now();

        Context {
            id: self.id,
            conversation: self.conversation,
            metadata: self.metadata,
            created_at: self.created_at,
            updated_at: self.updated_at,
            state: PhantomData,
        }
    }
}

impl<S: ContextState> Context<S> {
    /// Returns a reference to the underlying conversation.
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// Returns a reference to the context metadata.
    pub fn metadata(&self) -> &ContextMetadata {
        &self.metadata
    }

    /// Returns the context ID.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Returns when the context was created.
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Returns when the context was last updated.
    pub fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }
}

impl Context<Ready> {
    /// Consumes the context and returns the final conversation.
    pub fn into_conversation(self) -> Conversation {
        self.conversation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuromance_common::chat::Conversation;

    #[test]
    fn test_context_creation() {
        let conv = Conversation::new();
        let context = Context::new(conv.clone());

        assert_eq!(context.conversation().id, conv.id);
        assert_eq!(context.metadata().transformation_count(), 0);
    }

    #[test]
    fn test_state_transitions() {
        let conv = Conversation::new();
        let context = Context::new(conv);

        let context = context
            .filter(FilterCriteria::default())
            .transform()
            .ready();

        assert_eq!(context.metadata().transformation_count(), 3);
    }

    #[test]
    fn test_skip_transform() {
        let conv = Conversation::new();
        let context = Context::new(conv);

        let context = context
            .filter(FilterCriteria::default())
            .skip_transform()
            .ready();

        assert_eq!(context.metadata().transformation_count(), 3);
    }

    #[test]
    fn test_into_conversation() {
        let conv = Conversation::new();
        let conv_id = conv.id;
        let context = Context::new(conv);

        let final_conv = context
            .filter(FilterCriteria::default())
            .transform()
            .ready()
            .into_conversation();

        assert_eq!(final_conv.id, conv_id);
    }
}
