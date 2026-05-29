//! Shared accumulator for assembling [`Message`]s from provider wire formats.
//!
//! Each LLM provider returns responses in a different shape — Anthropic
//! gives content blocks, the Chat Completions API gives a single message
//! object, and the Responses API gives an output-item stream. Every
//! provider eventually has to fold its shape into our common [`Message`],
//! and that fold is where the duplication lives: the boilerplate around
//! `id`, `timestamp`, `metadata`, plus the per-provider walks that
//! accumulate text, push tool calls, and stitch reasoning content
//! together.
//!
//! [`MessageBuilder`] centralises that fold. It exposes only the
//! incremental operations the wire-format walks actually need
//! (`append_content`, `push_tool_call`, `append_reasoning`, …) so the
//! provider conversions stay thin.
//!
//! [`Message`]: neuromance_common::chat::Message

use std::collections::HashMap;

use chrono::Utc;
use smallvec::SmallVec;
use uuid::Uuid;

use neuromance_common::chat::{Message, MessageRole, ReasoningContent};
use neuromance_common::tools::ToolCall;

/// Incremental builder for a single [`Message`].
///
/// Construct one per response, walk the wire format, and call [`build`]
/// at the end. The defaults for `id` (random UUID), `timestamp` (now),
/// and `metadata` (empty) are filled in at build time.
///
/// [`build`]: Self::build
pub struct MessageBuilder {
    conversation_id: Uuid,
    role: MessageRole,
    content: String,
    tool_calls: SmallVec<[ToolCall; 2]>,
    reasoning_content: String,
    reasoning_signature: Option<String>,
    has_reasoning: bool,
    tool_call_id: Option<String>,
    name: Option<String>,
}

impl MessageBuilder {
    /// Start a build for `conversation_id` with the given `role`.
    pub fn new(conversation_id: Uuid, role: MessageRole) -> Self {
        Self {
            conversation_id,
            role,
            content: String::new(),
            tool_calls: SmallVec::new(),
            reasoning_content: String::new(),
            reasoning_signature: None,
            has_reasoning: false,
            tool_call_id: None,
            name: None,
        }
    }

    /// Append `text` to the content, prefixed by `\n` when content is
    /// non-empty. Use this when walking a list of text blocks.
    pub fn append_content(&mut self, text: &str) {
        if !self.content.is_empty() {
            self.content.push('\n');
        }
        self.content.push_str(text);
    }

    /// Replace the content with `content`. Use this when the wire format
    /// supplies the full content as a single field.
    pub fn set_content(&mut self, content: String) {
        self.content = content;
    }

    /// Push a tool call onto the build.
    pub fn push_tool_call(&mut self, tool_call: ToolCall) {
        self.tool_calls.push(tool_call);
    }

    /// Append `text` to the reasoning content, prefixed by `separator`
    /// only when reasoning was already populated. Use `"\n"` for
    /// line-joined blocks or `"\n\n"` for paragraph-joined blocks.
    ///
    /// Calling this method always marks reasoning as present, so
    /// [`build`](Self::build) will emit a [`ReasoningContent`] even if
    /// `text` happens to be empty.
    pub fn append_reasoning(&mut self, text: &str, separator: &str) {
        if self.has_reasoning {
            self.reasoning_content.push_str(separator);
        }
        self.reasoning_content.push_str(text);
        self.has_reasoning = true;
    }

    /// Set the reasoning signature, replacing any prior value. Anthropic
    /// uses this to verify thinking-content integrity on resubmission.
    pub fn set_reasoning_signature(&mut self, signature: String) {
        self.reasoning_signature = Some(signature);
        self.has_reasoning = true;
    }

    /// Set the `tool_call_id` field — required for messages with the
    /// `Tool` role.
    pub fn set_tool_call_id(&mut self, id: String) {
        self.tool_call_id = Some(id);
    }

    /// Set the `name` field — required for messages with the `Tool` role.
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Finalize into a [`Message`].
    ///
    /// `reasoning` is emitted only if any reasoning method was called on
    /// this builder; otherwise it stays `None`.
    #[must_use]
    pub fn build(self) -> Message {
        let reasoning = if self.has_reasoning {
            Some(ReasoningContent {
                content: self.reasoning_content,
                signature: self.reasoning_signature,
            })
        } else {
            None
        };
        Message {
            id: Uuid::new_v4(),
            conversation_id: self.conversation_id,
            role: self.role,
            content: self.content,
            tool_calls: self.tool_calls,
            tool_call_id: self.tool_call_id,
            name: self.name,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            reasoning,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use neuromance_common::tools::FunctionCall;

    use super::*;

    fn cid() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn empty_builder_yields_assistant_message_with_defaults() {
        let conversation_id = cid();
        let msg = MessageBuilder::new(conversation_id, MessageRole::Assistant).build();

        assert_eq!(msg.conversation_id, conversation_id);
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.content, "");
        assert!(msg.tool_calls.is_empty());
        assert!(msg.reasoning.is_none());
        assert!(msg.tool_call_id.is_none());
        assert!(msg.name.is_none());
    }

    #[test]
    fn append_content_joins_blocks_with_newline() {
        let mut b = MessageBuilder::new(cid(), MessageRole::Assistant);
        b.append_content("hello");
        b.append_content("world");
        b.append_content("again");
        assert_eq!(b.build().content, "hello\nworld\nagain");
    }

    #[test]
    fn set_content_replaces_prior_content() {
        let mut b = MessageBuilder::new(cid(), MessageRole::Assistant);
        b.append_content("first");
        b.set_content("second".to_string());
        assert_eq!(b.build().content, "second");
    }

    #[test]
    fn push_tool_call_collects_in_order() {
        let mut b = MessageBuilder::new(cid(), MessageRole::Assistant);
        b.push_tool_call(ToolCall {
            id: "a".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "f".into(),
                arguments: "{}".into(),
            },
            index: None,
        });
        b.push_tool_call(ToolCall {
            id: "b".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "g".into(),
                arguments: "{}".into(),
            },
            index: None,
        });
        let msg = b.build();
        assert_eq!(msg.tool_calls.len(), 2);
        assert_eq!(msg.tool_calls[0].id, "a");
        assert_eq!(msg.tool_calls[1].id, "b");
    }

    #[test]
    fn append_reasoning_skips_separator_on_first_call() {
        let mut b = MessageBuilder::new(cid(), MessageRole::Assistant);
        b.append_reasoning("alpha", "\n\n");
        b.append_reasoning("beta", "\n\n");
        b.append_reasoning("gamma", "\n\n");
        let r = b.build().reasoning.unwrap();
        assert_eq!(r.content, "alpha\n\nbeta\n\ngamma");
        assert!(r.signature.is_none());
    }

    #[test]
    fn append_reasoning_supports_different_separators() {
        let mut b = MessageBuilder::new(cid(), MessageRole::Assistant);
        b.append_reasoning("one", "\n");
        b.append_reasoning("two", "\n");
        assert_eq!(b.build().reasoning.unwrap().content, "one\ntwo");
    }

    #[test]
    fn reasoning_signature_set_without_content_still_emits_reasoning() {
        let mut b = MessageBuilder::new(cid(), MessageRole::Assistant);
        b.set_reasoning_signature("sig".to_string());
        let r = b.build().reasoning.unwrap();
        assert_eq!(r.content, "");
        assert_eq!(r.signature.as_deref(), Some("sig"));
    }

    #[test]
    fn empty_reasoning_text_still_emits_reasoning_content() {
        // Matches existing ChatCompletions behaviour: Some("") on the wire
        // becomes Some(ReasoningContent::new("")) — distinct from None.
        let mut b = MessageBuilder::new(cid(), MessageRole::Assistant);
        b.append_reasoning("", "\n");
        let r = b.build().reasoning.unwrap();
        assert_eq!(r.content, "");
        assert!(r.signature.is_none());
    }

    #[test]
    fn no_reasoning_methods_means_no_reasoning_emitted() {
        let mut b = MessageBuilder::new(cid(), MessageRole::Assistant);
        b.append_content("just text");
        assert!(b.build().reasoning.is_none());
    }

    #[test]
    fn set_tool_call_id_and_name_round_trip() {
        let mut b = MessageBuilder::new(cid(), MessageRole::Tool);
        b.set_tool_call_id("call_123".to_string());
        b.set_name("get_weather".to_string());
        b.set_content("result".to_string());
        let msg = b.build();
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(msg.name.as_deref(), Some("get_weather"));
        assert_eq!(msg.role, MessageRole::Tool);
    }
}
