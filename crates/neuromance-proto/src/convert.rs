//! Conversions between proto types and domain types.

use chrono::{DateTime, Utc};
use neuromance_common::chat::{Message, MessageRole, ReasoningContent};
use neuromance_common::client::{InputTokensDetails, OutputTokensDetails, Usage};
use neuromance_common::protocol::{ConversationSummary, ModelProfile};
use neuromance_common::tools::{FunctionCall, ToolApproval, ToolCall};
use smallvec::SmallVec;

use crate::proto;

// --- MessageRole ---

impl From<MessageRole> for proto::MessageRole {
    fn from(role: MessageRole) -> Self {
        match role {
            MessageRole::System => Self::System,
            MessageRole::User => Self::User,
            MessageRole::Assistant => Self::Assistant,
            MessageRole::Tool => Self::Tool,
            _ => Self::Unspecified,
        }
    }
}

impl From<proto::MessageRole> for MessageRole {
    fn from(role: proto::MessageRole) -> Self {
        match role {
            proto::MessageRole::System => Self::System,
            proto::MessageRole::User | proto::MessageRole::Unspecified => Self::User,
            proto::MessageRole::Assistant => Self::Assistant,
            proto::MessageRole::Tool => Self::Tool,
        }
    }
}

// --- ToolCall / FunctionCall ---

impl From<&ToolCall> for proto::ToolCallProto {
    fn from(tc: &ToolCall) -> Self {
        Self {
            id: tc.id.clone(),
            function: Some(proto::FunctionCallProto {
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
            }),
            call_type: tc.call_type.clone(),
        }
    }
}

impl From<proto::ToolCallProto> for ToolCall {
    fn from(tc: proto::ToolCallProto) -> Self {
        let function = tc.function.map_or_else(
            || FunctionCall {
                name: String::new(),
                arguments: Vec::new(),
            },
            |f| FunctionCall {
                name: f.name,
                arguments: f.arguments,
            },
        );
        Self {
            id: tc.id,
            function,
            call_type: tc.call_type,
        }
    }
}

// --- ToolApproval ---

impl From<&ToolApproval> for proto::ToolApprovalDecision {
    fn from(approval: &ToolApproval) -> Self {
        match approval {
            ToolApproval::Approved => Self {
                decision: Some(proto::tool_approval_decision::Decision::Approved(true)),
            },
            ToolApproval::Denied(reason) => Self {
                decision: Some(proto::tool_approval_decision::Decision::DeniedReason(
                    reason.clone(),
                )),
            },
            ToolApproval::Quit => Self {
                decision: Some(proto::tool_approval_decision::Decision::Quit(true)),
            },
        }
    }
}

impl From<proto::ToolApprovalDecision> for ToolApproval {
    fn from(decision: proto::ToolApprovalDecision) -> Self {
        match decision.decision {
            Some(proto::tool_approval_decision::Decision::Approved(true)) => Self::Approved,
            Some(proto::tool_approval_decision::Decision::DeniedReason(reason)) => {
                Self::Denied(reason)
            }
            Some(proto::tool_approval_decision::Decision::Quit(true)) => Self::Quit,
            _ => Self::Denied("Unknown approval decision".to_string()),
        }
    }
}

// --- ReasoningContent ---

impl From<&ReasoningContent> for proto::ReasoningContentProto {
    fn from(rc: &ReasoningContent) -> Self {
        Self {
            content: rc.content.clone(),
            signature: rc.signature.clone(),
        }
    }
}

impl From<proto::ReasoningContentProto> for ReasoningContent {
    fn from(rc: proto::ReasoningContentProto) -> Self {
        Self {
            content: rc.content,
            signature: rc.signature,
        }
    }
}

// --- Message ---

impl From<&Message> for proto::MessageProto {
    fn from(msg: &Message) -> Self {
        let metadata = msg
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect();

        Self {
            id: msg.id.to_string(),
            conversation_id: msg.conversation_id.to_string(),
            role: proto::MessageRole::from(msg.role).into(),
            content: msg.content.clone(),
            metadata,
            timestamp: msg.timestamp.to_rfc3339(),
            tool_calls: msg
                .tool_calls
                .iter()
                .map(proto::ToolCallProto::from)
                .collect(),
            tool_call_id: msg.tool_call_id.clone(),
            name: msg.name.clone(),
            reasoning: msg
                .reasoning
                .as_ref()
                .map(proto::ReasoningContentProto::from),
        }
    }
}

/// Converts a proto message back to a domain `Message`.
///
/// # Errors
///
/// Returns an error if UUID or timestamp parsing fails.
pub fn message_from_proto(msg: proto::MessageProto) -> Result<Message, String> {
    let id = msg
        .id
        .parse()
        .map_err(|e| format!("Invalid message id: {e}"))?;
    let conversation_id = msg
        .conversation_id
        .parse()
        .map_err(|e| format!("Invalid conversation_id: {e}"))?;
    let timestamp: DateTime<Utc> = msg
        .timestamp
        .parse()
        .map_err(|e| format!("Invalid timestamp: {e}"))?;

    let role: MessageRole = proto::MessageRole::try_from(msg.role)
        .map_err(|_| format!("Invalid message role: {}", msg.role))?
        .into();

    let metadata = msg
        .metadata
        .into_iter()
        .map(|(k, v)| {
            let val = serde_json::from_str(&v).unwrap_or(serde_json::Value::String(v));
            (k, val)
        })
        .collect();

    let tool_calls: SmallVec<[ToolCall; 2]> =
        msg.tool_calls.into_iter().map(ToolCall::from).collect();

    let reasoning = msg.reasoning.map(ReasoningContent::from);

    Ok(Message {
        id,
        conversation_id,
        role,
        content: msg.content,
        metadata,
        timestamp,
        tool_calls,
        tool_call_id: msg.tool_call_id,
        name: msg.name,
        reasoning,
    })
}

// --- Usage ---

impl From<&Usage> for proto::UsageProto {
    fn from(u: &Usage) -> Self {
        Self {
            conversation_id: String::new(),
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cost: u.cost,
            input_tokens_details: u.input_tokens_details.as_ref().map(|d| {
                proto::InputTokensDetailsProto {
                    cached_tokens: d.cached_tokens,
                    cache_creation_tokens: d.cache_creation_tokens,
                }
            }),
            output_tokens_details: u.output_tokens_details.as_ref().map(|d| {
                proto::OutputTokensDetailsProto {
                    reasoning_tokens: d.reasoning_tokens,
                }
            }),
        }
    }
}

impl From<&proto::UsageProto> for Usage {
    fn from(u: &proto::UsageProto) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cost: u.cost,
            input_tokens_details: u.input_tokens_details.as_ref().map(|d| InputTokensDetails {
                cached_tokens: d.cached_tokens,
                cache_creation_tokens: d.cache_creation_tokens,
            }),
            output_tokens_details: u
                .output_tokens_details
                .as_ref()
                .map(|d| OutputTokensDetails {
                    reasoning_tokens: d.reasoning_tokens,
                }),
        }
    }
}

// --- ConversationSummary ---

impl From<&ConversationSummary> for proto::ConversationSummaryProto {
    fn from(cs: &ConversationSummary) -> Self {
        Self {
            id: cs.id.clone(),
            short_id: cs.short_id.clone(),
            title: cs.title.clone(),
            message_count: cs.message_count as u64,
            created_at: cs.created_at.to_rfc3339(),
            updated_at: cs.updated_at.to_rfc3339(),
            bookmarks: cs.bookmarks.clone(),
            model: cs.model.clone(),
        }
    }
}

/// Converts a proto conversation summary back to a domain type.
///
/// # Errors
///
/// Returns an error if timestamp parsing fails.
pub fn conversation_summary_from_proto(
    cs: proto::ConversationSummaryProto,
) -> Result<ConversationSummary, String> {
    let created_at: DateTime<Utc> = cs
        .created_at
        .parse()
        .map_err(|e| format!("Invalid created_at: {e}"))?;
    let updated_at: DateTime<Utc> = cs
        .updated_at
        .parse()
        .map_err(|e| format!("Invalid updated_at: {e}"))?;

    Ok(ConversationSummary {
        id: cs.id,
        short_id: cs.short_id,
        title: cs.title,
        #[allow(clippy::cast_possible_truncation)]
        message_count: cs.message_count as usize,
        created_at,
        updated_at,
        bookmarks: cs.bookmarks,
        model: cs.model,
    })
}

// --- ModelProfile ---

impl From<&ModelProfile> for proto::ModelProfileProto {
    fn from(mp: &ModelProfile) -> Self {
        Self {
            nickname: mp.nickname.clone(),
            provider: mp.provider.clone(),
            model: mp.model.clone(),
            api_key_env: mp.api_key_env.clone(),
            base_url: mp.base_url.clone(),
        }
    }
}

impl From<proto::ModelProfileProto> for ModelProfile {
    fn from(mp: proto::ModelProfileProto) -> Self {
        Self {
            nickname: mp.nickname,
            provider: mp.provider,
            model: mp.model,
            api_key_env: mp.api_key_env,
            base_url: mp.base_url,
        }
    }
}

// --- ErrorCode ---

impl From<proto::ErrorCode> for neuromance_common::protocol::ErrorCode {
    fn from(code: proto::ErrorCode) -> Self {
        match code {
            proto::ErrorCode::ConversationNotFound => Self::ConversationNotFound,
            proto::ErrorCode::ModelNotFound => Self::ModelNotFound,
            proto::ErrorCode::BookmarkNotFound => Self::BookmarkNotFound,
            proto::ErrorCode::BookmarkExists => Self::BookmarkExists,
            proto::ErrorCode::NoActiveConversation => Self::NoActiveConversation,
            proto::ErrorCode::InvalidConversationId => Self::InvalidConversationId,
            proto::ErrorCode::LlmError => Self::LlmError,
            proto::ErrorCode::ConfigError => Self::ConfigError,
            proto::ErrorCode::StorageError => Self::StorageError,
            proto::ErrorCode::InvalidRequest => Self::InvalidRequest,
            proto::ErrorCode::Internal | proto::ErrorCode::Unspecified => Self::Internal,
        }
    }
}

impl From<neuromance_common::protocol::ErrorCode> for proto::ErrorCode {
    fn from(code: neuromance_common::protocol::ErrorCode) -> Self {
        match code {
            neuromance_common::protocol::ErrorCode::ConversationNotFound => {
                Self::ConversationNotFound
            }
            neuromance_common::protocol::ErrorCode::ModelNotFound => Self::ModelNotFound,
            neuromance_common::protocol::ErrorCode::BookmarkNotFound => Self::BookmarkNotFound,
            neuromance_common::protocol::ErrorCode::BookmarkExists => Self::BookmarkExists,
            neuromance_common::protocol::ErrorCode::NoActiveConversation => {
                Self::NoActiveConversation
            }
            neuromance_common::protocol::ErrorCode::InvalidConversationId => {
                Self::InvalidConversationId
            }
            neuromance_common::protocol::ErrorCode::LlmError => Self::LlmError,
            neuromance_common::protocol::ErrorCode::ConfigError => Self::ConfigError,
            neuromance_common::protocol::ErrorCode::StorageError => Self::StorageError,
            neuromance_common::protocol::ErrorCode::InvalidRequest => Self::InvalidRequest,
            _ => Self::Internal,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::panic)]
    #![allow(clippy::match_same_arms)]

    use super::*;

    #[test]
    fn test_message_role_roundtrip() {
        let roles = [
            MessageRole::System,
            MessageRole::User,
            MessageRole::Assistant,
            MessageRole::Tool,
        ];

        for role in roles {
            let proto_role = proto::MessageRole::from(role);
            let back: MessageRole = proto_role.into();
            assert_eq!(role, back);
        }
    }

    #[test]
    fn test_tool_call_roundtrip() {
        let tc = ToolCall::new("test_tool", vec!["arg1".to_string()]);
        let proto_tc = proto::ToolCallProto::from(&tc);
        let back = ToolCall::from(proto_tc);

        assert_eq!(tc.id, back.id);
        assert_eq!(tc.function.name, back.function.name);
        assert_eq!(tc.function.arguments, back.function.arguments);
    }

    #[test]
    fn test_tool_approval_roundtrip() {
        let cases = [
            ToolApproval::Approved,
            ToolApproval::Denied("reason".to_string()),
            ToolApproval::Quit,
        ];

        for approval in cases {
            let proto_approval = proto::ToolApprovalDecision::from(&approval);
            let back = ToolApproval::from(proto_approval);

            match (&approval, &back) {
                (ToolApproval::Approved, ToolApproval::Approved) => {}
                (ToolApproval::Denied(a), ToolApproval::Denied(b)) => assert_eq!(a, b),
                (ToolApproval::Quit, ToolApproval::Quit) => {}
                _ => panic!("Mismatch: {approval:?} != {back:?}"),
            }
        }
    }

    #[test]
    fn test_usage_roundtrip() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cost: Some(0.01),
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: 10,
                cache_creation_tokens: 5,
            }),
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 20,
            }),
        };

        let proto_usage = proto::UsageProto::from(&usage);
        let back = Usage::from(&proto_usage);

        assert_eq!(usage.prompt_tokens, back.prompt_tokens);
        assert_eq!(usage.completion_tokens, back.completion_tokens);
        assert_eq!(usage.total_tokens, back.total_tokens);
        assert_eq!(usage.cost, back.cost);
        assert_eq!(
            usage.input_tokens_details.as_ref().unwrap().cached_tokens,
            back.input_tokens_details.as_ref().unwrap().cached_tokens
        );
    }

    #[test]
    fn test_conversation_summary_roundtrip() {
        let cs = ConversationSummary {
            id: "abc12345-6789-0000-0000-000000000000".to_string(),
            short_id: "abc1234".to_string(),
            title: Some("Test".to_string()),
            message_count: 5,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            bookmarks: vec!["bm1".to_string()],
            model: "sonnet".to_string(),
        };

        let proto_cs = proto::ConversationSummaryProto::from(&cs);
        let back = conversation_summary_from_proto(proto_cs).unwrap();

        assert_eq!(cs.id, back.id);
        assert_eq!(cs.short_id, back.short_id);
        assert_eq!(cs.title, back.title);
        assert_eq!(cs.message_count, back.message_count);
        assert_eq!(cs.bookmarks, back.bookmarks);
        assert_eq!(cs.model, back.model);
    }

    #[test]
    fn test_model_profile_roundtrip() {
        let mp = ModelProfile {
            nickname: "sonnet".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            base_url: None,
        };

        let proto_mp = proto::ModelProfileProto::from(&mp);
        let back = ModelProfile::from(proto_mp);

        assert_eq!(mp.nickname, back.nickname);
        assert_eq!(mp.provider, back.provider);
        assert_eq!(mp.model, back.model);
        assert_eq!(mp.api_key_env, back.api_key_env);
        assert_eq!(mp.base_url, back.base_url);
    }

    #[test]
    fn test_message_roundtrip() {
        use std::collections::HashMap;
        use uuid::Uuid;

        let msg = Message {
            id: Uuid::new_v4(),
            conversation_id: Uuid::new_v4(),
            role: MessageRole::Assistant,
            content: "Hello".to_string(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            tool_calls: SmallVec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
        };

        let proto_msg = proto::MessageProto::from(&msg);
        let back = message_from_proto(proto_msg).unwrap();

        assert_eq!(msg.id, back.id);
        assert_eq!(msg.conversation_id, back.conversation_id);
        assert_eq!(msg.role, back.role);
        assert_eq!(msg.content, back.content);
    }

    #[test]
    fn test_message_from_proto_invalid_role() {
        use std::collections::HashMap;
        use uuid::Uuid;

        let proto_msg = proto::MessageProto {
            id: Uuid::new_v4().to_string(),
            conversation_id: Uuid::new_v4().to_string(),
            role: 999,
            content: "Hello".to_string(),
            metadata: HashMap::default(),
            timestamp: Utc::now().to_rfc3339(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning: None,
        };

        let err = message_from_proto(proto_msg).unwrap_err();
        assert!(err.contains("Invalid message role"), "got: {err}");
    }

    #[test]
    fn test_error_code_roundtrip() {
        use neuromance_common::protocol::ErrorCode as DomainErrorCode;

        let codes = [
            DomainErrorCode::ConversationNotFound,
            DomainErrorCode::ModelNotFound,
            DomainErrorCode::BookmarkNotFound,
            DomainErrorCode::BookmarkExists,
            DomainErrorCode::NoActiveConversation,
            DomainErrorCode::InvalidConversationId,
            DomainErrorCode::LlmError,
            DomainErrorCode::ConfigError,
            DomainErrorCode::StorageError,
            DomainErrorCode::InvalidRequest,
            DomainErrorCode::Internal,
        ];

        for code in codes {
            let proto_code = proto::ErrorCode::from(code);
            let back = DomainErrorCode::from(proto_code);
            assert_eq!(code, back);
        }
    }
}
