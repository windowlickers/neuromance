//! Serializing message content for the opt-in GenAI content attributes.
//!
//! The conventions model messages as `{role, parts: [...]}`, and OTLP
//! attributes cannot nest, so each structure is serialized to a JSON string.
//!
//! Every `*_json` builder returns `None` when
//! `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is off. That is
//! deliberate: the gate is checked in this module and nowhere else, so no
//! caller can forget it and ship a prompt to a trace backend.
//!
//! Each is a thin gate over a pure builder of the same name. The split keeps
//! the serialized shape testable: the gate reads the environment once into a
//! `OnceLock`, so a test cannot flip it, and setting an environment variable
//! is `unsafe` in this edition.

use serde::Serialize;
use serde_json::json;

use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::ChatResponse;
use neuromance_common::telemetry::capture_message_content;

/// One entry of `gen_ai.input.messages` or `gen_ai.output.messages`.
#[derive(Serialize)]
struct CapturedMessage {
    role: String,
    parts: Vec<serde_json::Value>,
}

/// The `gen_ai.system_instructions` value: the system-role messages.
///
/// Returns `None` when capture is off or the history carries no system
/// message.
pub fn system_instructions_json(messages: &[Message]) -> Option<String> {
    capture_message_content()
        .then(|| system_instructions(messages))
        .flatten()
}

fn system_instructions(messages: &[Message]) -> Option<String> {
    let parts: Vec<serde_json::Value> = messages
        .iter()
        .filter(|message| message.role == MessageRole::System)
        .map(|message| json!({"type": "text", "content": message.content}))
        .collect();
    if parts.is_empty() {
        return None;
    }
    serde_json::to_string(&parts).ok()
}

/// The `gen_ai.input.messages` value: everything but the system instructions,
/// which the conventions carry in their own attribute.
pub fn input_messages_json(messages: &[Message]) -> Option<String> {
    capture_message_content()
        .then(|| input_messages(messages))
        .flatten()
}

fn input_messages(messages: &[Message]) -> Option<String> {
    let captured: Vec<CapturedMessage> = messages
        .iter()
        .filter(|message| message.role != MessageRole::System)
        .map(capture_message)
        .collect();
    if captured.is_empty() {
        return None;
    }
    serde_json::to_string(&captured).ok()
}

/// The `gen_ai.output.messages` value for a non-streaming response.
pub fn output_messages_json(response: &ChatResponse) -> Option<String> {
    capture_message_content()
        .then(|| output_messages(response))
        .flatten()
}

fn output_messages(response: &ChatResponse) -> Option<String> {
    let mut captured = capture_message(&response.message);
    if let Some(reason) = response.finish_reason {
        return serde_json::to_string(&vec![json!({
            "role": captured.role.clone(),
            "parts": std::mem::take(&mut captured.parts),
            "finish_reason": reason.to_string(),
        })])
        .ok();
    }
    serde_json::to_string(&vec![captured]).ok()
}

/// The `gen_ai.output.messages` value for a stream, whose text and tool calls
/// are accumulated rather than carried on one message.
pub fn streamed_output_messages_json(
    content: &str,
    tool_calls: &[neuromance_common::tools::ToolCall],
    finish_reason: Option<&str>,
) -> Option<String> {
    capture_message_content()
        .then(|| streamed_output_messages(content, tool_calls, finish_reason))
        .flatten()
}

fn streamed_output_messages(
    content: &str,
    tool_calls: &[neuromance_common::tools::ToolCall],
    finish_reason: Option<&str>,
) -> Option<String> {
    let mut parts = Vec::new();
    if !content.is_empty() {
        parts.push(json!({"type": "text", "content": content}));
    }
    parts.extend(tool_calls.iter().map(tool_call_part));
    if parts.is_empty() {
        return None;
    }
    serde_json::to_string(&vec![json!({
        "role": "assistant",
        "parts": parts,
        "finish_reason": finish_reason,
    })])
    .ok()
}

fn capture_message(message: &Message) -> CapturedMessage {
    let mut parts = Vec::new();
    if let Some(ref reasoning) = message.reasoning
        && !reasoning.content.is_empty()
    {
        parts.push(json!({"type": "reasoning", "content": reasoning.content}));
    }
    if !message.content.is_empty() {
        parts.push(json!({"type": "text", "content": message.content}));
    }
    if message.role == MessageRole::Tool {
        parts.push(json!({
            "type": "tool_call_response",
            "id": message.tool_call_id,
            "response": message.content,
        }));
    }
    parts.extend(message.tool_calls.iter().map(tool_call_part));

    CapturedMessage {
        role: role_name(message.role),
        parts,
    }
}

fn tool_call_part(call: &neuromance_common::tools::ToolCall) -> serde_json::Value {
    json!({
        "type": "tool_call",
        "id": call.id,
        "name": call.function.name,
        "arguments": call.function.arguments,
    })
}

/// The wire name of a role.
///
/// Taken from the serde representation rather than a match: `MessageRole` is
/// `#[non_exhaustive]`, so a match here would need a catch-all that silently
/// mislabels any role added later. The serde rename attributes already carry
/// the exact names the conventions expect.
fn role_name(role: MessageRole) -> String {
    serde_json::to_value(role)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_common::tools::{FunctionCall, ToolCall};
    use uuid::Uuid;

    fn history() -> Vec<Message> {
        let conversation = Uuid::new_v4();
        let mut assistant = Message::assistant(conversation, "on it");
        assistant.tool_calls.push(ToolCall {
            id: "call_1".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: r#"{"text":"hi"}"#.to_string(),
            },
            call_type: "function".to_string(),
            index: Some(0),
        });
        vec![
            Message::system(conversation, "be useful"),
            Message::user(conversation, "echo hi"),
            assistant,
        ]
    }

    /// The default must be silence. A test process has the gate unset, so
    /// these calls exercise the closed path.
    #[test]
    fn test_every_builder_is_silent_when_capture_is_off() {
        assert!(!capture_message_content(), "the gate must default to off");

        let messages = history();
        assert_eq!(input_messages_json(&messages), None);
        assert_eq!(system_instructions_json(&messages), None);
        assert_eq!(streamed_output_messages_json("hi", &[], None), None);
    }

    /// The shape below is what a backend renders, so it is worth pinning even
    /// though the gate keeps it out of production traces by default.
    #[test]
    fn test_captured_message_splits_text_from_tool_calls() {
        let messages = history();
        let assistant = messages.last().expect("an assistant message");

        let captured = capture_message(assistant);
        let value = serde_json::to_value(&captured).expect("serializable");

        assert_eq!(value["role"], "assistant");
        assert_eq!(value["parts"][0]["type"], "text");
        assert_eq!(value["parts"][0]["content"], "on it");
        assert_eq!(value["parts"][1]["type"], "tool_call");
        assert_eq!(value["parts"][1]["name"], "echo");
        assert_eq!(value["parts"][1]["id"], "call_1");
    }

    /// System instructions have their own attribute, so leaving them in the
    /// input list would ship the system prompt twice.
    #[test]
    fn test_system_messages_are_not_part_of_the_input_list() {
        let json = input_messages(&history()).expect("a non-empty history");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");

        let roles: Vec<&str> = value
            .as_array()
            .expect("an array")
            .iter()
            .filter_map(|entry| entry["role"].as_str())
            .collect();
        assert_eq!(roles, vec!["user", "assistant"]);
        assert!(
            !json.contains("be useful"),
            "the system prompt belongs in gen_ai.system_instructions only: {json}"
        );
    }

    #[test]
    fn test_system_instructions_carry_only_the_system_messages() {
        let json = system_instructions(&history()).expect("a system message");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");

        assert_eq!(value.as_array().map(Vec::len), Some(1));
        assert_eq!(value[0]["type"], "text");
        assert_eq!(value[0]["content"], "be useful");
    }

    /// A history with nothing to say produces no attribute at all, rather than
    /// an empty array a backend would render as a message.
    #[test]
    fn test_absent_content_produces_no_attribute() {
        assert_eq!(system_instructions(&[]), None);
        assert_eq!(input_messages(&[]), None);
        assert_eq!(streamed_output_messages("", &[], None), None);
    }

    /// A stream's text and tool calls arrive as deltas, so the accumulated
    /// output is assembled rather than read off one message.
    #[test]
    fn test_streamed_output_carries_text_tool_calls_and_finish_reason() {
        let calls = [ToolCall {
            id: "call_1".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: r#"{"text":"hi"}"#.to_string(),
            },
            call_type: "function".to_string(),
            index: Some(0),
        }];

        let json =
            streamed_output_messages("partial answer", &calls, Some("tool_calls")).expect("output");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");

        assert_eq!(value[0]["role"], "assistant");
        assert_eq!(value[0]["finish_reason"], "tool_calls");
        assert_eq!(value[0]["parts"][0]["content"], "partial answer");
        assert_eq!(value[0]["parts"][1]["name"], "echo");
    }
}
