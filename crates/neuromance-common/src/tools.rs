//! Tool calling and function execution types for LLM interactions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use uuid::Uuid;

/// Represents the approval status of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum ToolApproval {
    /// The tool call is approved and should be executed.
    Approved,
    /// The tool call is denied with a reason.
    Denied(String),
    /// Quit the current operation.
    Quit,
}

/// Describes a single property in a function parameter schema.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Property {
    /// The JSON type (e.g., "string", "number", "object").
    #[serde(rename = "type")]
    pub prop_type: String,
    /// Human-readable description of this property.
    pub description: String,
}

/// Defines the parameter schema for a function using JSON Schema conventions.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Parameters {
    /// The JSON type, typically "object".
    #[serde(rename = "type")]
    pub param_type: String,
    /// Map of parameter names to their property definitions.
    pub properties: HashMap<String, Property>,
    /// List of required parameter names.
    pub required: Vec<String>,
}

/// Describes a function that can be called by an LLM.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Function {
    /// The name of the function.
    pub name: String,
    /// Human-readable description of what the function does.
    pub description: String,
    /// JSON Schema definition of the function's parameters.
    pub parameters: serde_json::Value,
}

/// Represents a tool available to the LLM, typically wrapping a function.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, Eq, PartialEq)]
pub struct Tool {
    /// The type of tool (defaults to "function").
    #[serde(rename = "type")]
    #[builder(default = "function".to_string())]
    pub r#type: String,
    /// The function definition.
    pub function: Function,
}

/// Represents an invocation of a function with arguments.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    /// The name of the function being called.
    pub name: String,
    /// The arguments passed to the function as strings.
    pub arguments: Vec<String>,
}

/// Represents a complete tool call from an LLM, including ID and function details.
///
/// Arguments in `function.arguments` are passed through as-is from API responses.
/// Users should validate and parse arguments when executing tools.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The function being invoked.
    pub function: FunctionCall,
    /// The type of call, typically "function".
    pub call_type: String,
}

impl ToolCall {
    /// Creates a new tool call with a generated ID.
    pub fn new<I, T>(name: impl Into<String>, arguments: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            id: Uuid::new_v4().to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into_iter().map(Into::into).collect(),
            },
            call_type: "function".to_string(),
        }
    }

    /// Merges tool call deltas by ID, concatenating argument fragments.
    ///
    /// Used when processing streaming LLM responses where tool calls arrive incrementally.
    /// Argument fragments are concatenated for matching IDs.
    #[must_use]
    pub fn merge_deltas(mut accumulated: Vec<Self>, deltas: &[Self]) -> Vec<Self> {
        for delta in deltas {
            if let Some(existing) = accumulated.iter_mut().find(|tc| tc.id == delta.id) {
                // Merge arguments by concatenating fragments
                for arg in &delta.function.arguments {
                    if let Some(last_arg) = existing.function.arguments.last_mut() {
                        // Append to the last argument (streaming sends fragments)
                        last_arg.push_str(arg);
                    } else {
                        // No existing arguments, add as new
                        existing.function.arguments.push(arg.clone());
                    }
                }
            } else {
                // New tool call, add it
                accumulated.push(delta.clone());
            }
        }

        accumulated
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_tool_approval_variants() {
        let approved = ToolApproval::Approved;
        let denied = ToolApproval::Denied("Invalid request".to_string());
        let quit = ToolApproval::Quit;

        assert_eq!(approved, ToolApproval::Approved);
        assert_eq!(denied, ToolApproval::Denied("Invalid request".to_string()));
        assert_eq!(quit, ToolApproval::Quit);
    }

    #[test]
    fn test_tool_approval_serialization() {
        let approved = ToolApproval::Approved;
        let json = serde_json::to_string(&approved).expect("Failed to serialize");
        let deserialized: ToolApproval =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(approved, deserialized);

        let denied = ToolApproval::Denied("Reason".to_string());
        let json = serde_json::to_string(&denied).expect("Failed to serialize");
        let deserialized: ToolApproval =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(denied, deserialized);
    }

    #[test]
    fn test_property_creation() {
        let prop = Property {
            prop_type: "string".to_string(),
            description: "The user's name".to_string(),
        };

        assert_eq!(prop.prop_type, "string");
        assert_eq!(prop.description, "The user's name");
    }

    #[test]
    fn test_property_serialization() {
        let prop = Property {
            prop_type: "number".to_string(),
            description: "Age in years".to_string(),
        };

        let json = serde_json::to_value(&prop).expect("Failed to serialize");
        assert_eq!(json["type"], "number");
        assert_eq!(json["description"], "Age in years");

        let deserialized: Property = serde_json::from_value(json).expect("Failed to deserialize");
        assert_eq!(prop, deserialized);
    }

    #[test]
    fn test_parameters_creation() {
        let mut properties = HashMap::new();
        properties.insert(
            "location".to_string(),
            Property {
                prop_type: "string".to_string(),
                description: "City name".to_string(),
            },
        );

        let params = Parameters {
            param_type: "object".to_string(),
            properties,
            required: vec!["location".to_string()],
        };

        assert_eq!(params.param_type, "object");
        assert_eq!(params.properties.len(), 1);
        assert_eq!(params.required, vec!["location"]);
    }

    #[test]
    fn test_parameters_serialization() {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            Property {
                prop_type: "string".to_string(),
                description: "Name".to_string(),
            },
        );

        let params = Parameters {
            param_type: "object".to_string(),
            properties,
            required: vec!["name".to_string()],
        };

        let json = serde_json::to_value(&params).expect("Failed to serialize");
        assert_eq!(json["type"], "object");
        assert!(json["properties"].is_object());
        assert!(json["required"].is_array());

        let deserialized: Parameters = serde_json::from_value(json).expect("Failed to deserialize");
        assert_eq!(params, deserialized);
    }

    #[test]
    fn test_function_creation() {
        let func = Function {
            name: "get_weather".to_string(),
            description: "Get weather for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        };

        assert_eq!(func.name, "get_weather");
        assert_eq!(func.description, "Get weather for a location");
        assert!(func.parameters.is_object());
    }

    #[test]
    fn test_function_serialization() {
        let func = Function {
            name: "calculate".to_string(),
            description: "Perform calculation".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        };

        let json = serde_json::to_value(&func).expect("Failed to serialize");
        assert_eq!(json["name"], "calculate");
        assert_eq!(json["description"], "Perform calculation");

        let deserialized: Function = serde_json::from_value(json).expect("Failed to deserialize");
        assert_eq!(func, deserialized);
    }

    #[test]
    fn test_tool_builder() {
        let tool = Tool::builder()
            .function(Function {
                name: "test_func".to_string(),
                description: "A test function".to_string(),
                parameters: serde_json::json!({}),
            })
            .build();

        assert_eq!(tool.r#type, "function");
        assert_eq!(tool.function.name, "test_func");
    }

    #[test]
    fn test_tool_builder_with_custom_type() {
        let tool = Tool::builder()
            .r#type("custom".to_string())
            .function(Function {
                name: "custom_func".to_string(),
                description: "Custom function".to_string(),
                parameters: serde_json::json!({}),
            })
            .build();

        assert_eq!(tool.r#type, "custom");
        assert_eq!(tool.function.name, "custom_func");
    }

    #[test]
    fn test_tool_serialization() {
        let tool = Tool::builder()
            .function(Function {
                name: "test".to_string(),
                description: "Test".to_string(),
                parameters: serde_json::json!({}),
            })
            .build();

        let json = serde_json::to_value(&tool).expect("Failed to serialize");
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "test");

        let deserialized: Tool = serde_json::from_value(json).expect("Failed to deserialize");
        assert_eq!(tool, deserialized);
    }

    #[test]
    fn test_function_call_creation() {
        let call = FunctionCall {
            name: "my_function".to_string(),
            arguments: vec!["arg1".to_string(), "arg2".to_string()],
        };

        assert_eq!(call.name, "my_function");
        assert_eq!(call.arguments.len(), 2);
        assert_eq!(call.arguments[0], "arg1");
    }

    #[test]
    fn test_tool_call_new() {
        let call = ToolCall::new("get_weather", vec!["NYC".to_string()]);

        assert!(!call.id.is_empty());
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(call.function.arguments, vec!["NYC"]);
        assert_eq!(call.call_type, "function");
    }

    #[test]
    fn test_tool_call_new_with_array_literal() {
        let call = ToolCall::new("test_func", [r#"{"key": "value"}"#]);

        assert_eq!(call.function.name, "test_func");
        assert_eq!(call.function.arguments.len(), 1);
        assert_eq!(call.function.arguments[0], r#"{"key": "value"}"#);
    }

    #[test]
    fn test_tool_call_new_empty_args() {
        let call = ToolCall::new("no_args_func", Vec::<String>::new());

        assert_eq!(call.function.name, "no_args_func");
        assert!(call.function.arguments.is_empty());
    }

    #[test]
    fn test_tool_call_new_multiple_args() {
        let call = ToolCall::new(
            "multi_arg_func",
            vec!["arg1".to_string(), "arg2".to_string(), "arg3".to_string()],
        );

        assert_eq!(call.function.arguments.len(), 3);
        assert_eq!(call.function.arguments[0], "arg1");
        assert_eq!(call.function.arguments[1], "arg2");
        assert_eq!(call.function.arguments[2], "arg3");
    }

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall::new("test_function", vec!["test_arg".to_string()]);

        let json = serde_json::to_value(&call).expect("Failed to serialize");
        assert_eq!(json["function"]["name"], "test_function");
        assert_eq!(json["call_type"], "function");

        let deserialized: ToolCall = serde_json::from_value(json).expect("Failed to deserialize");
        assert_eq!(call.function.name, deserialized.function.name);
        assert_eq!(call.function.arguments, deserialized.function.arguments);
    }

    #[test]
    fn test_tool_call_unique_ids() {
        let call1 = ToolCall::new("func", Vec::<String>::new());
        let call2 = ToolCall::new("func", Vec::<String>::new());

        assert_ne!(call1.id, call2.id);
    }

    #[test]
    fn test_tool_call_delta_merging() {
        // Simulate streaming chunks with tool call deltas that need to be merged
        // This tests the delta merging logic used in streaming responses
        let deltas = vec![
            // First delta: tool call with ID and partial arguments
            ToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "test_function".to_string(),
                    arguments: vec![r#"{"param1": ""#.to_string()],
                },
            },
            // Second delta: same ID, more argument fragments
            ToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "test_function".to_string(),
                    arguments: vec![r#"hello", "param2": "#.to_string()],
                },
            },
            // Third delta: final argument fragment
            ToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "test_function".to_string(),
                    arguments: vec![r"123}".to_string()],
                },
            },
        ];

        // Use the actual production merge function
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for delta in &deltas {
            tool_calls = ToolCall::merge_deltas(tool_calls, std::slice::from_ref(delta));
        }

        // Verify the final merged result
        assert_eq!(tool_calls.len(), 1);

        let merged = &tool_calls[0];
        assert_eq!(merged.id, "call_123");
        assert_eq!(merged.function.name, "test_function");
        assert_eq!(merged.function.arguments.len(), 1);
        assert_eq!(
            merged.function.arguments[0],
            r#"{"param1": "hello", "param2": 123}"#
        );

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&merged.function.arguments[0])
            .expect("Merged arguments should be valid JSON");
        assert_eq!(parsed["param1"], "hello");
        assert_eq!(parsed["param2"], 123);
    }

    #[test]
    fn test_multiple_tool_call_delta_merging() {
        // Test merging multiple different tool calls in the same stream
        let deltas = vec![
            // First tool call starts
            ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "func1".to_string(),
                    arguments: vec![r#"{"a":"#.to_string()],
                },
            },
            // Second tool call starts
            ToolCall {
                id: "call_2".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "func2".to_string(),
                    arguments: vec![r#"{"b":"#.to_string()],
                },
            },
            // First tool call continues
            ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "func1".to_string(),
                    arguments: vec![r"1}".to_string()],
                },
            },
            // Second tool call continues
            ToolCall {
                id: "call_2".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "func2".to_string(),
                    arguments: vec![r"2}".to_string()],
                },
            },
        ];

        // Use the actual production merge function
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for delta in &deltas {
            tool_calls = ToolCall::merge_deltas(tool_calls, std::slice::from_ref(delta));
        }

        // Verify both tool calls were properly merged
        assert_eq!(tool_calls.len(), 2);

        let call1 = &tool_calls[0];
        assert_eq!(call1.id, "call_1");
        assert_eq!(call1.function.name, "func1");
        assert_eq!(call1.function.arguments[0], r#"{"a":1}"#);

        let call2 = &tool_calls[1];
        assert_eq!(call2.id, "call_2");
        assert_eq!(call2.function.name, "func2");
        assert_eq!(call2.function.arguments[0], r#"{"b":2}"#);

        // Verify both are valid JSON
        serde_json::from_str::<serde_json::Value>(&call1.function.arguments[0])
            .expect("First call should be valid JSON");
        serde_json::from_str::<serde_json::Value>(&call2.function.arguments[0])
            .expect("Second call should be valid JSON");
    }
}

#[cfg(test)]
mod proptests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn fuzz_tool_call_deserialization(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            // Should not panic on arbitrary bytes
            let _ = serde_json::from_slice::<ToolCall>(&data);
        }

        #[test]
        fn fuzz_function_call_with_arbitrary_args(
            name in ".*",
            args in prop::collection::vec(".*", 0..10),
        ) {
            let call = FunctionCall {
                name,
                arguments: args,
            };

            // Should serialize and deserialize
            let json = serde_json::to_string(&call).unwrap();
            let parsed: FunctionCall = serde_json::from_str(&json).unwrap();
            assert_eq!(call.name, parsed.name);
            assert_eq!(call.arguments, parsed.arguments);
        }

        #[test]
        fn fuzz_tool_call_new_with_special_chars(
            func_name in r"[a-zA-Z0-9_\-\.]{1,50}",
            args in prop::collection::vec(r"[\\x00-\\x7F]*", 0..5),
        ) {
            let call = ToolCall::new(func_name.clone(), args.clone());

            assert_eq!(call.function.name, func_name);
            assert_eq!(call.function.arguments, args);
            assert_eq!(call.call_type, "function");
            assert!(!call.id.is_empty());
        }

        #[test]
        fn fuzz_tool_deserialization(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            // Should not panic on arbitrary bytes
            let _ = serde_json::from_slice::<Tool>(&data);
        }

        #[test]
        fn fuzz_function_with_arbitrary_json_params(
            name in ".*",
            description in ".*",
        ) {
            // Create various JSON parameter structures
            let params_variants = vec![
                serde_json::json!({}),
                serde_json::json!({"type": "object"}),
                serde_json::json!({"type": "object", "properties": {}, "required": []}),
                serde_json::json!(null),
                serde_json::json!([]),
                serde_json::json!("string"),
            ];

            for params in params_variants {
                let func = Function {
                    name: name.clone(),
                    description: description.clone(),
                    parameters: params.clone(),
                };

                // Should serialize and deserialize
                let json = serde_json::to_string(&func).unwrap();
                let parsed: Function = serde_json::from_str(&json).unwrap();
                assert_eq!(func.name, parsed.name);
                assert_eq!(func.description, parsed.description);
            }
        }

        #[test]
        fn fuzz_parameters_with_arbitrary_properties(
            num_props in 0usize..10,
        ) {
            let mut properties = HashMap::new();

            for i in 0..num_props {
                properties.insert(
                    format!("prop_{i}"),
                    Property {
                        prop_type: format!("type_{}", i % 3),
                        description: format!("desc_{i}"),
                    },
                );
            }

            let params = Parameters {
                param_type: "object".to_string(),
                properties: properties.clone(),
                required: (0..num_props).map(|i| format!("prop_{i}")).collect(),
            };

            // Should serialize and deserialize
            let json = serde_json::to_string(&params).unwrap();
            let parsed: Parameters = serde_json::from_str(&json).unwrap();
            assert_eq!(params.param_type, parsed.param_type);
            assert_eq!(params.properties.len(), parsed.properties.len());
            assert_eq!(params.required, parsed.required);
        }

        #[test]
        fn fuzz_tool_approval_serialization(
            approval_type in 0usize..3,
            reason in ".*",
        ) {
            let approval = match approval_type {
                0 => ToolApproval::Approved,
                1 => ToolApproval::Denied(reason),
                _ => ToolApproval::Quit,
            };

            // Should serialize and deserialize
            let json = serde_json::to_string(&approval).unwrap();
            let parsed: ToolApproval = serde_json::from_str(&json).unwrap();
            assert_eq!(approval, parsed);
        }

        #[test]
        fn fuzz_tool_call_with_malformed_json_args(
            func_name in ".*",
            num_args in 0usize..10,
        ) {
            // Generate various potentially malformed JSON strings
            let malformed_jsons = [
                "{",
                "}",
                "[",
                "]",
                "null",
                "undefined",
                r#"{"incomplete": "#,
                r#"{"key": "value"}"#,
                "",
                "   ",
            ];

            let args: Vec<String> = (0..num_args)
                .map(|i| malformed_jsons[i % malformed_jsons.len()].to_string())
                .collect();

            let call = ToolCall::new(func_name.clone(), args.clone());

            // Should create the call even with malformed JSON args
            assert_eq!(call.function.name, func_name);
            assert_eq!(call.function.arguments, args);

            // Should serialize and deserialize the ToolCall itself
            let json = serde_json::to_string(&call).unwrap();
            let parsed: ToolCall = serde_json::from_str(&json).unwrap();
            assert_eq!(call.function.arguments, parsed.function.arguments);
        }
    }
}
