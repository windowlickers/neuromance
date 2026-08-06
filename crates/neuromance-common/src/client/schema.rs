//! JSON Schema constraints for structured model output.
//!
//! Every provider we support can enforce a JSON Schema on the model's response natively:
//!
//! - **`OpenAI` Chat Completions** — `response_format.json_schema`
//! - **`OpenAI` Responses** — `text.format`
//! - **Anthropic Messages** — `output_config.format`
//!
//! [`OutputSchema`] is the provider-neutral carrier. It validates the two constraints all three
//! providers share up front, so a malformed schema fails locally with an actionable message
//! instead of surfacing as a provider `400`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// A schema the model's response must conform to.
///
/// Construct with [`OutputSchema::new`], which validates the schema. The fields are public for
/// reading, but building one directly bypasses validation, so prefer the constructor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputSchema {
    /// Name identifying the schema. Sent to `OpenAI`; Anthropic has no field for it.
    pub name: String,
    /// The JSON Schema itself.
    pub schema: Value,
    /// Whether the provider must enforce the schema exactly rather than treat it as a hint.
    ///
    /// Always `true` for schemas built by [`OutputSchema::new`] — the validation it performs is
    /// exactly what `OpenAI` strict mode requires. Anthropic has no equivalent field and enforces
    /// unconditionally.
    pub strict: bool,
}

/// A schema that no provider will accept.
#[derive(Debug, Error)]
pub enum SchemaError {
    /// The root of the schema is not an object schema.
    #[error(
        "output schema root must be a JSON Schema object with `type: \"object\"`, found {found}"
    )]
    NonObjectRoot {
        /// What the root declared instead.
        found: String,
    },
    /// An object schema does not close itself to extra properties.
    #[error(
        "object schema at `{path}` must set `additionalProperties: false` — \
         structured outputs reject open objects"
    )]
    OpenObject {
        /// Dotted path to the offending subschema, rooted at `$`.
        path: String,
    },
}

impl OutputSchema {
    /// Validates a JSON Schema and wraps it for use as a structured-output constraint.
    ///
    /// # Arguments
    ///
    /// * `name` - Identifier for the schema, surfaced to `OpenAI` providers
    /// * `schema` - The JSON Schema; its root must be an object schema
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError::NonObjectRoot`] if the root is not `type: "object"`, or
    /// [`SchemaError::OpenObject`] if any object subschema omits `additionalProperties: false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::client::OutputSchema;
    /// use serde_json::json;
    ///
    /// let schema = OutputSchema::new(
    ///     "verdict",
    ///     json!({
    ///         "type": "object",
    ///         "properties": {"ok": {"type": "boolean"}},
    ///         "required": ["ok"],
    ///         "additionalProperties": false,
    ///     }),
    /// )?;
    /// assert!(schema.strict);
    /// # Ok::<(), neuromance_common::client::SchemaError>(())
    /// ```
    pub fn new(name: impl Into<String>, schema: Value) -> Result<Self, SchemaError> {
        if schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(SchemaError::NonObjectRoot {
                found: describe_type(&schema),
            });
        }
        validate_node(&schema, "$")?;

        Ok(Self {
            name: name.into(),
            schema,
            strict: true,
        })
    }
}

/// Parse a model response that an [`OutputSchema`] constrained.
///
/// Providers enforce the schema themselves, so this only catches what enforcement cannot: a
/// truncated response, or a provider that dropped the constraint. It requires a JSON object,
/// matching the `type: "object"` root every [`OutputSchema`] demands — `from_str::<Value>` would
/// accept a bare `42`.
///
/// # Errors
///
/// Returns the `serde_json` error when `content` is not a JSON object.
///
/// # Examples
///
/// ```
/// use neuromance_common::client::parse_structured_response;
///
/// let parsed = parse_structured_response(r#"{"ok": true}"#)?;
/// assert_eq!(parsed["ok"], true);
/// assert!(parse_structured_response("42").is_err());
/// # Ok::<(), serde_json::Error>(())
/// ```
pub fn parse_structured_response(content: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str::<serde_json::Map<String, Value>>(content.trim()).map(Value::Object)
}

fn describe_type(schema: &Value) -> String {
    schema.get("type").and_then(Value::as_str).map_or_else(
        || "no `type` key".to_owned(),
        |declared| format!("`type: \"{declared}\"`"),
    )
}

fn validate_node(node: &Value, path: &str) -> Result<(), SchemaError> {
    let Some(map) = node.as_object() else {
        return Ok(());
    };

    if map.get("type").and_then(Value::as_str) == Some("object")
        && map.get("additionalProperties") != Some(&Value::Bool(false))
    {
        return Err(SchemaError::OpenObject {
            path: path.to_owned(),
        });
    }

    validate_named_children(node, path)?;
    validate_indexed_children(node, path)?;

    if let Some(items) = map.get("items") {
        validate_node(items, &format!("{path}[]"))?;
    }

    Ok(())
}

/// Recurses into subschemas keyed by name: `properties`, `$defs`, `definitions`.
fn validate_named_children(node: &Value, path: &str) -> Result<(), SchemaError> {
    for key in ["properties", "$defs", "definitions"] {
        let Some(children) = node.get(key).and_then(Value::as_object) else {
            continue;
        };
        for (name, child) in children {
            validate_node(child, &format!("{path}.{name}"))?;
        }
    }
    Ok(())
}

/// Recurses into subschemas held in arrays: `anyOf`, `allOf`, `oneOf`, `prefixItems`.
fn validate_indexed_children(node: &Value, path: &str) -> Result<(), SchemaError> {
    for key in ["anyOf", "allOf", "oneOf", "prefixItems"] {
        let Some(children) = node.get(key).and_then(Value::as_array) else {
            continue;
        };
        for (index, child) in children.iter().enumerate() {
            validate_node(child, &format!("{path}.{key}[{index}]"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use serde_json::json;

    fn closed_object(properties: &Value) -> Value {
        json!({
            "type": "object",
            "properties": properties,
            "additionalProperties": false,
        })
    }

    /// Extracts the path from an [`SchemaError::OpenObject`], or `None` for any other variant.
    fn open_path(error: &SchemaError) -> Option<&str> {
        match error {
            SchemaError::OpenObject { path } => Some(path.as_str()),
            SchemaError::NonObjectRoot { .. } => None,
        }
    }

    #[test]
    fn test_valid_schema_is_strict() {
        let schema = OutputSchema::new(
            "verdict",
            closed_object(&json!({"ok": {"type": "boolean"}})),
        )
        .unwrap();

        assert_eq!(schema.name, "verdict");
        assert!(schema.strict);
    }

    #[test]
    fn test_non_object_root_is_rejected() {
        let error = OutputSchema::new("list", json!({"type": "array"})).unwrap_err();

        assert!(matches!(error, SchemaError::NonObjectRoot { .. }));
        assert!(error.to_string().contains("array"));
    }

    #[test]
    fn test_root_without_type_is_rejected() {
        let error = OutputSchema::new("untyped", json!({"properties": {}})).unwrap_err();

        assert!(matches!(error, SchemaError::NonObjectRoot { .. }));
        assert!(error.to_string().contains("no `type` key"));
    }

    #[test]
    fn test_open_root_is_rejected() {
        let error =
            OutputSchema::new("open", json!({"type": "object", "properties": {}})).unwrap_err();

        assert_eq!(open_path(&error), Some("$"));
    }

    #[test]
    fn test_open_nested_property_names_its_path() {
        let error = OutputSchema::new(
            "nested",
            closed_object(&json!({"inner": {"type": "object", "properties": {}}})),
        )
        .unwrap_err();

        assert_eq!(open_path(&error), Some("$.inner"));
    }

    #[test]
    fn test_open_object_inside_array_items_names_its_path() {
        let error = OutputSchema::new(
            "rows",
            closed_object(&json!({
                "rows": {"type": "array", "items": {"type": "object", "properties": {}}}
            })),
        )
        .unwrap_err();

        assert_eq!(open_path(&error), Some("$.rows[]"));
    }

    #[test]
    fn test_open_object_inside_any_of_names_its_path() {
        let error = OutputSchema::new(
            "choice",
            closed_object(&json!({
                "choice": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {}}]}
            })),
        )
        .unwrap_err();

        assert_eq!(open_path(&error), Some("$.choice.anyOf[1]"));
    }

    #[test]
    fn test_open_object_inside_defs_names_its_path() {
        let mut schema = closed_object(&json!({"ref": {"$ref": "#/$defs/node"}}));
        schema["$defs"] = json!({"node": {"type": "object", "properties": {}}});

        let error = OutputSchema::new("defs", schema).unwrap_err();

        assert_eq!(open_path(&error), Some("$.node"));
    }
}
