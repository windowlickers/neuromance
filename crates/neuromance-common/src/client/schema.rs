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
/// Construct with [`OutputSchema::new`] or [`OutputSchema::from_value`]. Deserialization routes
/// through the same validation, so an `OutputSchema` that exists is always one the providers
/// accept. The fields are public for reading; building one with a struct literal is the only way
/// to skip the checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawOutputSchema")]
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
    /// The schema name is not one `OpenAI` accepts.
    #[error(
        "output schema name `{name}` must be 1-{MAX_NAME_LEN} characters of \
         `a-z`, `A-Z`, `0-9`, `_` or `-`"
    )]
    InvalidName {
        /// The rejected name.
        name: String,
    },
    /// The schema nests deeper than [`MAX_SCHEMA_DEPTH`].
    #[error("output schema nests deeper than {MAX_SCHEMA_DEPTH} levels at `{path}`")]
    TooDeep {
        /// Dotted path to the subschema that exceeded the limit, rooted at `$`.
        path: String,
    },
}

/// Longest schema name `OpenAI` accepts in `response_format.json_schema.name`.
pub const MAX_NAME_LEN: usize = 64;

/// Deepest subschema nesting [`OutputSchema::new`] will validate.
///
/// The recursive walk needs a bound of its own: today every untrusted schema arrives through
/// `serde_json`'s 128-level parse limit, but [`OutputSchema::new`] is public and a caller can
/// build a `Value` programmatically. 64 levels is far beyond any real structured-output schema.
pub const MAX_SCHEMA_DEPTH: usize = 64;

/// The wire shape of an [`OutputSchema`], before validation.
#[derive(Deserialize)]
struct RawOutputSchema {
    name: String,
    schema: Value,
}

impl TryFrom<RawOutputSchema> for OutputSchema {
    type Error = SchemaError;

    fn try_from(raw: RawOutputSchema) -> Result<Self, Self::Error> {
        Self::new(raw.name, raw.schema)
    }
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
    /// Returns [`SchemaError::InvalidName`] if `name` is empty, longer than [`MAX_NAME_LEN`], or
    /// holds a character outside `[A-Za-z0-9_-]`; [`SchemaError::NonObjectRoot`] if the root is
    /// not `type: "object"`; [`SchemaError::OpenObject`] if any object subschema omits
    /// `additionalProperties: false`; and [`SchemaError::TooDeep`] past [`MAX_SCHEMA_DEPTH`]
    /// levels of nesting.
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
        let name = name.into();
        validate_name(&name)?;
        if schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(SchemaError::NonObjectRoot {
                found: describe_type(&schema),
            });
        }
        validate_node(&schema, "$", 0)?;

        Ok(Self {
            name,
            schema,
            strict: true,
        })
    }

    /// Validates a JSON Schema, naming it from its own `title` key.
    ///
    /// The name defaults to `output` when the schema declares no string `title`. This is the
    /// entry point for schemas that arrive as loose JSON — a request body or a config file — where
    /// the name is not carried separately.
    ///
    /// # Arguments
    ///
    /// * `schema` - The JSON Schema; its root must be an object schema
    ///
    /// # Errors
    ///
    /// The same errors as [`OutputSchema::new`], including [`SchemaError::InvalidName`] when the
    /// schema's own `title` is not a name the providers accept.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromance_common::client::OutputSchema;
    /// use serde_json::json;
    ///
    /// let schema = OutputSchema::from_value(json!({
    ///     "title": "verdict",
    ///     "type": "object",
    ///     "properties": {"ok": {"type": "boolean"}},
    ///     "additionalProperties": false,
    /// }))?;
    /// assert_eq!(schema.name, "verdict");
    /// # Ok::<(), neuromance_common::client::SchemaError>(())
    /// ```
    pub fn from_value(schema: Value) -> Result<Self, SchemaError> {
        let name = schema
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("output")
            .to_owned();
        Self::new(name, schema)
    }
}

/// Rejects names `OpenAI` will bounce with a `400` after the task has already consumed a worker.
fn validate_name(name: &str) -> Result<(), SchemaError> {
    let acceptable = (1..=MAX_NAME_LEN).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');

    if acceptable {
        Ok(())
    } else {
        Err(SchemaError::InvalidName {
            name: name.to_owned(),
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

fn validate_node(node: &Value, path: &str, depth: usize) -> Result<(), SchemaError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaError::TooDeep {
            path: path.to_owned(),
        });
    }

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

    validate_named_children(node, path, depth)?;
    validate_indexed_children(node, path, depth)?;

    if let Some(items) = map.get("items") {
        validate_node(items, &format!("{path}[]"), depth + 1)?;
    }

    Ok(())
}

/// Recurses into subschemas keyed by name: `properties`, `$defs`, `definitions`.
fn validate_named_children(node: &Value, path: &str, depth: usize) -> Result<(), SchemaError> {
    for key in ["properties", "$defs", "definitions"] {
        let Some(children) = node.get(key).and_then(Value::as_object) else {
            continue;
        };
        for (name, child) in children {
            validate_node(child, &format!("{path}.{name}"), depth + 1)?;
        }
    }
    Ok(())
}

/// Recurses into subschemas held in arrays: `anyOf`, `allOf`, `oneOf`, `prefixItems`.
fn validate_indexed_children(node: &Value, path: &str, depth: usize) -> Result<(), SchemaError> {
    for key in ["anyOf", "allOf", "oneOf", "prefixItems"] {
        let Some(children) = node.get(key).and_then(Value::as_array) else {
            continue;
        };
        for (index, child) in children.iter().enumerate() {
            validate_node(child, &format!("{path}.{key}[{index}]"), depth + 1)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use proptest::prelude::*;

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
            SchemaError::NonObjectRoot { .. }
            | SchemaError::InvalidName { .. }
            | SchemaError::TooDeep { .. } => None,
        }
    }

    /// Builds a closed object schema nested `depth` levels below the root.
    fn nested_to_depth(depth: usize) -> Value {
        (0..depth).fold(closed_object(&json!({})), |inner, _| {
            closed_object(&json!({"inner": inner}))
        })
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

    #[test]
    fn test_name_with_spaces_is_rejected() {
        let error = OutputSchema::new("my verdict", closed_object(&json!({}))).unwrap_err();

        assert!(matches!(error, SchemaError::InvalidName { .. }));
        assert!(error.to_string().contains("my verdict"));
    }

    #[test]
    fn test_empty_name_is_rejected() {
        let error = OutputSchema::new("", closed_object(&json!({}))).unwrap_err();

        assert!(matches!(error, SchemaError::InvalidName { .. }));
    }

    #[test]
    fn test_name_longer_than_the_provider_limit_is_rejected() {
        let error =
            OutputSchema::new("n".repeat(MAX_NAME_LEN + 1), closed_object(&json!({}))).unwrap_err();

        assert!(matches!(error, SchemaError::InvalidName { .. }));
    }

    #[test]
    fn test_name_at_the_provider_limit_is_accepted() {
        let name = "n".repeat(MAX_NAME_LEN);

        let schema = OutputSchema::new(name.clone(), closed_object(&json!({}))).unwrap();

        assert_eq!(schema.name, name);
    }

    #[test]
    fn test_schema_nested_past_the_depth_limit_is_rejected() {
        let error = OutputSchema::new("deep", nested_to_depth(MAX_SCHEMA_DEPTH + 1)).unwrap_err();

        assert!(matches!(error, SchemaError::TooDeep { .. }));
    }

    #[test]
    fn test_schema_at_the_depth_limit_is_accepted() {
        assert!(OutputSchema::new("deep", nested_to_depth(MAX_SCHEMA_DEPTH)).is_ok());
    }

    #[test]
    fn test_from_value_takes_its_name_from_the_title() {
        let mut schema = closed_object(&json!({}));
        schema["title"] = json!("verdict");

        assert_eq!(OutputSchema::from_value(schema).unwrap().name, "verdict");
    }

    #[test]
    fn test_from_value_without_a_title_defaults_the_name() {
        let schema = OutputSchema::from_value(closed_object(&json!({}))).unwrap();

        assert_eq!(schema.name, "output");
    }

    #[test]
    fn test_deserialize_rejects_a_schema_new_would_reject() {
        let wire = json!({
            "name": "open",
            "schema": {"type": "object", "properties": {}},
            "strict": true,
        });

        let error = serde_json::from_value::<OutputSchema>(wire).unwrap_err();

        assert!(error.to_string().contains("additionalProperties"));
    }

    #[test]
    fn test_deserialize_cannot_forge_a_non_strict_schema() {
        let wire = json!({
            "name": "verdict",
            "schema": closed_object(&json!({})),
            "strict": false,
        });

        let schema = serde_json::from_value::<OutputSchema>(wire).unwrap();

        assert!(schema.strict);
    }

    /// The container key each recursion path in `validate_node` follows, and how to nest a
    /// subschema under it.
    fn nest_under(key: &str, child: &Value) -> Value {
        match key {
            "items" => json!({"type": "array", "items": child}),
            "properties" | "$defs" | "definitions" => json!({key: {"child": child}}),
            _ => json!({key: [child]}),
        }
    }

    /// Every container the validator claims to recurse into must actually catch an open object.
    /// A missed path is the failure mode that matters: the schema ships, and the provider 400s.
    #[test]
    fn test_every_recursion_path_catches_an_open_object() {
        let open = json!({"type": "object", "properties": {}});

        for key in [
            "properties",
            "$defs",
            "definitions",
            "items",
            "anyOf",
            "allOf",
            "oneOf",
            "prefixItems",
        ] {
            let mut schema = closed_object(&json!({"field": nest_under(key, &open)}));
            schema["additionalProperties"] = json!(false);

            let error = OutputSchema::new("probe", schema).unwrap_err();

            assert!(
                open_path(&error).is_some(),
                "`{key}` did not reach the open object: {error}"
            );
        }
    }

    proptest! {
        /// Any JSON object survives a serialize/parse round trip through the structured-output
        /// parser — it narrows the type without altering the value.
        #[test]
        fn parse_structured_response_round_trips_objects(
            keys in prop::collection::vec("[a-z]{1,8}", 0..8),
            values in prop::collection::vec(any::<i64>(), 0..8),
        ) {
            let object: serde_json::Map<String, Value> = keys
                .into_iter()
                .zip(values)
                .map(|(k, v)| (k, Value::from(v)))
                .collect();
            let encoded = serde_json::to_string(&object).unwrap();

            let parsed = parse_structured_response(&encoded).unwrap();

            prop_assert_eq!(parsed, Value::Object(object));
        }

        /// Surrounding whitespace is not part of the value.
        #[test]
        fn parse_structured_response_ignores_surrounding_whitespace(
            pad in "[ \t\r\n]{0,8}",
        ) {
            let parsed = parse_structured_response(&format!("{pad}{{\"ok\":true}}{pad}")).unwrap();

            prop_assert_eq!(parsed, json!({"ok": true}));
        }

        /// The parser exists to reject the non-objects `from_str::<Value>` would accept.
        #[test]
        fn parse_structured_response_rejects_non_objects(
            scalar in prop_oneof![
                any::<i64>().prop_map(|n| n.to_string()),
                any::<bool>().prop_map(|b| b.to_string()),
                Just("null".to_owned()),
                Just("[1, 2]".to_owned()),
                "\"[a-z]{0,8}\"",
            ],
        ) {
            prop_assert!(parse_structured_response(&scalar).is_err());
        }

        /// Validation is a pure inspection: an accepted schema comes back byte-identical, and
        /// re-validating it accepts again.
        #[test]
        fn validation_preserves_the_schema_and_is_idempotent(
            field in "[a-z]{1,8}",
            depth in 0usize..8,
        ) {
            let source = closed_object(&json!({field: nested_to_depth(depth)}));

            let schema = OutputSchema::new("probe", source.clone()).unwrap();

            prop_assert_eq!(&schema.schema, &source);
            prop_assert!(OutputSchema::new("probe", schema.schema).is_ok());
        }
    }

    #[test]
    fn test_serialize_round_trips_through_validation() {
        let original = OutputSchema::new(
            "verdict",
            closed_object(&json!({"ok": {"type": "boolean"}})),
        )
        .unwrap();

        let encoded = serde_json::to_string(&original).unwrap();
        let decoded: OutputSchema = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, original);
    }
}
