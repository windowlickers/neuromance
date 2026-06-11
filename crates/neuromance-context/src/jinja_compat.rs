//! Jinja2 compatibility layer for minijinja.
//!
//! This module adds Python/Jinja2 string methods and functions that are commonly used
//! in chat templates but not natively supported by minijinja.
//!
//! ## Supported String Methods
//!
//! - `strip()` / `strip(chars)` - Remove leading/trailing whitespace or specified characters
//! - `lstrip()` / `lstrip(chars)` - Remove leading whitespace or specified characters
//! - `rstrip()` / `rstrip(chars)` - Remove trailing whitespace or specified characters
//! - `split(sep)` - Split string by separator
//! - `endswith(suffix)` - Check if string ends with suffix
//! - `startswith(prefix)` - Check if string starts with prefix
//! - `replace(old, new)` - Replace occurrences of old with new
//! - `upper()` - Convert to uppercase
//! - `lower()` - Convert to lowercase
//! - `items()` - Get dictionary items as list of [key, value] pairs
//!
//! ## Supported Functions
//!
//! - `namespace(**kwargs)` - Create a mutable namespace object
//! - `raise_exception(msg)` - Raise a template error
//!
//! ## Supported Filters
//!
//! - `tojson` / `tojson(ensure_ascii=False)` - Convert to JSON string

use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Environment, Error, ErrorKind, State, Value};

/// Configure a minijinja Environment with Jinja2 compatibility features.
pub fn configure_environment(env: &mut Environment<'_>) {
    // Set up unknown method callback to handle string methods
    env.set_unknown_method_callback(handle_unknown_method);

    // Add namespace function for mutable variables (use minijinja's built-in)
    // This creates a special object that allows `{% set ns.var = value %}` assignments
    env.add_function("namespace", minijinja::functions::namespace);

    // Add raise_exception function
    env.add_function("raise_exception", raise_exception_fn);

    // Add dict function (creates a dictionary)
    env.add_function("dict", dict_fn);

    // Add tojson filter with kwargs support
    env.add_filter("tojson", tojson_filter);
}

/// Handle unknown method calls on values (implements Python string methods).
fn handle_unknown_method(
    _state: &State<'_, '_>,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    match value.kind() {
        ValueKind::String => handle_string_method(value, method, args),
        ValueKind::Map => handle_map_method(value, method, args),
        ValueKind::Seq => handle_seq_method(value, method, args),
        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!(
                "object of type '{}' has no method named '{}'",
                value.kind(),
                method
            ),
        )),
    }
}

/// Handle string method calls.
fn handle_string_method(value: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    let s = value.as_str().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            "expected string value for string method",
        )
    })?;

    match method {
        "strip" => {
            let result = if args.is_empty() {
                s.trim().to_string()
            } else {
                let chars = args[0].as_str().unwrap_or("");
                s.trim_matches(|c| chars.contains(c)).to_string()
            };
            Ok(Value::from(result))
        }

        "lstrip" => {
            let result = if args.is_empty() {
                s.trim_start().to_string()
            } else {
                let chars = args[0].as_str().unwrap_or("");
                s.trim_start_matches(|c| chars.contains(c)).to_string()
            };
            Ok(Value::from(result))
        }

        "rstrip" => {
            let result = if args.is_empty() {
                s.trim_end().to_string()
            } else {
                let chars = args[0].as_str().unwrap_or("");
                s.trim_end_matches(|c| chars.contains(c)).to_string()
            };
            Ok(Value::from(result))
        }

        "split" => {
            let sep = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                Error::new(ErrorKind::MissingArgument, "split requires a separator")
            })?;

            let parts: Vec<Value> = s.split(sep).map(Value::from).collect();
            Ok(Value::from(parts))
        }

        "endswith" => {
            let suffix = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                Error::new(ErrorKind::MissingArgument, "endswith requires a suffix")
            })?;
            Ok(Value::from(s.ends_with(suffix)))
        }

        "startswith" => {
            let prefix = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                Error::new(ErrorKind::MissingArgument, "startswith requires a prefix")
            })?;
            Ok(Value::from(s.starts_with(prefix)))
        }

        "replace" => {
            if args.len() < 2 {
                return Err(Error::new(
                    ErrorKind::MissingArgument,
                    "replace requires old and new arguments",
                ));
            }
            let old = args[0].as_str().unwrap_or("");
            let new = args[1].as_str().unwrap_or("");
            Ok(Value::from(s.replace(old, new)))
        }

        "upper" => Ok(Value::from(s.to_uppercase())),

        "lower" => Ok(Value::from(s.to_lowercase())),

        "title" => {
            // Capitalize first letter of each word
            let result = s
                .split_whitespace()
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => {
                            first.to_uppercase().to_string()
                                + chars.as_str().to_lowercase().as_str()
                        }
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            Ok(Value::from(result))
        }

        "find" => {
            let needle = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                Error::new(ErrorKind::MissingArgument, "find requires a substring")
            })?;
            let pos = s.find(needle).map(|p| p as i64).unwrap_or(-1);
            Ok(Value::from(pos))
        }

        "count" => {
            let needle = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                Error::new(ErrorKind::MissingArgument, "count requires a substring")
            })?;
            let count = s.matches(needle).count();
            Ok(Value::from(count as i64))
        }

        "join" => {
            // This is actually used when a string is the separator: ", ".join(items)
            // But in Jinja2, join is typically a filter, not a method
            // We'll handle this case anyway
            let items = args.first().ok_or_else(|| {
                Error::new(ErrorKind::MissingArgument, "join requires an iterable")
            })?;

            match items.try_iter() {
                Ok(iter) => {
                    let parts: Vec<String> = iter
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    Ok(Value::from(parts.join(s)))
                }
                Err(_) => Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "join argument must be iterable",
                )),
            }
        }

        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("string has no method named '{}'", method),
        )),
    }
}

/// Handle map/dict method calls.
fn handle_map_method(value: &Value, method: &str, _args: &[Value]) -> Result<Value, Error> {
    match method {
        "items" => {
            // Convert map to list of [key, value] pairs
            match value.try_iter() {
                Ok(keys) => {
                    let items: Vec<Value> = keys
                        .filter_map(|k| {
                            let v = value.get_item(&k).ok()?;
                            Some(Value::from(vec![k, v]))
                        })
                        .collect();
                    Ok(Value::from(items))
                }
                Err(_) => Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "cannot iterate over map",
                )),
            }
        }

        "keys" => match value.try_iter() {
            Ok(keys) => {
                let key_list: Vec<Value> = keys.collect();
                Ok(Value::from(key_list))
            }
            Err(_) => Err(Error::new(
                ErrorKind::InvalidOperation,
                "cannot get keys from map",
            )),
        },

        "values" => match value.try_iter() {
            Ok(keys) => {
                let values: Vec<Value> = keys.filter_map(|k| value.get_item(&k).ok()).collect();
                Ok(Value::from(values))
            }
            Err(_) => Err(Error::new(
                ErrorKind::InvalidOperation,
                "cannot get values from map",
            )),
        },

        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("map has no method named '{}'", method),
        )),
    }
}

/// Handle sequence method calls.
fn handle_seq_method(_value: &Value, method: &str, _args: &[Value]) -> Result<Value, Error> {
    match method {
        "append" | "extend" | "pop" | "insert" => {
            // These mutating methods don't make sense in templates
            Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("sequence method '{}' is not supported in templates", method),
            ))
        }

        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("sequence has no method named '{}'", method),
        )),
    }
}

// Note: We use minijinja's built-in namespace() function which creates a special
// mutable object that supports `{% set ns.var = value %}` syntax in templates.

/// Raise a template exception.
fn raise_exception_fn(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// Create a dictionary from keyword arguments.
fn dict_fn(kwargs: Kwargs) -> Result<Value, Error> {
    let mut map = std::collections::BTreeMap::new();

    for key in kwargs.args() {
        if let Ok(value) = kwargs.get::<Value>(key) {
            map.insert(key.to_string(), value);
        }
    }

    Ok(Value::from_iter(map))
}

/// Convert a value to JSON string.
///
/// Supports `tojson(ensure_ascii=False)` kwarg for compatibility.
fn tojson_filter(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    // serde_json emits UTF-8, so ensure_ascii=false is already our behavior and is
    // a silent no-op. ensure_ascii=true (escape non-ASCII) is not supported; warn
    // rather than silently ignore it so template authors aren't misled.
    if let Ok(true) = kwargs.get::<bool>("ensure_ascii") {
        tracing::warn!("tojson(ensure_ascii=true) is not supported; emitting UTF-8 unescaped");
    }

    serde_json::to_string(&value).map_err(|e| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("Failed to serialize to JSON: {}", e),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_env() -> Environment<'static> {
        let mut env = Environment::new();
        configure_environment(&mut env);
        env
    }

    #[test]
    fn test_strip() {
        let mut env = create_env();
        env.add_template("test", "{{ '  hello  '.strip() }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_strip_chars() {
        let mut env = create_env();
        env.add_template("test", "{{ 'xxhelloxx'.strip('x') }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_lstrip() {
        let mut env = create_env();
        env.add_template("test", "{{ '  hello  '.lstrip() }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "hello  ");
    }

    #[test]
    fn test_rstrip() {
        let mut env = create_env();
        env.add_template("test", "{{ '  hello  '.rstrip() }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "  hello");
    }

    #[test]
    fn test_split() {
        let mut env = create_env();
        env.add_template(
            "test",
            "{% for p in 'a,b,c'.split(',') %}{{ p }}-{% endfor %}",
        )
        .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "a-b-c-");
    }

    #[test]
    fn test_endswith() {
        let mut env = create_env();
        env.add_template("test", "{{ 'hello.txt'.endswith('.txt') }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "true");
    }

    #[test]
    fn test_startswith() {
        let mut env = create_env();
        env.add_template("test", "{{ 'hello world'.startswith('hello') }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "true");
    }

    #[test]
    fn test_replace() {
        let mut env = create_env();
        env.add_template("test", "{{ 'hello world'.replace('world', 'rust') }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "hello rust");
    }

    #[test]
    fn test_upper_lower() {
        let mut env = create_env();
        env.add_template("test", "{{ 'Hello'.upper() }}-{{ 'Hello'.lower() }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "HELLO-hello");
    }

    #[test]
    fn test_namespace() {
        let mut env = create_env();
        // Test namespace with initial value and mutation
        env.add_template(
            "test",
            "{%- set ns = namespace(count=0) -%}{%- set ns.count = ns.count + 1 -%}{{ ns.count }}",
        )
        .unwrap();
        let result = env.get_template("test").unwrap().render(()).unwrap();
        assert_eq!(result, "1");
    }

    #[test]
    fn test_tojson() {
        let mut env = create_env();
        env.add_template("test", "{{ data | tojson }}").unwrap();
        let ctx = serde_json::json!({"data": {"key": "value"}});
        let result = env.get_template("test").unwrap().render(&ctx).unwrap();
        assert_eq!(result, r#"{"key":"value"}"#);
    }

    #[test]
    fn test_dict_items() {
        let mut env = create_env();
        env.add_template(
            "test",
            "{%- for k, v in data.items() %}{{ k }}={{ v }};{% endfor -%}",
        )
        .unwrap();
        let ctx = serde_json::json!({"data": {"a": 1, "b": 2}});
        let result = env.get_template("test").unwrap().render(&ctx).unwrap();
        // Note: order may vary, but should contain both
        assert!(result.contains("a=1"));
        assert!(result.contains("b=2"));
    }

    #[test]
    fn test_raise_exception() {
        let mut env = create_env();
        env.add_template("test", "{{ raise_exception('test error') }}")
            .unwrap();
        let result = env.get_template("test").unwrap().render(());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test error"));
    }
}
