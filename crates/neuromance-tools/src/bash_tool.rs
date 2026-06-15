use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;
use tokio::time::timeout;

use crate::factory::ToolFactory;
use crate::truncate::{TruncatedBy, truncate_tail};
use crate::{ToolError, ToolImplementation, ToolRegistry};
use neuromance_common::tools::{Function, Parameters, Property, Tool};

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
/// Maximum bytes retained from each of stdout / stderr before truncation.
const MAX_STREAM_BYTES: usize = 64 * 1024;
/// Maximum lines retained from each of stdout / stderr before truncation.
const MAX_STREAM_LINES: usize = 2000;

/// Base environment variables always forwarded into the shell subprocess.
/// Everything else — including the runtime's own secrets injected by k8s as env
/// vars (`OPENAI_API_KEY`, the database DSN, `KUBERNETES_*`, projected
/// service-account paths, etc.) — is stripped via `env_clear` so it cannot leak
/// into tool output. Deployments that inject *tool* credentials as env vars
/// (e.g. a tokenizer-proxy token or `GIT_CONFIG_*`) name them explicitly via
/// the bash tool's `env_passthrough` config; see [`BashTool::env_passthrough`].
///
/// Setting [`BashTool::inherit_env`] disables this stripping entirely, letting
/// the shell inherit the full parent environment.
const ENV_ALLOWLIST: &[&str] = &["PATH", "HOME", "LANG", "LC_ALL", "TERM"];

/// Executes a shell command via `sh -c` and returns its exit code, stdout,
/// and stderr.
///
/// Not auto-approved: arbitrary command execution requires explicit approval.
pub struct BashTool {
    /// Additional env var names forwarded to the shell beyond [`ENV_ALLOWLIST`].
    ///
    /// `env_clear` strips the whole environment by default so the runtime's
    /// secrets never reach an agent-run command. Operator-injected *tool*
    /// credentials (a tokenizer-proxy token, `GIT_CONFIG_*`, a CLI's
    /// `*_ENDPOINT`) are the exception: the deployment lists exactly those
    /// names here so the tools the agent shells out to can authenticate, while
    /// the runtime's own provider keys and DSN — which are not in this list —
    /// stay stripped.
    env_passthrough: Vec<String>,
    /// When `true`, the shell inherits the full parent environment: `env_clear`
    /// and `env_passthrough` are both bypassed. Intended for deployments where
    /// secrets are delivered out-of-band (e.g. sealed tokens injected by a
    /// proxy) so the pod's environment holds nothing worth stripping. Defaults
    /// to `false`, preserving the allowlist-only behavior.
    inherit_env: bool,
    /// Timeout applied when a call omits `timeout_ms`. Defaults to
    /// [`DEFAULT_TIMEOUT_MS`].
    default_timeout_ms: u64,
    /// Upper bound a per-call `timeout_ms` is clamped to. Defaults to
    /// [`MAX_TIMEOUT_MS`].
    max_timeout_ms: u64,
}

impl Default for BashTool {
    fn default() -> Self {
        Self {
            env_passthrough: Vec::new(),
            inherit_env: false,
            default_timeout_ms: DEFAULT_TIMEOUT_MS,
            max_timeout_ms: MAX_TIMEOUT_MS,
        }
    }
}

impl BashTool {
    /// Construct a bash tool that forwards `env_passthrough` (in addition to
    /// [`ENV_ALLOWLIST`]) from the runtime's environment into each command,
    /// using the default and maximum timeouts.
    #[must_use]
    pub fn new(env_passthrough: Vec<String>) -> Self {
        Self {
            env_passthrough,
            ..Self::default()
        }
    }
}

#[async_trait]
impl ToolImplementation for BashTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "command".to_string(),
            Property::string("Shell command to execute via `sh -c`."),
        );
        properties.insert(
            "timeout_ms".to_string(),
            Property::number(format!(
                "Optional timeout in milliseconds. Defaults to {}, max {}.",
                self.default_timeout_ms, self.max_timeout_ms
            )),
        );
        properties.insert(
            "cwd".to_string(),
            Property::string(
                "Optional absolute working directory. Defaults to the current directory.",
            ),
        );

        Tool::builder()
            .function(Function {
                name: "bash".to_string(),
                description: "Execute a shell command via `sh -c` and return its exit code, \
                              stdout, and stderr. Each output stream is capped at 64 KiB."
                    .to_string(),
                parameters: Parameters::new(properties, vec!["command".into()]).into(),
            })
            .build()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        let obj = args
            .as_object()
            .ok_or_else(|| ToolError::InvalidArguments("expected object arguments".into()))?;

        let command = obj
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments("missing 'command' parameter".into()))?;

        let timeout_ms = match obj.get("timeout_ms") {
            None | Some(Value::Null) => self.default_timeout_ms,
            Some(v) => v.as_u64().ok_or_else(|| {
                ToolError::InvalidArguments("'timeout_ms' must be a positive integer".into())
            })?,
        };
        let timeout_ms = timeout_ms.clamp(1, self.max_timeout_ms);

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.kill_on_drop(true);
        if !self.inherit_env {
            cmd.env_clear();
            for key in ENV_ALLOWLIST
                .iter()
                .copied()
                .chain(self.env_passthrough.iter().map(String::as_str))
            {
                if let Ok(value) = std::env::var(key) {
                    cmd.env(key, value);
                }
            }
        }

        if let Some(cwd) = obj.get("cwd").and_then(Value::as_str) {
            let cwd = PathBuf::from(cwd);
            if !cwd.is_absolute() {
                return Err(ToolError::InvalidArguments(format!(
                    "'cwd' must be absolute, got: {}",
                    cwd.display()
                )));
            }
            if !cwd.is_dir() {
                return Err(ToolError::InvalidArguments(format!(
                    "'cwd' is not a directory: {}",
                    cwd.display()
                )));
            }
            cmd.current_dir(cwd);
        }

        let child = cmd.spawn().map_err(|e| {
            ToolError::execution(format!("failed to spawn shell for command: {command}: {e}"))
        })?;

        match timeout(Duration::from_millis(timeout_ms), child.wait_with_output()).await {
            Ok(Ok(output)) => Ok(render_output(
                output.status.code().unwrap_or(-1),
                output.stdout,
                output.stderr,
                None,
            )
            .await),
            Ok(Err(e)) => Err(ToolError::execution(format!("error running command: {e}"))),
            Err(_) => Ok(render_output(
                -1,
                Vec::new(),
                Vec::new(),
                Some(format!("[timed out after {timeout_ms}ms]")),
            )
            .await),
        }
    }
}

async fn render_output(
    exit_code: i32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    note: Option<String>,
) -> String {
    let stdout_section = render_stream(stdout).await;
    let stderr_section = render_stream(stderr).await;

    let mut out = format!("exit_code: {exit_code}\n");
    if let Some(note) = note {
        let _ = writeln!(out, "{note}");
    }
    out.push_str("--- stdout ---\n");
    out.push_str(&stdout_section);
    if !stdout_section.is_empty() && !stdout_section.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("--- stderr ---\n");
    out.push_str(&stderr_section);
    out
}

/// Renders one captured stream, keeping the **tail** (where errors and final
/// results live) and spilling the full stream to a temp file when truncated.
async fn render_stream(bytes: Vec<u8>) -> String {
    let lossy = String::from_utf8_lossy(&bytes);
    let truncated = truncate_tail(&lossy, MAX_STREAM_LINES, MAX_STREAM_BYTES);
    drop(lossy);

    let Some(by) = truncated.truncated_by else {
        return truncated.content;
    };

    let mut s = truncated.content;
    if !s.is_empty() && !s.ends_with('\n') {
        s.push('\n');
    }
    let unit = match by {
        TruncatedBy::Lines => "lines",
        TruncatedBy::Bytes => "bytes",
    };
    match spill_to_temp(bytes).await {
        Some(path) => {
            let _ = write!(
                s,
                "[output truncated by {unit}: showing last {} of {} lines; full output: {}]",
                truncated.shown_lines,
                truncated.total_lines,
                path.display()
            );
        }
        None => {
            let _ = write!(
                s,
                "[output truncated by {unit}: showing last {} of {} lines]",
                truncated.shown_lines, truncated.total_lines
            );
        }
    }
    s
}

/// Writes the full stream to a persisted temp file so nothing is lost to
/// truncation. Returns `None` if the spill fails — best-effort, never fatal.
async fn spill_to_temp(bytes: Vec<u8>) -> Option<PathBuf> {
    tokio::task::spawn_blocking(move || {
        let mut file = tempfile::Builder::new()
            .prefix("neuromance-bash-")
            .suffix(".log")
            .tempfile()
            .ok()?;
        std::io::Write::write_all(file.as_file_mut(), &bytes).ok()?;
        let (_, path) = file.keep().ok()?;
        Some(path)
    })
    .await
    .ok()
    .flatten()
}

/// Factory that registers [`BashTool`] under the name `bash`.
///
/// Optional config:
/// - `env_passthrough`: an array of env var names forwarded from the runtime's
///   environment into each shell command on top of [`ENV_ALLOWLIST`].
///   Deployments use it to let agent-run tools see the credentials injected for
///   them (e.g. a tokenizer-proxy token, `GIT_CONFIG_*`) without exposing the
///   runtime's own secrets.
/// - `inherit_env`: a boolean (default `false`). When `true`, the shell inherits
///   the full parent environment and both `env_clear` and `env_passthrough` are
///   bypassed. Use only where the pod's environment holds no secrets worth
///   stripping (e.g. credentials delivered out-of-band by a proxy).
/// - `default_timeout_ms`: timeout applied when a call omits `timeout_ms`
///   (defaults to [`DEFAULT_TIMEOUT_MS`]).
/// - `max_timeout_ms`: upper bound a per-call `timeout_ms` is clamped to
///   (defaults to [`MAX_TIMEOUT_MS`]). Must be >= `default_timeout_ms`.
pub struct BashToolFactory;

impl ToolFactory for BashToolFactory {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn build(&self, config: &Value, registry: &ToolRegistry) -> Result<(), ToolError> {
        let env_passthrough = match config.get("env_passthrough") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| {
                    v.as_str().map(str::to_string).ok_or_else(|| {
                        ToolError::InvalidArguments(
                            "bash 'env_passthrough' entries must be strings".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => {
                return Err(ToolError::InvalidArguments(
                    "bash 'env_passthrough' must be an array of strings".into(),
                ));
            }
        };

        let inherit_env = match config.get("inherit_env") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                return Err(ToolError::InvalidArguments(
                    "bash 'inherit_env' must be a boolean".into(),
                ));
            }
        };

        let default_timeout_ms =
            parse_positive_ms(config, "default_timeout_ms")?.unwrap_or(DEFAULT_TIMEOUT_MS);
        let max_timeout_ms = parse_positive_ms(config, "max_timeout_ms")?.unwrap_or(MAX_TIMEOUT_MS);
        if default_timeout_ms > max_timeout_ms {
            return Err(ToolError::InvalidArguments(format!(
                "bash 'default_timeout_ms' ({default_timeout_ms}) must not exceed \
                 'max_timeout_ms' ({max_timeout_ms})"
            )));
        }

        registry.register(Arc::new(BashTool {
            env_passthrough,
            inherit_env,
            default_timeout_ms,
            max_timeout_ms,
        }));
        Ok(())
    }
}

/// Reads an optional positive-integer millisecond field from a config blob.
///
/// Returns `None` when absent or null. Rejects non-integers, negatives, and
/// zero (a zero timeout would fire immediately).
fn parse_positive_ms(config: &Value, key: &str) -> Result<Option<u64>, ToolError> {
    match config.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(0) | None => Err(ToolError::InvalidArguments(format!(
                "bash '{key}' must be a positive integer"
            ))),
            Some(ms) => Ok(Some(ms)),
        },
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_bash_echo() {
        let tool = BashTool::default();
        let result = tool
            .execute(&json!({ "command": "echo hello" }))
            .await
            .unwrap();
        assert!(result.contains("exit_code: 0"));
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn test_bash_nonzero_exit() {
        let tool = BashTool::default();
        let result = tool.execute(&json!({ "command": "false" })).await.unwrap();
        assert!(result.contains("exit_code: 1"));
    }

    #[tokio::test]
    async fn test_bash_captures_stderr() {
        let tool = BashTool::default();
        let result = tool
            .execute(&json!({ "command": "echo oops 1>&2" }))
            .await
            .unwrap();
        assert!(result.contains("--- stderr ---"));
        assert!(result.contains("oops"));
    }

    #[tokio::test]
    async fn test_bash_timeout() {
        let tool = BashTool::default();
        let result = tool
            .execute(&json!({
                "command": "sleep 5",
                "timeout_ms": 100,
            }))
            .await
            .unwrap();
        assert!(result.contains("exit_code: -1"));
        assert!(result.contains("timed out after 100ms"));
    }

    #[tokio::test]
    async fn test_bash_cwd_must_exist() {
        let tool = BashTool::default();
        let err = tool
            .execute(&json!({
                "command": "pwd",
                "cwd": "/this/path/should/not/exist/anywhere",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a directory"));
    }

    #[tokio::test]
    async fn test_bash_cwd_used() {
        let dir = tempdir().unwrap();
        let tool = BashTool::default();
        let result = tool
            .execute(&json!({
                "command": "pwd",
                "cwd": dir.path().to_str().unwrap(),
            }))
            .await
            .unwrap();
        assert!(result.contains("exit_code: 0"));
    }

    #[tokio::test]
    async fn test_bash_relative_cwd_rejected() {
        let tool = BashTool::default();
        let err = tool
            .execute(&json!({
                "command": "pwd",
                "cwd": "rel/dir",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must be absolute"));
    }

    #[test]
    fn test_is_not_auto_approved() {
        assert!(!BashTool::default().is_auto_approved());
    }

    #[tokio::test]
    async fn test_bash_truncation_keeps_tail() {
        // 5000 lines exceeds MAX_STREAM_LINES (2000); the *last* line must
        // survive (errors live at the end) and the full output is spilled.
        let tool = BashTool::default();
        let result = tool
            .execute(&json!({ "command": "seq 1 5000" }))
            .await
            .unwrap();
        assert!(result.contains("\n5000\n"), "tail line missing:\n{result}");
        assert!(result.contains("output truncated by lines"));
        assert!(result.contains("showing last 2000 of 5000 lines"));
        assert!(result.contains("full output:"));
    }

    #[tokio::test]
    async fn test_bash_spilled_file_has_full_output() {
        let tool = BashTool::default();
        let result = tool
            .execute(&json!({ "command": "seq 1 5000" }))
            .await
            .unwrap();
        let path = result
            .lines()
            .find_map(|l| l.split_once("full output: "))
            .map(|(_, rest)| rest.trim_end_matches(']').to_string())
            .expect("spill path should be present");
        let spilled = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(spilled.lines().count(), 5000);
        assert!(spilled.starts_with("1\n"));
        assert!(spilled.trim_end().ends_with("5000"));
        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_bash_forwards_path_env() {
        // PATH is in ENV_ALLOWLIST and is set in any sane test environment.
        let tool = BashTool::default();
        let result = tool
            .execute(&json!({ "command": "test -n \"$PATH\" && echo PATH_OK" }))
            .await
            .unwrap();
        assert!(
            result.contains("PATH_OK"),
            "PATH should be forwarded into the shell:\n{result}"
        );
    }

    #[tokio::test]
    async fn test_bash_strips_non_allowlisted_env() {
        // `cargo test` always sets `CARGO`; it is not in ENV_ALLOWLIST, so
        // `env_clear` followed by the allowlist forwarding should drop it.
        // If $CARGO is empty inside the shell, env_clear is working.
        let tool = BashTool::default();
        let result = tool
            .execute(&json!({
                "command": "if [ -z \"$CARGO\" ]; then echo CLEAN; else echo LEAKED; fi"
            }))
            .await
            .unwrap();
        assert!(
            result.contains("CLEAN"),
            "non-allowlisted env (CARGO) leaked into shell:\n{result}"
        );
        assert!(!result.contains("LEAKED"));
    }

    #[tokio::test]
    async fn test_bash_env_passthrough_forwards_named_var() {
        // `cargo test` always sets `CARGO`; it is not in ENV_ALLOWLIST, so it is
        // stripped by default (see test above). Naming it in `env_passthrough`
        // forwards it — the mechanism that lets operator-injected tool
        // credentials reach the subprocess.
        let tool = BashTool::new(vec!["CARGO".to_string()]);
        let result = tool
            .execute(&json!({
                "command": "if [ -n \"$CARGO\" ]; then echo FORWARDED; else echo MISSING; fi"
            }))
            .await
            .unwrap();
        assert!(
            result.contains("FORWARDED"),
            "env_passthrough var (CARGO) should reach the shell:\n{result}"
        );
        assert!(!result.contains("MISSING"));
    }

    #[tokio::test]
    async fn test_bash_env_passthrough_does_not_widen_other_vars() {
        // Forwarding one name must not forward unrelated names: PWD is set in the
        // runtime env but absent from both ENV_ALLOWLIST and env_passthrough.
        let tool = BashTool::new(vec!["CARGO".to_string()]);
        let result = tool
            .execute(&json!({
                "command": "if [ -z \"$CARGO_MANIFEST_DIR\" ]; then echo CLEAN; else echo LEAKED; fi"
            }))
            .await
            .unwrap();
        assert!(result.contains("CLEAN"), "unexpected env leaked:\n{result}");
        assert!(!result.contains("LEAKED"));
    }

    #[tokio::test]
    async fn test_bash_inherit_env_forwards_all() {
        // `cargo test` always sets `CARGO`, which is not in ENV_ALLOWLIST and so
        // is stripped by default. With `inherit_env`, the full parent
        // environment is inherited and it reaches the shell unnamed.
        let tool = BashTool {
            inherit_env: true,
            ..BashTool::default()
        };
        let result = tool
            .execute(&json!({
                "command": "if [ -n \"$CARGO\" ]; then echo INHERITED; else echo MISSING; fi"
            }))
            .await
            .unwrap();
        assert!(
            result.contains("INHERITED"),
            "inherit_env should pass the full parent environment to the shell:\n{result}"
        );
        assert!(!result.contains("MISSING"));
    }

    #[test]
    fn test_factory_parses_inherit_env() {
        let registry = ToolRegistry::new();
        BashToolFactory
            .build(&json!({ "inherit_env": true }), &registry)
            .unwrap();
        assert!(registry.contains("bash"));
    }

    #[test]
    fn test_factory_rejects_non_bool_inherit_env() {
        let registry = ToolRegistry::new();
        let err = BashToolFactory
            .build(&json!({ "inherit_env": "yes" }), &registry)
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_factory_registers_bash_with_no_config() {
        let registry = ToolRegistry::new();
        BashToolFactory.build(&Value::Null, &registry).unwrap();
        assert!(registry.contains("bash"));
    }

    #[test]
    fn test_factory_accepts_env_passthrough_array() {
        let registry = ToolRegistry::new();
        BashToolFactory
            .build(
                &json!({ "env_passthrough": ["PLANE_ENDPOINT", "GIT_CONFIG_COUNT"] }),
                &registry,
            )
            .unwrap();
        assert!(registry.contains("bash"));
    }

    #[test]
    fn test_factory_rejects_non_array_env_passthrough() {
        let registry = ToolRegistry::new();
        let err = BashToolFactory
            .build(&json!({ "env_passthrough": "PLANE_ENDPOINT" }), &registry)
            .unwrap_err();
        assert!(err.to_string().contains("must be an array"));
    }

    #[test]
    fn test_factory_rejects_non_string_env_passthrough_entries() {
        let registry = ToolRegistry::new();
        let err = BashToolFactory
            .build(&json!({ "env_passthrough": [42] }), &registry)
            .unwrap_err();
        assert!(err.to_string().contains("must be strings"));
    }

    #[tokio::test]
    async fn test_bash_uses_configured_default_timeout() {
        // No per-call timeout_ms, so the configured default (100ms) applies and
        // a 5s sleep is killed.
        let tool = BashTool {
            default_timeout_ms: 100,
            ..BashTool::default()
        };
        let result = tool
            .execute(&json!({ "command": "sleep 5" }))
            .await
            .unwrap();
        assert!(result.contains("timed out after 100ms"), "{result}");
    }

    #[tokio::test]
    async fn test_bash_clamps_per_call_timeout_to_configured_max() {
        // The call requests 5000ms but the configured max is 100ms, so the
        // command is clamped and killed at 100ms.
        let tool = BashTool {
            max_timeout_ms: 100,
            ..BashTool::default()
        };
        let result = tool
            .execute(&json!({ "command": "sleep 5", "timeout_ms": 5000 }))
            .await
            .unwrap();
        assert!(result.contains("timed out after 100ms"), "{result}");
    }

    #[test]
    fn test_factory_accepts_timeout_overrides() {
        let registry = ToolRegistry::new();
        BashToolFactory
            .build(
                &json!({ "default_timeout_ms": 30_000, "max_timeout_ms": 300_000 }),
                &registry,
            )
            .unwrap();
        assert!(registry.contains("bash"));
    }

    #[test]
    fn test_factory_rejects_default_timeout_exceeding_max() {
        let registry = ToolRegistry::new();
        let err = BashToolFactory
            .build(
                &json!({ "default_timeout_ms": 300_000, "max_timeout_ms": 1_000 }),
                &registry,
            )
            .unwrap_err();
        assert!(err.to_string().contains("must not exceed"), "{err}");
    }

    #[test]
    fn test_factory_rejects_zero_timeout() {
        let registry = ToolRegistry::new();
        let err = BashToolFactory
            .build(&json!({ "default_timeout_ms": 0 }), &registry)
            .unwrap_err();
        assert!(
            err.to_string().contains("must be a positive integer"),
            "{err}"
        );
    }

    #[test]
    fn test_factory_rejects_non_integer_timeout() {
        let registry = ToolRegistry::new();
        let err = BashToolFactory
            .build(&json!({ "max_timeout_ms": "soon" }), &registry)
            .unwrap_err();
        assert!(
            err.to_string().contains("must be a positive integer"),
            "{err}"
        );
    }
}
