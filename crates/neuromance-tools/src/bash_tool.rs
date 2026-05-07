use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;
use tokio::time::timeout;

use crate::factory::ToolFactory;
use crate::{ToolImplementation, ToolRegistry};
use neuromance_common::tools::{Function, Parameters, Property, Tool};

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
/// Maximum bytes captured from each of stdout / stderr.
const MAX_STREAM_BYTES: usize = 64 * 1024;

/// Environment variables forwarded into the shell subprocess. Anything else —
/// including secrets injected by k8s as env vars (`OPENAI_API_KEY`,
/// `KUBERNETES_*`, projected service-account paths, etc.) — is stripped via
/// `env_clear` so it cannot leak into tool output.
const ENV_ALLOWLIST: &[&str] = &["PATH", "HOME", "LANG", "LC_ALL", "TERM"];

/// Executes a shell command via `sh -c` and returns its exit code, stdout,
/// and stderr.
///
/// Not auto-approved: arbitrary command execution requires explicit approval.
pub struct BashTool;

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
                "Optional timeout in milliseconds. Defaults to {DEFAULT_TIMEOUT_MS}, max {MAX_TIMEOUT_MS}."
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

    async fn execute(&self, args: &Value) -> Result<String> {
        let obj = args
            .as_object()
            .ok_or_else(|| anyhow!("expected object arguments"))?;

        let command = obj
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'command' parameter"))?;

        let timeout_ms = match obj.get("timeout_ms") {
            None | Some(Value::Null) => DEFAULT_TIMEOUT_MS,
            Some(v) => v
                .as_u64()
                .ok_or_else(|| anyhow!("'timeout_ms' must be a positive integer"))?,
        };
        let timeout_ms = timeout_ms.clamp(1, MAX_TIMEOUT_MS);

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.kill_on_drop(true);
        cmd.env_clear();
        for key in ENV_ALLOWLIST {
            if let Ok(value) = std::env::var(key) {
                cmd.env(key, value);
            }
        }

        if let Some(cwd) = obj.get("cwd").and_then(Value::as_str) {
            let cwd = PathBuf::from(cwd);
            if !cwd.is_absolute() {
                bail!("'cwd' must be absolute, got: {}", cwd.display());
            }
            if !cwd.is_dir() {
                bail!("'cwd' is not a directory: {}", cwd.display());
            }
            cmd.current_dir(cwd);
        }

        let child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn shell for command: {command}"))?;

        match timeout(Duration::from_millis(timeout_ms), child.wait_with_output()).await {
            Ok(Ok(output)) => Ok(format_output(
                output.status.code().unwrap_or(-1),
                &output.stdout,
                &output.stderr,
                None,
            )),
            Ok(Err(e)) => Err(anyhow!("error running command: {e}")),
            Err(_) => Ok(format_output(
                -1,
                &[],
                &[],
                Some(format!("[timed out after {timeout_ms}ms]")),
            )),
        }
    }
}

fn format_output(exit_code: i32, stdout: &[u8], stderr: &[u8], note: Option<String>) -> String {
    let stdout_str = truncate_stream(stdout);
    let stderr_str = truncate_stream(stderr);

    let mut out = format!("exit_code: {exit_code}\n");
    if let Some(note) = note {
        let _ = writeln!(out, "{note}");
    }
    out.push_str("--- stdout ---\n");
    out.push_str(&stdout_str);
    if !stdout_str.is_empty() && !stdout_str.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("--- stderr ---\n");
    out.push_str(&stderr_str);
    out
}

fn truncate_stream(bytes: &[u8]) -> String {
    if bytes.len() <= MAX_STREAM_BYTES {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    let mut s = String::from_utf8_lossy(&bytes[..MAX_STREAM_BYTES]).into_owned();
    let extra = bytes.len() - MAX_STREAM_BYTES;
    let _ = write!(s, "\n[truncated, {extra} more bytes]");
    s
}

/// Factory that registers [`BashTool`] under the name `bash`.
/// Takes no configuration.
pub struct BashToolFactory;

impl ToolFactory for BashToolFactory {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<()> {
        registry.register(Arc::new(BashTool));
        Ok(())
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
        let tool = BashTool;
        let result = tool
            .execute(&json!({ "command": "echo hello" }))
            .await
            .unwrap();
        assert!(result.contains("exit_code: 0"));
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn test_bash_nonzero_exit() {
        let tool = BashTool;
        let result = tool.execute(&json!({ "command": "false" })).await.unwrap();
        assert!(result.contains("exit_code: 1"));
    }

    #[tokio::test]
    async fn test_bash_captures_stderr() {
        let tool = BashTool;
        let result = tool
            .execute(&json!({ "command": "echo oops 1>&2" }))
            .await
            .unwrap();
        assert!(result.contains("--- stderr ---"));
        assert!(result.contains("oops"));
    }

    #[tokio::test]
    async fn test_bash_timeout() {
        let tool = BashTool;
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
        let tool = BashTool;
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
        let tool = BashTool;
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
        let tool = BashTool;
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
        assert!(!BashTool.is_auto_approved());
    }

    #[tokio::test]
    async fn test_bash_forwards_path_env() {
        // PATH is in ENV_ALLOWLIST and is set in any sane test environment.
        let tool = BashTool;
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
        let tool = BashTool;
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
}
