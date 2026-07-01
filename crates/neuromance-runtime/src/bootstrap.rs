//! One-time startup setup for tools that keep their own credential state.
//!
//! The agent pod has no persistent storage, so any tool that caches auth in a
//! config file (rather than reading it from the environment on each call) must
//! be logged in at container start. Each `[[bootstrap]]` entry names a command
//! to run; if it carries a `token_env`, that env var's value is fed to the
//! command on stdin (so a sealed token never lands in argv). The runtime has no
//! per-tool knowledge — the operator bakes the full argv into the config.
//!
//! Bootstrap runs in whichever process hosts the tool executor, because a
//! credential written to one container's filesystem is invisible to another: in
//! sandbox deployments the sandbox process bootstraps (that is where tools run),
//! and in single-process deployments the orchestrator does. See the call sites
//! in `main`.

use std::process::Stdio;

use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::config::BootstrapCommand;

/// Run best-effort tool bootstrapping for each configured entry.
///
/// Never fails the process: a tool that can't be set up just isn't available,
/// which the agent will discover and report, exactly as it would for any other
/// tool error.
pub async fn run(commands: &[BootstrapCommand]) {
    for cmd in commands {
        run_command(cmd).await;
    }
}

/// Run a single bootstrap command. When `token_env` is set, its value is fed on
/// stdin and the token never appears in argv. Idempotent commands (those that
/// detect existing credentials and exit zero) are safe to re-run across pod
/// restarts.
async fn run_command(cmd: &BootstrapCommand) {
    let token = match &cmd.token_env {
        Some(env) => match std::env::var(env) {
            Ok(token) if !token.trim().is_empty() => Some(token),
            Ok(_) => {
                warn!(
                    bootstrap = %cmd.name,
                    "{env} is set but empty; skipping bootstrap"
                );
                return;
            }
            Err(_) => {
                debug!(
                    bootstrap = %cmd.name,
                    "skipping bootstrap; set {env} to enable it"
                );
                return;
            }
        },
        None => None,
    };

    let mut child = match Command::new(&cmd.command)
        .args(&cmd.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            warn!(
                bootstrap = %cmd.name,
                command = %cmd.command,
                error = %e,
                "could not spawn bootstrap command (is it on PATH?)"
            );
            return;
        }
    };

    // Close the child's stdin: write the token first when present, then drop
    // the write end so the child sees EOF. This must happen even without a
    // token — a command that reads stdin (e.g. an auth CLI prompting for input)
    // would otherwise block forever and wedge startup.
    if let Some(mut stdin) = child.stdin.take() {
        if let Some(token) = token
            && let Err(e) = stdin
                .write_all(format!("{}\n", token.trim()).as_bytes())
                .await
        {
            warn!(bootstrap = %cmd.name, error = %e, "could not write token to stdin");
            return;
        }
        drop(stdin);
    }

    match child.wait_with_output().await {
        Ok(out) if out.status.success() => {
            info!(bootstrap = %cmd.name, "bootstrap command succeeded");
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            warn!(
                bootstrap = %cmd.name,
                status = ?out.status.code(),
                stderr = %stderr.trim(),
                "bootstrap command failed"
            );
        }
        Err(e) => warn!(bootstrap = %cmd.name, error = %e, "waiting on bootstrap command failed"),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_token_is_not_placed_in_argv() {
        let cmd = BootstrapCommand {
            name: "forgejo".to_string(),
            command: "fj".to_string(),
            args: vec![
                "--host".to_string(),
                "git.windowlicke.rs".to_string(),
                "auth".to_string(),
                "add-tokenizer".to_string(),
            ],
            token_env: Some("FORGEJO_TOKEN".to_string()),
        };
        // The token is fed on stdin; only the env var *name* may appear, never
        // the value, and the args carry no credential material.
        assert!(!cmd.args.iter().any(|a| a.contains("FORGEJO_TOKEN")));
        assert!(!cmd.args.iter().any(|a| a.contains("Tokenizer ")));
    }

    #[tokio::test]
    async fn test_command_reading_stdin_without_token_env_does_not_hang() {
        // `cat` reads stdin until EOF. If stdin is left open when no token_env
        // is set, it blocks forever; closing stdin unconditionally lets it exit.
        let cmd = BootstrapCommand {
            name: "reads-stdin".to_string(),
            command: "cat".to_string(),
            args: vec![],
            token_env: None,
        };
        tokio::time::timeout(std::time::Duration::from_secs(5), run_command(&cmd))
            .await
            .expect("bootstrap without token_env must not block on stdin");
    }

    #[test]
    fn test_command_without_token_env_is_well_formed() {
        let cmd = BootstrapCommand {
            name: "noop".to_string(),
            command: "true".to_string(),
            args: vec![],
            token_env: None,
        };
        assert!(cmd.token_env.is_none());
        assert!(cmd.args.is_empty());
    }
}
