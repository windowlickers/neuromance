//! Process management utilities.

use std::process::{Command, Stdio};

/// Checks whether a process with the given PID is still running.
///
/// Sends signal 0 via `kill`, which checks process existence
/// without actually delivering a signal.
#[must_use]
#[cfg(unix)]
pub fn is_process_running(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

#[must_use]
#[cfg(not(unix))]
pub fn is_process_running(_pid: u32) -> bool {
    true
}
