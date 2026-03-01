//! Stdout/stderr capture via Python's `io.StringIO`.

use pyo3::prelude::*;

use crate::ReplError;

/// Captured stream state for stdout/stderr redirection.
pub(super) struct CapturedStreams<'py> {
    stdout_capture: Bound<'py, PyAny>,
    stderr_capture: Bound<'py, PyAny>,
    old_stdout: Bound<'py, PyAny>,
    old_stderr: Bound<'py, PyAny>,
}

/// Redirect `sys.stdout` and `sys.stderr` to `StringIO`
/// captures, returning the old streams.
pub(super) fn redirect_streams(
    py: Python<'_>,
) -> Result<CapturedStreams<'_>, ReplError> {
    let io_module = py.import("io")?;
    let string_io = io_module.getattr("StringIO")?;

    let stdout_capture = string_io.call0()?;
    let stderr_capture = string_io.call0()?;

    let sys_module = py.import("sys")?;
    let old_stdout = sys_module.getattr("stdout")?;
    let old_stderr = sys_module.getattr("stderr")?;

    sys_module.setattr("stdout", &stdout_capture)?;
    sys_module.setattr("stderr", &stderr_capture)?;

    Ok(CapturedStreams {
        stdout_capture,
        stderr_capture,
        old_stdout,
        old_stderr,
    })
}

impl CapturedStreams<'_> {
    /// Restore `sys.stdout` / `sys.stderr` and return the
    /// captured (stdout, stderr) strings.
    pub(super) fn restore(
        self,
        py: Python<'_>,
    ) -> Result<(String, String), ReplError> {
        let sys_module = py.import("sys")?;
        let _ = sys_module.setattr("stdout", &self.old_stdout);
        let _ = sys_module.setattr("stderr", &self.old_stderr);

        let stdout = self
            .stdout_capture
            .call_method0("getvalue")?
            .extract::<String>()?;

        let stderr = self
            .stderr_capture
            .call_method0("getvalue")?
            .extract::<String>()?;

        Ok((stdout, stderr))
    }
}
