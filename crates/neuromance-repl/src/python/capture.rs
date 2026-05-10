//! Stdout/stderr capture via Python's `io.StringIO`.

use pyo3::prelude::*;

use crate::ReplError;
use crate::error::PyResultExt;

/// Captured stream state for stdout/stderr redirection.
pub(super) struct CapturedStreams<'py> {
    stdout_capture: Bound<'py, PyAny>,
    stderr_capture: Bound<'py, PyAny>,
    old_stdout: Bound<'py, PyAny>,
    old_stderr: Bound<'py, PyAny>,
}

/// Redirect `sys.stdout` and `sys.stderr` to `StringIO`
/// captures, returning the old streams.
pub(super) fn redirect_streams(py: Python<'_>) -> Result<CapturedStreams<'_>, ReplError> {
    let io_module = py.import("io").at("import io")?;
    let string_io = io_module.getattr("StringIO").at("getattr io.StringIO")?;

    let stdout_capture = string_io.call0().at("call io.StringIO() for stdout")?;
    let stderr_capture = string_io.call0().at("call io.StringIO() for stderr")?;

    let sys_module = py.import("sys").at("import sys")?;
    let old_stdout = sys_module.getattr("stdout").at("getattr sys.stdout")?;
    let old_stderr = sys_module.getattr("stderr").at("getattr sys.stderr")?;

    sys_module
        .setattr("stdout", &stdout_capture)
        .at("setattr sys.stdout")?;
    sys_module
        .setattr("stderr", &stderr_capture)
        .at("setattr sys.stderr")?;

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
    pub(super) fn restore(self, py: Python<'_>) -> Result<(String, String), ReplError> {
        let sys_module = py.import("sys").at("import sys (restore)")?;
        sys_module
            .setattr("stdout", &self.old_stdout)
            .at("setattr sys.stdout (restore)")?;
        sys_module
            .setattr("stderr", &self.old_stderr)
            .at("setattr sys.stderr (restore)")?;

        let stdout = self
            .stdout_capture
            .call_method0("getvalue")
            .at("call stdout_capture.getvalue")?
            .extract::<String>()
            .at("extract stdout_capture.getvalue -> String")?;

        let stderr = self
            .stderr_capture
            .call_method0("getvalue")
            .at("call stderr_capture.getvalue")?
            .extract::<String>()
            .at("extract stderr_capture.getvalue -> String")?;

        Ok((stdout, stderr))
    }
}
