//! Process lifecycle: signal handling and shutdown coordination.

use tokio::signal::unix::{SignalKind, signal};
use tokio_util::sync::CancellationToken;
use tracing::info;

/// Spawn a task that listens for SIGTERM and SIGINT and cancels `token`
/// on the first received signal.
///
/// # Errors
/// Returns the underlying `io::Error` if signal handlers cannot be installed.
pub fn install_shutdown_handler(token: CancellationToken) -> std::io::Result<()> {
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;
    tokio::spawn(async move {
        tokio::select! {
            _ = sigterm.recv() => info!("SIGTERM received; beginning shutdown"),
            _ = sigint.recv() => info!("SIGINT received; beginning shutdown"),
        }
        token.cancel();
    });
    Ok(())
}
