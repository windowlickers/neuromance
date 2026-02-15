//! Unix socket client for communicating with the daemon.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};
use neuromance_common::protocol::{DaemonRequest, DaemonResponse};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Client for communicating with the Neuromance daemon.
pub struct DaemonClient {
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
}

impl DaemonClient {
    /// Connects to the daemon, auto-spawning if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails after spawn attempts.
    pub async fn connect() -> Result<Self> {
        let socket_path = Self::socket_path()?;

        // Try to connect
        match UnixStream::connect(&socket_path).await {
            Ok(stream) => Ok(Self::from_stream(stream)),
            Err(_) => {
                // Daemon not running, spawn it
                Self::spawn_daemon()?;

                // Wait for socket to appear (5 second timeout)
                for _ in 0..50 {
                    if let Ok(stream) = UnixStream::connect(&socket_path).await {
                        return Ok(Self::from_stream(stream));
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }

                anyhow::bail!("Failed to connect to daemon after spawn")
            }
        }
    }

    /// Creates a client from an existing Unix stream.
    fn from_stream(stream: UnixStream) -> Self {
        let (reader, writer) = stream.into_split();
        Self {
            reader: BufReader::new(reader),
            writer,
        }
    }

    /// Sends a request to the daemon.
    ///
    /// # Errors
    ///
    /// Returns an error if sending fails.
    pub async fn send_request(&mut self, request: &DaemonRequest) -> Result<()> {
        let json = serde_json::to_string(request)?;
        self.writer.write_all(json.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;
        Ok(())
    }

    /// Reads a single response from the daemon.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or parsing fails.
    pub async fn read_response(&mut self) -> Result<DaemonResponse> {
        let mut line = String::new();
        self.reader.read_line(&mut line).await?;

        if line.is_empty() {
            anyhow::bail!("Connection closed by daemon");
        }

        let response = serde_json::from_str(&line)?;
        Ok(response)
    }

    /// Reads responses until a completion condition is met.
    ///
    /// Calls the handler for each response. Returns when the handler returns `false`.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or parsing fails.
    pub async fn read_until_complete<F>(&mut self, mut handler: F) -> Result<()>
    where
        F: FnMut(&DaemonResponse) -> bool,
    {
        loop {
            let response = self.read_response().await?;
            if !handler(&response) {
                break;
            }
        }
        Ok(())
    }

    /// Returns the socket path.
    fn socket_path() -> Result<PathBuf> {
        let data_dir = dirs::data_local_dir()
            .context("Failed to determine data directory")?
            .join("neuromance");

        Ok(data_dir.join("neuromance.sock"))
    }

    /// Spawns the daemon in the background.
    fn spawn_daemon() -> Result<()> {
        Command::new("neuromance-daemon")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn daemon. Is neuromance-daemon in PATH?")?;

        Ok(())
    }
}
