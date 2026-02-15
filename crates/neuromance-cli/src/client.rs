//! Unix socket client for communicating with the daemon.

use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};
use fs2::FileExt;
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
    /// This method handles race conditions by:
    /// 1. Checking for an existing daemon process via PID file
    /// 2. Using a lock file to prevent concurrent spawns
    /// 3. Validating that the PID is actually running before attempting spawn
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails after spawn attempts.
    pub async fn connect() -> Result<Self> {
        let socket_path = Self::socket_path()?;
        let pid_file = Self::pid_file()?;
        let lock_file = Self::lock_file()?;

        // Try to connect to existing daemon
        if let Ok(stream) = UnixStream::connect(&socket_path).await {
            return Ok(Self::from_stream(stream));
        }

        // Check if daemon process exists via PID file
        if let Some(pid) = Self::read_pid(&pid_file)? {
            if Self::is_process_running(pid) {
                // Daemon is running but socket not ready yet, wait for it
                for _ in 0..50 {
                    if let Ok(stream) = UnixStream::connect(&socket_path).await {
                        return Ok(Self::from_stream(stream));
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                anyhow::bail!("Daemon process exists (PID {pid}) but socket unavailable");
            }
            // Stale PID file, clean it up
            let _ = std::fs::remove_file(&pid_file);
        }

        // Acquire lock to prevent concurrent spawns
        let lock = Self::acquire_spawn_lock(&lock_file)?;

        // Double-check after acquiring lock (another process may have spawned)
        if let Ok(stream) = UnixStream::connect(&socket_path).await {
            drop(lock); // Release lock
            return Ok(Self::from_stream(stream));
        }

        // Spawn daemon
        Self::spawn_daemon()?;

        // Wait for socket to appear (10 second timeout with backoff)
        let mut delay_ms = 50;
        for attempt in 0..20 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;

            if let Ok(stream) = UnixStream::connect(&socket_path).await {
                drop(lock); // Release lock
                return Ok(Self::from_stream(stream));
            }

            // Exponential backoff up to 500ms
            if attempt < 5 {
                delay_ms = (delay_ms * 2).min(500);
            }
        }

        drop(lock); // Release lock
        anyhow::bail!("Failed to connect to daemon after spawn")
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

    /// Returns the data directory path.
    fn data_dir() -> Result<PathBuf> {
        dirs::data_local_dir()
            .context("Failed to determine data directory")
            .map(|dir| dir.join("neuromance"))
    }

    /// Returns the socket path.
    fn socket_path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("neuromance.sock"))
    }

    /// Returns the PID file path.
    fn pid_file() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("neuromance.pid"))
    }

    /// Returns the lock file path.
    fn lock_file() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("neuromance.lock"))
    }

    /// Reads the daemon PID from the PID file.
    ///
    /// Returns `None` if the file doesn't exist or is invalid.
    fn read_pid(pid_file: &PathBuf) -> Result<Option<u32>> {
        if !pid_file.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(pid_file)
            .context("Failed to read PID file")?;

        let pid = content.trim().parse::<u32>().ok();
        Ok(pid)
    }

    /// Checks if a process with the given PID is running.
    ///
    /// Uses platform-specific methods to verify process existence.
    #[cfg(unix)]
    fn is_process_running(pid: u32) -> bool {
        // Check if /proc/<pid> exists on Linux
        // This avoids unsafe code while being reliable on Linux
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }

    #[cfg(not(unix))]
    fn is_process_running(_pid: u32) -> bool {
        // Conservative fallback: assume process exists
        true
    }

    /// Acquires an exclusive lock for daemon spawning.
    ///
    /// This prevents multiple clients from spawning daemons simultaneously.
    ///
    /// # Errors
    ///
    /// Returns an error if lock file creation or locking fails.
    fn acquire_spawn_lock(lock_file: &PathBuf) -> Result<File> {
        // Ensure parent directory exists
        if let Some(parent) = lock_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_file)
            .context("Failed to open lock file")?;

        // Try to acquire exclusive lock (blocking with timeout)
        file.try_lock_exclusive()
            .context("Failed to acquire spawn lock - another client may be spawning daemon")?;

        Ok(file)
    }

    /// Spawns the daemon in the background.
    ///
    /// # Errors
    ///
    /// Returns an error if the daemon executable cannot be found or spawned.
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
