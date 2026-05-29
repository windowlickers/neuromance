use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("config: {0}")]
    Config(String),

    #[error("environment variable '{0}' is not set")]
    MissingEnv(String),

    #[error("read proxy token file '{}': {source}", path.display())]
    ProxyTokenRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("approval webhook: {0}")]
    Approval(String),

    #[error("metrics: {0}")]
    Metrics(String),

    #[error("telemetry: {0}")]
    Telemetry(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    ToolBuild(#[from] neuromance_tools::ToolError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
