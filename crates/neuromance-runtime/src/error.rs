use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("config: {0}")]
    Config(String),

    #[error("environment variable '{0}' is not set")]
    MissingEnv(String),

    #[error("approval webhook: {0}")]
    Approval(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
