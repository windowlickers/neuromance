//! Tokenizer proxy support for secure credential handling in tools.
//!
//! This module provides utilities for tool implementations that need to make
//! authenticated HTTP requests through a tokenizer proxy. The proxy intercepts
//! requests with sealed tokens and injects real credentials before forwarding
//! to the target API.
//!
//! # Overview
//!
//! When building tools that need to access external APIs (GitHub, Slack, etc.),
//! you often need authentication. Rather than giving tools direct access to
//! real API keys, you can use a tokenizer proxy:
//!
//! ```text
//! Tool → ProxyAwareClient → Tokenizer Proxy → External API
//!              ↑                    ↓
//!         Sealed Token        Real API Key
//! ```
//!
//! # Example
//!
//! ```no_run
//! use neuromance_tools::proxy::{ProxyAwareClient, ToolProxyConfig};
//! use secrecy::SecretString;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = ToolProxyConfig {
//!     proxy_url: "http://tokenizer.internal:8080".to_string(),
//!     token_header: "X-Tokenizer-Token".to_string(),
//!     sealed_token: SecretString::new("sealed.abc123xyz".to_string().into()),
//!     target_host_header: Some("X-Target-Host".to_string()),
//!     connect_timeout: None,  // Uses default (30s)
//!     timeout: None,          // Uses default (60s)
//! };
//!
//! let client = ProxyAwareClient::new(config)?;
//!
//! // All requests route through the proxy, which injects real credentials
//! let response = client.get("https://api.github.com/user").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Security Benefits
//!
//! - Tools never see real API credentials
//! - Proxy can audit all API calls
//! - Sealed tokens can be scoped, rotated, or revoked
//! - Real credentials stay within the trusted proxy boundary

mod client;

pub use client::{ProxyAwareClient, ToolProxyConfig};
