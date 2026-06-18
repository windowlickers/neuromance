//! Orchestrator-side gRPC client for the sandbox tool service.

use tonic::transport::{Channel, Endpoint};

use super::MAX_MESSAGE_SIZE;
use super::proto::sandbox_tool_service_client::SandboxToolServiceClient;
use super::proto::{
    CloseSessionRequest, ExecuteToolRequest, ExecuteToolResponse, ListToolsRequest, ToolDefinition,
};
use crate::error::RuntimeError;

/// A cloneable handle to the sandbox tool service.
///
/// The channel connects lazily, so constructing a client never fails on a
/// not-yet-ready sandbox; connection errors surface on the first RPC.
#[derive(Clone)]
pub struct SandboxClient {
    inner: SandboxToolServiceClient<Channel>,
}

impl SandboxClient {
    /// Build a client for the sandbox at `endpoint` (e.g. `http://127.0.0.1:50051`).
    ///
    /// # Errors
    /// Returns [`RuntimeError::Config`] if `endpoint` is not a valid URI.
    pub fn connect(endpoint: &str) -> Result<Self, RuntimeError> {
        let channel = Endpoint::from_shared(endpoint.to_string())
            .map_err(|e| {
                RuntimeError::Config(format!("invalid sandbox endpoint '{endpoint}': {e}"))
            })?
            .connect_lazy();
        let inner = SandboxToolServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);
        Ok(Self { inner })
    }

    /// List the tools the sandbox hosts.
    ///
    /// # Errors
    /// Returns the gRPC [`Status`](tonic::Status) on transport or server error.
    pub async fn list_tools(&self) -> Result<Vec<ToolDefinition>, tonic::Status> {
        let mut client = self.inner.clone();
        let response = client.list_tools(ListToolsRequest {}).await?;
        Ok(response.into_inner().tools)
    }

    /// Execute a tool in the sandbox.
    ///
    /// # Errors
    /// Returns the gRPC [`Status`](tonic::Status) on transport or server error.
    /// A tool that runs but fails is reported via [`ExecuteToolResponse::is_error`],
    /// not as an error here.
    pub async fn execute_tool(
        &self,
        name: String,
        arguments_json: String,
        session_id: String,
    ) -> Result<ExecuteToolResponse, tonic::Status> {
        let mut client = self.inner.clone();
        let response = client
            .execute_tool(ExecuteToolRequest {
                name,
                arguments_json,
                session_id,
            })
            .await?;
        Ok(response.into_inner())
    }

    /// Release a stateful interpreter session.
    ///
    /// # Errors
    /// Returns the gRPC [`Status`](tonic::Status) on transport or server error.
    pub async fn close_session(&self, session_id: String) -> Result<(), tonic::Status> {
        let mut client = self.inner.clone();
        client
            .close_session(CloseSessionRequest { session_id })
            .await?;
        Ok(())
    }
}
