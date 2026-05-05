//! Tool approval policies.
//!
//! `auto` mode auto-approves every tool call (in addition to per-tool
//! `is_auto_approved`). `async` mode delegates to an approval webhook —
//! POSTs the tool call and waits for an `{approved, reason?}` response,
//! denying on timeout or any failure.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::warn;

use neuromance_common::tools::{ToolApproval, ToolCall};

use crate::error::RuntimeError;

/// Owned, `'static` future type returned by [`WebhookApprover::approve`].
///
/// Boxing makes the type free of input lifetimes so it satisfies the
/// `Fut: Future + Send + 'static` bound on
/// [`neuromance::Core::with_tool_approval_callback`].
pub type ApprovalFuture = Pin<Box<dyn Future<Output = ToolApproval> + Send>>;

#[derive(Debug, Serialize)]
struct WebhookRequest<'a> {
    agent_id: &'a str,
    tool_call_id: &'a str,
    tool_name: &'a str,
    arguments: &'a str,
}

#[derive(Debug, Deserialize)]
struct WebhookResponse {
    approved: bool,
    #[serde(default)]
    reason: Option<String>,
}

/// Posts each non-auto-approved tool call to a webhook and awaits a decision.
///
/// Cloning is cheap — the inner reqwest client and `Arc<str>` fields are
/// shared. The struct is used as the `tool_approval_callback` on
/// [`neuromance::Core`] via [`Self::approve`].
#[derive(Clone)]
pub struct WebhookApprover {
    client: reqwest::Client,
    agent_id: Arc<str>,
    webhook_url: Arc<str>,
}

impl WebhookApprover {
    /// Construct an approver. `timeout` bounds each webhook call; on timeout
    /// the call is treated as a denial.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Approval`] if the HTTP client cannot be built.
    pub fn new(
        agent_id: impl Into<Arc<str>>,
        webhook_url: impl Into<Arc<str>>,
        timeout: Duration,
    ) -> Result<Self, RuntimeError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| RuntimeError::Approval(format!("build webhook client: {e}")))?;
        Ok(Self {
            client,
            agent_id: agent_id.into(),
            webhook_url: webhook_url.into(),
        })
    }

    /// Returns a boxed `'static` future that performs the approval check for
    /// `tool_call`. Suitable for use as the body of a
    /// [`neuromance::Core`] approval callback.
    #[must_use]
    pub fn approve(&self, tool_call: &ToolCall) -> ApprovalFuture {
        let client = self.client.clone();
        let agent_id = Arc::clone(&self.agent_id);
        let webhook_url = Arc::clone(&self.webhook_url);
        let tool_call_id = tool_call.id.clone();
        let tool_name = tool_call.function.name.clone();
        let arguments = tool_call.function.arguments_json().to_string();

        Box::pin(async move {
            let req = WebhookRequest {
                agent_id: &agent_id,
                tool_call_id: &tool_call_id,
                tool_name: &tool_name,
                arguments: &arguments,
            };

            match client.post(&*webhook_url).json(&req).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        warn!(%status, tool=%tool_name, "approval webhook returned non-success");
                        return ToolApproval::Denied(format!("approval webhook returned {status}"));
                    }
                    match resp.json::<WebhookResponse>().await {
                        Ok(parsed) if parsed.approved => ToolApproval::Approved,
                        Ok(parsed) => ToolApproval::Denied(
                            parsed
                                .reason
                                .unwrap_or_else(|| "denied by webhook".to_string()),
                        ),
                        Err(e) => {
                            warn!(error=%e, tool=%tool_name, "approval response parse failed");
                            ToolApproval::Denied(format!("approval response parse failed: {e}"))
                        }
                    }
                }
                Err(e) => {
                    warn!(error=%e, tool=%tool_name, "approval webhook call failed");
                    ToolApproval::Denied(format!("approval webhook unreachable: {e}"))
                }
            }
        })
    }
}
