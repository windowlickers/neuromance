//! Tool approval policies.
//!
//! `auto` mode auto-approves every tool call (in addition to per-tool
//! `is_auto_approved`). `async` mode delegates to an approval webhook —
//! POSTs the tool call and waits for an `{approved, reason?}` response,
//! denying on timeout or any failure.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use metrics::{counter, histogram};
use serde::{Deserialize, Serialize};
use tracing::{Instrument, debug, info, info_span, warn};

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
            .redirect(reqwest::redirect::Policy::none())
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

        let span = info_span!(
            "approval_webhook",
            tool = %tool_name,
            call_id = %tool_call_id,
        );
        Box::pin(
            async move {
                let started = Instant::now();
                let req = WebhookRequest {
                    agent_id: &agent_id,
                    tool_call_id: &tool_call_id,
                    tool_name: &tool_name,
                    arguments: &arguments,
                };

                let outcome = match client.post(&*webhook_url).json(&req).send().await {
                    Ok(resp) => {
                        let status = resp.status();
                        let duration_ms =
                            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                        if status.is_success() {
                            match resp.json::<WebhookResponse>().await {
                                Ok(parsed) if parsed.approved => {
                                    let duration_ms = u64::try_from(started.elapsed().as_millis())
                                        .unwrap_or(u64::MAX);
                                    info!(duration_ms, "approval granted");
                                    ("approved", ToolApproval::Approved)
                                }
                                Ok(parsed) => {
                                    let duration_ms = u64::try_from(started.elapsed().as_millis())
                                        .unwrap_or(u64::MAX);
                                    let reason = parsed
                                        .reason
                                        .unwrap_or_else(|| "denied by webhook".to_string());
                                    info!(duration_ms, reason = %reason, "approval denied");
                                    ("denied", ToolApproval::Denied(reason))
                                }
                                Err(e) => {
                                    let duration_ms = u64::try_from(started.elapsed().as_millis())
                                        .unwrap_or(u64::MAX);
                                    // Don't surface `e` to the caller: reqwest's parse
                                    // errors include the URL, which may carry an auth
                                    // token in its query string.
                                    debug!(error = %e, "approval response parse failed");
                                    warn!(duration_ms, "approval response parse failed");
                                    (
                                        "failed",
                                        ToolApproval::Denied(
                                            "approval response parse failed".to_string(),
                                        ),
                                    )
                                }
                            }
                        } else {
                            warn!(
                                %status,
                                duration_ms,
                                "approval webhook returned non-success",
                            );
                            (
                                "failed",
                                ToolApproval::Denied(format!("approval webhook returned {status}")),
                            )
                        }
                    }
                    Err(e) => {
                        let duration_ms =
                            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                        // Same reasoning: `e.to_string()` from reqwest typically
                        // contains the full request URL.
                        debug!(error = %e, "approval webhook call failed");
                        let status_hint = e
                            .status()
                            .map_or_else(|| "unreachable".to_string(), |s| format!("status={s}"));
                        warn!(
                            status = %status_hint,
                            duration_ms,
                            "approval webhook call failed",
                        );
                        (
                            "failed",
                            ToolApproval::Denied(format!(
                                "approval webhook call failed: {status_hint}"
                            )),
                        )
                    }
                };
                let elapsed = started.elapsed().as_secs_f64();
                histogram!("neuromance_approval_duration_seconds").record(elapsed);
                counter!("neuromance_approvals_total", "outcome" => outcome.0).increment(1);
                outcome.1
            }
            .instrument(span),
        )
    }
}
