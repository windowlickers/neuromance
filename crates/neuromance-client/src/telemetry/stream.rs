//! Keeping a chat span open for the whole life of a stream.
//!
//! `chat_stream` returns before a single token has arrived, so ending the span
//! when the function returns would time the connection setup and nothing else,
//! and would record no usage at all. [`InstrumentedChunkStream`] carries the
//! operation instead: it passes chunks through untouched, folds the terminal
//! state as they go, and closes the span when the underlying stream ends.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;

use neuromance_common::client::{ChatChunk, FinishReason, Usage};

use super::GenAiOp;
use crate::error::ClientError;
use crate::streaming::ChatChunkStream;

/// The outcome of a stream, assembled chunk by chunk.
///
/// This mirrors what `Core` derives while consuming the same stream. The
/// duplication is deliberate: the client must close its own span without
/// waiting for a callback from an orchestrator that may not exist.
#[derive(Debug, Default)]
struct StreamAccumulator {
    response_model: Option<String>,
    response_id: Option<String>,
    finish_reasons: Vec<FinishReason>,
    usage: Option<Usage>,
}

impl StreamAccumulator {
    fn observe(&mut self, chunk: &ChatChunk) {
        if self.response_model.is_none() && !chunk.model.is_empty() {
            self.response_model = Some(chunk.model.clone());
        }
        if self.response_id.is_none() {
            self.response_id.clone_from(&chunk.response_id);
        }
        if let Some(reason) = chunk.finish_reason
            && !self.finish_reasons.contains(&reason)
        {
            self.finish_reasons.push(reason);
        }
        if let Some(ref usage) = chunk.usage {
            match self.usage {
                // Providers split usage across chunks — Anthropic sends input
                // tokens first and output tokens last — so a later report
                // zeroes an earlier field unless the counts are merged.
                Some(ref mut accumulated) => accumulated.merge_max(usage),
                None => self.usage = Some(usage.clone()),
            }
        }
    }
}

/// A passthrough stream that owns the chat span until the stream terminates.
pub struct InstrumentedChunkStream {
    inner: ChatChunkStream,
    /// Taken by whichever terminal branch fires first. The SSE driver can
    /// yield an error and then a `None`, so closing must happen once.
    op: Option<GenAiOp>,
    accumulated: StreamAccumulator,
}

impl InstrumentedChunkStream {
    pub(super) fn new(op: GenAiOp, inner: ChatChunkStream) -> Self {
        Self {
            inner,
            op: Some(op),
            accumulated: StreamAccumulator::default(),
        }
    }
}

impl Stream for InstrumentedChunkStream {
    type Item = Result<ChatChunk, ClientError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Enter the span for the poll so anything the transport logs, and any
        // span it opens, lands under this operation rather than under whoever
        // happens to be driving the stream.
        let polled = {
            let _entered = this.op.as_ref().map(|op| op.span().enter());
            this.inner.as_mut().poll_next(cx)
        };

        match polled {
            Poll::Ready(Some(Ok(chunk))) => {
                this.accumulated.observe(&chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(op) = this.op.take() {
                    op.finish_error(&error);
                }
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                if let Some(op) = this.op.take() {
                    let accumulated = &this.accumulated;
                    op.finish_parts(
                        accumulated.response_model.as_deref(),
                        accumulated.response_id.as_deref(),
                        &accumulated.finish_reasons,
                        accumulated.usage.as_ref(),
                    );
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use futures::StreamExt as _;
    use neuromance_common::client::{ChatRequest, Config};
    use neuromance_common::telemetry::genai;
    use opentelemetry::Value;

    use crate::telemetry::tests::{attribute, exported_spans, exported_spans_observed};

    fn chunk(usage: Option<Usage>, finish_reason: Option<FinishReason>) -> ChatChunk {
        ChatChunk {
            model: "gpt-4o-2024-08-06".to_string(),
            delta_content: Some("hi".to_string()),
            delta_reasoning_content: None,
            delta_role: None,
            delta_tool_calls: None,
            finish_reason,
            usage,
            response_id: Some("resp_123".to_string()),
            created_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        }
    }

    /// Drive a stream to completion on a private tokio runtime.
    ///
    /// Not `futures::executor::block_on`: closing the span runs the simple
    /// span processor, which blocks on the exporter with that same executor,
    /// and nesting it panics.
    fn drain(mut stream: InstrumentedChunkStream) {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build runtime")
            .block_on(async { while stream.next().await.is_some() {} });
    }

    fn instrumented(chunks: Vec<Result<ChatChunk, ClientError>>) -> InstrumentedChunkStream {
        let request = ChatRequest::new(Vec::new()).with_model("gpt-4o");
        let op = GenAiOp::chat(&Config::new("openai", "gpt-4o"), &request);
        InstrumentedChunkStream::new(op, Box::pin(futures::stream::iter(chunks)))
    }

    /// The whole point of the adapter: `chat_stream` returns before any token
    /// arrives, so a span closed at return would time the handshake and record
    /// no usage.
    #[test]
    fn test_span_closes_after_the_last_chunk_not_when_the_stream_is_built() {
        let mut closed_before_draining = usize::MAX;

        let spans = exported_spans_observed(|exporter| {
            let stream = instrumented(vec![
                Ok(chunk(Some(usage(900, 0)), None)),
                Ok(chunk(Some(usage(0, 40)), Some(FinishReason::Stop))),
            ]);
            closed_before_draining = exporter
                .get_finished_spans()
                .expect("in-memory spans")
                .len();

            drain(stream);
        });

        assert_eq!(
            closed_before_draining, 0,
            "no span may close before the stream is drained"
        );
        let span = spans.first().expect("one chat span");
        assert_eq!(
            attribute(span, genai::USAGE_INPUT_TOKENS),
            Some(&Value::I64(900))
        );
        assert_eq!(
            attribute(span, genai::USAGE_OUTPUT_TOKENS),
            Some(&Value::I64(40)),
            "usage split across chunks must be merged, not overwritten"
        );
    }

    #[test]
    fn test_stream_span_records_the_response_identity_and_finish_reason() {
        let spans = exported_spans(|| {
            drain(instrumented(vec![Ok(chunk(
                None,
                Some(FinishReason::ToolCalls),
            ))]));
        });

        let span = spans.first().expect("one chat span");
        assert_eq!(
            attribute(span, genai::RESPONSE_MODEL),
            Some(&Value::from("gpt-4o-2024-08-06"))
        );
        assert_eq!(
            attribute(span, genai::RESPONSE_ID),
            Some(&Value::from("resp_123"))
        );
        let value = attribute(span, genai::RESPONSE_FINISH_REASONS).expect("finish reasons");
        assert_eq!(value.to_string(), "[\"tool_calls\"]");
    }

    #[test]
    fn test_stream_error_closes_the_span_as_a_failure() {
        let spans = exported_spans(|| {
            drain(instrumented(vec![
                Ok(chunk(None, None)),
                Err(ClientError::TimeoutError),
            ]));
        });

        let span = spans.first().expect("one chat span");
        assert_eq!(
            attribute(span, genai::ERROR_TYPE),
            Some(&Value::from("timeout"))
        );
    }

    /// Cancelling a turn drops the stream mid-flight. Without the marker the
    /// span exports with no outcome, which reads as a call that succeeded and
    /// happened to report nothing.
    #[test]
    fn test_dropping_the_stream_early_marks_the_operation_cancelled() {
        let spans = exported_spans(|| {
            let mut stream = instrumented(vec![Ok(chunk(None, None)), Ok(chunk(None, None))]);
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("build runtime")
                .block_on(async {
                    let _first = stream.next().await;
                });
            drop(stream);
        });

        let span = spans.first().expect("one chat span");
        assert_eq!(
            attribute(span, genai::ERROR_TYPE),
            Some(&Value::from("cancelled"))
        );
    }
}
