//! Benchmarks for streaming response processing.
//!
//! These benchmarks measure the performance of deserializing and processing
//! `OpenAI` streaming chunks, which is a hot path in real-time streaming.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::json;

use neuromance_client::openai::ChatCompletionChunk;

/// Create a typical streaming chunk with text content
fn create_content_chunk_json() -> serde_json::Value {
    json!({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1_677_652_288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "Hello, how can I help you today?"
            },
            "finish_reason": null
        }]
    })
}

/// Create a streaming chunk with tool calls (common in agentic workflows)
fn create_tool_call_chunk_json(num_tools: usize) -> serde_json::Value {
    let tool_calls: Vec<_> = (0..num_tools)
        .map(|i| {
            json!({
                "index": i,
                "id": format!("call_{}", i),
                "type": "function",
                "function": {
                    "name": format!("tool_{}", i),
                    "arguments": "{\"param\": \"value\"}"
                }
            })
        })
        .collect();

    json!({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1_677_652_288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": tool_calls
            },
            "finish_reason": null
        }]
    })
}

/// Create a final chunk with usage information
fn create_usage_chunk_json() -> serde_json::Value {
    json!({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1_677_652_288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    })
}

/// Benchmark deserializing a simple content chunk
#[allow(clippy::unwrap_used)]
fn bench_deserialize_content_chunk(c: &mut Criterion) {
    let json = create_content_chunk_json();
    let json_str = serde_json::to_string(&json).unwrap();

    c.bench_function("deserialize_content_chunk", |b| {
        b.iter(|| {
            #[allow(clippy::used_underscore_binding)]
            let _chunk: ChatCompletionChunk = serde_json::from_str(black_box(&json_str)).unwrap();
            black_box(_chunk);
        });
    });
}

/// Benchmark deserializing chunks with varying numbers of tool calls
#[allow(clippy::unwrap_used)]
fn bench_deserialize_tool_call_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialize_tool_call_chunks");

    for num_tools in &[1, 2, 4, 8] {
        let json = create_tool_call_chunk_json(*num_tools);
        let json_str = serde_json::to_string(&json).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(num_tools), num_tools, |b, _| {
            b.iter(|| {
                #[allow(clippy::used_underscore_binding)]
                let _chunk: ChatCompletionChunk =
                    serde_json::from_str(black_box(&json_str)).unwrap();
                black_box(_chunk);
            });
        });
    }

    group.finish();
}

/// Benchmark deserializing a chunk with usage information
#[allow(clippy::unwrap_used)]
fn bench_deserialize_usage_chunk(c: &mut Criterion) {
    let json = create_usage_chunk_json();
    let json_str = serde_json::to_string(&json).unwrap();

    c.bench_function("deserialize_usage_chunk", |b| {
        b.iter(|| {
            #[allow(clippy::used_underscore_binding)]
            let _chunk: ChatCompletionChunk = serde_json::from_str(black_box(&json_str)).unwrap();
            black_box(_chunk);
        });
    });
}

/// Benchmark a realistic streaming scenario with mixed chunks
#[allow(clippy::unwrap_used)]
fn bench_streaming_sequence(c: &mut Criterion) {
    // Simulate a typical streaming response:
    // 1. Initial role chunk
    // 2. Several content chunks
    // 3. Tool call chunk
    // 4. Final usage chunk
    let json_chunks = [
        create_content_chunk_json(),
        create_content_chunk_json(),
        create_content_chunk_json(),
        create_tool_call_chunk_json(2),
        create_usage_chunk_json(),
    ];

    let json_strs: Vec<String> = json_chunks
        .iter()
        .map(|j| serde_json::to_string(j).unwrap())
        .collect();

    c.bench_function("deserialize_streaming_sequence", |b| {
        b.iter(|| {
            for json_str in &json_strs {
                #[allow(clippy::used_underscore_binding)]
                let _chunk: ChatCompletionChunk =
                    serde_json::from_str(black_box(json_str)).unwrap();
                black_box(_chunk);
            }
        });
    });
}

/// Benchmark the chunk conversion function (hot path in streaming)
#[allow(clippy::unwrap_used)]
fn bench_convert_chunk(c: &mut Criterion) {
    use neuromance_client::openai::convert_chunk_to_chat_chunk;

    let content_chunk_json = create_content_chunk_json();
    let content_chunk: ChatCompletionChunk = serde_json::from_value(content_chunk_json).unwrap();

    let tool_chunk_json = create_tool_call_chunk_json(2);
    let tool_chunk: ChatCompletionChunk = serde_json::from_value(tool_chunk_json).unwrap();

    let usage_chunk_json = create_usage_chunk_json();
    let usage_chunk: ChatCompletionChunk = serde_json::from_value(usage_chunk_json).unwrap();

    let mut group = c.benchmark_group("convert_chunk");

    group.bench_function("content_chunk", |b| {
        b.iter(|| {
            let chat_chunk = convert_chunk_to_chat_chunk(black_box(&content_chunk));
            black_box(chat_chunk);
        });
    });

    group.bench_function("tool_call_chunk", |b| {
        b.iter(|| {
            let chat_chunk = convert_chunk_to_chat_chunk(black_box(&tool_chunk));
            black_box(chat_chunk);
        });
    });

    group.bench_function("usage_chunk", |b| {
        b.iter(|| {
            let chat_chunk = convert_chunk_to_chat_chunk(black_box(&usage_chunk));
            black_box(chat_chunk);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_deserialize_content_chunk,
    bench_deserialize_tool_call_chunks,
    bench_deserialize_usage_chunk,
    bench_streaming_sequence,
    bench_convert_chunk
);

criterion_main!(benches);
