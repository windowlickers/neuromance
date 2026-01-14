# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build all crates
cargo build

# Run tests
cargo test

# Lint with clippy (strict settings enforced)
cargo clippy --all-targets --all-features

# Format code
cargo fmt

# Run the CLI
cargo run --bin neuromance-cli -- --mcp-config mcp_config.toml
```

### Nix-based Development

The project uses Nix flakes for reproducible builds and CI:

```bash
# Run all checks (fmt, clippy, tests, build)
nix flake check

# Enter dev shell with tools
nix develop

# Build the library
nix build

# Build the CLI
nix build .#neuromance-cli
```

## Architecture

Neuromance is a Rust library for LLM orchestration, organized as a Cargo workspace.

### Crates

- **neuromance-common**: Foundational types shared across all crates. Defines `Conversation`, `Message`, `ChatRequest`, `ChatResponse`, `Tool`, `ToolCall`, `Config`, and cross-provider abstractions like `ThinkingMode` and `ReasoningLevel`.

- **neuromance-client**: LLM provider clients implementing the `LLMClient` trait. Includes `OpenAIClient` and `AnthropicClient` with streaming, tool calling, and retry support.

- **neuromance-tools**: Tool execution framework. Defines `ToolImplementation` trait for custom tools, `ToolRegistry` for registration, and `ToolExecutor` for execution. Includes MCP (Model Context Protocol) client for connecting to external tool servers.

- **neuromance**: Main orchestration library. The `Core<C: LLMClient>` struct manages conversation loops with tool execution, using an event-driven architecture with callbacks for streaming content, tool approval, and usage tracking.

- **neuromance-agent**: Agent framework for autonomous multi-turn task execution. The `Agent` trait and `BaseAgent` implementation provide state management, memory, and sequential execution with tool support.

- **neuromance-cli**: Interactive command-line interface for LLM conversations with MCP tool support.

## Common Patterns

- **Builder pattern**: Structs use `typed-builder` for ergonomic construction (e.g., `Tool::builder()`, `Config::new().with_api_key()`)
- **Async-first**: All I/O uses `tokio` runtime. Async traits use `async-trait` crate.
- **Secrets handling**: API keys are wrapped in `secrecy::SecretString` to prevent accidental logging
- **Streaming**: SSE streams use `reqwest-eventsource` with custom `NoRetryPolicy` for retry control at the application layer

## Error Handling

The codebase uses a two-tier error strategy:

- **`thiserror`** for typed library errors with specific variants (see `ClientError` in `neuromance-client/src/error.rs`, `CoreError` in `neuromance/src/error.rs`)
- **`anyhow`** for application-level error propagation

`ClientError` includes an `is_retryable()` method to distinguish transient failures (network, rate limits, 5xx) from permanent errors (auth, validation).

## MCP Configuration

MCP servers are configured via TOML, YAML, or JSON files. Example TOML:

```toml
[settings]
max_retries = 3

[[servers]]
id = "filesystem"
name = "Local Filesystem"
protocol = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
auto_approve = false

[[servers]]
id = "search"
name = "Web Search"
protocol = "sse"
url = "https://example.com/mcp"
auto_approve = true
```

Transport protocols: `stdio` (subprocess), `sse` (Server-Sent Events), `http` (HTTP streaming).

## Versioning

All crates share a single version defined in the root `Cargo.toml` under `[workspace.package]`. When bumping versions, update only the workspace version.

## Benchmarks

```bash
# Run streaming benchmark
cargo bench -p neuromance-client
```

Benchmarks are in `crates/neuromance-client/benches/`.

## Code Standards

### Strict Clippy Configuration

The workspace enforces strict lints (see `Cargo.toml`):
- `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `dbg_macro` are **denied**
- Tests can use `#![allow(clippy::unwrap_used)]` and `#![allow(clippy::expect_used)]`
- `unsafe_code` is **forbidden**

### Commit Messages

Commits must follow [Conventional Commits](https://www.conventionalcommits.org/) format (enforced by commitlint in CI).

## Rust Version

Requires Rust 1.90+ (Edition 2024). The `rust-toolchain.toml` pins to stable channel.
