# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build all crates
cargo build

# Run tests
cargo test

# Run tests for a specific crate
cargo test -p neuromance-runtime

# Run a single test by name (with output)
cargo test -p neuromance-runtime test_name -- --nocapture

# Lint with clippy (strict settings enforced)
cargo clippy --all-targets --all-features

# Format code
cargo fmt
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
```

`nix flake check` runs clippy with `-D warnings`, so any warning fails CI even if `cargo clippy` is clean locally.

Individual checks (cached via crane):

```bash
nix run .#fmt
nix run .#clippy
nix run .#test
```

### Container Images

The flake produces two runtime image variants:

```bash
# Build images
nix build .#neuromance-image           # minimal runtime image
nix build .#neuromance-toolkit-image   # adds busybox, git, curl, jq, node, python3, shell

# Generic image verbs — pass the image short-name as the first arg:
#   <image> ∈ { neuromance, neuromance-toolkit }
nix run .#list                                        # list available images
nix run .#load    -- <image>
nix run .#push    -- <image> <registry> <namespace> [tag]
nix run .#release -- <image> <registry> <namespace>   # pushes :imageTag + :floatingTag
nix run .#inspect -- <image>                          # opens dive
nix run .#info    -- <image>                          # name, tags, store path, size
```

## Architecture

Neuromance is a Rust library for LLM orchestration, organized as a Cargo workspace.

### Crates

- **neuromance-common**: Foundational types shared across all crates. Defines `Conversation`, `Message`, `ChatRequest`, `ChatResponse`, `Tool`, `ToolCall`, `Config`, and cross-provider abstractions like `ThinkingMode` and `ReasoningLevel`.

- **neuromance-client**: LLM provider clients implementing the `LLMClient` trait. Includes `ChatCompletionsClient` (for `OpenAI` and any compatible provider), `ResponsesClient` (`OpenAI` Responses API), and `AnthropicClient`, all with streaming, tool calling, and retry support.

- **neuromance-tools**: Tool execution framework. Defines `ToolImplementation` trait for custom tools, `ToolRegistry` for registration, and `ToolExecutor` for execution. Includes MCP (Model Context Protocol) client for connecting to external tool servers.

- **neuromance**: Main orchestration library. The `Core<C: LLMClient>` struct manages conversation loops with tool execution, using an event-driven architecture with callbacks for streaming content, tool approval, and usage tracking.

- **neuromance-agent**: Agent framework for autonomous multi-turn task execution. `Agent<C: LLMClient>` wraps `Core` with state, memory, and a sequential tool-using execution loop; constructed via `AgentBuilder` (`Agent::builder(id, client)`).

- **neuromance-repl**: Embedded Python REPL via PyO3. Stateful sessions, Rust-backed callbacks, and a `ToolImplementation` wrapper so agents can drive a Python interpreter as a tool.

- **neuromance-runtime**: Container runtime binary that boots an `Agent` from TOML config and runs in `oneshot` mode (single task, write JSON, exit — for k8s `Job`s) or `serve` mode (HTTP intake at `POST /tasks/new` / `GET /tasks` / `GET /tasks/{id}` until SIGTERM — for `Deployment`s). Health and readiness are exposed on `runtime.health_addr` (default `:8081`); the task port is not a probe target. Tools are registered at startup via `ToolFactoryRegistry::with_builtin()`. Approval is `auto` or `async` (webhook). Serving state is in-memory; an optional `[database]` section writes conversation history through to postgres (via `neuromance-db`; see README). Set `RUST_LOG_FORMAT=json` for structured logs (k8s ingestion); default is human-readable.

- **neuromance-db**: Postgres persistence for conversations and messages (sqlx, compile-time checked queries with committed `.sqlx/` offline metadata). `PgConversationStore` stores messages as an append-only log — idempotent per message id, ordered by a per-conversation `seq` — plus the narrow `ConversationSink` trait that `Core` consumes behind the `db` feature (`Core::with_persistence`). Schema migrations are embedded and run at startup. When changing queries or schema, regenerate metadata with `cargo sqlx prepare` from `crates/neuromance-db` against a live postgres (crate docs have the workflow); integration tests run with `cargo test -p neuromance-db -- --ignored` and `DATABASE_URL` set.

## Tokenizer Proxy

When the runtime is deployed behind a tokenizer proxy, the agent pod never holds the plaintext provider credential — it sends a sealed token under `X-Tokenizer-Token` and the proxy injects the real provider key server-side.

Two distinct URLs are involved:

- `agent.base_url` — the **upstream** LLM endpoint. Falls back to the provider default from the `model` prefix (e.g. `openai:gpt-4o` → `https://api.openai.com/v1`).
- `[proxy].base_url` — the **tokenizer proxy** itself. The client attaches it as an HTTP forward proxy, so requests leave the pod in absolute-form with the upstream authority in the request URL; no side-band routing header is needed.

See `README.md` for the full TOML config example.

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

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) format.

## CI

CI runs on Forgejo (`.forgejo/workflows/nix-ci.yaml`), not GitHub Actions.

## Rust Version

Requires Rust 1.95+ (Edition 2024, set in `[workspace.package]`). The `rust-toolchain.toml` pins to stable channel.
