# Neuromance

A Rust library and runtime for controlling and orchestrating LLMs.

[![Crates.io](https://img.shields.io/crates/v/neuromance.svg)](https://crates.io/crates/neuromance)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)

## Overview

Neuromance provides high-level abstractions for building LLM-powered applications in Rust.

- **`neuromance`** - Main library providing unified interface for LLM orchestration
- **`neuromance-common`** - Common types and data structures for conversations, messages, and tools
- **`neuromance-client`** - Client implementations for various LLM providers
- **`neuromance-agent`** - Agent framework for autonomous task execution with LLMs
- **`neuromance-tools`** - Tool execution framework with MCP support

Neuromance also provides a container runtime for kubernetes.

- **`neuromance-runtime`** - Container runtime binary; runs an agent in `oneshot` or `serve` mode

## Development

### Prerequisites

- [Nix](https://nixos.org/download) with flakes enabled

Nix provides the pinned Rust toolchain (stable channel, via `rust-toolchain.toml`) and all build dependencies. No separate Rust install is needed.

### Dev shell

```bash
nix develop
```

### Building

```bash
nix build              # library
nix build .#neuromance-runtime
```

### Checks

```bash
nix flake check        # fmt + clippy + tests + build (all crates)
```

Or run individual checks (cached via crane):

```bash
nix run .#fmt
nix run .#clippy
nix run .#test
```

### Container images

```bash
nix build .#neuromance-image           # minimal runtime
nix build .#neuromance-toolkit-image   # adds busybox, git, curl, jq, node, python3, shell
```

Generic image verbs take the image short-name (`neuromance` or `neuromance-toolkit`) as the first argument:

```bash
nix run .#list                                        # list available images
nix run .#load    -- <image>                          # load into local docker
nix run .#push    -- <image> <registry> <namespace> [tag]
nix run .#release -- <image> <registry> <namespace>   # pushes :imageTag + :floatingTag
nix run .#inspect -- <image>                          # opens dive
nix run .#info    -- <image>                          # name, tags, store path, size
```

## Workspace Structure

```
neuromance/
├── crates/
│   ├── neuromance/           # Main library
│   ├── neuromance-common/    # Common types and data structures
│   ├── neuromance-client/    # Client implementations
│   ├── neuromance-agent/     # Agent framework
│   ├── neuromance-tools/     # Tool execution framework
│   ├── neuromance-db/        # Postgres persistence for conversations
│   └── neuromance-runtime/   # Container runtime binary
├── Cargo.toml                # Workspace configuration
└── README.md
```

## Providers

Endpoints, credentials, and default models are grouped into named `[[providers]]` entries. The `[agent]` references one by name; each subagent inherits the agent's provider unless it names its own. A provider supplies credentials via exactly one of two paths — `api_key_env` (a raw key read from the environment) or an inline `[providers.proxy]` table (a sealed token routed through a tokenizer proxy). At least one provider is required.

```toml
mode = "serve"

[[providers]]
name = "primary"
model = "openai:gpt-4o"          # default model; an agent/subagent may override it
api_key_env = "OPENAI_API_KEY"   # raw key from the environment

[agent]
id = "research-agent"
provider = "primary"             # references a [[providers]] entry
# model omitted — inherits the provider's "openai:gpt-4o"
system_prompt = "..."
```

The `provider:` prefix on a `model` string (`openai:`, `anthropic:`, `chat_completions:`, …) selects the client type and the default endpoint; the provider's `base_url` overrides that endpoint. A model with a generic prefix (`chat_completions:`, `responses:`) has no default endpoint, so its provider must set `base_url`.

### Tokenizer proxy

When a provider is deployed behind a tokenizer proxy, the agent pod never holds the plaintext credential. The pod reads a sealed token from a projected secret, sends it to the proxy under `X-Tokenizer-Token`, and the proxy decrypts the token and injects the real provider key server-side before forwarding to the upstream.

The provider's `base_url` and `[providers.proxy].base_url` carry distinct URLs:

- `base_url` is the **upstream** LLM endpoint. Falls back to the provider default from the `model` prefix (e.g. `openai:gpt-4o` → `https://api.openai.com/v1`).
- `[providers.proxy].base_url` is the **tokenizer proxy** itself. The client attaches it as an HTTP forward proxy so requests leave the pod in absolute-form (RFC 7230 §5.3.2) with the upstream authority in the request URL — no side-band routing header is needed.

```toml
[[providers]]
name = "primary"
model = "openai:gpt-4o"
# Upstream provider URL. Optional when the model prefix has a default
# (here, openai → api.openai.com). Set this to pin a different upstream,
# e.g. https://openrouter.ai/api/v1.
base_url = "https://openrouter.ai/api/v1"
# api_key_env is omitted — the proxy is the credential source.

[providers.proxy]
# Tokenizer proxy Service inside the cluster. Attached as the HTTP forward proxy.
base_url = "http://tokenizer-proxy.windowlickers.svc.cluster.local:8080"
# Projected Secret volume holding the sealed token.
token_file = "/var/run/neuromance/tokens/llm"
# Header carrying the sealed token. Defaults to X-Tokenizer-Token.
token_header = "X-Tokenizer-Token"
```

## Conversation persistence

With an optional `[database]` section, the runtime writes conversation history through to postgres as tasks run — a shared durable record any number of agents can write into. Messages are stored as an append-only log (idempotent per message id, ordered by a per-conversation sequence), so the full history including tool calls and tool results survives restarts and context compaction. In-memory state stays authoritative for serving; persistence is best-effort and never blocks a running task, while a failed connection at startup fails fast.

```toml
[database]
# Environment variable holding the postgres URL — the URL embeds a
# credential, so it never lives in this file (same policy as a provider's
# api_key_env).
url_env = "DATABASE_URL"
max_connections = 5            # optional, default 5
acquire_timeout_seconds = 5    # optional, default 5
run_migrations = true          # optional, default true
```

Migrations are embedded in `neuromance-db` and applied automatically at startup. Set `run_migrations = false` when an external owner (an operator or a shared schema service) manages the database — the runtime then connects to and uses the existing schema without attempting any DDL. The embedded migrations remain the source of truth for neuromance's tables regardless of who applies them. See the `neuromance-db` crate docs for the schema and the `cargo sqlx prepare` workflow when changing queries.

## MCP tools

Each `[[mcp_servers]]` entry connects an MCP server at startup. Every tool the server lists becomes an agent tool named `<id>__<tool>`, so `fs` serving `read_file` appears as `fs__read_file`. Characters a provider rejects in a tool name become `_`. The main agent and every subagent share the same connections, and a stdio server spawns once per process.

A server that stays unreachable after about five seconds of retries fails startup. This covers a sidecar that starts after the runtime. A tool needs explicit approval unless its server sets `auto_approve = true`, so the `approval.mode = "auto"` startup gate applies to MCP tools too.

```toml
# Subprocess speaking MCP over stdin/stdout.
[[mcp_servers]]
id = "fs"
name = "Filesystem"
protocol = "stdio"
command = "mcp-server-filesystem"
args = ["/work"]
env = { LOG_LEVEL = "warn" }   # optional
working_directory = "/work"    # optional
auto_approve = true

# Remote server over streamable HTTP ("sse" is accepted as an alias).
[[mcp_servers]]
id = "hands"
name = "Remote hands"
protocol = "http"
url = "http://hands.agents.svc:9000/mcp"
```

## Subagents

A `[[subagents]]` section declares subagents the main agent can delegate to. A subagent that pins neither `provider` nor `model` runs whatever the main agent runs for that task — a serve task's per-task `provider`/`model` override included — so it only needs an `id` and `system_prompt`. Pinning either one keeps the subagent's own identity and ignores the per-task override: the effective model is then the subagent's `model`, then the chosen provider's default `model`, then the agent's configured model.

A subagent is provisioned with the **same toolset as the main agent** — the capability tools from `[[tools]]`, the `[[mcp_servers]]` tools, and the delegate tools — so it can both use tools and delegate further. Its conversation loop is wired the same way too: it persists through `[database]`, compacts under `[context]`, carries the `[skills]` menu in its system prompt with `$mention` expansion in-loop, and receives `[rules]`. Two things are deliberately left off. Subagents never stream — a delegate tool returns only the final content. And subagent tool calls auto-approve inside the pod: they run autonomously within a single parent delegation, with no interactive approver in the loop, so the pod boundary (e.g. kata containers) is the isolation, the same way the whole agent pod is already sandboxed.

**Self-delegation.** Setting `runtime.self_delegation = true` gives the main agent a `task` delegate tool that runs a fresh copy of *itself* — the same system prompt, provider, model, and turn budget — on a scoped subtask, with no `[[subagents]]` entry to write. It is exactly a synthesized subagent named `task`, so it joins the same `max_delegation_depth` tower and gets the same shared toolset. `task` is reserved while this is on: an explicit subagent or tool of that name is rejected at startup. Combine it with `[[subagents]]` freely — the `task` self-clone is added alongside any named subagents.

One divergence: a serve task that supplies its own `system_prompt` overrides the main agent's seed message only. The `task` clone still carries `agent.system_prompt`, because a subagent's prompt is fixed when the delegation tower is built and the per-task prompt is not known then.

Nested delegation is bounded by `runtime.max_delegation_depth`, which counts subagent hops from the main agent (depth 0). At `1` the main agent reaches subagents but those subagents carry no delegate tools; at `2` (the default) a subagent may delegate one further hop, and so on. The deepest subagents are still fully tool-capable — they simply cannot delegate. The bound is enforced structurally (a finite tower of subagent instances built at startup), so the chain depth cannot run away; it is capped at 5.

The bound is on depth alone. Each delegated run carries its own `agent.max_turns` budget rather than drawing down the parent's, so within the depth limit an agent may still delegate many times. Set a finite `agent.max_turns` to bound the work a single task can generate, especially with `self_delegation` on, where the clone inherits the same instructions that told the parent to delegate.

Each configured subagent is a delegate tool named after its `id`, taking `instructions` and optional `context`. A delegate tool is not auto-approved, so under `approval.mode = "auto"` it is subject to the startup safety gate (set `approval.allow_unsafe_tools = true` or use async approval).

```toml
[runtime]
# Max subagent hops in a delegation chain. Optional; defaults to 2.
max_delegation_depth = 2

[[subagents]]
id = "researcher"
system_prompt = "You research a question and report findings."
# provider, model, max_turns all optional; provider defaults to the agent's,
# model defaults to the provider's default then the agent's effective model.

[[subagents]]
id = "critic"
system_prompt = "You critique a draft and list concrete fixes."
description = "Delegate a draft to the critic for review."
model = "anthropic:claude-opus-4-8"
max_turns = 4
```

## Structured outputs

A task can require the agent's final answer to be a JSON object matching a schema. The schema is
enforced by the provider, not by a prompt: the runtime passes it through to the model's native
structured-output parameter (`response_format` on Chat Completions, `text.format` on the Responses
API, `output_config.format` on Anthropic). A provider that does not support it fails the run rather
than answering unconstrained.

In `serve` mode the schema is per task, in the `POST /tasks/new` body:

```json
{
  "task": "Rate the pull request.",
  "output_schema": {
    "title": "verdict",
    "type": "object",
    "properties": {
      "score": { "type": "integer" },
      "summary": { "type": "string" }
    },
    "required": ["score", "summary"],
    "additionalProperties": false
  }
}
```

In `oneshot` mode it is config, under `[oneshot].output_schema`, with the same shape.

The schema must be a closed JSON object: the root declares `"type": "object"`, and every object
subschema — including those under `properties`, `$defs`, `items`, and `anyOf` — sets
`"additionalProperties": false`. Both are checked locally, so an unenforceable schema is rejected
at intake (`400`, before a task is minted) or at startup validation rather than mid-run, and the
error names the path of the offending subschema. OpenAI strict mode additionally requires every
property to appear in `required`; that one is enforced by the provider, not here. Nesting deeper
than 64 levels is rejected too. The optional `title` names the schema on the wire; it defaults to
`output` and must be 1–64 characters of `a-z`, `A-Z`, `0-9`, `_` or `-`, the range OpenAI accepts.

The constraint applies to whatever turn carries it, so it never rides on a turn that offers tools:
the model would answer in the schema where it should call a tool. The agent runs its tool loop
unconstrained, and when the model stops calling tools it is asked once more — with the schema and
no tools — for the answer. That final tool-free turn is the one that must parse as a JSON object;
the unconstrained answer it replaces is discarded and never reaches the conversation history. A
task with no tools configured skips the extra call: its first turn is already the final one.

Conformance beyond that is the provider's job. The runtime checks that the final turn parses as a
JSON object; it does not re-validate the object against the schema, so a provider that ignores the
constraint returns a `structured` field that satisfies no more than "is an object".

A successful task returns the parsed object in a `structured` field alongside the prose `output`,
on `GET /tasks/{id}` and in the `oneshot` JSON. With `[database]` configured it is stored in the
`tasks.structured` JSONB column. A run whose final turn is not a schema-valid object fails with
`schema_violation`; a refusal fails with `schema_refusal` and a response cut off at the token limit
fails with `schema_truncated`. None of the three are retried — the provider already enforced the
schema, so a retry would only repeat the failure.

## Tool bootstrap

Some tools cache their credentials to disk rather than reading them from the environment on each call. The agent pod has no persistent storage, so those tools must be logged in at container start. Each `[[bootstrap]]` entry names a command the runtime runs once before tasks begin; failures are logged but never fatal — a tool that can't be set up just isn't available, the same as any other tool error.

The runtime has no per-tool knowledge: the full command and arguments are supplied in config, so the deployer (typically an operator) bakes them in. A secret is **never** placed in `args` — `config.toml` ships in a `ConfigMap`. Instead, set `token_env` to the name of an environment variable (sourced from a `Secret`); the runtime feeds that variable's value to the command on stdin, so the credential never lands in argv.

```toml
[[bootstrap]]
name = "forgejo"                 # label for logs only
command = "fj"                   # executable, must be on PATH
args = [
  "--host", "git.example.com",
  "auth", "add-tokenizer",
  "--proxy", "http://tokenizer-proxy.windowlickers.svc.cluster.local:8080",
  "--name", "forgejo",
]
# Optional. When set, the value of this env var is fed on stdin, never in args.
token_env = "FORGEJO_TOKEN"
```

## Context compaction

An optional `[context]` section enables automatic conversation compaction in the runtime. Once the conversation grows past a ratio of the model's context window, older turns are summarized by the LLM and replaced with the summary, preserving the system prompt and the most recent turns.

Conversation size is measured from the provider-reported usage of the most recent response, so no tokenizer is downloaded at startup. One known lag: the first request of a resumed conversation is sent uncompacted because no usage exists yet in that run — compaction at the end of the previous run keeps stored histories under target.

Subagents compact too, on the same settings and against their own conversations. Each subagent run gets its own compactor, so concurrent runs never influence each other's decisions. Long delegated runs therefore summarize instead of walking into the context window, at the cost of the summarization calls that implies.

```toml
[context]
# Model context window in tokens. Required; everything else is optional.
context_window_size = 128000
# Compact once usage exceeds this ratio of the window (default 0.8).
compaction_threshold_ratio = 0.8
# Aim for this ratio of the window after compaction (default 0.5).
target_ratio = 0.5
# Recent user+assistant turns preserved verbatim (default 3).
preserve_recent_turns = 3
# one_shot (default) | hierarchical | truncate
strategy = "one_shot"
```

## Skills

An optional `[skills]` section gives the agent **skills** — directories containing a `SKILL.md` (YAML frontmatter with `name` and `description`, plus optional [agentskills.io](https://agentskills.io) fields, followed by a Markdown body of reusable instructions). Skills use progressive disclosure: a cheap menu of `name: description (file: …)` is folded into the system prompt, and a skill's full body lives on disk, read only when needed. This keeps the system prompt stable and cache-friendly.

Skills are discovered from on-host directory `roots` (each immediate subdirectory containing a `SKILL.md` is a skill) and/or a corpus-shaped HTTP `endpoint` (`GET /skills` for the menu, `GET /skills/{id}` for a body). On-host roots take precedence over the endpoint when a skill name appears in both.

At startup the runtime **materializes** every discovered skill to a local temp directory and rebuilds the catalog over it, so the corpus is only a boot-time dependency. The menu then lists each skill's on-disk path, and the agent loads a skill by **reading that file** with its `read` tool — no fetch at reasoning time. In addition, when `mention` is enabled a `$name`, `skill://id`, or `[text](skill://id)` link in user input (for example a `/thing-skill` slash command) injects the body inline as a message; common shell variables (`$PATH`, `$HOME`, …) are ignored.

Subagents see the same catalog. The menu is appended to each subagent's configured `system_prompt` (a subagent has no seed message to fold it into), and `$mention` expansion works inside a delegated run the same way it does for the main agent.

```toml
[skills]
# On-host skill directories, highest precedence first.
roots = ["/etc/neuromance/skills", "./skills"]
# Optional corpus-shaped endpoint serving skills over HTTP.
endpoint = "https://corpus.internal/api/v1/skills"
# Optional env var holding a bearer token for the endpoint.
endpoint_token_env = "CORPUS_TOKEN"
# Expand `$name` mentions in user input into the skill body (default true).
mention = true
# Byte budgets for the menu and each mention-injected body (default 8192 each).
menu_budget_bytes = 8192
body_budget_bytes = 8192
```

## Rules

An optional `[rules]` section gives the agent **rule files** — Markdown files (`.md`/`.mdc`) with optional YAML frontmatter that inject instructions by *location* rather than by intent. Where a skill is summoned when its description matches the task, a rule is pushed in automatically: a rule marked `always_apply` is injected once at the start of every conversation, and a rule with `globs` is injected the first time a tool touches a file whose path matches one of its patterns. Each rule is injected at most once per conversation.

Recognized frontmatter keys are `globs` (a YAML sequence or a comma-separated scalar; `paths` is accepted as an alias), `always_apply` (`alwaysApply` is also accepted), and `description`. A file with no frontmatter is a body-only rule with no globs. Glob patterns follow gitignore-style semantics and are matched workspace-relative, so `*.rs` matches a Rust file in any directory and `src/**/*.rs` matches only under `src/`.

Rules are discovered from on-host directory `roots` (walked recursively; a rule's id is its path relative to the root) and/or a corpus-shaped HTTP `endpoint` (`GET /rules` for the listing, `GET /rules/{id}` for a body). On-host roots take precedence over the endpoint when a rule id appears in both.

Subagents receive rules from the same catalog on the same terms: always-apply rules at the start of a delegated run, glob-matched rules the first time the subagent's own tool calls touch a matching path.

```toml
[rules]
# On-host rule directories, highest precedence first (searched recursively).
roots = ["/etc/neuromance/rules", "./rules"]
# Optional corpus-shaped endpoint serving rules over HTTP.
endpoint = "https://corpus.internal/api/v1/rules"
# Optional env var holding a bearer token for the endpoint.
endpoint_token_env = "CORPUS_TOKEN"
# Byte budget for each injected rule body (default 8192).
body_budget_bytes = 8192
```

A rule file looks like:

```markdown
---
globs: "*.rs"
description: Rust conventions
---
Follow the repository's error-handling rules: no unwrap/expect outside tests.
```

## Observability

The runtime exports traces, logs, and GenAI metrics over OTLP, and operational
metrics over a Prometheus scrape endpoint.

### Enabling export

`OTEL_EXPORTER_OTLP_ENDPOINT` is the master switch. When it is unset, the
runtime emits stderr logs and Prometheus metrics only.

| Variable | Effect |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Enables OTLP export and sets the default endpoint |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` (default) or `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | Headers added to every export request |
| `OTEL_SERVICE_NAME` | `service.name`; falls back to `agent.id` |
| `OTEL_RESOURCE_ATTRIBUTES` | Merged on top of the default resource |
| `OTEL_EXPORTER_OTLP_{TRACES,LOGS,METRICS}_ENDPOINT` | Per-signal endpoint override |
| `OTEL_METRIC_EXPORT_INTERVAL` | Milliseconds between metric exports |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture prompts and completions; off by default |
| `RUST_LOG_FORMAT=json` | Structured stderr logs, for cluster log ingestion |

`RUST_LOG` gates span creation, not just log output. A `RUST_LOG=warn`
deployment with OTLP configured exports no `gen_ai` spans at all. Use
`RUST_LOG=info`, the runtime default.

### Spans

Instrumentation follows the [OpenTelemetry GenAI semantic
conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), pinned to
v1.38.0. Attribute keys live in
`crates/neuromance-common/src/telemetry/genai.rs`; update that file and its
version pin together.

```
invoke_agent {agent}              neuromance-agent    INTERNAL
└── chat_turn                     neuromance          INTERNAL
    ├── chat {model}              neuromance-client   CLIENT   (one per retry attempt)
    └── execute_tool {tool}       neuromance          INTERNAL
```

`chat_turn` is not a GenAI operation. It is one iteration of the orchestration
loop, spanning both the model call and the tool executions that follow, and it
keeps the turn grouping that makes a long agent run readable.

| Span | Attributes |
| --- | --- |
| `chat {model}` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.{model,temperature,top_p,max_tokens,frequency_penalty,presence_penalty,stop_sequences,choice.count}`, `gen_ai.output.type`, `gen_ai.response.{model,id,finish_reasons}`, `gen_ai.usage.{input_tokens,output_tokens}`, `server.address`, `server.port`, `error.type` |
| `embeddings {model}` | As above, minus the response and completion attributes. Prompt tokens only: an embeddings call generates nothing |
| `execute_tool {tool}` | `gen_ai.operation.name`, `gen_ai.tool.{name,type}`, `gen_ai.tool.call.id`, `error.type` |
| `invoke_agent {agent}` | `gen_ai.operation.name`, `gen_ai.agent.id`, `gen_ai.conversation.id`, `gen_ai.request.model`, `gen_ai.usage.{input_tokens,output_tokens}`, `cached_input_tokens`, `cache_creation_input_tokens`, `llm_requests`, `error.type` |

The usage attributes on `invoke_agent` are the **whole run's**, so one span
prices a run without a join across its `chat` children. Cached and
cache-creation tokens are broken out because they bill at different rates, and
`gen_ai.request.model` is the agent's configured model — a compactor
configured with a different model bills against its own `chat` spans.
The counts exclude subagents: each delegated run carries its own
`invoke_agent` span, which is what keeps a sum over a delegation tree from
counting the same tokens twice. A run that fails after calling the provider
still reports what it spent.

`server.address` names the **upstream provider**, not the tokenizer proxy: the
proxy is transport plumbing, and reporting it would make every provider look
like one server. A streaming call keeps its `chat` span open until the stream
ends, so the span times the whole generation and carries the final token
counts. A retried request produces one `chat` span per attempt.

### Metrics

Two pipelines run side by side, and nothing is emitted to both.

- **`gen_ai.client.token.usage`** and **`gen_ai.client.operation.duration`** go
  out natively over OTLP, with the units and bucket boundaries the conventions
  advise. A Prometheus round-trip would turn the dots into underscores and drop
  the buckets. Attributes: `gen_ai.operation.name`, `gen_ai.provider.name`,
  `gen_ai.request.model`, `server.address`, `server.port`, plus
  `gen_ai.response.model` and `error.type` when known, and `gen_ai.token.type`
  (`input` / `output`) on the token histogram.
- **`neuromance_*`** operational metrics (queue depth, task outcomes, tool
  durations, approvals, persistence) are Prometheus, scraped from
  `GET /metrics` on `runtime.health_addr` (default `:8081`).

A collector needs both a receiver for the OTLP push and a scrape job for
`/metrics`:

```yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
  prometheus:
    config:
      scrape_configs:
        - job_name: neuromance
          scrape_interval: 30s
          static_configs:
            - targets: ["neuromance:8081"]

service:
  pipelines:
    traces:  { receivers: [otlp], exporters: [clickhouse] }
    logs:    { receivers: [otlp], exporters: [clickhouse] }
    metrics: { receivers: [otlp, prometheus], exporters: [clickhouse] }
```

### Capturing prompts and completions

`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` adds
`gen_ai.input.messages`, `gen_ai.output.messages`,
`gen_ai.system_instructions`, `gen_ai.tool.call.arguments`, and
`gen_ai.tool.call.result` to the spans above.

**This ships conversation content to your trace backend.** Everything a user
types, everything a model answers, and every argument and result of every tool
call. API keys are not captured — they live in `SecretString` and are never
part of a captured structure — but user data in prompts is. The flag is off by
default; turn it on per-deployment, and only where the trace store has the same
access controls as the conversations themselves.

## Model Context Protocol (MCP) Support

Neuromance supports the [Model Context Protocol](https://modelcontextprotocol.io/) for connecting to external tool servers via the `neuromance-tools` crate.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Evan Dobry ([@ecdobry](https://github.com/ecdobry))

## Acknowledgments

Named after William Gibson's novel Neuromancer.
