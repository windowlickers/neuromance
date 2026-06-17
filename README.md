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
- **`neuromance-repl`** - Embedded Python REPL (PyO3) with stateful sessions and Rust-backed callbacks

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
│   ├── neuromance-repl/      # Embedded Python REPL
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

## Subagents

A `[[subagents]]` section declares subagents the main agent can delegate to. A subagent inherits the parent agent's provider (and so its endpoint, credential, and effective model) automatically; it only needs an `id` and `system_prompt`. It may optionally set its own `provider` and/or `model`: the effective model is the subagent's `model`, then the chosen provider's default `model`, then the agent's effective model.

A subagent is provisioned with the **same toolset as the main agent** — the capability tools from `[[tools]]`, the `execute_python` bridge, and the delegate tools — so it can both use tools and delegate further. Subagent tool calls auto-approve inside the pod: they run autonomously within a single parent delegation, with no interactive approver in the loop, so the pod boundary (e.g. kata containers) is the isolation, the same way the whole agent pod is already sandboxed.

Nested delegation is bounded by `runtime.max_delegation_depth`, which counts subagent hops from the main agent (depth 0). At `1` the main agent reaches subagents but those subagents carry no delegate tools; at `2` (the default) a subagent may delegate one further hop, and so on. The deepest subagents are still fully tool-capable — they simply cannot delegate. The bound is enforced structurally (a finite tower of subagent instances built at startup), so it cannot run away; it is capped at 5.

Every configured subagent is reachable two ways:

- **As a delegate tool** — an agent gets a tool named after the subagent's `id`, taking `instructions` and optional `context`. Like other tools, a delegate tool is not auto-approved, so the main agent's tool calls under `approval.mode = "auto"` are subject to the same startup safety gate (set `approval.allow_unsafe_tools = true` or use async approval).
- **From the Python REPL** — when an `execute_python` tool is also configured (requires the `python-repl` build feature), the runtime builds the REPL with the subagents bridged in as `run_subagent(name, instructions, context=None)` and `spawn_agents([Agent(name, instructions), ...])`, so the agent can write its own orchestration in Python. The bridge runs in restricted mode; an `execute_python` entry with `restricted = false` alongside subagents is rejected. The deepest subagents, having no children to delegate to, get a plain `execute_python` (no `run_subagent`/`spawn_agents`). Each subagent run gets its own fresh interpreter — including each concurrent run in a `spawn_agents` fan-out — so interpreter state never bleeds across runs.

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

# Capability tools and the Python bridge are shared by the main agent and every
# subagent. Bridge the subagents into Python (run_subagent / spawn_agents):
[[tools]]
name = "execute_python"
```

## Sandboxed tool execution

Tool execution can run in a separate **sandbox** process so that a misbehaving agent — one that escapes the in-process Python sandbox or abuses `bash` — has no path to the database or the LLM credentials. The orchestrator and the sandbox run as two containers in one pod under different service accounts: the orchestrator (`sa/mancer`) holds the DB and provider credentials and runs the agent loop, approval, and persistence; the sandbox (`sa/mancer-sandbox`) executes the capability tools (`bash`, file tools, `grep`/`find`/`ls`, `execute_python`) against the workspace, with an egress `NetworkPolicy` blocking Postgres and no place in the database `AuthorizationPolicy`.

The same binary runs both roles, selected by subcommand:

```bash
neuromance-runtime           # orchestrator (default; also `run`)
neuromance-runtime sandbox   # sandbox tool executor
```

Both processes read the same config file. The `[sandbox]` section configures the boundary:

```toml
[sandbox]
# Address the sandbox process binds its gRPC server to. Loopback-only: the
# two containers share the pod network namespace, so the channel never leaves
# the pod. Optional; defaults to 127.0.0.1:50051.
listen_addr = "127.0.0.1:50051"
# Endpoint the orchestrator dials. When set, the orchestrator advertises the
# sandbox's tools to the LLM and routes every approved tool call there instead
# of executing in-process. Unset (or no [sandbox] section) keeps tools local.
endpoint = "http://127.0.0.1:50051"
```

The orchestrator fetches the sandbox's tool definitions once at startup (retrying while the sandbox container comes up) and mirrors each tool's auto-approval requirement, so the startup approval gate behaves exactly as it does for local tools. Approval is always decided by the orchestrator before a call crosses the boundary; the sandbox only executes. A tool that runs but fails is returned as a normal tool error; a transport failure degrades to a failed tool call rather than crashing the orchestrator.

The channel is loopback-only within the pod (Istio mesh covers transport; the channel construction is left pluggable for future SPIFFE/SPIRE identities). The **workspace volume mounts into the sandbox container only** — the orchestrator never touches the workspace and holds no credentials reachable from tool code.

The gRPC build needs `protoc` (provided by the Nix dev shell and build inputs).

**Limitation (this release):** the Python `run_subagent`/`spawn_agents` bridge cannot cross the sandbox boundary — it needs the interpreter and the subagent tower in one process. A config that sets `sandbox.endpoint` together with both `[[subagents]]` and an `execute_python` tool is rejected at startup. Subagents (delegation) and a standalone `execute_python` each work under the sandbox individually.

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

An optional `[skills]` section gives the agent **skills** — directories containing a `SKILL.md` (YAML frontmatter with `name` and `description`, plus optional [agentskills.io](https://agentskills.io) fields, followed by a Markdown body of reusable instructions). Skills use progressive disclosure: a cheap menu of `name: description` is injected once per conversation, and a skill's full body is loaded into context only when it is summoned. This keeps the system prompt stable and cache-friendly.

Skills are discovered from on-host directory `roots` (each immediate subdirectory containing a `SKILL.md` is a skill) and/or a corpus-shaped HTTP `endpoint` (`GET /skills` for the menu, `GET /skills/{id}` for a body). On-host roots take precedence over the endpoint when a skill name appears in both.

A skill's body is summoned two ways, selected by `invocation`:

- **`load_skill` tool** — the model calls `load_skill(name)` and the body is returned as the tool result. Suited to autonomous runs.
- **`$mention`** — `$name`, `skill://id`, or a `[text](skill://id)` link in user input injects the body as a message. Common shell variables (`$PATH`, `$HOME`, …) are ignored.

```toml
[skills]
# On-host skill directories, highest precedence first.
roots = ["/etc/neuromance/skills", "./skills"]
# Optional corpus-shaped endpoint serving skills over HTTP.
endpoint = "https://corpus.internal/api/v1/skills"
# Optional env var holding a bearer token for the endpoint.
endpoint_token_env = "CORPUS_TOKEN"
# tool | mention | both (default)
invocation = "both"
# Byte budgets for the menu and each loaded body (default 8192 each).
menu_budget_bytes = 8192
body_budget_bytes = 8192
```

## Rules

An optional `[rules]` section gives the agent **rule files** — Markdown files (`.md`/`.mdc`) with optional YAML frontmatter that inject instructions by *location* rather than by intent. Where a skill is summoned when its description matches the task, a rule is pushed in automatically: a rule marked `always_apply` is injected once at the start of every conversation, and a rule with `globs` is injected the first time a tool touches a file whose path matches one of its patterns. Each rule is injected at most once per conversation.

Recognized frontmatter keys are `globs` (a YAML sequence or a comma-separated scalar; `paths` is accepted as an alias), `always_apply` (`alwaysApply` is also accepted), and `description`. A file with no frontmatter is a body-only rule with no globs. Glob patterns follow gitignore-style semantics and are matched workspace-relative, so `*.rs` matches a Rust file in any directory and `src/**/*.rs` matches only under `src/`.

Rules are discovered from on-host directory `roots` (walked recursively; a rule's id is its path relative to the root) and/or a corpus-shaped HTTP `endpoint` (`GET /rules` for the listing, `GET /rules/{id}` for a body). On-host roots take precedence over the endpoint when a rule id appears in both.

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

## Model Context Protocol (MCP) Support

Neuromance supports the [Model Context Protocol](https://modelcontextprotocol.io/) for connecting to external tool servers via the `neuromance-tools` crate.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Evan Dobry ([@ecdobry](https://github.com/ecdobry))

## Acknowledgments

Named after William Gibson's novel Neuromancer.
