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
│   └── neuromance-runtime/   # Container runtime binary
├── Cargo.toml                # Workspace configuration
└── README.md
```

## Tokenizer proxy

When the runtime is deployed behind a tokenizer proxy, the agent pod never holds the plaintext provider credential. The pod reads a sealed token from a projected secret, sends it to the proxy under `X-Tokenizer-Token`, and the proxy decrypts the token and injects the real provider key server-side before forwarding to the upstream.

`agent.base_url` and `[proxy].base_url` carry distinct URLs:

- `agent.base_url` is the **upstream** LLM endpoint. Falls back to the provider default from the `model` prefix (e.g. `openai:gpt-4o` → `https://api.openai.com/v1`).
- `[proxy].base_url` is the **tokenizer proxy** itself. The client attaches it as an HTTP forward proxy so requests leave the pod in absolute-form (RFC 7230 §5.3.2) with the upstream authority in the request URL — no side-band routing header is needed.

```toml
mode = "serve"

[agent]
id = "research-agent"
model = "openai:gpt-4o"
# Upstream provider URL. Optional when the model prefix has a default
# (here, openai → api.openai.com). Set this to pin a different upstream,
# e.g. https://openrouter.ai/api/v1.
base_url = "https://openrouter.ai/api/v1"
system_prompt = "..."
# api_key_env is omitted — [proxy] is the credential source.

[proxy]
# Tokenizer proxy Service inside the cluster. Attached as the HTTP forward proxy.
base_url = "http://tokenizer-proxy.windowlickers.svc.cluster.local:8080"
# Projected Secret volume holding the sealed token.
token_file = "/var/run/neuromance/tokens/llm"
# Header carrying the sealed token. Defaults to X-Tokenizer-Token.
token_header = "X-Tokenizer-Token"
```

## Model Context Protocol (MCP) Support

Neuromance supports the [Model Context Protocol](https://modelcontextprotocol.io/) for connecting to external tool servers via the `neuromance-tools` crate.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Evan Dobry ([@ecdobry](https://github.com/ecdobry))

## Acknowledgments

Named after William Gibson's novel Neuromancer.
