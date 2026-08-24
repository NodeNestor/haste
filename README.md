# haste

A micro agent harness built for wafer-speed inference (Cerebras-class, 2000+ tok/s),
where the model stops being the bottleneck and wall-clock is
`turns × (RTT + prefill + gen + tools + harness)`. haste attacks every term:

- **Minimum output tokens** — a line DSL instead of JSON tool calls (a typical action
  is 8–20 tokens), file interning (`#3` instead of paths), line-range edits that never
  repeat old text, no reasoning, no prose.
- **Minimum turns** — batched commands per message, parallel background subagents,
  auto-verify that deletes the "run the tests" turn, a plan state machine that refuses
  premature finishes.
- **Minimum context** — a lossless append-only ledger where *compression is a rendering
  decision*: write-time dedup (identical results become pointers), delta file legends,
  model self-compaction sealed into the cached prefix, phase-boundary seals that hold a
  long task's context near-constant — turn 300 pays the same prefill as turn 30.
- **~Zero harness time** — commands execute **mid-stream** the moment the lexer
  completes them (tool time hides inside generation time), a persistent shell daemon
  (~2ms per command instead of ~150ms of shell startup), background test runs, no async
  runtime. Overhead is a CI assertion, not a hope.

~4.5k LOC, 7 deps, one self-contained binary.

## Install

```powershell
# Windows
irm https://raw.githubusercontent.com/NodeNestor/haste/master/install.ps1 | iex
```
```sh
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/NodeNestor/haste/master/install.sh | sh
```

`haste update` self-updates from the latest release. Or `cargo build --release`.

## Run

```
haste                          # TUI in the current directory
haste [-p profile] [-C root] <task...>   # headless one-shot
```

Point `[model].base_url` at any OpenAI-compatible endpoint. The TUI is a chat: type to
steer the agent mid-run, Esc stops, F2 toggles the raw stream, and a sidebar shows live
subagents and the plan. Exit prints the speed report: turns, commands, wall / model /
tool time, real token usage (with cached %), and seal count.

## The DSL

```
R <id|path> [a:b]    read (numbered lines)
E <id> <a>:<b>       replace lines a..b — content follows, terminated by a lone "."
I <id> <a>           insert after line a (0 = top)
N <path>             create file
G <regex> [target]   search (ripgrep when installed)  ->  #id:line:text
O [id|dir]           outline: signatures only, orient before reading
X <command>          shell (persistent daemon on Windows)
V <id|path>          view an image — the model sees it next turn
A <profile> <task>   background subagent (uncapped, parallel)
S <text>             say one line to the user without finishing
D <message>          done — ends the run
<any other letter>   config-declared tool from haste.toml
```

Payload lines that must start with `.` get one extra leading dot.

## Context economics

- `append` mode (default): the rendered document is byte-stable, so provider prefix
  caches hit on everything but the newest turn. Budget decisions use the provider's
  **real** `prompt_tokens`, never estimates. When the doc crosses the budget — or a
  plan step completes past the phase floor — the model writes its own compaction
  summary in one prompt-cached call and history reseals behind it.
- `working_set` mode: re-rendered aggressively every turn for providers without any
  prefix cache — stale reads superseded, duplicates pointered, old results folded.

Note on Cerebras: their prompt caching is automatic and exact-prefix (append mode fits
it perfectly) but cached tokens bill at the **full** input rate — set
`compact_phase_tokens` low there; the cache buys latency, compaction buys money.

## Pruning tiers

1. **Structural** (free): `head_tail:A,B`, `first_failure`, `errors_only`, `keep:RE`,
   `drop:RE` — chained with `|`, per tool, in config.
2. **Dedup** (free): a result byte-identical to an earlier one is stored as a pointer
   at write time — the duplicate never enters the context at all.
3. **Distill** (~one cheap model call): `distill` in a prune chain routes output
   through the model with the task in the prompt — 40K tokens of web page in the
   ledger, 300 tokens in the context.

## Subagents

A profile in `haste.toml` = system prompt + allowed verbs + own budget. `A researcher
find how X does retries` runs the same loop recursively in a thread; batched `A` lines
run in parallel (no cap — a duplicate-spawn guard and depth limit are the rails); only
the final `D` brief enters the parent ledger. The TUI shows each one live.

## Mods

Drop a folder into `~/.haste/mods/` with a `mod.toml` and haste grows new verbs:

```toml
name = "mcp"
prompt = "extra system-prompt lines"
[env]
MY_KEY = "..."
[tool.M]
desc = "call an MCP tool"
cmd  = "python {mod}/bridge.py {args}"
```

A mod (or `[tool.*]` in haste.toml) can also **replace a native verb**, game-mod
style: declare `override = true` on `R`, `G`, `X`, `O`, or `V` and the command routes
to your process instead of the built-in. Payload verbs (E/I/N) and protocol verbs
(S/D/A) stay native.

Project instruction files (`HASTE.md`, `AGENTS.md`, `CLAUDE.md`) at the workspace root
are pinned into the bootstrap automatically.

## Testing

```
cargo test
```

50+ tests: lexer, edit renumbering, pruners, profile restrictions, dedup, compaction
and phase seals, the shell daemon, and full e2e loops against an in-process scripted
SSE server. CI **asserts harness overhead stays under 250ms per task** and <5ms per
render at 500 ledger entries. Tag a `v*` and CI ships Windows/Linux/macOS binaries to
a GitHub release.

## Design lineage

Ledger/renderer split and the sealed-prefix rule from nestor; weak/fast-model
discipline from stim; pruning instincts from nestor-lean — redesigned to fit in one
context window, so the agent can read its own harness in a single `R`.
