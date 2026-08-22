# haste

A micro agent harness built for wafer-speed inference (Cerebras-class, 2000+ tok/s),
where the model stops being the bottleneck and wall-clock is
`turns × (RTT + prefill + gen + tools + harness)`. haste attacks every term:

- **Minimum output tokens** — a line DSL instead of JSON tool calls (a typical action
  is 8–20 tokens), file interning (`#3` instead of paths), line-range edits that never
  repeat old text, no reasoning, no prose.
- **Minimum turns** — batched commands per message, compound config tools, subagents
  that burn context in their own ledger and return only a distilled brief.
- **Minimum context** — a lossless append-only ledger where *compression is a rendering
  decision* (nestor's invariant): stale reads superseded, duplicates pointered, old
  results folded, budget enforced — all recomputed per turn, reversible by design.
- **~Zero harness time** — sequential blocking loop, no async runtime, in-process tools,
  streaming lexer that completes commands mid-generation. Measured overhead is a CI
  assertion, not a hope.

~1,300 LOC, 5 deps, one binary. Everything moddable lives in `haste.toml`.

## Run

```
haste [-c haste.toml] [-p profile] [-C root] <task...>
```

Point `[model].base_url` at any OpenAI-compatible endpoint. Exit prints a speed report:
turns, commands, wall / model / tool / render time, tokens sent.

## The DSL

```
R <id|path> [a:b]    read (numbered lines)
E <id> <a>:<b>       replace lines a..b — content follows, terminated by a lone "."
I <id> <a>           insert after line a (0 = top)
N <path>             create file
G <regex> [target]   search  ->  #id:line:text
X <command>          shell
A <profile> <task>   subagent (parallel when batched)
D <message>          done
<any other letter>   config-declared tool from haste.toml
```

Payload lines that must start with `.` get one extra leading dot.

## Context modes

- `working_set` — re-render aggressively every turn. For providers **without** a prefix
  cache (Cerebras): mutation is free, so take maximum compression.
- `append` — byte-stable prefix; folding only happens at reseal points and is then
  frozen. For providers **with** a prefix cache (llama.cpp, vLLM).

Same ledger, same renderer, one config switch.

## Pruning tiers

1. **Structural** (free): `head_tail:A,B`, `first_failure`, `errors_only`, `keep:RE`,
   `drop:RE` — chained with `|`, per tool, in config.
2. **Dedupe** (free): identical results render once; repeats become pointers.
3. **Distill** (~one cheap model call): `distill` in a prune chain routes output through
   the model with the task in the prompt — 40K tokens of web page in the ledger, 300
   tokens in the context.

## Subagents

A profile in `haste.toml` = system prompt + allowed verbs + own budget. `A researcher
find how X does retries` runs the same loop recursively in a thread; batched `A` lines
run in parallel; only the final `D` brief enters the parent ledger.

## Testing

```
cargo test
```

Unit tests cover the lexer, edit renumbering, pruners, and profile restrictions. The
e2e test runs the full loop against an in-process scripted SSE server and **asserts
harness overhead stays under 250ms per task** (loopback network included); the render
bench asserts <5ms per render at 500 ledger entries. Numbers print with `--nocapture`.

## Design lineage

Ledger/renderer split and the sealed-prefix rule from [nestor]; weak/fast-model
discipline from stim; pruning instincts from nestor-lean — redesigned to fit in one
context window, so the agent can read its own harness in a single `R`.
