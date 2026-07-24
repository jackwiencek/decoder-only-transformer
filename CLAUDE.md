# CLAUDE.md

Repo-scoped instructions. Loaded automatically at the start of every session.

## What this project is

A decoder-only transformer language model implemented from PyTorch primitives
(`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, raw tensor ops) and trained for
next-token prediction on the TinyStories dataset.

Explicit non-goal: this is not a wrapper around `nn.TransformerDecoder` or
HuggingFace `transformers`. Every architectural component is hand-written. If a
shortcut would hide the mechanism being learned, do not take it.

**Target result:** validation cross-entropy <= 1.5 nats/token, measured against
this repo's own 8k BPE tokenizer (see `docs/adr/0005`).

## Working agreement

This repo is being built as a learning exercise. The owner writes the core code;
Claude scaffolds, explains, tests, and reviews. Concretely:

- **Do not pre-implement model or tokenizer code.** No stub files, no
  placeholder classes, no "here's the finished version" ahead of the
  conversation that builds it. See `docs/adr/0007`.
- Work **one section at a time**: explain the concept, let the owner attempt it,
  review and correct, then move on.
- Explaining shapes is usually more useful than explaining syntax. Annotate
  tensor dimensions with the `(B, T, C)` convention in comments and prose.
- When a decision has multiple defensible answers, surface the trade-off rather
  than silently picking one.

## Naming conventions

Standard GPT shorthand is used throughout; keep it consistent.

| Symbol | Meaning                                     |
| ------ | ------------------------------------------- |
| `B`    | batch size                                  |
| `T`    | time / sequence position (up to `block_size`) |
| `C`    | channels, i.e. `n_embd`                     |
| `V`    | vocabulary size (8192)                      |
| `nh`   | number of attention heads                   |
| `hs`   | head size, `n_embd // n_head`               |

## Layout

```
src/tinygpt/     the package - modules appear here as we build them
configs/         YAML run configs (smoke.yaml for CPU, base.yaml for Colab)
tests/           pytest; every component gets a shape/behaviour test
notebooks/       thin Colab driver that clones + installs this package
docs/adr/        numbered architecture decision records
docs/learning-log.md   running journal of concepts covered
scratch/         throwaway experiments, not part of the package
```

## Environments

Two environments, deliberately different:

- **Laptop (Windows, no GPU):** CPU-only PyTorch. Used for writing code, running
  tests, and the `smoke` config only. Never used for real training.
- **Google Colab (T4/L4):** all real training runs. The notebook clones this
  repo and `pip install -e .`s it, so anything that must work on the GPU must be
  committed first.

Because of this split, **never hardcode `cuda`**. Device comes from config, with
a runtime fallback to CPU.

## Commands

```powershell
# activate (PowerShell)
.\.venv\Scripts\Activate.ps1

python -m pytest          # tests
ruff check . ; ruff format .
```

## Conventions

- Config is YAML, loaded into a dataclass. No argparse sprawl, no magic numbers
  buried in function bodies.
- Anything under `data/` or `checkpoints/` is regenerable and gitignored.
- Every non-obvious decision gets an ADR in `docs/adr/`. Every concept the owner
  learns gets an entry in `docs/learning-log.md`.
- Loss is always reported in **nats/token** and is only comparable within a fixed
  tokenizer. Always state the tokenizer alongside the number.
