# ADR 0004: Model scale and context length

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

The model must be large enough to reach the target validation loss and produce
recognisably coherent stories, and small enough that a full training run fits
inside a single Colab session (ADR 0002) with time left over to iterate.

TinyStories documents are short — roughly 200–300 tokens each under an 8k BPE
vocabulary. A context window much larger than that is mostly wasted: attention
cost grows as `O(T^2)` while the extra positions see only padding or unrelated
neighbouring stories.

## Decision

Default configuration:

| Hyperparameter | Value |
| -------------- | ----- |
| `n_layer`      | 6     |
| `n_head`       | 6     |
| `n_embd` (`C`) | 384   |
| `head_size`    | 64 (`n_embd // n_head`) |
| `block_size` (`T`) | 256 |
| `vocab_size` (`V`) | 8192 |

That is roughly **13M parameters** with tied input/output embeddings.

Every one of these is read from a YAML config, never hardcoded in the model. Two
presets ship: `smoke` (tiny, CPU, seconds — for correctness) and `base` (the
table above, Colab).

## Consequences

- Fits a T4 comfortably at batch size 64; a run to the target loss is
  expected in the 1–2 hour range.
- Three to four experiments per Colab session, which is the property that
  actually drives learning — a single expensive run teaches very little.
- Head size lands at 64, matching the value used across the GPT family. Not a
  coincidence worth ignoring: it keeps per-head attention matrices at a size
  that GPU tensor cores handle well.
- `block_size = 256` truncates the tail of longer stories. Accepted: the model
  learns local narrative structure, which is the point, and the cost of `T=512`
  is roughly 4x the attention compute.
- Anything trained now cannot be compared to a later run at a different scale
  without re-running the baseline. Config files are therefore committed
  alongside results.

## Alternatives considered

- **8L / 8H / 512d / ctx 512 (~30M params)** — closer to the largest model in
  the TinyStories paper, better samples, lower loss. Rejected for now: 4–6 hours
  per run on a T4 means one attempt per session and expensive mistakes. A
  reasonable "scale up at the end" target once the pipeline is proven.
- **4L / 4H / 256d (~5M params)** — ~30 min runs, but visibly weak generations
  and reaching 1.5 val loss is uncertain.
