# ADR 0003: Hand-written BPE tokenizer with an 8192-token vocabulary

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

Two questions, entangled: *who writes the tokenizer* and *how large is the
vocabulary*.

Vocabulary size is not a cosmetic choice at small scale. The token embedding
table has `V x C` parameters, and the output projection has another `C x V`. At
`C = 384`:

| Vocab `V` | Embedding params | Share of a ~13M model |
| --------- | ---------------- | --------------------- |
| 50,257 (GPT-2)  | ~19.3M (untied) | dominates the model entirely |
| 8,192           | ~3.1M (untied)  | ~24%, ~12% if tied |

TinyStories is deliberately written with the vocabulary of a 3–4 year old. Most
of GPT-2's 50k vocabulary would be dead weight — embedding rows that receive
almost no gradient signal while consuming memory and optimizer state.

Separately, the project's stated goal includes "building tokenization"
pipelines. Calling `tiktoken.get_encoding("gpt2")` does not constitute building
one.

## Decision

Implement **byte-level BPE from scratch in pure Python** — training (pair
counting, iterative merges), `encode`, and `decode` — with a vocabulary of
**8192** tokens.

The tokenizer is trained on a ~100MB slice of the TinyStories training text, not
the full corpus. Merge statistics converge long before the full dataset is
consumed, and this keeps the one-time training cost tolerable in pure Python.
The learned merges and vocabulary are cached to disk so this happens once.

Byte-level (starting from the 256 possible bytes) rather than character-level so
the tokenizer can never encounter an out-of-vocabulary input.

## Consequences

- The embedding table is small enough that the transformer blocks — the part
  actually being learned — hold the majority of the parameters.
- Cross-entropy values are **not comparable** to published TinyStories numbers
  that use the GPT-Neo/GPT-2 tokenizer. See ADR 0005.
- Pure-Python BPE training is slow (minutes, not seconds). Acceptable as a
  cached one-time cost; if it becomes painful, the escape hatch is the Rust
  `tokenizers` library with an identical on-disk format.
- Encoding the full corpus is also pure Python and will need to be done once,
  with the token array cached as a flat `uint16` file (8192 < 65536, so
  `uint16` is exactly wide enough — a deliberate consequence of this vocab size).

## Alternatives considered

- **`tiktoken` GPT-2 BPE** — zero work, directly comparable to nanoGPT.
  Rejected: two-thirds of the parameter budget would go to an unused vocabulary.
- **HuggingFace `tokenizers`** — fast and correct, but you configure a pipeline
  rather than implement the merge algorithm. Weaker learning payoff.
- **Character-level** — trivial, but ~4x longer sequences for the same text and
  a much weaker claim to a "tokenization pipeline".
