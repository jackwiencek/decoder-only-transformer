# ADR 0005: Loss measurement convention

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

The project's headline result is "1.5 validation cross-entropy loss". That
number is meaningless without stating how it was measured, and it is easy to
report a flattering version of it by accident.

Three ways the same model can produce different "validation loss" numbers:

1. **Tokenizer.** Cross-entropy is per *token*, and tokens differ between
   tokenizers. A smaller vocabulary generally yields lower per-token loss while
   producing more tokens per story — the two effects trade off, and the numbers
   are not interchangeable. Quoting 1.5 against an 8k BPE vocabulary is a
   different claim than 1.5 against GPT-2's 50k.
2. **Log base.** `nats/token` (natural log, what `F.cross_entropy` returns) vs
   `bits/token` (log base 2). These differ by a factor of ~1.44.
3. **Sampling noise.** Loss on one batch of validation data is noticeably
   noisier than loss over a fixed, sufficiently large held-out set. Reporting
   the minimum over training steps is also a subtle form of cheating.

## Decision

The reported metric is:

> **Mean cross-entropy in nats per token**, over the **held-out TinyStories
> validation split**, tokenized with **this repo's 8192-token BPE tokenizer**,
> averaged over a **fixed number of deterministic batches** (fixed seed, fixed
> ordering), evaluated with the model in `eval()` mode under `torch.no_grad()`.

Rules:

- The validation split is never used for training and never used to select
  merges during tokenizer training.
- The headline number is the loss at the **final** checkpoint, or at an
  explicitly-chosen early-stopping checkpoint — not the minimum ever observed.
- Every reported loss is accompanied by the tokenizer and config it was measured
  under. `README.md` states both.

## Consequences

- Numbers in this repo are internally comparable across runs and honestly
  reportable, but **not** directly comparable to the TinyStories paper or to
  nanoGPT results, which use different tokenizers. The README says so.
- Requires a deterministic eval path — fixed seed, fixed batch count — which
  costs a little extra machinery in the eval loop.
- If the tokenizer is ever retrained with different merges, all historical loss
  numbers are invalidated and must be re-measured. Tokenizer artifacts are
  therefore versioned alongside checkpoints.
