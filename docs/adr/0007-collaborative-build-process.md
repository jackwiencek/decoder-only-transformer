# ADR 0007: Collaborative build process — no pre-written model code

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

The primary purpose of this repo is for its owner to learn how a decoder-only
transformer works well enough to defend every line of it. A working artifact
that its author cannot explain has negative value in that setting.

There is a spectrum of possible working arrangements, from "Claude writes
everything and then explains it" to "Claude writes nothing".

The failure mode at one end is obvious: reading code is a far weaker teacher
than writing it. The failure mode at the other end is less obvious but just as
real — days lost to dataloader indexing bugs teach very little per hour spent.

A middle option, "ship documented stubs with failing tests", was considered and
rejected: a stub with a filled-in docstring, exact shape contract, and a test
that pins the answer leaves the interesting decisions already made.

## Decision

Build **section by section, in conversation**.

- Claude does **not** pre-implement or stub model, tokenizer, or training code.
  `src/tinygpt/` starts empty by design.
- For each section: Claude explains the concept and the shape contract, the
  owner writes the implementation, Claude reviews and corrects, then a test is
  added and we move on.
- Claude owns repo scaffolding, environment setup, tests, docs, and review.
- Build order: PyTorch tensor fundamentals and attention from first principles →
  the rest of the model → tokenizer → data pipeline → training loop →
  generation.

Note that this build order is deliberately *not* pipeline order. Attention is
the conceptual core; starting there means the hardest idea gets attacked while
motivation is highest, rather than after a week of tokenizer plumbing.

## Consequences

- The repo takes longer to reach a runnable end-to-end state, and there is no
  test safety net for a component until after it is written.
- Every line is defensible in an interview, which is the actual deliverable.
- Sessions are stateful in a way that matters: `CLAUDE.md` and
  `docs/learning-log.md` carry the thread across context resets, so both must be
  kept current as we go.
- If this becomes too slow in practice, the fallback is to move plumbing (data,
  checkpointing, LR schedule) to Claude while keeping the model owner-written.
  That reversal should be recorded as a superseding ADR.
