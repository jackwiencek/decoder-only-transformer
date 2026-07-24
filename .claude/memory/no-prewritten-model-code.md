---
name: no-prewritten-model-code
description: Jack writes the model/tokenizer code himself; Claude must not pre-implement or stub it
metadata:
  type: feedback
---

On the decoder-only-transformer project, do not pre-implement, stub, or "show
the finished version of" model, tokenizer, or training code. `src/tinygpt/`
starts empty on purpose. Work one section at a time: explain the concept and
shape contract, let Jack write it, review and correct, add a test, move on.

Explicitly rejected during setup: the "ship documented stubs with a failing
test" middle ground.

**Why:** the deliverable is Jack being able to defend every line of this in an
interview, not a working artifact. A stub with a filled-in docstring and a test
that pins the answer leaves the interesting decisions already made.

**How to apply:** when a new component comes up, lead with the concept and the
`(B, T, C)` shape contract, then hand it over. Scaffolding, tests, docs, and
review are Claude's; the model is not. If this pace becomes a problem, the
agreed fallback is to take over plumbing (data, checkpointing, LR schedule)
while the model stays Jack-written — recorded as a superseding ADR.

Related: [[pytorch-learning-goal]], [[colab-only-gpu-access]]
