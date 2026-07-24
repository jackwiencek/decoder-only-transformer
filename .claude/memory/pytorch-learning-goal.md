---
name: pytorch-learning-goal
description: Jack is learning PyTorch from scratch via this project; it backs a dated resume bullet
metadata:
  type: project
---

Jack is new to PyTorch and is using the decoder-only-transformer repo to learn
it hands-on. The project exists to back a resume bullet dated June–July 2026
claiming a hand-built decoder-only transformer trained on TinyStories reaching
**1.5 validation cross-entropy**.

**Why:** this shapes both the teaching level and what counts as done. Assume no
prior PyTorch fluency — tensor shapes, broadcasting, and autograd mechanics need
explaining, not just transformer theory. And the claimed number has to be real
and honestly measured, because he may be asked about it in an interview.

**How to apply:** teach shapes before syntax. Keep the resume claim falsifiable —
the measurement convention (nats/token, this repo's 8k BPE, fixed deterministic
eval batches, final checkpoint not the minimum) is fixed in ADR 0005 and the
README must state the tokenizer alongside any loss number. Build order starts
with tensors and attention from first principles, not pipeline order.

Related: [[no-prewritten-model-code]], [[colab-only-gpu-access]]
