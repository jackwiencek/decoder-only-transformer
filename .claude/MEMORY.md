# Project memory index

One line per memory. The harness loads this file each session. Memory bodies
live in `.claude/memory/`. This directory is the source of truth and is
committed to git; see `.claude/README.md` for how it is bridged to the harness's
per-machine memory path.

- [PyTorch learning goal](memory/pytorch-learning-goal.md) — why this repo exists and what the 1.5 loss claim has to survive
- [No pre-written model code](memory/no-prewritten-model-code.md) — Jack writes the model; Claude scaffolds, explains, reviews
- [Colab-only GPU access](memory/colab-only-gpu-access.md) — CPU laptop vs CUDA Colab, and what that breaks if forgotten
