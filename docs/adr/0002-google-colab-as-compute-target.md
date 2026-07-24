# ADR 0002: Google Colab as the compute target

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

Development happens on a Windows laptop with no dedicated GPU. Training a ~13M
parameter transformer on ~500M tokens of TinyStories is not feasible on CPU —
it would take days rather than hours.

Compute must therefore be remote. The candidates were Google Colab, rented
instances (RunPod / Vast.ai), serverless GPU (Modal), and Kaggle Notebooks.

## Decision

Use **Google Colab** (free T4, upgrading to Pro if session limits bite) as the
training environment.

This forces three structural requirements on the repo:

1. The project is a **pip-installable package** (`src/` layout,
   `pyproject.toml`). The Colab notebook clones the repo and runs
   `pip install -e .`; it contains no model code of its own.
2. **Checkpoints are written to Google Drive**, not the VM's local disk, because
   the VM is destroyed when the session ends.
3. Training must be **resumable from a checkpoint**, because free-tier sessions
   disconnect without warning.

## Consequences

- Zero infrastructure cost and near-zero setup friction; the fast iteration loop
  matters more for learning than raw throughput does.
- Anything that must run on the GPU has to be **committed and pushed first** —
  no testing uncommitted local edits on Colab. This is friction, but it enforces
  a clean repo.
- Session caps (12h, with idle disconnects on free tier) mean a single run
  should target 1–2 hours. This directly motivates the model scale in ADR 0004.
- The notebook is a thin driver, so the repo stays reviewable as source code
  rather than as a wall of notebook JSON.

## Alternatives considered

- **RunPod / Vast.ai** — a real SSH box with a persistent volume and VS Code
  Remote. More production-like, but you pay for idle time and the setup tax
  (keys, images, rsync) is real. Worth revisiting if runs outgrow Colab.
- **Modal** — elegant (`@app.function(gpu="A10G")`), keeps code local. Rejected
  because it couples the training entrypoint to a vendor API, and the
  edit → cold start → debug loop is slower than a live notebook.
- **Kaggle** — longer free sessions than Colab, but a clunkier environment.
