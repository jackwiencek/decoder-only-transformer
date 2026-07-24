# ADR 0006: Experiment tracking with Weights & Biases

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

Training happens on ephemeral Colab VMs (ADR 0002) that can disconnect without
warning. Anything written only to the VM's local disk — including scrollback in
the notebook output cell — is lost when the session dies.

Learning also depends on *comparing* runs: this LR against that one, tied
embeddings against untied. That comparison has to survive across sessions and
across weeks.

## Decision

Use **Weights & Biases**, logging train loss, validation loss, learning rate,
gradient norm, tokens/second, and the full run config.

Logging is wrapped behind a thin internal interface so that a missing
`WANDB_API_KEY` degrades to a **no-op logger** rather than an error. CPU smoke
runs on the laptop must not require network access or an account.

## Consequences

- Loss curves survive disconnects, and runs remain comparable months later.
- Produces a shareable public run page — useful evidence to link alongside the
  project.
- Adds a network dependency and an account to the real training path. Mitigated
  by the no-op fallback.
- The wrapper is a small amount of indirection, but it also means swapping to
  TensorBoard or CSV later touches one file rather than the training loop.

## Alternatives considered

- **TensorBoard** — no account, no network, but event files die with the VM
  unless synced to Drive, and cross-run comparison is clumsier.
- **CSV + matplotlib** — maximally transparent and version-controllable, but
  comparing six runs becomes manual work.
