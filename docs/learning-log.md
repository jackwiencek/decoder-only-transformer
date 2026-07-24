# Learning log

A running journal of concepts as they come up. One entry per session or topic.
The goal is that re-reading this months later reconstructs the understanding,
not just the code.

Format per entry: what the concept is, why it exists (what breaks without it),
and the shape/mechanics detail that was non-obvious at the time.

---

## Shape conventions used throughout

| Symbol | Meaning |
| ------ | ------- |
| `B`    | batch size |
| `T`    | time / sequence position, up to `block_size` |
| `C`    | channels, i.e. `n_embd` (384) |
| `V`    | vocabulary size (8192) |
| `nh`   | number of heads (6) |
| `hs`   | head size, `C // nh` (64) |

---

## 2026-07-22 — Project setup

Decisions made and recorded as ADRs 0001–0007: Colab for compute, hand-written
8k BPE, 6L/6H/384d/ctx-256, W&B tracking, and a collaborative build process with
no pre-written model code.

Two setup details worth remembering:

- **`src/` layout forces `pip install -e .`.** Tests then import the *installed*
  package, the same way Colab will. A flat layout would let tests pass via an
  import path that does not exist on the GPU box.
- **CPU-only PyTorch locally** (`--index-url https://download.pytorch.org/whl/cpu`)
  is ~200MB instead of ~2.5GB. The laptop and Colab environments differ on
  purpose, which is why `device` must come from config rather than being
  hardcoded to `cuda`.

---

## Next up: tensors and attention from first principles

Questions to be able to answer by the end of that session:

1. Why does self-attention need a **causal mask**, and what exactly breaks in
   training if you omit it?
2. In `q @ k.transpose(-2, -1)`, what does each entry of the resulting
   `(B, nh, T, T)` matrix mean?
3. Why divide by `sqrt(head_size)` before the softmax? What happens to the
   softmax as head size grows if you do not?
4. Why does multi-head attention split `C` into `nh` heads of size `hs` instead
   of running `nh` full-width attentions?
