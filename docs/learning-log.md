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

## 2026-08-04 — Byte-level BPE tokenizer

Built the tokenizer in `tokenizer.py` from primitives, one function at a time,
with a shape/behaviour test locking in each piece (`tests/test_tokenizer.py`).

The algorithm is two stateless kernels plus a stateful class wrapping them:

- `get_stats(ids)` — count adjacent pairs. `zip(ids, ids[1:])` yields the pairs;
  the *unequal* lengths are what make it stop one short (hence `strict=False`).
- `merge(ids, pair, new_id)` — collapse a pair, non-overlapping, left to right.
  The correctness traps were both in the index loop: forgetting to advance `i`
  (infinite loop) and over-tight bounds dropping the tail. The guard belongs in
  the `if` (`i < len(ids) - 1 and ...`), not the `while`, so the last element is
  still appended; `and` short-circuits so `ids[i+1]` is never read out of range.
- `BPETokenizer.train/encode/decode` — hold the learned `merges` on `self`.

Non-obvious things worth keeping:

- **Merges are hierarchical and ordered.** New ids merge with other new ids, so
  `merges` is a topological sort (parents after children). `decode` exploits this
  to build its `id -> bytes` vocab in one forward pass — no recursion — because a
  child's bytes are always already computed. `encode` exploits the same order in
  reverse: apply the *lowest* merge id present first (`min(..., key=merges.get)`),
  never the highest, or you'd try to merge tokens that don't exist yet.
- **Byte-level means never out-of-vocabulary.** Start from the 256 byte values;
  any string UTF-8-encodes into them, so `decode(encode(x)) == x` holds even for
  unicode never seen in training. Verified by the round-trip test on `"ééé"`.
- `bytes([i])` (a one-byte object) vs `bytes(i)` (i zero-bytes) — the `[i]` is
  load-bearing. And `argmax`/`argmin` over a dict is `max/min(d, key=d.get)`.
- `train` needs a `len(ids) < 2: break` guard: a small corpus can exhaust all
  pairs before reaching `vocab_size`, and `max({})` raises. The 100MB real corpus
  won't hit it, but the smoke-sized inputs do.

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
