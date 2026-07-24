# Decoder-Only Transformer Language Model

A GPT-style decoder-only transformer implemented from PyTorch primitives and
trained for next-token prediction on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

No `nn.TransformerDecoder`, no HuggingFace `transformers`. Token embeddings,
causal multi-head self-attention, feed-forward networks, residual connections,
layer normalization, and autoregressive generation are all hand-written.

## Status

🚧 **In progress.** Repo scaffolding and environment are set up; model
components are being built one at a time. See
[docs/learning-log.md](docs/learning-log.md) for progress.

## Results

_To be filled in once the first full run completes._

| Metric | Value |
| ------ | ----- |
| Validation cross-entropy | — (target: ≤ 1.5 nats/token) |
| Parameters | ~13M |
| Tokenizer | hand-written byte-level BPE, 8192 vocab |
| Config | [configs/base.yaml](configs/base.yaml) |

> **On comparing this number:** cross-entropy is per *token*, so it is only
> meaningful relative to a tokenizer. These numbers use this repo's own 8k BPE
> vocabulary and are **not** directly comparable to published TinyStories or
> nanoGPT results, which use GPT-2/GPT-Neo's 50k vocabulary. The full
> measurement convention is in [docs/adr/0005](docs/adr/0005-loss-measurement-convention.md).

## Architecture

| Hyperparameter | Value |
| -------------- | ----- |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dim (`n_embd`) | 384 |
| Head size | 64 |
| Context length (`block_size`) | 256 |
| Vocab size | 8192 |

Rationale for each choice is in [docs/adr/](docs/adr/).

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m pytest
```

CPU-only PyTorch is intentional — the development laptop has no GPU. Training
runs on Google Colab, which supplies its own CUDA build. See
[docs/adr/0002](docs/adr/0002-google-colab-as-compute-target.md).

## Training

Training happens in Colab via [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb),
which clones this repo, installs it, and calls into the package. The notebook
contains no model code — anything that runs on the GPU must be committed first.

## Layout

```
src/tinygpt/          the package
configs/              YAML run configs (smoke = CPU, base = Colab)
tests/                pytest
notebooks/            thin Colab driver
docs/adr/             architecture decision records
docs/learning-log.md  running journal of concepts
scratch/              throwaway experiments
```
