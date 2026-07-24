"""tinygpt - a decoder-only transformer language model built from PyTorch primitives.

Modules are added as we build them together, in this order:

    tokenizer.py   byte-pair encoding: train, encode, decode
    data.py        TinyStories download, tokenization to a flat uint16 array, batching
    model.py       embeddings, causal self-attention, MLP, block, GPT
    train.py       training loop, optimizer, LR schedule, eval, checkpointing
    generate.py    autoregressive sampling

Nothing is stubbed ahead of time on purpose - see docs/adr/0007.
"""

__version__ = "0.1.0"
