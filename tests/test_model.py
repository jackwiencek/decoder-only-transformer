"""Shape/behaviour tests for the GPT model, built one component at a time."""

import torch

from tinygpt.model import GPT


def test_forward_maps_BT_ids_to_BTV_logits():
    # Small, fast config - we are testing shapes and wiring, not learning.
    vocab_size, n_embd, block_size = 32, 16, 8
    B, T = 4, 5  # T must be <= block_size, or position lookup goes out of range

    model = GPT(vocab_size, n_embd, block_size)

    # A batch of random token ids in [0, vocab_size). Integer dtype: these are
    # indices into the embedding table, not floats.
    idx = torch.randint(0, vocab_size, (B, T))

    logits = model(idx)  # calls forward() via nn.Module.__call__

    # TODO(human): assert the output shape is (B, T, vocab_size).
    # Use logits.shape and compare against the expected tuple.
    assert logits.shape == (B, T, vocab_size)


