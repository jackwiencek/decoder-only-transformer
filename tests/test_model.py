"""Shape/behaviour tests for the GPT model, built one component at a time."""

import torch

from tinygpt.model import GPT, Head, MultiHeadAttention


def test_head_maps_BTC_to_BThs():
    B, T, C, hs = 2, 5, 8, 4
    head = Head(n_embd=C, head_size=hs)
    x = torch.randn(B, T, C)
    out = head(x)
    assert out.shape == (B, T, hs)


def test_head_is_causal():
    # The defining property of causal attention: a token's output depends only
    # on itself and earlier positions - NEVER on the future. We verify it by
    # perturbing only the LAST time step and checking what does / doesn't move.
    torch.manual_seed(0)
    B, T, C, hs = 2, 6, 8, 8
    head = Head(n_embd=C, head_size=hs)

    x = torch.randn(B, T, C)
    out_before = head(x)

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] = torch.randn(B, C)  # change ONLY the final token
    out_after = head(x_perturbed)

    # TODO(human): write two assertions that encode causality:
    #   1. every position EXCEPT the last is unchanged by the perturbation
    assert torch.allclose(out_before[:, :-1, :],out_after[:,:-1,:])
    #   2. the last position IS changed by it
    # Use torch.allclose(a, b) -> bool, and slicing [:, :-1, :] / [:, -1, :].
    assert not torch.allclose(out_before[:,-1,:],out_after[:,-1,:])


def test_multihead_preserves_BTC_shape():
    # nh heads of size C/nh, concatenated + projected, must return to (B, T, C)
    # so the block is stackable.
    B, T, C, nh = 2, 5, 12, 3  # C divisible by nh
    mha = MultiHeadAttention(n_embd=C, n_head=nh)
    x = torch.randn(B, T, C)
    out = mha(x)
    assert out.shape == (B, T, C)


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


