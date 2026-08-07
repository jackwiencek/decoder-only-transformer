import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """A single causal self-attention head."""

    def __init__(self, n_embd: int, head_size: int):
        super().__init__()
        self.head_size = head_size
        # Three separate projections, each mapping C -> head_size.
        # bias=False is conventional here: these are pure change-of-basis
        # projections, and the downstream softmax is shift-invariant anyway.
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape                                       # x: (B, T, C)
        q = self.query(x)                                       # (B, T, hs)
        k = self.key(x)                                         # (B, T, hs)
        v = self.value(x)                                       # (B, T, hs)

        scores = q @ k.transpose(-2, -1)                        # (B,T,hs)@(B,hs,T) -> (B, T, T)
        scores = scores * (self.head_size ** -0.5)              # (B, T, T)  scaled
        mask = torch.triu(torch.ones(T, T,device=x.device), diagonal=1)         # (T, T)  1s = future
        scores = scores.masked_fill(mask == 1, float("-inf"))   # (B, T, T)  future -> -inf
        scores = F.softmax(scores, dim=-1)                      # (B, T, T)  rows sum to 1
        out = scores @ v                                     # (B,T,T)@(B,T,hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """nh causal heads run in parallel, concatenated and projected back to C."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head  # so nh heads of size hs concat back to C
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)  # mixes the heads' outputs together

    def forward(self, x: torch.Tensor):
        # x: (B, T, C). Each head returns (B, T, hs); there are nh of them.
        tensors = [h(x) for h in self.heads]        # nh tensors, each (B, T, hs)
        tensor_concat = torch.cat(tensors, dim=-1)  # (B, T, nh*hs) = (B, T, C)
        return self.proj(tensor_concat)             # (B, T, C)


class FeedForward(nn.Module):
    """Position-wise MLP: expand to 4*C, apply a nonlinearity, project back to C."""

    def __init__(self, n_embd: int):
        super().__init__()
        # TODO(human): build the 2-layer MLP as self.net.
        # nn.Sequential(...) chaining: Linear(n_embd -> 4*n_embd), a nonlinearity
        # (nn.ReLU() or nn.GELU()), then Linear(4*n_embd -> n_embd).

    def forward(self, x: torch.Tensor):
        # x: (B, T, C) -> (B, T, C), applied independently per position.
        return self.net(x)


class GPT(nn.Module):
    def __init__(self,vocab_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.unembedding_matrix = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx):
        B, T = idx.shape
        tok_embd = self.token_embedding(idx) # B,T,C

        #position embedding
        pos_list = torch.arange(0, T, device=idx.device) #tensor of size T
        pos_embd = self.position_embedding(pos_list) # T by C matrix
        #add token + position
        x = tok_embd + pos_embd

        logits = self.unembedding_matrix(x)
        return logits
