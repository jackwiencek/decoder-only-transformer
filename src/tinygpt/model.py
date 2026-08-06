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
