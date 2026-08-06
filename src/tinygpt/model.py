import torch
import torch.nn as nn


class Head(nn.Module):
    """A single causal self-attention head."""

    def __init__(self, n_embd: int, head_size: int):
        super().__init__()
        # Three separate projections, each mapping C -> head_size.
        # bias=False is conventional here: these are pure change-of-basis
        # projections, and the downstream softmax is shift-invariant anyway.
        # TODO(human): create self.query, self.key, self.value as
        # nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        # x is (B, T, C); we will fill this in next, after the projections.
        pass


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
