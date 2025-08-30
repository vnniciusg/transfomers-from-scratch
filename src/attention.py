"""
Attention mechanisms for Transformer
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention mechanism with query, key, value projections"""

    def __init__(self, *, d_model: int, output_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, output_size)
        self.key = nn.Linear(d_model, output_size)
        self.value = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        dim_k = key.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float("inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        outputs = torch.bmm(weights, value)

        return outputs


class MultiHeadedAttention(nn.Module):
    """Self-attention mechanism with query, key, value projections"""

    def __init__(self, *, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_output_size = self.d_model // self.num_heads

        self.attentions = nn.ModuleList(
            [
                SelfAttention(
                    d_model=d_model, output_size=self.attn_output_size, dropout=dropout
                )
                for _ in range(self.num_heads)
            ]
        )

        self.output = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v, mask=None):
        x = torch.cat([layer(q, k, v, mask) for layer in self.attentions], dim=1)
        x = self.output(x)
        return x
