"""
Auxiliary layers for Transformers
"""

from math import log

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model, *, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class PositionalEncoding(nn.Module):
    """Positional encoding using sinusoidal functions"""

    def __init__(self, d_model, max_len: int = 5_000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]
