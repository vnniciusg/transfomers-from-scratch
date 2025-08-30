"""
Encoder components for Transformer
"""

import torch.nn as nn

from .attention import MultiHeadedAttention
from .layers import FeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self, *, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1
    ) -> None:
        """Single encode layer with self.attention and feed-forward"""
        super().__init__()

        self.attn = MultiHeadedAttention(d_model, num_heads, dropout)

        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = src
        x = x + self.attn(x, x, x, mask=src_mask)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x


class Encoder(nn.Module):
    """Stack of encoder layers"""

    def __init__(self, *, d_model: int, num_heads: int, num_encoders: int) -> None:
        super().__init__()
        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(d_model=d_model, num_heads=num_heads)
                for _ in range(num_encoders)
            ]
        )

    def forward(self, src, src_mask):
        output = src
        for layer in self.enc_layers:
            output = layer(output, src_mask)

        return output
