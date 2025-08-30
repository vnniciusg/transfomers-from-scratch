"""
Decoder components for Transformer
"""

import torch.nn as nn

from .attention import MultiHeadedAttention
from .layers import FeedForward


class DecoderLayer(nn.Module):
    """Single decoder layer with masked self-attention, encoder-decoder attention and feed-forward"""

    def __init__(
        self, *, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.masked_attn = MultiHeadedAttention(d_model, num_heads, dropout)

        self.masked_attn_norm = nn.LayerNorm(d_model)

        self.attn = MultiHeadedAttention(d_model, num_heads, dropout)

        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, *, tgt, enc, tgt_mask=None, enc_mask=None) -> None:
        x = tgt
        x = x + self.masked_attn(x, x, x, mask=tgt_mask)
        x = self.masked_attn_norm(x)
        x = x + self.attn(x, enc, enc, mask=enc_mask)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x


class Decoder(nn.Module):
    """Stack of decoder layers"""

    def __init__(self, *, d_model: int, num_heads: int, num_decoders: int) -> None:
        super().__init__()
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_decoders)]
        )

    def forward(self, tgt, enc, tgt_mask, enc_mask):
        output = tgt
        for layer in self.dec_layers:
            output = layer(output, enc, tgt_mask, enc_mask)
        return output
