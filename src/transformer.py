"""
Main Transformer model implementation
"""

from math import sqrt

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .layers import PositionalEncoding


class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks"""

    def __init__(
        self,
        *,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoders: int = 6,
        num_decoders: int = 6,
        src_vocab_size: int = 100_000,
        tgt_vocab_size: int = 100_000,
        max_len: int = 5_000,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        self.encoder = Encoder(d_model, num_heads=num_heads, num_encoders=num_encoders)
        self.decoder = Decoder(d_model, num_heads=num_heads, num_decoders=num_decoders)

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.output = nn.Linear(d_model, tgt_vocab_size)

    def create_pad_mask(self, seq, pad_token):
        return (seq != pad_token).unsqueeze(1).unsqueeze(1)

    def create_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt, src_pad_token: 0, tgt_pad_token=0):
        src_mask = self.create_pad_mask(src, src_pad_token)
        tgt_mask = self.create_pad_mask(src, tgt_pad_token)
        subsequent_mask = self.create_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_mask = tgt_mask & subsequent_mask

        src_emb = self.src_embedding(src) + sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(src) + sqrt(self.d_model)

        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)

        enc_out = self.encoder(src_emb, src_mask)
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask, src_mask)

        return self.output(dec_out)
