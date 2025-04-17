#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class MiniCrystalFormer(nn.Module):
    def __init__(self, vocab_size, dim=256, depth=4, heads=4, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))  # learned positional embedding

        transformer_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=depth)

        self.ln = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        B, T = x.size()
        token_embed = self.token_emb(x)  # (B, T, dim)
        pos_embed = self.pos_emb[:, :T, :]  # (1, T, dim)
        x = token_embed + pos_embed

        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))

        # Transformer decoder input = self
        x = self.transformer(x, memory=x, tgt_mask=causal_mask)
        x = self.ln(x)
        return self.to_logits(x)  # (B, T, vocab_size)
