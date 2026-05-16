"""Time-series transformer over state-vector sequences.

The model receives `(batch, sequence_length, state_dim)` tensors and outputs logits
over outcome buckets (or a single binary logit). A learned [CLS] token is prepended;
its final hidden state feeds the prediction head. Positional encoding is sinusoidal
(simple and fast; switch to RoPE if a market shows benefit).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from quant.config import MarketConfig


class SinusoidalPosEnc(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class StateVectorTransformer(nn.Module):
    def __init__(self, market: MarketConfig):
        super().__init__()
        cfg = market.transformer
        self.input_proj = nn.Linear(market.state_vector.dim, cfg.model_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.model_dim))
        self.pos = SinusoidalPosEnc(cfg.model_dim, max_len=market.sequence_length + 2)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.heads,
            dim_feedforward=cfg.model_dim * cfg.ffn_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers)
        self.norm = nn.LayerNorm(cfg.model_dim)
        if market.label.kind == "binary":
            self.head = nn.Linear(cfg.model_dim, 1)
        else:
            self.head = nn.Linear(cfg.model_dim, market.label.num_classes)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D_state) -> project to (B, L, model_dim)
        h = self.input_proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h[:, 0, :])
        return self.head(h)
