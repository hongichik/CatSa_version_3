"""Transformer encoder dùng bởi CORE-trm (tách khỏi RecBole)."""

from __future__ import annotations

import math

import torch
from torch import nn


class _TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        n_heads: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size phải chia hết cho n_heads")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        act = hidden_act.lower()
        if act == "gelu":
            self.intermediate = nn.GELU()
        elif act == "relu":
            self.intermediate = nn.ReLU()
        else:
            raise ValueError(f"hidden_act không hỗ trợ: {hidden_act}")

        self.up = nn.Linear(hidden_size, inner_size)
        self.down = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # mask: (B, L, L) hoặc (B, 1, L, L) — KHÔNG unsqueeze thêm nếu đã 4D
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1, dtype=torch.float)
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        x = self.norm1(x + self.dropout(self.out(ctx)))

        hidden = self.up(x)
        hidden = self.intermediate(hidden)
        hidden = self.down(hidden)
        return self.norm2(x + self.dropout(hidden))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerLayer(
                hidden_size, inner_size, n_heads,
                hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = False,
    ) -> list[torch.Tensor] | torch.Tensor:
        all_outputs: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_outputs.append(hidden_states)
        if output_all_encoded_layers:
            return all_outputs
        return hidden_states
