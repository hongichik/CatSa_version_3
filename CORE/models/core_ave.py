"""CORE-ave — session encoder bằng weighted average (RCE + RDM)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class COREave(nn.Module):
    def __init__(
        self,
        n_items: int,
        embedding_size: int,
        sess_dropout: float,
        item_dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.temperature = temperature
        self.sess_dropout = nn.Dropout(sess_dropout)
        self.item_dropout = nn.Dropout(item_dropout)
        # index 0 = padding; item thật 1..n_items
        self.item_embedding = nn.Embedding(n_items + 1, embedding_size, padding_idx=0)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _ave_weights(self, item_seq: torch.Tensor) -> torch.Tensor:
        mask = item_seq.gt(0)
        alpha = mask.to(torch.float) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return alpha.unsqueeze(-1)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        alpha = self._ave_weights(item_seq)
        seq_output = torch.sum(alpha * x, dim=1)
        return F.normalize(seq_output, dim=-1)

    def compute_logits(self, item_seq: torch.Tensor) -> torch.Tensor:
        seq_output = self.forward(item_seq)
        all_item_emb = self.item_dropout(self.item_embedding.weight)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        return torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature

    def loss(self, item_seq: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.compute_logits(item_seq)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def predict_scores(self, item_seq: torch.Tensor) -> torch.Tensor:
        seq_output = self.forward(item_seq)
        test_item_emb = F.normalize(self.item_embedding.weight, dim=-1)
        return torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
