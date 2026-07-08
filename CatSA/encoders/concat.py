"""Baseline: item_emb + category_emb, sequential GNN trên item."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv

from .common import EmbeddingTables, SoftAttentionReadout, apply_hetero_layers


class ConcatEncoder(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_cats: int,
        item2cat: dict[int, int],
        d: int = 100,
        n_layers: int = 2,
        use_taxonomy: bool = True,
        dropout: float = 0.1,
        **_kwargs,
    ):
        super().__init__()
        self.d = d
        self.emb = EmbeddingTables(n_items, n_cats, d)
        self.register_buffer(
            "item_cat_idx",
            torch.tensor([item2cat.get(i, 0) for i in range(n_items)], dtype=torch.long),
        )
        self.proj = nn.Linear(d, d)
        self.convs = nn.ModuleList(
            [SAGEConv(d, d) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = SoftAttentionReadout(d)

    def forward(self, batch: Batch) -> torch.Tensor:
        node_ids = batch["item"].node_id
        h = self.emb.item_emb(node_ids) + self.emb.cat_emb(self.item_cat_idx[node_ids])
        h = self.proj(h)
        edge_index = batch["item", "sequential", "item"].edge_index
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        return self.readout(h, batch)

    def scores(self, z_s):
        return self.emb.scores(z_s)
