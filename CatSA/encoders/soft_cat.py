"""Soft category membership: item embedding + weighted category context."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, SAGEConv

from .common import (
    EmbeddingTables,
    SoftAttentionReadout,
    apply_hetero_layers,
    build_hetero_rels,
)


class SoftCatEncoder(nn.Module):
    """Học trọng số mềm item→category, kết hợp RGCN message passing."""

    def __init__(
        self,
        n_items: int,
        n_cats: int,
        d: int = 100,
        n_layers: int = 2,
        use_taxonomy: bool = True,
        dropout: float = 0.1,
        **_kwargs,
    ):
        super().__init__()
        self.use_taxonomy = use_taxonomy
        self.d = d
        self.emb = EmbeddingTables(n_items, n_cats, d)
        self.cat_logits = nn.Linear(d, n_cats, bias=False)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            rels = {r: SAGEConv(d, d) for r in build_hetero_rels(use_taxonomy)}
            self.convs.append(HeteroConv(rels, aggr="sum"))
        self.dropout = nn.Dropout(dropout)
        self.readout = SoftAttentionReadout(d)

    def forward(self, batch: Batch) -> torch.Tensor:
        item_ids = batch["item"].node_id
        h_item = self.emb.item_emb(item_ids)
        weights = F.softmax(self.cat_logits(h_item), dim=-1)
        h_cat_ctx = weights @ self.emb.cat_emb.weight
        x_dict = {
            "item": h_item + h_cat_ctx,
            "category": self.emb.cat_emb(batch["category"].node_id),
        }
        if self.use_taxonomy:
            x_dict["parent"] = self.emb.cat_emb(batch["parent"].node_id)
        x_dict = apply_hetero_layers(self.convs, x_dict, batch, self.dropout)
        return self.readout(x_dict["item"], batch)

    def scores(self, z_s):
        return self.emb.scores(z_s)
