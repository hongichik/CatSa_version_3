"""HGT: relation-aware attention thay SAGEConv."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import HGTConv, Linear

from .common import (
    EmbeddingTables,
    SoftAttentionReadout,
    build_hetero_rels,
    init_x_dict,
)


class HGTEncoder(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_cats: int,
        d: int = 100,
        n_layers: int = 2,
        use_taxonomy: bool = True,
        dropout: float = 0.1,
        n_heads: int = 2,
        **_kwargs,
    ):
        super().__init__()
        self.use_taxonomy = use_taxonomy
        self.d = d
        self.emb = EmbeddingTables(n_items, n_cats, d)

        node_types = ["item", "category"]
        if use_taxonomy:
            node_types.append("parent")
        edge_types = build_hetero_rels(use_taxonomy)
        self.metadata = (node_types, edge_types)

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(HGTConv(d, d, self.metadata, heads=n_heads))
            self.lins.append(Linear(d, d, bias=False))

        self.dropout = nn.Dropout(dropout)
        self.readout = SoftAttentionReadout(d)

    def forward(self, batch: Batch) -> torch.Tensor:
        x_dict = init_x_dict(self.emb, batch, self.use_taxonomy)
        for conv, lin in zip(self.convs, self.lins):
            x_dict = conv(x_dict, batch.edge_index_dict)
            x_dict = {k: lin(F.relu(v)) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        return self.readout(x_dict["item"], batch)

    def scores(self, z_s):
        return self.emb.scores(z_s)
