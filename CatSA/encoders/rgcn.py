"""RGCN-style encoder (CatSA gốc)."""

from __future__ import annotations

from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, SAGEConv

from .common import (
    EmbeddingTables,
    SoftAttentionReadout,
    apply_hetero_layers,
    build_hetero_rels,
    init_x_dict,
)


class RGCNEncoder(nn.Module):
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
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            rels = {
                r: SAGEConv(d, d) for r in build_hetero_rels(use_taxonomy)
            }
            self.convs.append(HeteroConv(rels, aggr="sum"))
        self.dropout = nn.Dropout(dropout)
        self.readout = SoftAttentionReadout(d)

    def forward(self, batch: Batch) -> torch.Tensor:
        x_dict = init_x_dict(self.emb, batch, self.use_taxonomy)
        x_dict = apply_hetero_layers(self.convs, x_dict, batch, self.dropout)
        return self.readout(x_dict["item"], batch)

    def scores(self, z_s):
        return self.emb.scores(z_s)
