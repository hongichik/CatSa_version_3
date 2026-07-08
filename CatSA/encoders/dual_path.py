"""Dual-path: sequential GNN + taxonomy GNN, fusion có thể cấu hình."""

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
    fuse_vectors,
    init_x_dict,
)


class DualPathEncoder(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_cats: int,
        d: int = 100,
        n_layers: int = 2,
        use_taxonomy: bool = True,
        dropout: float = 0.1,
        fusion_type: str = "cross_attn",
        **_kwargs,
    ):
        super().__init__()
        self.use_taxonomy = use_taxonomy
        self.d = d
        self.fusion_type = fusion_type
        self.emb = EmbeddingTables(n_items, n_cats, d)

        self.seq_convs = nn.ModuleList()
        self.tax_convs = nn.ModuleList()
        for _ in range(n_layers):
            seq_rels = {
                ("item", "sequential", "item"): SAGEConv(d, d),
                ("item", "rev_sequential", "item"): SAGEConv(d, d),
            }
            tax_rels = {
                ("item", "membership", "category"): SAGEConv(d, d),
                ("category", "rev_membership", "item"): SAGEConv(d, d),
            }
            if use_taxonomy:
                tax_rels[("category", "taxonomy", "parent")] = SAGEConv(d, d)
                tax_rels[("parent", "rev_taxonomy", "category")] = SAGEConv(d, d)
            self.seq_convs.append(HeteroConv(seq_rels, aggr="sum"))
            self.tax_convs.append(HeteroConv(tax_rels, aggr="sum"))

        self.dropout = nn.Dropout(dropout)
        self.readout_seq = SoftAttentionReadout(d)
        self.readout_tax = SoftAttentionReadout(d)
        self.gate = nn.Linear(d * 2, d) if fusion_type == "gate" else None
        self.out_proj = nn.Linear(d, d)

    def forward(self, batch: Batch) -> torch.Tensor:
        x0 = init_x_dict(self.emb, batch, self.use_taxonomy)
        x_seq = {k: v.clone() for k, v in x0.items()}
        x_tax = {k: v.clone() for k, v in x0.items()}

        for seq_conv, tax_conv in zip(self.seq_convs, self.tax_convs):
            out_s = seq_conv(x_seq, batch.edge_index_dict)
            x_seq = {k: F.relu(out_s[k]) if k in out_s else v for k, v in x_seq.items()}
            x_seq = {k: self.dropout(v) for k, v in x_seq.items()}

            out_t = tax_conv(x_tax, batch.edge_index_dict)
            x_tax = {k: F.relu(out_t[k]) if k in out_t else v for k, v in x_tax.items()}
            x_tax = {k: self.dropout(v) for k, v in x_tax.items()}

        z_seq = self.readout_seq(x_seq["item"], batch)
        z_tax = self.readout_tax(x_tax["item"], batch)
        z = fuse_vectors(z_seq, z_tax, self.fusion_type, self.gate)
        return self.out_proj(z)

    def scores(self, z_s):
        return self.emb.scores(z_s)
