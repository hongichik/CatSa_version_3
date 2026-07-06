"""Giai đoạn 3 — Module 1 phần B: RGCN-style encoder + soft-attention readout.

CatSAEncoder nhận Batch các heterogeneous graph (từ CatSA/graph.py) và trả về
session embedding z_s cho từng phiên trong batch (shape (B, d)).

Kiến trúc:
    - Embedding table item (|I| x d) và category (|C| x d);
      parent DÙNG CHUNG bảng category (parent là category cấp cao hơn).
    - L lớp HeteroConv, mỗi loại quan hệ (sequential, membership, taxonomy
      + các cạnh ngược) có bộ trọng số RIÊNG (tinh thần RGCN: W_r cho từng r).
    - Soft-attention readout theo SR-GNN: chỉ dùng ITEM node (không dùng
      category/parent) để z_s cùng không gian với item embedding.
    - Tied-weight: cùng item embedding table dùng cho input encoder và
      output ranking (scores = z_s @ item_emb.T).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, SAGEConv, global_add_pool
from torch_geometric.utils import softmax as scatter_softmax


class CatSAEncoder(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_cats: int,
        d: int = 100,
        n_layers: int = 2,
        use_taxonomy: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_taxonomy = use_taxonomy
        self.d = d

        self.item_emb = nn.Embedding(n_items, d)
        self.cat_emb = nn.Embedding(n_cats, d)  # dùng chung cho category và parent
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.normal_(self.cat_emb.weight, std=0.1)

        # Mỗi quan hệ một conv riêng (W_r riêng cho từng r — tinh thần RGCN).
        # SAGEConv hỗ trợ cạnh giữa hai loại node khác nhau (bipartite) và có
        # trọng số root riêng (đóng vai trò W_self).
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            rels = {
                ("item", "sequential", "item"): SAGEConv(d, d),
                ("item", "rev_sequential", "item"): SAGEConv(d, d),
                ("item", "membership", "category"): SAGEConv(d, d),
                ("category", "rev_membership", "item"): SAGEConv(d, d),
            }
            if use_taxonomy:
                rels[("category", "taxonomy", "parent")] = SAGEConv(d, d)
                rels[("parent", "rev_taxonomy", "category")] = SAGEConv(d, d)
            self.convs.append(HeteroConv(rels, aggr="sum"))

        self.dropout = nn.Dropout(dropout)

        # Soft-attention readout (SR-GNN): alpha_j = q^T tanh(W1 h_j + W2 h_last)
        self.W1 = nn.Linear(d, d)
        self.W2 = nn.Linear(d, d)
        self.q = nn.Linear(d, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Trả về session embedding z_s shape (B, d) cho B phiên trong batch."""
        x_dict = {
            "item": self.item_emb(batch["item"].node_id),
            "category": self.cat_emb(batch["category"].node_id),
        }
        if self.use_taxonomy:
            x_dict["parent"] = self.cat_emb(batch["parent"].node_id)

        # L lớp message passing
        for conv in self.convs:
            out = conv(x_dict, batch.edge_index_dict)
            # HeteroConv chỉ trả về node type có cạnh đến; giữ nguyên phần còn lại
            x_dict = {k: F.relu(out[k]) if k in out else v for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        # --- Readout: CHỈ từ item node ---
        h_items = x_dict["item"]                      # (tổng số item node, d)
        item_batch = batch["item"].batch              # graph id của từng item node
        ptr = batch["item"].ptr                       # offset node của từng graph

        # Vị trí toàn cục của item cuối mỗi phiên = offset graph + vị trí local
        last_global = ptr[:-1] + batch["item"].last_idx.view(-1)
        h_last = h_items[last_global]                 # (B, d)

        alpha = self.q(torch.tanh(self.W1(h_items) + self.W2(h_last)[item_batch]))
        alpha = scatter_softmax(alpha, item_batch)    # softmax riêng từng graph
        z_s = global_add_pool(alpha * h_items, item_batch)  # (B, d)
        return z_s

    def scores(self, z_s: torch.Tensor) -> torch.Tensor:
        """Điểm dự đoán trên toàn vocabulary (tied-weight với item_emb).

        Lưu ý: dùng embedding TĨNH từ bảng item_emb (không phải embedding sau
        message passing) — cần embedding của MỌI item để full-ranking.
        """
        return z_s @ self.item_emb.weight.T           # (B, |I|)
