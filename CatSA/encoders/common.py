"""Thành phần dùng chung cho mọi session encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as scatter_softmax


ENCODER_TYPES = frozenset({"catsa_plus", "catsa_plus_v2"})
FUSION_TYPES = frozenset({"cross_attn", "gate", "sum"})


class SoftAttentionReadout(nn.Module):
    """SR-GNN readout: alpha_j = q^T tanh(W1 h_j + W2 h_last)."""

    def __init__(self, d: int):
        super().__init__()
        self.W1 = nn.Linear(d, d)
        self.W2 = nn.Linear(d, d)
        self.q = nn.Linear(d, 1)

    def forward(self, h_items: torch.Tensor, batch: Batch) -> torch.Tensor:
        item_batch = batch["item"].batch
        ptr = batch["item"].ptr
        last_idx = batch["item"].last_idx.view(-1)
        num_graphs = ptr.numel() - 1
        assert last_idx.numel() == num_graphs, (
            f"last_idx phải có đúng 1 giá trị mỗi graph trong batch "
            f"(num_graphs={num_graphs}, last_idx.numel()={last_idx.numel()})"
        )
        last_global = ptr[:-1] + last_idx
        h_last = h_items[last_global]
        alpha = self.q(torch.tanh(self.W1(h_items) + self.W2(h_last)[item_batch]))
        alpha = scatter_softmax(alpha, item_batch)
        return global_add_pool(alpha * h_items, item_batch)


class MultiInterestReadout(nn.Module):
    """Readout đa-interest (ComiRec/Atten-Mixer style) — thay 1 vector soft-
    attention duy nhất bằng K "interest head" pool song song rồi gộp lại
    bằng 1 tầng attention thứ hai (hierarchical: item-level → interest-level).

    Mục đích: với phiên dài, nhiều intent khác nhau dễ bị loãng vào 1 vector
    duy nhất (SoftAttentionReadout gốc). K head cho phép mỗi head bắt 1 khía
    cạnh khác nhau của phiên, rồi tầng thứ 2 quyết định trộn theo ngữ cảnh
    item cuối — không thêm nhánh/loại quan hệ mới vào Module 1, chỉ đổi CÁCH
    tổng hợp embedding item sau message passing.
    """

    def __init__(self, d: int, n_interests: int = 4):
        super().__init__()
        self.n_interests = n_interests
        self.W1 = nn.Linear(d, d)
        self.W2 = nn.Linear(d, d)
        self.q = nn.Linear(d, n_interests)
        self.Wc = nn.Linear(d, d)
        self.Wc2 = nn.Linear(d, d)
        self.qc = nn.Linear(d, 1)

    def forward(self, h_items: torch.Tensor, batch: Batch) -> torch.Tensor:
        item_batch = batch["item"].batch
        ptr = batch["item"].ptr
        last_global = ptr[:-1] + batch["item"].last_idx.view(-1)
        h_last = h_items[last_global]

        scores = self.q(torch.tanh(self.W1(h_items) + self.W2(h_last)[item_batch]))
        alpha = torch.cat(
            [scatter_softmax(scores[:, k : k + 1], item_batch) for k in range(self.n_interests)],
            dim=-1,
        )
        z_k = torch.stack(
            [global_add_pool(alpha[:, k : k + 1] * h_items, item_batch) for k in range(self.n_interests)],
            dim=1,
        )

        beta = F.softmax(
            self.qc(torch.tanh(self.Wc(z_k) + self.Wc2(h_last).unsqueeze(1))), dim=1,
        )
        return (beta * z_k).sum(dim=1)


class EmbeddingTables(nn.Module):
    """Bảng embedding item/category dùng chung (tied-weight ranking)."""

    def __init__(self, n_items: int, n_cats: int, d: int):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, d)
        self.cat_emb = nn.Embedding(n_cats, d)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.normal_(self.cat_emb.weight, std=0.1)

    def scores(self, z_s: torch.Tensor) -> torch.Tensor:
        return z_s @ self.item_emb.weight.T


def fuse_vectors(
    a: torch.Tensor, b: torch.Tensor, fusion_type: str, gate: nn.Module | None = None,
) -> torch.Tensor:
    if fusion_type == "sum":
        return a + b
    if fusion_type == "gate":
        assert gate is not None
        g = torch.sigmoid(gate(torch.cat([a, b], dim=-1)))
        return g * a + (1 - g) * b
    # cross_attn: a là query, b là context (single vector)
    scale = a.size(-1) ** 0.5
    w = torch.sum(a * b, dim=-1, keepdim=True) / scale
    attn = torch.sigmoid(w)
    return a + attn * b


def build_hetero_rels(use_taxonomy: bool) -> list[tuple[str, str, str]]:
    rels = [
        ("item", "sequential", "item"),
        ("item", "rev_sequential", "item"),
        ("item", "membership", "category"),
        ("category", "rev_membership", "item"),
    ]
    if use_taxonomy:
        rels.extend([
            ("category", "taxonomy", "parent"),
            ("parent", "rev_taxonomy", "category"),
        ])
    return rels


def init_x_dict(
    emb: EmbeddingTables, batch: Batch, use_taxonomy: bool,
) -> dict[str, torch.Tensor]:
    x_dict = {
        "item": emb.item_emb(batch["item"].node_id),
        "category": emb.cat_emb(batch["category"].node_id),
    }
    # Check "parent" in batch.node_types (không chỉ dựa vào use_taxonomy) —
    # nhất quán với graph.py has_taxonomy (finding G1/G4): node type "parent"
    # chỉ tồn tại khi graph builder thực sự tạo ra nó.
    if use_taxonomy and "parent" in batch.node_types:
        x_dict["parent"] = emb.cat_emb(batch["parent"].node_id)
    return x_dict


def apply_hetero_layers(
    convs: nn.ModuleList,
    x_dict: dict[str, torch.Tensor],
    batch: Batch,
    dropout: nn.Dropout,
) -> dict[str, torch.Tensor]:
    for conv in convs:
        out = conv(x_dict, batch.edge_index_dict)
        # Dropout chỉ áp lên node type VỪA được cập nhật (finding M4) — tránh
        # áp dropout chồng lên representation cũ (stale) của type không nhận
        # message ở layer này.
        x_dict = {
            k: dropout(F.relu(out[k])) if k in out else v
            for k, v in x_dict.items()
        }
    return x_dict
