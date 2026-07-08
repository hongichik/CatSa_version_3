"""CatSA v2 — Multi-Granularity + CORE scoring + category-aware rerank.

Kết hợp ý tưởng từ 3 baseline trong repo:
- CORE: causal Transformer + L2-normalize + temperature-scaled dot product
- MSGIFSR: tách score in-session / in-category / novel (extra rerank)
- CatSA dual_path + transition: GNN 2 nhánh + GRU trên chuỗi category
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, SAGEConv

from CORE.models.core_trm import TransNet

from .common import (
    EmbeddingTables,
    SoftAttentionReadout,
    fuse_vectors,
    init_x_dict,
)
from .transition import _session_cat_tensor


def _sessions_to_trm_tensor(
    sessions: list[list[int]], max_len: int, device: torch.device,
) -> torch.Tensor:
    """Pad session → (B, L), id 1..n_items (0 = PAD) cho TransNet."""
    out = torch.zeros(len(sessions), max_len, dtype=torch.long, device=device)
    for i, s in enumerate(sessions):
        sl = min(len(s), max_len)
        if sl > 0:
            out[i, :sl] = torch.tensor([it + 1 for it in s[:sl]], device=device)
    return out


def _build_cat_pool_mask(
    sessions: list[list[int]],
    item2cat: dict[int, int],
    cat2items: dict[int, list[int]],
    n_items: int,
    device: torch.device,
) -> torch.Tensor:
    """(B, n_items) — True nếu item thuộc category đã xuất hiện trong phiên."""
    B = len(sessions)
    mask = torch.zeros(B, n_items, dtype=torch.bool, device=device)
    for i, sess in enumerate(sessions):
        cats = {item2cat.get(it, 0) for it in sess}
        items: set[int] = set()
        for c in cats:
            items.update(cat2items.get(c, ()))
        if items:
            idx = torch.tensor(list(items), dtype=torch.long, device=device)
            mask[i, idx] = True
    return mask


class MGCoreEncoder(nn.Module):
    """Encoder CatSA v2: Transformer + dual-path GNN + category GRU."""

    def __init__(
        self,
        n_items: int,
        n_cats: int,
        item2cat: dict[int, int],
        cat2items: dict[int, list[int]],
        d: int = 100,
        n_layers: int = 2,
        use_taxonomy: bool = True,
        dropout: float = 0.1,
        fusion_type: str = "gate",
        n_heads: int = 2,
        max_seq_length: int = 50,
        trm_layers: int = 2,
        trm_inner_size: int = 256,
        temperature: float = 0.07,
        sess_dropout: float = 0.2,
        item_dropout: float = 0.2,
        extra_rerank: bool = True,
        extra_beta: float = 12.0,
        **_kwargs,
    ):
        super().__init__()
        self.n_items = n_items
        self.item2cat = item2cat
        self.cat2items = cat2items
        self.d = d
        self.max_seq_length = max_seq_length
        self.temperature = temperature
        self.extra_rerank = extra_rerank
        self.extra_beta = extra_beta
        self.use_taxonomy = use_taxonomy

        self.emb = EmbeddingTables(n_items, n_cats, d)
        # Embedding riêng cho Transformer (1-indexed, 0 = pad)
        self.trm_emb = nn.Embedding(n_items + 1, d, padding_idx=0)

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
        self.sess_dropout = nn.Dropout(sess_dropout)
        self.item_dropout = nn.Dropout(item_dropout)
        self.readout_seq = SoftAttentionReadout(d)
        self.readout_tax = SoftAttentionReadout(d)
        self.gate_dual = nn.Linear(d * 2, d)

        self.cat_gru = nn.GRU(d, d, batch_first=True)

        self.trm = TransNet(
            max_seq_length=max_seq_length,
            n_layers=trm_layers,
            n_heads=n_heads,
            hidden_size=d,
            inner_size=trm_inner_size,
            hidden_dropout_prob=dropout,
            attn_dropout_prob=dropout,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            initializer_range=0.02,
        )

        self.fuse3 = nn.Linear(d * 3, 3)
        self.out_proj = nn.Linear(d, d)
        self.sc_head = nn.Linear(d, 3)

    def _dual_path(self, batch: Batch) -> torch.Tensor:
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
        g = torch.sigmoid(self.gate_dual(torch.cat([z_seq, z_tax], dim=-1)))
        return g * z_seq + (1 - g) * z_tax

    def _transformer_path(self, sessions: list[list[int]], device: torch.device) -> torch.Tensor:
        item_seq = _sessions_to_trm_tensor(sessions, self.max_seq_length, device)
        x = self.sess_dropout(self.trm_emb(item_seq))
        alpha = self.trm(item_seq, x)
        z = (alpha * x).sum(dim=1)
        return F.normalize(z, dim=-1)

    def _category_path(self, sessions: list[list[int]], device: torch.device) -> torch.Tensor:
        cat_pad, lengths = _session_cat_tensor(sessions, self.item2cat, device)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb.cat_emb(cat_pad), lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.cat_gru(packed)
        return h_n.squeeze(0)

    def forward(self, batch: Batch) -> torch.Tensor:
        sessions = getattr(batch, "session_lists", None)
        device = batch["item"].node_id.device

        z_gnn = self._dual_path(batch)
        if sessions is None:
            return self.out_proj(z_gnn)

        z_trm = self._transformer_path(sessions, device)
        z_cat = self._category_path(sessions, device)

        w = torch.softmax(self.fuse3(torch.cat([z_trm, z_gnn, z_cat], dim=-1)), dim=-1)
        z = w[:, 0:1] * z_trm + w[:, 1:2] * z_gnn + w[:, 2:3] * z_cat
        return F.normalize(self.out_proj(z), dim=-1)

    def _base_logits(self, z_s: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.emb.item_emb.weight, dim=-1)
        if self.training:
            w = self.item_dropout(w)
        z = F.normalize(z_s, dim=-1)
        return (z @ w.T) / self.temperature

    def scores(self, z_s: torch.Tensor, batch: Batch | None = None) -> torch.Tensor:
        logits = self._base_logits(z_s)
        if not self.extra_rerank or batch is None:
            return logits

        sessions = getattr(batch, "session_lists", None)
        if sessions is None:
            return logits

        device = z_s.device
        B, n_items = logits.shape
        cat_mask = _build_cat_pool_mask(
            sessions, self.item2cat, self.cat2items, n_items, device,
        )
        out = torch.empty_like(logits)
        beta = self.extra_beta
        phi = torch.softmax(self.sc_head(z_s), dim=-1)

        for i, sess in enumerate(sessions):
            li = logits[i]
            seen = torch.zeros(n_items, dtype=torch.bool, device=device)
            if sess:
                seen[torch.tensor(sess, dtype=torch.long, device=device)] = True

            def _softmax_masked(mask: torch.Tensor) -> torch.Tensor:
                m = li.masked_fill(~mask, float("-inf"))
                return F.softmax(beta * m, dim=-1)

            p_seen = _softmax_masked(seen)
            p_cat = _softmax_masked(cat_mask[i])
            p_novel = _softmax_masked(~cat_mask[i])
            mix = phi[i, 0] * p_seen + phi[i, 1] * p_cat + phi[i, 2] * p_novel
            out[i] = torch.log(mix.clamp(min=1e-12))

        return out
