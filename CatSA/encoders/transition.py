"""Category transition: GRU trên chuỗi category + sequential GNN trên item."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv

from .common import EmbeddingTables, SoftAttentionReadout, fuse_vectors


def _session_cat_tensor(
    sessions: list[list[int]], item2cat: dict[int, int], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad chuỗi category theo session prefix; trả về (B, L), lengths (B,)."""
    cat_seqs = [[item2cat.get(it, 0) for it in s] for s in sessions]
    lengths = torch.tensor([len(s) for s in cat_seqs], dtype=torch.long, device=device)
    max_len = max(int(lengths.max()), 1)
    padded = torch.zeros(len(cat_seqs), max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(cat_seqs):
        if seq:
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return padded, lengths


class TransitionEncoder(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_cats: int,
        item2cat: dict[int, int],
        d: int = 100,
        n_layers: int = 2,
        use_taxonomy: bool = True,
        dropout: float = 0.1,
        fusion_type: str = "cross_attn",
        **_kwargs,
    ):
        super().__init__()
        self.item2cat = item2cat
        self.d = d
        self.fusion_type = fusion_type
        self.emb = EmbeddingTables(n_items, n_cats, d)
        self.convs = nn.ModuleList([SAGEConv(d, d) for _ in range(n_layers)])
        self.cat_gru = nn.GRU(d, d, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.readout = SoftAttentionReadout(d)
        self.gate = nn.Linear(d * 2, d) if fusion_type == "gate" else None
        self.out_proj = nn.Linear(d, d)

    def forward(self, batch: Batch) -> torch.Tensor:
        node_ids = batch["item"].node_id
        h = self.emb.item_emb(node_ids)
        edge_index = batch["item", "sequential", "item"].edge_index
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        z_item = self.readout(h, batch)

        sessions = getattr(batch, "session_lists", None)
        if sessions is None:
            return self.out_proj(z_item)

        cat_pad, lengths = _session_cat_tensor(sessions, self.item2cat, h.device)
        cat_emb = self.emb.cat_emb(cat_pad)
        packed = nn.utils.rnn.pack_padded_sequence(
            cat_emb, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.cat_gru(packed)
        z_cat = h_n.squeeze(0)
        z = fuse_vectors(z_item, z_cat, self.fusion_type, self.gate)
        return self.out_proj(z)

    def scores(self, z_s):
        return self.emb.scores(z_s)
