"""CatSA+ — Module 1 + scoring ổn định + 2 phần tùy chọn (aux / post).

Theo Xây dựng_CatSA_May26.pdf:
  Module 1 — Category-Enhanced Session Graph
  Module 2 — Category-Structure-Guided Augmentation  [train.yaml]
  Session-level InfoNCE                              [train.yaml]

Bổ sung (bật bằng config, 3 biến thể ablation):
  use_error_aux  — nhánh phụ category-GRU residual trên logits (sửa soft leaf confusion)
  post_process   — post-processing taxonomy boost (same-leaf / sibling / other)

encoder_type: catsa_plus
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, SAGEConv

from CORE.models.core_trm import TransNet

from ..augment import compute_siblings
from .common import SoftAttentionReadout, fuse_vectors


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


def _sessions_to_trm_tensor(
    sessions: list[list[int]], max_len: int, device: torch.device,
) -> torch.Tensor:
    """Pad session → (B, L), id 1..n_items (0 = PAD)."""
    out = torch.zeros(len(sessions), max_len, dtype=torch.long, device=device)
    for i, s in enumerate(sessions):
        sl = min(len(s), max_len)
        if sl > 0:
            out[i, :sl] = torch.tensor([it + 1 for it in s[:sl]], device=device)
    return out


def _build_taxonomy_masks(
    sessions: list[list[int]],
    item2cat: dict[int, int],
    cat2items: dict[int, list[int]],
    siblings: dict[int, list[int]] | None,
    n_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(same_mask, sib_mask) shape (B, n_items)."""
    B = len(sessions)
    same = torch.zeros(B, n_items, dtype=torch.bool, device=device)
    sib = torch.zeros(B, n_items, dtype=torch.bool, device=device)
    for i, sess in enumerate(sessions):
        cats = {item2cat.get(it, 0) for it in sess} if sess else set()
        same_items: set[int] = set()
        sib_items: set[int] = set()
        for c in cats:
            same_items.update(cat2items.get(c, ()))
            if siblings:
                for sc in siblings.get(c, ()):
                    sib_items.update(cat2items.get(sc, ()))
        sib_items -= same_items
        if same_items:
            same[i, torch.tensor(list(same_items), dtype=torch.long, device=device)] = True
        if sib_items:
            sib[i, torch.tensor(list(sib_items), dtype=torch.long, device=device)] = True
    return same, sib


class CatSAPlusEncoder(nn.Module):
    """Module 1 dual-path hetero GNN + optional TransNet + aux residual + post."""

    def __init__(
        self,
        n_items: int,
        n_cats: int,
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
        trm_dropout: float = 0.5,
        use_seq_trm: bool = True,
        # --- V1: error-style auxiliary residual ---
        use_error_aux: bool = False,
        error_aux_alpha: float = 0.15,
        # --- V2: taxonomy post-processing ---
        post_process: bool = False,
        post_same_boost: float = 0.35,
        post_sib_boost: float = 0.12,
        post_other_penalty: float = 0.08,
        item2cat: dict[int, int] | None = None,
        cat2items: dict[int, list[int]] | None = None,
        cat_parent: dict[int, int] | None = None,
        **_kwargs,
    ):
        super().__init__()
        self.n_items = n_items
        self.d = d
        self.use_taxonomy = use_taxonomy
        self.fusion_type = fusion_type
        self.temperature = temperature
        self.max_seq_length = max_seq_length
        self.use_seq_trm = use_seq_trm

        self.use_error_aux = use_error_aux
        self.error_aux_alpha = error_aux_alpha
        self.post_process = post_process
        self.post_same_boost = post_same_boost
        self.post_sib_boost = post_sib_boost
        self.post_other_penalty = post_other_penalty

        self.item2cat = item2cat or {}
        self.cat2items = cat2items or {}
        self.siblings = (
            compute_siblings(cat_parent) if (cat_parent and post_process) else None
        )

        # Bảng duy nhất: 0=PAD, 1..n_items — dùng cho GNN, TRM và ranking
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, d)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.zeros_(self.item_emb.weight[0])
        nn.init.normal_(self.cat_emb.weight, std=0.1)

        self.dropout = nn.Dropout(dropout)
        self.sess_dropout = nn.Dropout(sess_dropout)
        self.item_dropout = nn.Dropout(item_dropout)

        # ----- Module 1: dual-path hetero GNN (đóng góp category chính) -----
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

        self.readout_seq = SoftAttentionReadout(d)
        self.readout_tax = SoftAttentionReadout(d)
        self.gate_m1 = nn.Linear(d * 2, d) if fusion_type == "gate" else None

        # ----- Nhánh tuần tự phụ (geometry mạnh, cùng emb) -----
        self.trm = None
        self.gate_plus = None
        if use_seq_trm:
            self.trm = TransNet(
                max_seq_length=max_seq_length,
                n_layers=trm_layers,
                n_heads=n_heads,
                hidden_size=d,
                inner_size=trm_inner_size,
                hidden_dropout_prob=trm_dropout,
                attn_dropout_prob=trm_dropout,
                hidden_act="gelu",
                layer_norm_eps=1e-12,
                initializer_range=0.02,
            )
            self.gate_plus = nn.Linear(d * 2, d)

        self.out_proj = nn.Linear(d, d)

        # ----- V1: category-path residual (aux nhánh sửa soft leaf / taxonomy) -----
        self.cat_gru = None
        self.err_proj = None
        self.err_gate = None
        if use_error_aux:
            if not self.item2cat:
                raise ValueError("use_error_aux cần item2cat")
            self.cat_gru = nn.GRU(d, d, batch_first=True)
            self.err_proj = nn.Linear(d, d)
            self.err_gate = nn.Linear(d * 2, 1)

    def _init_x_dict(self, batch: Batch) -> dict[str, torch.Tensor]:
        x: dict[str, torch.Tensor] = {
            "item": self.item_emb(batch["item"].node_id + 1),
            "category": self.cat_emb(batch["category"].node_id),
        }
        if self.use_taxonomy and "parent" in batch.node_types:
            x["parent"] = self.cat_emb(batch["parent"].node_id)
        return x

    def _module1(self, batch: Batch) -> torch.Tensor:
        x0 = self._init_x_dict(batch)
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
        return fuse_vectors(z_seq, z_tax, self.fusion_type, self.gate_m1)

    def _trm_encode(self, sessions: list[list[int]], device: torch.device) -> torch.Tensor:
        assert self.trm is not None
        item_seq = _sessions_to_trm_tensor(sessions, self.max_seq_length, device)
        x = self.sess_dropout(self.item_emb(item_seq))
        alpha = self.trm(item_seq, x)
        return (alpha * x).sum(dim=1)

    def _error_aux_encode(
        self, sessions: list[list[int]], device: torch.device,
    ) -> torch.Tensor:
        assert self.cat_gru is not None and self.err_proj is not None
        cat_pad, lengths = _session_cat_tensor(sessions, self.item2cat, device)
        # pack_padded_sequence yêu cầu length ≥ 1
        lengths = lengths.clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.cat_emb(cat_pad), lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.cat_gru(packed)
        return self.err_proj(h_n.squeeze(0))

    def forward(self, batch: Batch) -> torch.Tensor:
        z_m1 = self._module1(batch)
        sessions = getattr(batch, "session_lists", None)
        if self.use_seq_trm and self.trm is not None and sessions is not None:
            z_trm = self._trm_encode(sessions, batch["item"].node_id.device)
            z = fuse_vectors(z_m1, z_trm, "gate", self.gate_plus)
        else:
            z = z_m1
        z = F.normalize(self.out_proj(z), dim=-1)
        # Stash for scores() residual (cùng forward pass)
        if self.use_error_aux and sessions is not None:
            self._last_zerr = self._error_aux_encode(sessions, z.device)
            self._last_zs = z
        else:
            self._last_zerr = None
            self._last_zs = None
        return z

    def _item_weights(self) -> torch.Tensor:
        w = self.item_emb.weight[1:]
        if self.training:
            w = self.item_dropout(w)
        return F.normalize(w, dim=-1)

    def _base_logits(self, z_s: torch.Tensor) -> torch.Tensor:
        w = self._item_weights()
        z = F.normalize(z_s, dim=-1)
        return (z @ w.T) / self.temperature

    def _apply_error_aux(self, logits: torch.Tensor, z_s: torch.Tensor, batch: Batch) -> torch.Tensor:
        sessions = getattr(batch, "session_lists", None)
        if sessions is None or self.err_gate is None:
            return logits
        z_err = getattr(self, "_last_zerr", None)
        if z_err is None or z_err.shape[0] != z_s.shape[0]:
            z_err = self._error_aux_encode(sessions, z_s.device)
        z_err = F.normalize(z_err, dim=-1)
        w = self._item_weights()
        logits_err = (z_err @ w.T) / self.temperature
        g = torch.sigmoid(self.err_gate(torch.cat([z_s, z_err], dim=-1)))  # (B, 1)
        a = self.error_aux_alpha
        return (1.0 - a * g) * logits + (a * g) * logits_err

    def _apply_post(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        sessions = getattr(batch, "session_lists", None)
        if sessions is None or not self.item2cat or not self.cat2items:
            return logits
        same, sib = _build_taxonomy_masks(
            sessions, self.item2cat, self.cat2items, self.siblings,
            logits.size(1), logits.device,
        )
        other = ~(same | sib)
        return (
            logits
            + self.post_same_boost * same.float()
            + self.post_sib_boost * sib.float()
            - self.post_other_penalty * other.float()
        )

    def scores(self, z_s: torch.Tensor, batch: Batch | None = None) -> torch.Tensor:
        logits = self._base_logits(z_s)
        if batch is not None and self.use_error_aux:
            logits = self._apply_error_aux(logits, z_s, batch)
        # Post-processing: chỉ lúc eval (không ảnh hưởng gradient train)
        if batch is not None and self.post_process and not self.training:
            logits = self._apply_post(logits, batch)
        return logits
