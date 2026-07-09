"""CatSA++ (v2) — giữ đóng góp Q3, fusion có thể chọn qua config.

Đóng góp CatSA (May26) — KHÔNG đổi:
  Module 1 — dual-path hetero GNN (seq + membership + taxonomy)
  Module 2 — same / sibling / hybrid augmentation          [train.yaml]
  Session-level InfoNCE                                     [train.yaml]

Chỉ thay đổi CÁCH KẾT HỢP nhánh sequential (TRM, mượn geometry CORE) với Module 1:
  path_fusion:
    embed_gate   — CatSA+ cũ: gate embedding → 1 cosine (so sánh ablation)
    dual_score   — logits = logits_seq + β·logits_cat  (khuyến nghị Q3)
    trm_residual — z = norm(z_seq + γ·proj(z_cat)) → 1 cosine

Tùy chọn mở rộng (vẫn gắn taxonomy / Module 1):
  use_cat_bias — bias logit học theo category (train end-to-end)
  length_aware_gate — (chỉ áp dụng khi path_fusion=dual_score) thêm 1 feature
    độ dài phiên (log-scaled) vào score_gate hiện có, để beta (trọng số nhánh
    category) tự điều chỉnh theo độ dài phiên trong CÙNG MỘT model — thay vì
    train 2 model short/long riêng (catsa_plus_v2_len_short/long/dual) vốn làm
    giảm dữ liệu train cho phiên dài. Mặc định false — không đổi hành vi các
    phiên bản trước.

Ablation paper (chọn qua YAML):
  A   use_seq_trm=true,  use_module1=false, use_cl=false
  A2  use_seq_trm=false, use_module1=true,  use_cl=false
  B   dual_score + M1 + TRM,                 use_cl=false
  C   dual_score + M1 + TRM + M2 + InfoNCE  (full CatSA)
  D   C + length_aware_gate (1 model, gate biết độ dài phiên — thay cho
      cách tách 2 model len_short/len_long/len_dual)

encoder_type: catsa_plus_v2
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv, SAGEConv

from CORE.models.core_trm import TransNet

from .common import SoftAttentionReadout, fuse_vectors

PATH_FUSION_TYPES = frozenset({"embed_gate", "dual_score", "trm_residual"})


def _sessions_to_trm_tensor(
    sessions: list[list[int]], max_len: int, device: torch.device,
) -> torch.Tensor:
    out = torch.zeros(len(sessions), max_len, dtype=torch.long, device=device)
    for i, s in enumerate(sessions):
        sl = min(len(s), max_len)
        if sl > 0:
            out[i, :sl] = torch.tensor([it + 1 for it in s[:sl]], device=device)
    return out


class CatSAPlusV2Encoder(nn.Module):
    """CatSA++ — Module 1 + TRM với path_fusion chọn được; CL dùng embedding session."""

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
        use_module1: bool = True,
        path_fusion: str = "dual_score",
        dual_score_beta: float = 1.0,
        learn_score_beta: bool = True,
        trm_residual_gamma: float = 0.5,
        use_cat_bias: bool = False,
        length_aware_gate: bool = False,
        length_gate_max_len: int = 50,
        item2cat: dict[int, int] | None = None,
        **_kwargs,
    ):
        super().__init__()
        if path_fusion not in PATH_FUSION_TYPES:
            raise ValueError(f"path_fusion không hợp lệ: {path_fusion!r}")
        if not use_seq_trm and not use_module1:
            raise ValueError("Cần ít nhất một trong use_seq_trm hoặc use_module1")

        self.n_items = n_items
        self.d = d
        self.use_taxonomy = use_taxonomy
        self.fusion_type = fusion_type
        self.temperature = temperature
        self.max_seq_length = max_seq_length
        self.use_seq_trm = use_seq_trm
        self.use_module1 = use_module1
        self.path_fusion = path_fusion
        self.trm_residual_gamma = trm_residual_gamma
        self.use_cat_bias = use_cat_bias
        # length-aware dual_score gate — mặc định false, KHÔNG đổi hành vi cũ
        self.length_aware_gate = length_aware_gate and path_fusion == "dual_score"
        self.length_gate_max_len = max(int(length_gate_max_len), 1)

        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, d)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.zeros_(self.item_emb.weight[0])
        nn.init.normal_(self.cat_emb.weight, std=0.1)

        self.dropout = nn.Dropout(dropout)
        self.sess_dropout = nn.Dropout(sess_dropout)
        self.item_dropout = nn.Dropout(item_dropout)

        # ----- Module 1 (đóng góp category chính) -----
        self.seq_convs = nn.ModuleList()
        self.tax_convs = nn.ModuleList()
        if use_module1:
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

        # ----- Nhánh sequential (geometry CORE — không claim novelty) -----
        self.trm = None
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

        # Fusion helpers
        self.gate_embed = None
        self.gate_cl = None
        self.cat_proj = None
        self.log_beta = None
        self.score_gate = None
        if use_seq_trm and use_module1:
            if path_fusion == "embed_gate":
                self.gate_embed = nn.Linear(d * 2, d)
            elif path_fusion == "trm_residual":
                self.cat_proj = nn.Linear(d, d)
            elif path_fusion == "dual_score":
                if learn_score_beta:
                    import math
                    self.log_beta = nn.Parameter(
                        torch.tensor(math.log(max(dual_score_beta, 1e-6))),
                    )
                else:
                    self.register_buffer(
                        "fixed_beta", torch.tensor(float(dual_score_beta)),
                    )
                gate_in_dim = d * 2 + 1 if self.length_aware_gate else d * 2
                self.score_gate = nn.Linear(gate_in_dim, 1)
            self.gate_cl = nn.Linear(d * 2, d)

        self.out_proj = nn.Linear(d, d)

        # Category logit bias (Module 1 / taxonomy, train end-to-end)
        self.cat_logit_bias = None
        if use_cat_bias:
            if not item2cat:
                raise ValueError("use_cat_bias cần item2cat")
            idx = torch.zeros(n_items, dtype=torch.long)
            for it, c in item2cat.items():
                if 0 <= it < n_items:
                    idx[it] = int(c)
            self.register_buffer("item_cat_idx", idx)
            self.cat_logit_bias = nn.Embedding(n_cats, 1)
            nn.init.zeros_(self.cat_logit_bias.weight)

        self._z_m1: torch.Tensor | None = None
        self._z_trm: torch.Tensor | None = None
        self._len_feat: torch.Tensor | None = None

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

    def _length_feature(
        self, sessions: list[list[int]], device: torch.device,
    ) -> torch.Tensor:
        """(B, 1) — độ dài phiên chuẩn hoá log-scale, dùng cho length_aware_gate."""
        import math
        max_len = self.length_gate_max_len
        norm = math.log1p(max_len)
        vals = [math.log1p(min(len(s), max_len)) / norm for s in sessions]
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(-1)

    def _item_weights(self) -> torch.Tensor:
        w = self.item_emb.weight[1:]
        if self.training:
            w = self.item_dropout(w)
        return F.normalize(w, dim=-1)

    def _cosine_logits(self, z: torch.Tensor) -> torch.Tensor:
        w = self._item_weights()
        z = F.normalize(z, dim=-1)
        return (z @ w.T) / self.temperature

    def _beta_scale(self, z_trm: torch.Tensor, z_m1: torch.Tensor) -> torch.Tensor:
        """(B, 1) — scale nhánh category trong dual_score."""
        if self.log_beta is not None:
            base = torch.exp(self.log_beta).clamp(max=10.0)
        else:
            base = self.fixed_beta  # type: ignore[attr-defined]
        if self.score_gate is not None:
            gate_in = [z_trm, z_m1]
            if self.length_aware_gate:
                len_feat = self._len_feat
                if len_feat is None:
                    len_feat = torch.zeros(z_trm.size(0), 1, device=z_trm.device)
                gate_in.append(len_feat)
            g = torch.sigmoid(self.score_gate(torch.cat(gate_in, dim=-1)))
            return base * g
        return base.expand(z_trm.size(0), 1)

    def _session_embedding_for_cl(
        self, z_m1: torch.Tensor | None, z_trm: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embedding cho InfoNCE — không ảnh hưởng dual_score prediction path."""
        if z_m1 is not None and z_trm is not None and self.gate_cl is not None:
            z = fuse_vectors(z_m1, z_trm, "gate", self.gate_cl)
        elif z_trm is not None:
            z = z_trm
        else:
            assert z_m1 is not None
            z = z_m1
        return F.normalize(self.out_proj(z), dim=-1)

    def forward(self, batch: Batch) -> torch.Tensor:
        sessions = getattr(batch, "session_lists", None)
        device = batch["item"].node_id.device

        z_m1 = self._module1(batch) if self.use_module1 else None
        z_trm = (
            self._trm_encode(sessions, device)
            if self.use_seq_trm and self.trm is not None and sessions is not None
            else None
        )
        self._z_m1 = z_m1
        self._z_trm = z_trm
        self._len_feat = (
            self._length_feature(sessions, device)
            if self.length_aware_gate and sessions is not None
            else None
        )

        if self.path_fusion == "embed_gate" and z_m1 is not None and z_trm is not None:
            z = fuse_vectors(z_m1, z_trm, "gate", self.gate_embed)
            return F.normalize(self.out_proj(z), dim=-1)
        if self.path_fusion == "trm_residual" and z_m1 is not None and z_trm is not None:
            assert self.cat_proj is not None
            z = z_trm + self.trm_residual_gamma * self.cat_proj(z_m1)
            return F.normalize(self.out_proj(z), dim=-1)

        # dual_score hoặc single-path: CL dùng session embedding riêng
        return self._session_embedding_for_cl(z_m1, z_trm)

    def _resolve_paths(self, z_s: torch.Tensor, batch: Batch | None):
        z_m1 = self._z_m1
        z_trm = self._z_trm
        if z_m1 is None and z_trm is None:
            # Fallback khi scores() gọi không qua forward (hiếm)
            if self.use_module1 and batch is not None:
                z_m1 = self._module1(batch)
            sessions = getattr(batch, "session_lists", None) if batch else None
            if self.use_seq_trm and sessions is not None:
                device = z_s.device
                z_trm = self._trm_encode(sessions, device)
            if self.length_aware_gate and self._len_feat is None and sessions is not None:
                self._len_feat = self._length_feature(sessions, z_s.device)
        return z_m1, z_trm

    def scores(self, z_s: torch.Tensor, batch: Batch | None = None) -> torch.Tensor:
        z_m1, z_trm = self._resolve_paths(z_s, batch)

        if self.path_fusion == "dual_score" and z_m1 is not None and z_trm is not None:
            logits = self._cosine_logits(z_trm)
            logits_cat = self._cosine_logits(z_m1)
            beta = self._beta_scale(z_trm, z_m1)
            logits = logits + beta * logits_cat
        elif z_trm is not None and z_m1 is None:
            logits = self._cosine_logits(z_trm)
        elif z_m1 is not None and z_trm is None:
            logits = self._cosine_logits(z_m1)
        else:
            logits = self._base_logits_from_z(z_s)

        if self.use_cat_bias and self.cat_logit_bias is not None and self.item_cat_idx is not None:
            bias = self.cat_logit_bias(self.item_cat_idx).squeeze(-1)
            logits = logits + bias.unsqueeze(0)
        return logits

    def _base_logits_from_z(self, z_s: torch.Tensor) -> torch.Tensor:
        return self._cosine_logits(z_s)
