"""RGCN + global category retrieval (cross-session qua cat2items)."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

from .rgcn import RGCNEncoder


class GlobalCategoryIndex:
    """Index item theo category để retrieve nhanh."""

    def __init__(self, cat2items: dict[int, list[int]]):
        self.cat2items = {c: list(items) for c, items in cat2items.items() if items}

    def sample_items(
        self, cat_ids: list[int], topk: int, exclude: set[int] | None = None,
    ) -> list[int]:
        exclude = exclude or set()
        pool: list[int] = []
        for c in cat_ids:
            pool.extend(it for it in self.cat2items.get(c, []) if it not in exclude)
        if not pool:
            return []
        if len(pool) <= topk:
            return pool
        return random.sample(pool, topk)


class RetrievalEncoder(nn.Module):
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
        retrieval_topk: int = 5,
        **_kwargs,
    ):
        super().__init__()
        self.item2cat = item2cat
        self.retrieval_topk = retrieval_topk
        self.index = GlobalCategoryIndex(cat2items)
        self.base = RGCNEncoder(
            n_items, n_cats, d=d, n_layers=n_layers,
            use_taxonomy=use_taxonomy, dropout=dropout,
        )
        self.ret_attn = nn.Linear(d, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        z_s = self.base(batch)
        sessions = getattr(batch, "session_lists", None)
        if sessions is None:
            return z_s

        B = z_s.size(0)
        device = z_s.device
        retrieved = torch.zeros_like(z_s)
        counts = torch.zeros(B, 1, device=device)

        for i, sess in enumerate(sessions):
            cats = list({self.item2cat.get(it, 0) for it in sess})
            items = self.index.sample_items(cats, self.retrieval_topk, exclude=set(sess))
            if not items:
                continue
            h = self.base.emb.item_emb(torch.tensor(items, device=device))
            w = F.softmax(self.ret_attn(h), dim=0)
            retrieved[i] = (w * h).sum(dim=0)
            counts[i] = 1.0

        mask = counts > 0
        z_s = z_s.clone()
        z_s[mask.squeeze(-1)] = z_s[mask.squeeze(-1)] + retrieved[mask.squeeze(-1)]
        return z_s

    def scores(self, z_s):
        return self.base.scores(z_s)
