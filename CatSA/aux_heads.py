"""Head phụ cho auxiliary task: dự đoán category/parent của item tiếp theo."""

from __future__ import annotations

from torch import nn


class AuxiliaryHeads(nn.Module):
    def __init__(self, d: int, n_cats: int, aux_cat: bool, aux_parent: bool):
        super().__init__()
        self.aux_cat = aux_cat
        self.aux_parent = aux_parent
        self.cat_head = nn.Linear(d, n_cats) if aux_cat else None
        self.parent_head = nn.Linear(d, n_cats) if aux_parent else None

    def forward(self, z_s):
        out = {}
        if self.cat_head is not None:
            out["cat"] = self.cat_head(z_s)
        if self.parent_head is not None:
            out["parent"] = self.parent_head(z_s)
        return out
