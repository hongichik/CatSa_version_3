"""Giai đoạn 6 — Loss functions: InfoNCE, prototype CL, auxiliary tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def session_level_infonce(
    z_s: torch.Tensor, z_s_tilde: torch.Tensor, tau: float = 0.5,
) -> torch.Tensor:
    """Session-level InfoNCE symmetric (SimCLR-style)."""
    B = z_s.size(0)
    z_s = F.normalize(z_s, dim=1)
    z_s_tilde = F.normalize(z_s_tilde, dim=1)
    sim = torch.mm(z_s, z_s_tilde.T) / tau
    targets = torch.arange(B, device=z_s.device)
    loss_row = F.cross_entropy(sim, targets)
    loss_col = F.cross_entropy(sim.T, targets)
    return (loss_row + loss_col) / 2


def category_prototype_loss(
    z_s: torch.Tensor,
    targets: torch.Tensor,
    item2cat: dict[int, int],
    tau: float = 0.5,
    cat_parent: dict[int, int] | None = None,
    sibling_negatives: bool = True,
) -> torch.Tensor:
    """Kéo z_s về prototype category của target; âm từ category khác trong batch."""
    z = F.normalize(z_s, dim=1)
    target_cats = torch.tensor(
        [item2cat.get(int(t), 0) for t in targets.tolist()],
        device=z_s.device, dtype=torch.long,
    )
    unique_cats = target_cats.unique()
    if unique_cats.numel() <= 1:
        return torch.tensor(0.0, device=z_s.device)

    prototypes = []
    proto_labels = []
    for c in unique_cats.tolist():
        mask = target_cats == c
        if mask.any():
            prototypes.append(z[mask].mean(dim=0))
            proto_labels.append(c)
    proto = F.normalize(torch.stack(prototypes), dim=1)
    proto_labels_t = torch.tensor(proto_labels, device=z_s.device, dtype=torch.long)

    sim = torch.mm(z, proto.T) / tau
    label_map = {c: i for i, c in enumerate(proto_labels)}
    pos_idx = torch.tensor(
        [label_map[int(c)] for c in target_cats.tolist()],
        device=z_s.device, dtype=torch.long,
    )

    if sibling_negatives and cat_parent:
        # Giảm trọng số âm sibling (cùng parent) — mask mềm qua logits
        for i, c in enumerate(proto_labels):
            p = cat_parent.get(c)
            if p is None:
                continue
            for j, c2 in enumerate(proto_labels):
                if c2 != c and cat_parent.get(c2) == p:
                    sim[:, j] = sim[:, j] * 0.5

    return F.cross_entropy(sim, pos_idx)


def auxiliary_category_loss(
    logits: torch.Tensor, targets: torch.Tensor, item2cat: dict[int, int],
) -> torch.Tensor:
    cat_targets = torch.tensor(
        [item2cat.get(int(t), 0) for t in targets.tolist()],
        device=targets.device, dtype=torch.long,
    )
    return F.cross_entropy(logits, cat_targets)


def auxiliary_parent_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    item2cat: dict[int, int],
    cat_parent: dict[int, int],
    default_parent: int = 0,
) -> torch.Tensor:
    parent_targets = []
    for t in targets.tolist():
        c = item2cat.get(int(t), 0)
        parent_targets.append(cat_parent.get(c, default_parent))
    parent_t = torch.tensor(parent_targets, device=targets.device, dtype=torch.long)
    return F.cross_entropy(logits, parent_t)
