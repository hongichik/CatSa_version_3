"""Giai đoạn 4 — Evaluation full-ranking: HR@K, NDCG@K, MRR@K.

Với mỗi phiên: tính z_s, tính score cho TOÀN BỘ vocabulary (không sampled
negatives), lấy hạng của item ground-truth trong ranking giảm dần.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from .model import build_encoder


def _model_scores(model, z_s: torch.Tensor, batch=None) -> torch.Tensor:
    import inspect
    sig = inspect.signature(model.scores)
    if batch is not None and len(sig.parameters) >= 2:
        return model.scores(z_s, batch)
    return model.scores(z_s)


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    top_k: list[int],
) -> dict[str, float]:
    """Trả về dict metric, ví dụ {'hr@20': ..., 'ndcg@20': ..., 'mrr@20': ...}."""
    model.eval()
    hits = {k: 0 for k in top_k}
    ndcg = {k: 0.0 for k in top_k}
    mrr = {k: 0.0 for k in top_k}
    n_samples = 0

    for batch_orig, _, targets in loader:
        batch_orig = batch_orig.to(device)
        targets = targets.to(device)

        z_s = model(batch_orig)
        logits = _model_scores(model, z_s, batch_orig)

        # Hạng của target = số item có score cao hơn + 1
        target_scores = logits.gather(1, targets.view(-1, 1))
        ranks = (logits > target_scores).sum(dim=1) + 1  # (B,)

        for k in top_k:
            in_top = ranks <= k
            hits[k] += int(in_top.sum())
            ndcg[k] += float((1.0 / torch.log2(ranks[in_top].float() + 1)).sum())
            mrr[k] += float((1.0 / ranks[in_top].float()).sum())
        n_samples += targets.size(0)

    metrics: dict[str, float] = {}
    for k in top_k:
        metrics[f"hr@{k}"] = hits[k] / n_samples
        metrics[f"ndcg@{k}"] = ndcg[k] / n_samples
        metrics[f"mrr@{k}"] = mrr[k] / n_samples
    return metrics
