"""Đánh giá full-ranking HR@K, NDCG@K, MRR@K cho MSGIFSR."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _prepare_batch(batch, device: torch.device):
    inputs, labels = batch
    return [x.to(device) for x in inputs], labels.to(device)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: list[int],
) -> dict[str, float]:
    """MSGIFSR trả về log-softmax (B, n_items); thứ tự ranking giữ nguyên."""
    model.eval()
    hits = {k: 0 for k in top_k}
    ndcg = {k: 0.0 for k in top_k}
    mrr = {k: 0.0 for k in top_k}
    n_samples = 0

    for batch in loader:
        inputs, targets = _prepare_batch(batch, device)
        scores = model(*inputs)

        target_scores = scores.gather(1, targets.view(-1, 1))
        ranks = (scores > target_scores).sum(dim=1) + 1

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
