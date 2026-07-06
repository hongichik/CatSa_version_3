"""Đánh giá full-ranking HR@K, NDCG@K, MRR@K cho CORE."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from CORE.dataset import pad_batch


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: list[int],
) -> dict[str, float]:
    model.eval()
    hits = {k: 0 for k in top_k}
    ndcg = {k: 0.0 for k in top_k}
    mrr = {k: 0.0 for k in top_k}
    n_samples = 0

    for seqs, targets in loader:
        item_seq = pad_batch(seqs, device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=device)
        logits = model.predict_scores(item_seq)

        target_scores = logits.gather(1, targets_t.view(-1, 1))
        ranks = (logits > target_scores).sum(dim=1) + 1

        for k in top_k:
            in_top = ranks <= k
            hits[k] += int(in_top.sum())
            ndcg[k] += float((1.0 / torch.log2(ranks[in_top].float() + 1)).sum())
            mrr[k] += float((1.0 / ranks[in_top].float()).sum())
        n_samples += targets_t.size(0)

    metrics: dict[str, float] = {}
    for k in top_k:
        metrics[f"hr@{k}"] = hits[k] / n_samples
        metrics[f"ndcg@{k}"] = ndcg[k] / n_samples
        metrics[f"mrr@{k}"] = mrr[k] / n_samples
    return metrics
