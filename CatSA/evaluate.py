"""Giai đoạn 4 — Evaluation full-ranking: HR@K, NDCG@K, MRR@K.

Với mỗi phiên: tính z_s, tính score cho TOÀN BỘ vocabulary (không sampled
negatives), lấy hạng của item ground-truth trong ranking giảm dần.
"""

from __future__ import annotations

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
def _accumulate_ranks(
    ranks: torch.Tensor,
    top_k: list[int],
    hits: dict[int, int],
    ndcg: dict[int, float],
    mrr: dict[int, float],
) -> None:
    for k in top_k:
        in_top = ranks <= k
        hits[k] += int(in_top.sum())
        ndcg[k] += float((1.0 / torch.log2(ranks[in_top].float() + 1)).sum())
        mrr[k] += float((1.0 / ranks[in_top].float()).sum())


def _ranks_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_scores = logits.gather(1, targets.view(-1, 1))
    return (logits > target_scores).sum(dim=1) + 1


def _finalize_metrics(
    hits: dict[int, int],
    ndcg: dict[int, float],
    mrr: dict[int, float],
    n_samples: int,
    top_k: list[int],
    prefix: str = "",
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    p = f"{prefix}_" if prefix else ""
    for k in top_k:
        metrics[f"{p}hr@{k}"] = hits[k] / n_samples if n_samples else 0.0
        metrics[f"{p}ndcg@{k}"] = ndcg[k] / n_samples if n_samples else 0.0
        metrics[f"{p}mrr@{k}"] = mrr[k] / n_samples if n_samples else 0.0
    return metrics


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
        ranks = _ranks_from_logits(logits, targets)
        _accumulate_ranks(ranks, top_k, hits, ndcg, mrr)
        n_samples += targets.size(0)

    return _finalize_metrics(hits, ndcg, mrr, n_samples, top_k)


@torch.no_grad()
def evaluate_dual_length(
    model_short,
    model_long,
    loader: DataLoader,
    device: torch.device,
    top_k: list[int],
    threshold: int,
) -> dict[str, float]:
    """Routing: len(prefix) <= threshold → model_short, ngược lại → model_long."""
    model_short.eval()
    model_long.eval()

    hits = {k: 0 for k in top_k}
    ndcg = {k: 0.0 for k in top_k}
    mrr = {k: 0.0 for k in top_k}
    hits_s = {k: 0 for k in top_k}
    ndcg_s = {k: 0.0 for k in top_k}
    mrr_s = {k: 0.0 for k in top_k}
    hits_l = {k: 0 for k in top_k}
    ndcg_l = {k: 0.0 for k in top_k}
    mrr_l = {k: 0.0 for k in top_k}
    n_samples = n_short = n_long = 0

    for batch_orig, _, targets in loader:
        batch_orig = batch_orig.to(device)
        targets = targets.to(device)
        sessions = getattr(batch_orig, "session_lists", None)
        if sessions is None:
            raise ValueError("batch cần session_lists cho length routing")

        z_short = model_short(batch_orig)
        logits_short = _model_scores(model_short, z_short, batch_orig)
        z_long = model_long(batch_orig)
        logits_long = _model_scores(model_long, z_long, batch_orig)

        logits = logits_short.clone()
        short_mask = torch.zeros(targets.size(0), dtype=torch.bool, device=device)
        for i, sess in enumerate(sessions):
            if len(sess) > threshold:
                logits[i] = logits_long[i]
            else:
                short_mask[i] = True

        ranks = _ranks_from_logits(logits, targets)
        _accumulate_ranks(ranks, top_k, hits, ndcg, mrr)
        n_samples += targets.size(0)

        if short_mask.any():
            rs = _ranks_from_logits(logits[short_mask], targets[short_mask])
            _accumulate_ranks(rs, top_k, hits_s, ndcg_s, mrr_s)
            n_short += int(short_mask.sum())
        if (~short_mask).any():
            rl = _ranks_from_logits(logits[~short_mask], targets[~short_mask])
            _accumulate_ranks(rl, top_k, hits_l, ndcg_l, mrr_l)
            n_long += int((~short_mask).sum())

    metrics = _finalize_metrics(hits, ndcg, mrr, n_samples, top_k)
    metrics.update(_finalize_metrics(hits_s, ndcg_s, mrr_s, n_short, top_k, "short"))
    metrics.update(_finalize_metrics(hits_l, ndcg_l, mrr_l, n_long, top_k, "long"))
    metrics["n_samples"] = float(n_samples)
    metrics["n_short"] = float(n_short)
    metrics["n_long"] = float(n_long)
    return metrics
