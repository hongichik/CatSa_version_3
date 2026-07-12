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
        # NDCG dùng 1/log2(rank+1) — ĐÚNG chỉ vì mỗi mẫu có DUY NHẤT 1 item
        # ground-truth (IDCG=1 nên không cần chuẩn hoá riêng). Nếu sau này
        # có multi-label ground-truth, công thức này cần sửa (finding E6).
        ndcg[k] += float((1.0 / torch.log2(ranks[in_top].float() + 1)).sum())
        mrr[k] += float((1.0 / ranks[in_top].float()).sum())


def _ranks_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average-rank tie-handling (finding E1).

    Trước đây chỉ đếm strict > (best-case tie-break): item cùng điểm với
    target luôn được xếp SAU nó, làm phồng HR/NDCG/MRR một cách hệ thống
    (đặc biệt đầu training hoặc dưới float16, khi nhiều điểm trùng nhau).
    Ở đây dùng average rank: item cùng điểm với target được tính NỬA trọng
    số thay vì luôn xếp sau — chuẩn IR, không thiên vị theo hướng nào.

    Quy ước protocol khác (chưa đổi, xem CatSA_Correctness_Synthesis E2/E3):
    - KHÔNG mask item đã xuất hiện trong phiên (repeat purchase được coi là
      ứng viên hợp lệ) — nhất quán giữa CatSA và CORE trong repo này vì cả
      hai dùng chung full-ranking scores() trên toàn vocabulary.
    - Đánh giá theo PER-PREFIX (sliding window), không phải per-session
      last-item-only — xem docstring dataset.py. CatSA và CORE trong repo
      này dùng chung quy ước này (đã xác nhận qua n_samples=81372 khớp
      nhau giữa 2 model trên cùng data/retailrocket_item_hon_5).
    """
    target_scores = logits.gather(1, targets.view(-1, 1))
    higher = (logits > target_scores).sum(dim=1).float()
    tied = (logits == target_scores).sum(dim=1).float() - 1.0  # trừ chính target
    return higher + tied * 0.5 + 1.0


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
    return_ranks: bool = False,
):
    """Trả về dict metric, ví dụ {'hr@20': ..., 'ndcg@20': ..., 'mrr@20': ...}.

    return_ranks=True (finding E4): trả thêm tensor rank per-sample (CPU),
    dùng cho kiểm định ý nghĩa thống kê (Wilcoxon signed-rank) khi so sánh
    nhiều seed/config — mặc định False, không đổi kiểu trả về cũ.
    """
    model.eval()
    hits = {k: 0 for k in top_k}
    ndcg = {k: 0.0 for k in top_k}
    mrr = {k: 0.0 for k in top_k}
    n_samples = 0
    all_ranks: list[torch.Tensor] = [] if return_ranks else None

    for batch_orig, _, targets in loader:
        batch_orig = batch_orig.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        z_s = model(batch_orig)
        logits = _model_scores(model, z_s, batch_orig)
        ranks = _ranks_from_logits(logits, targets)
        _accumulate_ranks(ranks, top_k, hits, ndcg, mrr)
        n_samples += targets.size(0)
        if return_ranks:
            all_ranks.append(ranks.detach().cpu())

    metrics = _finalize_metrics(hits, ndcg, mrr, n_samples, top_k)
    if return_ranks:
        return metrics, torch.cat(all_ranks) if all_ranks else torch.empty(0)
    return metrics


@torch.no_grad()
def evaluate_by_length_buckets(
    model,
    loader: DataLoader,
    device: torch.device,
    top_k: list[int],
    buckets: list[tuple[int, int | None]],
) -> dict[str, float]:
    """Sub-population analysis theo độ dài phiên (finding T6) — MỘT model,
    báo cáo metric riêng cho từng khoảng độ dài prefix.

    buckets: danh sách (lo, hi) — hi=None nghĩa là không giới hạn trên.
    Ví dụ [(2,3), (4,7), (8,None)] → 3 bucket "2_3", "4_7", "8_plus".
    Yêu cầu batch có session_lists (giống evaluate_dual_length).
    """
    model.eval()

    def _label(lo: int, hi: int | None) -> str:
        return f"{lo}_{hi}" if hi is not None else f"{lo}_plus"

    labels = [_label(lo, hi) for lo, hi in buckets]
    hits = {lb: {k: 0 for k in top_k} for lb in labels}
    ndcg = {lb: {k: 0.0 for k in top_k} for lb in labels}
    mrr = {lb: {k: 0.0 for k in top_k} for lb in labels}
    n_samples = {lb: 0 for lb in labels}

    for batch_orig, _, targets in loader:
        batch_orig = batch_orig.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sessions = getattr(batch_orig, "session_lists", None)
        if sessions is None:
            raise ValueError("batch cần session_lists cho length sub-population analysis")

        z_s = model(batch_orig)
        logits = _model_scores(model, z_s, batch_orig)
        ranks = _ranks_from_logits(logits, targets)

        lens = torch.tensor([len(s) for s in sessions])
        for (lo, hi), lb in zip(buckets, labels):
            mask = (lens >= lo) & (lens <= hi if hi is not None else torch.ones_like(lens, dtype=torch.bool))
            if not mask.any():
                continue
            mask = mask.to(device)
            _accumulate_ranks(ranks[mask], top_k, hits[lb], ndcg[lb], mrr[lb])
            n_samples[lb] += int(mask.sum())

    metrics: dict[str, float] = {}
    for lb in labels:
        metrics.update(_finalize_metrics(hits[lb], ndcg[lb], mrr[lb], n_samples[lb], top_k, lb))
        metrics[f"{lb}_n_samples"] = float(n_samples[lb])
    return metrics


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
        batch_orig = batch_orig.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
