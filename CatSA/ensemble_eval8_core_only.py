"""Control cho bài báo — ensemble CORE-only (2 seed) + repeat-boost +
bucket-routing, đối chứng công bằng với ensemble thuần CatSA (phase 7).

Chạy:
    python CatSA/ensemble_eval8_core_only.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from common.config import load_config  # noqa: E402
from tienxuly import load_processed  # noqa: E402
from CatSA.dataset import make_loader  # noqa: E402
from CatSA.evaluate import _accumulate_ranks, _finalize_metrics, _ranks_from_logits  # noqa: E402
from CatSA.ensemble_eval import REF_RUN, TOP_K, _core_logits, _load_core_model  # noqa: E402

CORE_DIRS = [
    "checkpoints/CORE/retailrocket/core_trm",
    "checkpoints/CORE/retailrocket/core_trm_seed43",
]
BUCKETS = [(1, 3), (4, 7), (8, None)]
W42S = [0.3, 0.4, 0.5, 0.6, 0.7]
D_REPS = [8.0, 12.0]


def _bucket_label(lo, hi):
    return f"{lo}_{hi}" if hi is not None else f"{lo}_plus"


@torch.no_grad()
def _pass(loader, models, variants, n_items, device, routed_weights=None):
    """variants=[(w42, rep)] → per-bucket metrics; routed_weights → 1 metric chung."""
    labels = [_bucket_label(lo, hi) for lo, hi in BUCKETS]
    if routed_weights is None:
        keys = [f"w42_{w:g}|rep{dr:g}" for w, dr in variants]
        hits = {lb: {c: {k: 0 for k in TOP_K} for c in keys} for lb in labels}
        ndcg = {lb: {c: {k: 0.0 for k in TOP_K} for c in keys} for lb in labels}
        mrr = {lb: {c: {k: 0.0 for k in TOP_K} for c in keys} for lb in labels}
        n_s = {lb: 0 for lb in labels}
    else:
        hits = {k: 0 for k in TOP_K}
        ndcg = {k: 0.0 for k in TOP_K}
        mrr = {k: 0.0 for k in TOP_K}
        n_samples = 0

    for batch, _, targets in loader:
        batch = batch.to(device)
        targets = targets.to(device)
        sessions = batch.session_lists
        B = len(sessions)

        lg42 = _core_logits(models[0], sessions, device)
        lg43 = _core_logits(models[1], sessions, device)
        rep = torch.zeros(B, n_items, device=device)
        for i, s in enumerate(sessions):
            rep[i, torch.tensor(list(set(s)), device=device)] = 1.0
        lens = torch.tensor([len(s) for s in sessions], device=device)
        masks = {}
        for (lo, hi), lb in zip(BUCKETS, labels):
            m = lens >= lo
            if hi is not None:
                m &= lens <= hi
            masks[lb] = m

        if routed_weights is None:
            for lb in labels:
                n_s[lb] += int(masks[lb].sum())
            for (w, dr), c in zip(variants, [f"w42_{w:g}|rep{dr:g}" for w, dr in variants]):
                fused = w * lg42 + (1 - w) * lg43 + dr * rep
                ranks = _ranks_from_logits(fused, targets)
                for lb in labels:
                    m = masks[lb]
                    if m.any():
                        _accumulate_ranks(ranks[m], TOP_K, hits[lb][c], ndcg[lb][c], mrr[lb][c])
        else:
            fused = torch.empty_like(lg42)
            for lb in labels:
                m = masks[lb]
                if not m.any():
                    continue
                w, dr = routed_weights[lb]
                fused[m] = w * lg42[m] + (1 - w) * lg43[m] + dr * rep[m]
            ranks = _ranks_from_logits(fused, targets)
            _accumulate_ranks(ranks, TOP_K, hits, ndcg, mrr)
            n_samples += targets.size(0)

    if routed_weights is None:
        keys = [f"w42_{w:g}|rep{dr:g}" for w, dr in variants]
        return {
            lb: {c: _finalize_metrics(hits[lb][c], ndcg[lb][c], mrr[lb][c], n_s[lb], TOP_K) for c in keys}
            for lb in labels
        }, n_s
    return _finalize_metrics(hits, ndcg, mrr, n_samples, TOP_K), n_samples


def main() -> None:
    cfg = load_config("config", catsa_run=REF_RUN, catsa_suite="retailrocket")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(cfg.data)
    n_items = data["n_items"]
    max_prefix = int(data.get("max_prefix_length", 50))

    def loader_for(sessions):
        return make_loader(
            sessions, data["item2cat"], data["cat_parent"], True,
            cfg.training.batch_size, shuffle=False, num_workers=0,
            max_prefix_length=max_prefix,
        )

    print(f"[core-only] Load 2 CORE trên {device} ...", flush=True)
    models = [_load_core_model(d, n_items, device) for d in CORE_DIRS]
    variants = [(w, dr) for w in W42S for dr in D_REPS]

    print(f"[core-only] {len(variants)} biến thể × 3 bucket — VAL ...", flush=True)
    val_res, n_val = _pass(loader_for(data["val_sessions"]), models, variants, n_items, device)

    key2var = dict(zip([f"w42_{w:g}|rep{dr:g}" for w, dr in variants], variants))
    routed = {}
    print("\n[core-only] Chọn trọng số riêng từng bucket (val mrr@20):")
    for lb, res in val_res.items():
        best = max(res, key=lambda c: res[c]["mrr@20"])
        routed[lb] = key2var[best]
        print(f"  bucket {lb} (n={n_val[lb]}): {best} — val mrr@20={res[best]['mrr@20']:.4f}")

    print("\n[core-only] TEST với bộ trọng số ghép ...", flush=True)
    test_metrics, n_test = _pass(
        loader_for(data["test_sessions"]), models, variants, n_items, device, routed_weights=routed,
    )
    print(f"\n[core-only] KẾT QUẢ TEST ({n_test} mẫu):")
    print("  " + " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    print(f"\n[core-only] → TEST chính thức: mrr@20={test_metrics['mrr@20']:.4f}")

    with open("checkpoints/ensemble_eval8_core_only_result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "bucket_weights": {lb: list(w) for lb, w in routed.items()},
            "test_metrics": test_metrics,
        }, f, allow_unicode=True, sort_keys=False)
    print("[core-only] Đã ghi checkpoints/ensemble_eval8_core_only_result.yaml")


if __name__ == "__main__":
    main()
