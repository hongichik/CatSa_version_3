"""Bảng chính đề xuất cho bài báo — single model, điều kiện ngang bằng.

So sánh CatSA v3_full vs CORE, mỗi model được tune delta repeat-aware RIÊNG
trên VAL (delta tốt nhất của chính nó), rồi đánh giá TEST với delta đã chọn
và delta=0. Kèm phân tích bucket độ dài phiên (1-3 / 4-7 / 8+).

Repeat-aware scoring: logits[i, item ∈ prefix_i] += delta (quy tắc inference
cố định, 1 hyperparameter chọn trên val — tiền lệ RepeatNet; KHÔNG học trong
training vì CE-loss ép delta về 0 dù metric xếp hạng được lợi).

Chạy:
    python CatSA/eval_repeat_aware.py
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
from CatSA.ensemble_eval import (  # noqa: E402
    TOP_K, _core_logits, _load_catsa_model, _load_core_model,
)

REF_RUN = "catsa_plus_v2_len_gate.yaml"  # suite diginetica — không có thư mục baseline/

MODELS = {
    "catsa_len_gate": "checkpoints/CatSA/diginetica/catsa_plus_v2_len_gate",
    "core_trm": "checkpoints/CORE/diginetica/core_trm",
}
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
BUCKETS = [(1, 3), (4, 7), (8, None)]


def _bucket_label(lo, hi):
    return f"{lo}_{hi}" if hi is not None else f"{lo}_plus"


def _model_logits(name, model, batch, sessions, device):
    if name.startswith("core"):
        return _core_logits(model, sessions, device)
    z = model(batch)
    return model.scores(z, batch)


@torch.no_grad()
def _eval_deltas(loader, name, model, deltas, n_items, device, with_buckets=False):
    """metrics[delta] (toàn bộ) + nếu with_buckets: metrics[delta][bucket]."""
    labels = [_bucket_label(lo, hi) for lo, hi in BUCKETS]
    hits = {d: {k: 0 for k in TOP_K} for d in deltas}
    ndcg = {d: {k: 0.0 for k in TOP_K} for d in deltas}
    mrr = {d: {k: 0.0 for k in TOP_K} for d in deltas}
    bhits = {d: {lb: {k: 0 for k in TOP_K} for lb in labels} for d in deltas}
    bndcg = {d: {lb: {k: 0.0 for k in TOP_K} for lb in labels} for d in deltas}
    bmrr = {d: {lb: {k: 0.0 for k in TOP_K} for lb in labels} for d in deltas}
    n_samples = 0
    bn = {lb: 0 for lb in labels}

    for batch, _, targets in loader:
        batch = batch.to(device)
        targets = targets.to(device)
        sessions = batch.session_lists
        B = len(sessions)

        logits = _model_logits(name, model, batch, sessions, device)
        rep = torch.zeros(B, n_items, device=device)
        for i, s in enumerate(sessions):
            rep[i, torch.tensor(list(set(s)), device=device)] = 1.0

        masks = None
        if with_buckets:
            lens = torch.tensor([len(s) for s in sessions], device=device)
            masks = {}
            for (lo, hi), lb in zip(BUCKETS, labels):
                m = lens >= lo
                if hi is not None:
                    m &= lens <= hi
                masks[lb] = m

        for d in deltas:
            f = logits + d * rep if d else logits
            ranks = _ranks_from_logits(f, targets)
            _accumulate_ranks(ranks, TOP_K, hits[d], ndcg[d], mrr[d])
            if with_buckets:
                for lb in labels:
                    m = masks[lb]
                    if m.any():
                        _accumulate_ranks(ranks[m], TOP_K, bhits[d][lb], bndcg[d][lb], bmrr[d][lb])
        n_samples += targets.size(0)
        if with_buckets:
            for lb in labels:
                bn[lb] += int(masks[lb].sum())

    out = {d: _finalize_metrics(hits[d], ndcg[d], mrr[d], n_samples, TOP_K) for d in deltas}
    bout = None
    if with_buckets:
        bout = {
            d: {
                lb: _finalize_metrics(bhits[d][lb], bndcg[d][lb], bmrr[d][lb], bn[lb], TOP_K)
                for lb in labels
            }
            for d in deltas
        }
    return out, bout, n_samples, bn


def main() -> None:
    cfg = load_config("config", catsa_run=REF_RUN, catsa_suite="diginetica")
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

    print(f"[repeat_aware] Load models trên {device} ...", flush=True)
    models = {}
    for name, d in MODELS.items():
        if name.startswith("core"):
            models[name] = _load_core_model(d, n_items, device)
        else:
            models[name] = _load_catsa_model(d, data, device)

    val_loader = loader_for(data["val_sessions"])
    test_loader = loader_for(data["test_sessions"])

    result = {}
    for name, model in models.items():
        print(f"\n[repeat_aware] === {name} ===", flush=True)
        val_res, _, n_val, _ = _eval_deltas(val_loader, name, model, DELTAS, n_items, device)
        best_d = max(DELTAS, key=lambda d: val_res[d]["mrr@20"])
        print(f"  VAL ({n_val} mẫu) — quét delta:")
        for d in DELTAS:
            mark = " ←chọn" if d == best_d else ""
            print(f"    delta={d:g}: mrr@20={val_res[d]['mrr@20']:.4f}{mark}")

        test_deltas = sorted({0.0, best_d})
        test_res, test_buckets, n_test, bn = _eval_deltas(
            test_loader, name, model, test_deltas, n_items, device, with_buckets=True,
        )
        print(f"  TEST ({n_test} mẫu):")
        for d in test_deltas:
            m = test_res[d]
            tag = "repeat-aware" if d else "gốc"
            print(f"    [{tag}, delta={d:g}] mrr@20={m['mrr@20']:.4f} | hr@20={m['hr@20']:.4f} | ndcg@20={m['ndcg@20']:.4f}")
            for lb, bm in test_buckets[d].items():
                print(f"        bucket {lb} (n={bn[lb]}): mrr@20={bm['mrr@20']:.4f}")

        result[name] = {
            "delta_chon_tren_val": best_d,
            "val_sweep": {f"{d:g}": val_res[d] for d in DELTAS},
            "test": {f"{d:g}": test_res[d] for d in test_deltas},
            "test_buckets": {f"{d:g}": test_buckets[d] for d in test_deltas},
        }

    with open("checkpoints/repeat_aware_diginetica_result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    print("\n[repeat_aware] Đã ghi checkpoints/repeat_aware_diginetica_result.yaml")


if __name__ == "__main__":
    main()
