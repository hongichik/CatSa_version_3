"""Phase 6 — trọng số ensemble RIÊNG THEO BUCKET độ dài phiên.

Quan sát: bản A (cat_intent+repeat) mạnh nhất ở phiên ngắn (1-3: 0.4379 vs
len_gate 0.4339), CORE/len_gate nhỉnh hơn ở phiên dài → 1 bộ trọng số chung
là thoả hiệp. Ở đây: tune bộ trọng số TỐI ƯU RIÊNG cho từng bucket
(≤3, 4-7, ≥8) trên VAL (hợp lệ — không đụng test), rồi ghép lại đánh giá
TEST một lần.

Tự động thêm len_gate_seed43 làm model thứ 8 nếu đã train xong.

Chạy:
    python CatSA/ensemble_eval6.py
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
    CATSA_CKPTS, REF_RUN, TOP_K,
    _core_logits, _load_catsa_model, _load_core_model,
)

CKPT_A = "checkpoints/CatSA/retailrocket/catsa_plus_v3_cat_intent"
CKPT_LG43 = "checkpoints/CatSA/retailrocket/catsa_plus_v2_len_gate_seed43"
CORE_DIRS: list[str] = []  # THUẦN CatSA — bỏ CORE khỏi ensemble
_CORE_DIRS_GOC = [
    "checkpoints/CORE/retailrocket/core_trm",
    "checkpoints/CORE/retailrocket/core_trm_seed43",
]

BUCKETS = [(1, 3), (4, 7), (8, None)]
W_OLDS = [0.2, 0.3, 0.4, 0.5]
W_AS = [0.1, 0.2, 0.3, 0.4]
D_REPS = [8.0, 12.0]


def _lg43_ready() -> bool:
    info = Path(CKPT_LG43) / "info.yaml"
    if not info.exists():
        return False
    d = yaml.safe_load(open(info, encoding="utf-8"))
    return "test_metrics" in d


def _bucket_label(lo: int, hi: int | None) -> str:
    return f"{lo}_{hi}" if hi is not None else f"{lo}_plus"


def _variants():
    """(w_old, w_A, w_rest, rep) — w_rest chia đều cho các model 'rest'."""
    out = []
    for wo in W_OLDS:
        for wa in W_AS:
            wr = 1.0 - wo - wa
            if wr <= 0:
                continue
            for dr in D_REPS:
                out.append((wo, wa, wr, dr))
    return out


@torch.no_grad()
def _collect_logits(batch, old_models, model_a, rest_models, sessions, device):
    old_logits = None
    for m in old_models:
        z = m(batch)
        lg = m.scores(z, batch)
        old_logits = lg if old_logits is None else old_logits + lg
    old_logits = old_logits / len(old_models)
    z = model_a(batch)
    a_logits = model_a.scores(z, batch)
    rest_logits = None
    for kind, m in rest_models:
        lg = _core_logits(m, sessions, device) if kind == "core" else None
        if lg is None:
            z = m(batch)
            lg = m.scores(z, batch)
        rest_logits = lg if rest_logits is None else rest_logits + lg
    rest_logits = rest_logits / len(rest_models)
    return old_logits, a_logits, rest_logits


@torch.no_grad()
def _eval_per_bucket(loader, old_models, model_a, rest_models, variants, n_items, device):
    """Trả metrics[bucket_label][variant_key] trên loader."""
    labels = [_bucket_label(lo, hi) for lo, hi in BUCKETS]
    keys = [f"old{wo:g}|A{wa:g}|rest{wr:g}|rep{dr:g}" for wo, wa, wr, dr in variants]
    hits = {lb: {c: {k: 0 for k in TOP_K} for c in keys} for lb in labels}
    ndcg = {lb: {c: {k: 0.0 for k in TOP_K} for c in keys} for lb in labels}
    mrr = {lb: {c: {k: 0.0 for k in TOP_K} for c in keys} for lb in labels}
    n_s = {lb: 0 for lb in labels}

    for batch, _, targets in loader:
        batch = batch.to(device)
        targets = targets.to(device)
        sessions = batch.session_lists
        B = len(sessions)

        old_l, a_l, rest_l = _collect_logits(batch, old_models, model_a, rest_models, sessions, device)
        rep_flat = torch.zeros(B, n_items, device=device)
        for i, s in enumerate(sessions):
            rep_flat[i, torch.tensor(list(set(s)), device=device)] = 1.0
        lens = torch.tensor([len(s) for s in sessions], device=device)

        masks = {}
        for (lo, hi), lb in zip(BUCKETS, labels):
            m = lens >= lo
            if hi is not None:
                m &= lens <= hi
            masks[lb] = m
            n_s[lb] += int(m.sum())

        for (wo, wa, wr, dr), c in zip(variants, keys):
            fused = wo * old_l + wa * a_l + wr * rest_l + dr * rep_flat
            ranks = _ranks_from_logits(fused, targets)
            for lb in labels:
                m = masks[lb]
                if m.any():
                    _accumulate_ranks(ranks[m], TOP_K, hits[lb][c], ndcg[lb][c], mrr[lb][c])

    out = {}
    for lb in labels:
        out[lb] = {
            c: _finalize_metrics(hits[lb][c], ndcg[lb][c], mrr[lb][c], n_s[lb], TOP_K)
            for c in keys
        }
    return out, n_s


@torch.no_grad()
def _eval_routed(loader, old_models, model_a, rest_models, bucket_weights, n_items, device):
    """Đánh giá 1 lần với bộ trọng số riêng theo bucket."""
    hits = {k: 0 for k in TOP_K}
    ndcg = {k: 0.0 for k in TOP_K}
    mrr = {k: 0.0 for k in TOP_K}
    n_samples = 0
    labels = [_bucket_label(lo, hi) for lo, hi in BUCKETS]

    for batch, _, targets in loader:
        batch = batch.to(device)
        targets = targets.to(device)
        sessions = batch.session_lists
        B = len(sessions)

        old_l, a_l, rest_l = _collect_logits(batch, old_models, model_a, rest_models, sessions, device)
        rep_flat = torch.zeros(B, n_items, device=device)
        for i, s in enumerate(sessions):
            rep_flat[i, torch.tensor(list(set(s)), device=device)] = 1.0
        lens = torch.tensor([len(s) for s in sessions], device=device)

        fused = torch.empty_like(old_l)
        for (lo, hi), lb in zip(BUCKETS, labels):
            m = lens >= lo
            if hi is not None:
                m &= lens <= hi
            if not m.any():
                continue
            wo, wa, wr, dr = bucket_weights[lb]
            fused[m] = (
                wo * old_l[m] + wa * a_l[m] + wr * rest_l[m] + dr * rep_flat[m]
            )
        ranks = _ranks_from_logits(fused, targets)
        _accumulate_ranks(ranks, TOP_K, hits, ndcg, mrr)
        n_samples += targets.size(0)

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

    use_lg43 = _lg43_ready()
    print(f"[phase7-pure] len_gate_seed43 sẵn sàng: {use_lg43}", flush=True)
    print(f"[phase7-pure] Load models trên {device} ...", flush=True)
    old_models = [_load_catsa_model(d, data, device) for _, d in CATSA_CKPTS]
    model_a = _load_catsa_model(CKPT_A, data, device)
    rest_models = [("core", _load_core_model(d, n_items, device)) for d in CORE_DIRS]
    if use_lg43:
        rest_models.append(("catsa", _load_catsa_model(CKPT_LG43, data, device)))

    variants = _variants()
    keys = [f"old{wo:g}|A{wa:g}|rest{wr:g}|rep{dr:g}" for wo, wa, wr, dr in variants]
    key2var = dict(zip(keys, variants))

    print(f"[phase7-pure] {len(variants)} biến thể × 3 bucket — VAL ...", flush=True)
    val_res, n_val = _eval_per_bucket(
        loader_for(data["val_sessions"]), old_models, model_a, rest_models, variants, n_items, device,
    )

    bucket_weights = {}
    print("\n[phase7-pure] Chọn trọng số riêng từng bucket (theo val mrr@20):")
    for lb, res in val_res.items():
        best_key = max(res, key=lambda c: res[c]["mrr@20"])
        bucket_weights[lb] = key2var[best_key]
        print(f"  bucket {lb} (n={n_val[lb]}): {best_key} — val mrr@20={res[best_key]['mrr@20']:.4f}")

    print("\n[phase7-pure] TEST với bộ trọng số ghép theo bucket ...", flush=True)
    test_metrics, n_test = _eval_routed(
        loader_for(data["test_sessions"]), old_models, model_a, rest_models, bucket_weights, n_items, device,
    )
    print(f"\n[phase7-pure] KẾT QUẢ TEST ({n_test} mẫu):")
    print("  " + " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    print(f"\n[phase7-pure] → TEST chính thức: mrr@20={test_metrics['mrr@20']:.4f}")

    with open("checkpoints/ensemble_eval7_pure_catsa_result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "use_lengate_seed43": use_lg43,
            "bucket_weights": {lb: list(w) for lb, w in bucket_weights.items()},
            "test_metrics": test_metrics,
        }, f, allow_unicode=True, sort_keys=False)
    print("[phase7-pure] Đã ghi checkpoints/ensemble_eval7_pure_catsa_result.yaml")


if __name__ == "__main__":
    main()
