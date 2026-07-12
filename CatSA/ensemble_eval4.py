"""Phase 4 — ensemble 6 model: 4 CatSA cũ + bản A (cat_intent) + CORE.

Bản A (catsa_plus_v3_cat_intent, test đơn 0.3700) có repeat-boost NỘI TẠI và
nhánh category-intent → logits đa dạng so với nhóm cũ. Grid: trọng số riêng
cho (mean 4 CatSA cũ, bản A, CORE) + repeat-boost ngoài. Chọn trên VAL,
báo cáo TEST. Mốc phase 3: test 0.3878 (w0.65|rep8).

Chạy:
    python CatSA/ensemble_eval4.py
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
    CATSA_CKPTS, CORE_CKPT, REF_RUN, TOP_K,
    _core_logits, _load_catsa_model, _load_core_model,
)

CKPT_A = "checkpoints/CatSA/retailrocket/catsa_plus_v3_cat_intent"

W_AS = [0.0, 0.15, 0.3, 0.45]
W_CORES = [0.2, 0.3, 0.35]
D_REPS = [4.0, 6.0, 8.0, 10.0]


def _variants():
    out = []
    for wa in W_AS:
        for wc in W_CORES:
            wo = 1.0 - wa - wc
            if wo <= 0:
                continue
            for dr in D_REPS:
                out.append((wo, wa, wc, dr))
    return out


@torch.no_grad()
def _eval(loader, old_models, model_a, core_model, variants, n_items, device):
    keys = [f"old{wo:g}|A{wa:g}|core{wc:g}|rep{dr:g}" for wo, wa, wc, dr in variants]
    hits = {c: {kk: 0 for kk in TOP_K} for c in keys}
    ndcg = {c: {kk: 0.0 for kk in TOP_K} for c in keys}
    mrr = {c: {kk: 0.0 for kk in TOP_K} for c in keys}
    n_samples = 0

    for batch, _, targets in loader:
        batch = batch.to(device)
        targets = targets.to(device)
        sessions = batch.session_lists
        B = len(sessions)

        old_logits = None
        for m in old_models:
            z = m(batch)
            lg = m.scores(z, batch)
            old_logits = lg if old_logits is None else old_logits + lg
        old_logits = old_logits / len(old_models)
        z = model_a(batch)
        a_logits = model_a.scores(z, batch)
        core_logits = _core_logits(core_model, sessions, device)

        rep_flat = torch.zeros(B, n_items, device=device)
        for i, s in enumerate(sessions):
            rep_flat[i, torch.tensor(list(set(s)), device=device)] = 1.0

        for (wo, wa, wc, dr), c in zip(variants, keys):
            fused = wo * old_logits + wa * a_logits + wc * core_logits + dr * rep_flat
            ranks = _ranks_from_logits(fused, targets)
            _accumulate_ranks(ranks, TOP_K, hits[c], ndcg[c], mrr[c])
        n_samples += targets.size(0)

    return {
        c: _finalize_metrics(hits[c], ndcg[c], mrr[c], n_samples, TOP_K) for c in keys
    }, n_samples


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

    print(f"[phase4] Load 4 CatSA cũ + bản A + CORE trên {device} ...", flush=True)
    old_models = [_load_catsa_model(d, data, device) for _, d in CATSA_CKPTS]
    model_a = _load_catsa_model(CKPT_A, data, device)
    core_model = _load_core_model(CORE_CKPT[1], n_items, device)

    variants = _variants()
    print(f"[phase4] {len(variants)} biến thể — VAL ...", flush=True)
    val_res, n_val = _eval(
        loader_for(data["val_sessions"]), old_models, model_a, core_model, variants, n_items, device,
    )
    ranked = sorted(val_res.items(), key=lambda kv: kv[1]["mrr@20"], reverse=True)
    print(f"\n[phase4] VAL ({n_val} mẫu) — top 15:")
    for c, m in ranked[:15]:
        print(f"  {m['mrr@20']:.4f}  hr@20={m['hr@20']:.4f}  {c}")

    key2var = dict(zip(
        [f"old{wo:g}|A{wa:g}|core{wc:g}|rep{dr:g}" for wo, wa, wc, dr in variants], variants,
    ))
    top_vars = [key2var[c] for c, _ in ranked[:6]]
    print("\n[phase4] TEST với top-6 theo val ...", flush=True)
    test_res, n_test = _eval(
        loader_for(data["test_sessions"]), old_models, model_a, core_model, top_vars, n_items, device,
    )
    print(f"\n[phase4] KẾT QUẢ TEST ({n_test} mẫu):")
    for c, m in sorted(test_res.items(), key=lambda kv: kv[1]["mrr@20"], reverse=True):
        print(f"  mrr@20={m['mrr@20']:.4f}  hr@20={m['hr@20']:.4f}  ndcg@20={m['ndcg@20']:.4f}  {c}")

    best = ranked[0][0]
    if best in test_res:
        m = test_res[best]
        print(f"\n[phase4] Combo chọn theo VAL: {best}")
        print(f"[phase4] → TEST chính thức: mrr@20={m['mrr@20']:.4f} | hr@20={m['hr@20']:.4f} | ndcg@20={m['ndcg@20']:.4f}")

    with open("checkpoints/ensemble_eval4_result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "val_top15": {c: v for c, v in ranked[:15]},
            "test": test_res,
            "combo_chon_theo_val": best,
        }, f, allow_unicode=True, sort_keys=False)
    print("[phase4] Đã ghi checkpoints/ensemble_eval4_result.yaml")


if __name__ == "__main__":
    main()
