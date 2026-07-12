"""Phase 2 — tinh chỉnh quanh combo tốt nhất của ensemble_eval.py.

Phát hiện phase 1: catsa_mean*0.7+core*0.3 với repeat-boost delta=2 (BIÊN grid)
→ test 0.3858. Phase 2 quét: delta lớn hơn, boost riêng item CUỐI (dữ liệu giữ
duplicate liên tiếp → P(next == last) cao), boost theo SỐ LẦN xuất hiện, và
trọng số catsa/core mịn hơn. Chọn trên VAL, báo cáo TEST.

Chạy:
    python CatSA/ensemble_eval2.py
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

# (w_catsa, d_rep, d_last, count_scale)
WEIGHTS = [0.6, 0.65, 0.7, 0.75, 0.8]
D_REPS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
D_LASTS = [0.0, 1.0, 2.0]
COUNT_SCALES = [False, True]  # True: boost × min(count, 3)


def _variants():
    out = []
    for w in WEIGHTS:
        for dr in D_REPS:
            for dl in D_LASTS:
                for cs in COUNT_SCALES:
                    out.append((w, dr, dl, cs))
    return out


@torch.no_grad()
def _eval(loader, catsa_models, core_model, variants, n_items, device):
    keys = [f"w{w:g}|rep{dr:g}|last{dl:g}|{'cnt' if cs else 'flat'}" for w, dr, dl, cs in variants]
    hits = {c: {kk: 0 for kk in TOP_K} for c in keys}
    ndcg = {c: {kk: 0.0 for kk in TOP_K} for c in keys}
    mrr = {c: {kk: 0.0 for kk in TOP_K} for c in keys}
    n_samples = 0

    for batch, _, targets in loader:
        batch = batch.to(device)
        targets = targets.to(device)
        sessions = batch.session_lists
        B = len(sessions)

        catsa_logits = None
        for m in catsa_models:
            z = m(batch)
            lg = m.scores(z, batch)
            catsa_logits = lg if catsa_logits is None else catsa_logits + lg
        catsa_logits = catsa_logits / len(catsa_models)
        core_logits = _core_logits(core_model, sessions, device)

        # mask flat (0/1), mask theo count (capped 3), mask item cuối
        rep_flat = torch.zeros(B, n_items, device=device)
        rep_cnt = torch.zeros(B, n_items, device=device)
        last_mask = torch.zeros(B, n_items, device=device)
        for i, s in enumerate(sessions):
            idx = torch.tensor(s, device=device)
            rep_flat[i, idx] = 1.0
            rep_cnt[i].index_add_(0, idx, torch.ones(len(s), device=device))
            last_mask[i, s[-1]] = 1.0
        rep_cnt.clamp_(max=3.0)

        for (w, dr, dl, cs), c in zip(variants, keys):
            fused = w * catsa_logits + (1 - w) * core_logits
            boost = dr * (rep_cnt if cs else rep_flat)
            if dl:
                boost = boost + dl * last_mask
            ranks = _ranks_from_logits(fused + boost, targets)
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

    print(f"[phase2] Load models trên {device} ...", flush=True)
    catsa_models = [_load_catsa_model(d, data, device) for _, d in CATSA_CKPTS]
    core_model = _load_core_model(CORE_CKPT[1], n_items, device)

    variants = _variants()
    print(f"[phase2] {len(variants)} biến thể — VAL ...", flush=True)
    val_res, n_val = _eval(
        loader_for(data["val_sessions"]), catsa_models, core_model, variants, n_items, device,
    )
    ranked = sorted(val_res.items(), key=lambda kv: kv[1]["mrr@20"], reverse=True)
    print(f"\n[phase2] VAL ({n_val} mẫu) — top 20:")
    for c, m in ranked[:20]:
        print(f"  {m['mrr@20']:.4f}  hr@20={m['hr@20']:.4f}  {c}")

    key2var = dict(zip(
        [f"w{w:g}|rep{dr:g}|last{dl:g}|{'cnt' if cs else 'flat'}" for w, dr, dl, cs in variants],
        variants,
    ))
    top_vars = [key2var[c] for c, _ in ranked[:8]]
    print(f"\n[phase2] TEST với top-8 theo val ...", flush=True)
    test_res, n_test = _eval(
        loader_for(data["test_sessions"]), catsa_models, core_model, top_vars, n_items, device,
    )
    print(f"\n[phase2] KẾT QUẢ TEST ({n_test} mẫu):")
    for c, m in sorted(test_res.items(), key=lambda kv: kv[1]["mrr@20"], reverse=True):
        print(f"  mrr@20={m['mrr@20']:.4f}  hr@20={m['hr@20']:.4f}  ndcg@20={m['ndcg@20']:.4f}  {c}")

    best = ranked[0][0]
    if best in test_res:
        m = test_res[best]
        print(f"\n[phase2] Combo chọn theo VAL: {best}")
        print(f"[phase2] → TEST chính thức: mrr@20={m['mrr@20']:.4f} | hr@20={m['hr@20']:.4f} | ndcg@20={m['ndcg@20']:.4f}")

    with open("checkpoints/ensemble_eval2_result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "val_top20": {c: v for c, v in ranked[:20]},
            "test": test_res,
            "combo_chon_theo_val": best,
        }, f, allow_unicode=True, sort_keys=False)
    print("[phase2] Đã ghi checkpoints/ensemble_eval2_result.yaml")


if __name__ == "__main__":
    main()
