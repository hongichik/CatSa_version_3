"""Phase 3 — thêm tín hiệu Sequential Rules (SR/Markov) từ TRAIN data.

Nghiên cứu Ludewig & Jannach (2018) "Evaluation of Session-based Recommendation
Algorithms" chỉ ra heuristic dạng SR/vSKNN cực mạnh trên RetailRocket. Ở đây:
xây ma trận chuyển tiếp có trọng số suy giảm theo khoảng cách (SR: w=1/gap)
từ train sessions, chuẩn hoá hàng theo log1p rồi cộng vào logits ensemble.

Nền tảng: phase 2 tốt nhất = w_catsa 0.65-0.7 + repeat-boost flat delta 5-6
→ test mrr@20=0.3874. Phase 3 quét thêm w_sr và delta quanh vùng đó.

Chạy:
    python CatSA/ensemble_eval3.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

from common.config import load_config  # noqa: E402
from tienxuly import load_processed  # noqa: E402
from CatSA.dataset import make_loader  # noqa: E402
from CatSA.evaluate import _accumulate_ranks, _finalize_metrics, _ranks_from_logits  # noqa: E402
from CatSA.ensemble_eval import (  # noqa: E402
    CATSA_CKPTS, CORE_CKPT, REF_RUN, TOP_K,
    _core_logits, _load_catsa_model, _load_core_model,
)

W_CATSA = [0.65, 0.7]
D_REPS = [4.0, 5.0, 6.0, 8.0]
W_SRS = [0.0, 0.5, 1.0, 2.0, 4.0]
SR_MAX_GAP = 3  # SR: chỉ đếm cặp cách nhau <= gap, trọng số 1/gap


def build_sr_matrix(train_sessions: list[list[int]], n_items: int) -> csr_matrix:
    """Ma trận SR (n_items × n_items): m[a,b] = Σ 1/gap với mọi cặp (a … b)."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for s in train_sessions:
        for i in range(len(s) - 1):
            for j in range(i + 1, min(i + 1 + SR_MAX_GAP, len(s))):
                rows.append(s[i])
                cols.append(s[j])
                vals.append(1.0 / (j - i))
    m = csr_matrix(
        (np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(n_items, n_items),
    )
    # log1p nén đuôi phân phối count, rồi chia max hàng → [0, 1]
    m.data = np.log1p(m.data)
    row_max = m.max(axis=1).toarray().ravel()
    row_max[row_max == 0] = 1.0
    d = 1.0 / row_max
    m = csr_matrix((m.data * np.repeat(d, np.diff(m.indptr)), m.indices, m.indptr), shape=m.shape)
    return m


def _sr_rows(sr: csr_matrix, last_items: list[int], device: torch.device) -> torch.Tensor:
    dense = np.asarray(sr[last_items].todense(), dtype=np.float32)
    return torch.from_numpy(dense).to(device)


@torch.no_grad()
def _eval(loader, catsa_models, core_model, sr, variants, n_items, device):
    keys = [f"w{w:g}|rep{dr:g}|sr{ws:g}" for w, dr, ws in variants]
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

        rep_flat = torch.zeros(B, n_items, device=device)
        for i, s in enumerate(sessions):
            rep_flat[i, torch.tensor(list(set(s)), device=device)] = 1.0
        sr_rows = _sr_rows(sr, [s[-1] for s in sessions], device)

        for (w, dr, ws), c in zip(variants, keys):
            fused = w * catsa_logits + (1 - w) * core_logits + dr * rep_flat
            if ws:
                fused = fused + ws * sr_rows
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

    print("[phase3] Xây ma trận SR từ train ...", flush=True)
    sr = build_sr_matrix(data["train_sessions"], n_items)
    print(f"[phase3] SR: {sr.nnz} cặp khác 0", flush=True)

    print(f"[phase3] Load models trên {device} ...", flush=True)
    catsa_models = [_load_catsa_model(d, data, device) for _, d in CATSA_CKPTS]
    core_model = _load_core_model(CORE_CKPT[1], n_items, device)

    variants = [(w, dr, ws) for w in W_CATSA for dr in D_REPS for ws in W_SRS]
    print(f"[phase3] {len(variants)} biến thể — VAL ...", flush=True)
    val_res, n_val = _eval(
        loader_for(data["val_sessions"]), catsa_models, core_model, sr, variants, n_items, device,
    )
    ranked = sorted(val_res.items(), key=lambda kv: kv[1]["mrr@20"], reverse=True)
    print(f"\n[phase3] VAL ({n_val} mẫu) — top 15:")
    for c, m in ranked[:15]:
        print(f"  {m['mrr@20']:.4f}  hr@20={m['hr@20']:.4f}  {c}")

    key2var = dict(zip([f"w{w:g}|rep{dr:g}|sr{ws:g}" for w, dr, ws in variants], variants))
    top_vars = [key2var[c] for c, _ in ranked[:6]]
    print("\n[phase3] TEST với top-6 theo val ...", flush=True)
    test_res, n_test = _eval(
        loader_for(data["test_sessions"]), catsa_models, core_model, sr, top_vars, n_items, device,
    )
    print(f"\n[phase3] KẾT QUẢ TEST ({n_test} mẫu):")
    for c, m in sorted(test_res.items(), key=lambda kv: kv[1]["mrr@20"], reverse=True):
        print(f"  mrr@20={m['mrr@20']:.4f}  hr@20={m['hr@20']:.4f}  ndcg@20={m['ndcg@20']:.4f}  {c}")

    best = ranked[0][0]
    if best in test_res:
        m = test_res[best]
        print(f"\n[phase3] Combo chọn theo VAL: {best}")
        print(f"[phase3] → TEST chính thức: mrr@20={m['mrr@20']:.4f} | hr@20={m['hr@20']:.4f} | ndcg@20={m['ndcg@20']:.4f}")

    with open("checkpoints/ensemble_eval3_result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "val_top15": {c: v for c, v in ranked[:15]},
            "test": test_res,
            "combo_chon_theo_val": best,
        }, f, allow_unicode=True, sort_keys=False)
    print("[phase3] Đã ghi checkpoints/ensemble_eval3_result.yaml")


if __name__ == "__main__":
    main()
