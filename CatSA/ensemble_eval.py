"""Bản B (không ràng buộc) — ensemble logits nhiều checkpoint để đẩy MRR@20.

Ý tưởng: các model CatSA (len_gate/proto/multi/v2) và CORE-trm đều chấm điểm
cosine/temperature trên CÙNG vocabulary + CÙNG per-prefix protocol → có thể
cộng logits trực tiếp. Chọn tổ hợp trọng số trên VAL, báo cáo trên TEST
(không đụng test khi chọn combo). Thêm "repeat-boost": cộng delta vào logit
các item đã xuất hiện trong prefix (protocol không mask repeat item).

Chạy:
    python CatSA/ensemble_eval.py
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from common.config import ModelConfig, load_config  # noqa: E402
from tienxuly import load_processed  # noqa: E402
from CatSA.dataset import make_loader  # noqa: E402
from CatSA.encoders import build_encoder  # noqa: E402
from CatSA.evaluate import _accumulate_ranks, _finalize_metrics, _ranks_from_logits  # noqa: E402
from CORE.models import COREtrm  # noqa: E402

TOP_K = [10, 20]
REF_RUN = "baseline/catsa_plus_v2_len_gate.yaml"

CATSA_CKPTS = [
    ("len_gate", "checkpoints/CatSA/retailrocket/catsa_plus_v2_len_gate"),
    ("proto", "checkpoints/CatSA/retailrocket/catsa_plus_v2_proto"),
    ("multi", "checkpoints/CatSA/retailrocket/catsa_plus_v2_multi"),
    ("v2", "checkpoints/CatSA/retailrocket/catsa_plus_v2"),
]
CORE_CKPT = ("core", "checkpoints/CORE/retailrocket/core_trm")

REPEAT_DELTAS = [0.0, 0.5, 1.0, 2.0]


def _filter_fields(cls, d: dict) -> dict:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def _load_catsa_model(ckpt_dir: str, data: dict, device: torch.device):
    info = yaml.safe_load(open(Path(ckpt_dir) / "info.yaml", encoding="utf-8"))
    mc = ModelConfig(**_filter_fields(ModelConfig, info["cau_hinh"]["model"]))
    assert not mc.use_star_node, f"{ckpt_dir}: star-node cần loader riêng, bỏ qua"
    model = build_encoder(
        mc, data["n_items"], data["n_cats"],
        item2cat=data["item2cat"], cat2items=data.get("cat2items"),
        cat_parent=data["cat_parent"] if mc.use_taxonomy else None,
    ).to(device)
    state = torch.load(Path(ckpt_dir) / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    return model


def _load_core_model(ckpt_dir: str, n_items: int, device: torch.device):
    info = yaml.safe_load(open(Path(ckpt_dir) / "info.yaml", encoding="utf-8"))
    m = info["cau_hinh"]["core_model"]
    model = COREtrm(
        n_items=n_items,
        embedding_size=m["embedding_size"],
        sess_dropout=m["sess_dropout"],
        item_dropout=m["item_dropout"],
        temperature=m["temperature"],
        max_seq_length=m["max_seq_length"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        inner_size=m["inner_size"],
        hidden_dropout_prob=m["hidden_dropout_prob"],
        attn_dropout_prob=m["attn_dropout_prob"],
        hidden_act=m["hidden_act"],
        layer_norm_eps=float(m["layer_norm_eps"]),
        initializer_range=m["initializer_range"],
    ).to(device)
    model.load_state_dict(torch.load(Path(ckpt_dir) / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    return model


def _core_logits(model, sessions: list[list[int]], device: torch.device) -> torch.Tensor:
    max_len = max(len(s) for s in sessions)
    seq = torch.zeros(len(sessions), max_len, dtype=torch.long, device=device)
    for i, s in enumerate(sessions):
        seq[i, : len(s)] = torch.tensor([it + 1 for it in s], device=device)
    return model.predict_scores(seq)[:, 1:]  # bỏ cột padding → khớp id CatSA


def _make_combos(names: list[str]) -> list[tuple[str, torch.Tensor]]:
    """Danh sách (tên, vector trọng số K) — singles, subset uniform, pair sweep."""
    k = len(names)
    combos: list[tuple[str, list[float]]] = []
    for i, nm in enumerate(names):
        w = [0.0] * k
        w[i] = 1.0
        combos.append((nm, w))
    for r in range(2, k + 1):
        for subset in itertools.combinations(range(k), r):
            w = [0.0] * k
            for i in subset:
                w[i] = 1.0 / r
            combos.append(("+".join(names[i] for i in subset), w))
    # pair sweep len_gate ↔ core và catsa_mean ↔ core
    i_lg, i_core = names.index("len_gate"), names.index("core")
    catsa_idx = [i for i, nm in enumerate(names) if nm != "core"]
    for wc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        w = [0.0] * k
        w[i_lg], w[i_core] = 1 - wc, wc
        combos.append((f"len_gate*{1-wc:.1f}+core*{wc:.1f}", w))
        w2 = [0.0] * k
        for i in catsa_idx:
            w2[i] = (1 - wc) / len(catsa_idx)
        w2[i_core] = wc
        combos.append((f"catsa_mean*{1-wc:.1f}+core*{wc:.1f}", w2))
    return [(nm, torch.tensor(w)) for nm, w in combos]


@torch.no_grad()
def _eval_combos(loader, catsa_models, core_model, combos, deltas, n_items, device):
    keys = [f"{nm}|d{dl:g}" for nm, _ in combos for dl in deltas]
    hits = {c: {kk: 0 for kk in TOP_K} for c in keys}
    ndcg = {c: {kk: 0.0 for kk in TOP_K} for c in keys}
    mrr = {c: {kk: 0.0 for kk in TOP_K} for c in keys}
    n_samples = 0

    for batch, _, targets in loader:
        batch = batch.to(device)
        targets = targets.to(device)
        sessions = batch.session_lists

        logits_list = []
        for m in catsa_models:
            z = m(batch)
            logits_list.append(m.scores(z, batch))
        logits_list.append(_core_logits(core_model, sessions, device))
        stack = torch.stack(logits_list)  # (K, B, N)

        rep_mask = torch.zeros(len(sessions), n_items, device=device)
        for i, s in enumerate(sessions):
            rep_mask[i, torch.tensor(list(set(s)), device=device)] = 1.0

        for nm, w in combos:
            fused = torch.einsum("k,kbn->bn", w.to(device), stack)
            for dl in deltas:
                f2 = fused + dl * rep_mask if dl else fused
                ranks = _ranks_from_logits(f2, targets)
                c = f"{nm}|d{dl:g}"
                _accumulate_ranks(ranks, TOP_K, hits[c], ndcg[c], mrr[c])
        n_samples += targets.size(0)

    return {
        c: _finalize_metrics(hits[c], ndcg[c], mrr[c], n_samples, TOP_K)
        for c in keys
    }, n_samples


def main() -> None:
    cfg = load_config("config", catsa_run=REF_RUN, catsa_suite="retailrocket")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(cfg.data)
    n_items = data["n_items"]
    item2cat = data["item2cat"]
    cat_parent = data["cat_parent"]
    max_prefix = int(data.get("max_prefix_length", 50))

    def loader_for(sessions):
        return make_loader(
            sessions, item2cat, cat_parent, True,
            cfg.training.batch_size, shuffle=False, num_workers=0,
            max_prefix_length=max_prefix,
        )

    val_loader = loader_for(data["val_sessions"])
    test_loader = loader_for(data["test_sessions"])

    names = [nm for nm, _ in CATSA_CKPTS] + [CORE_CKPT[0]]
    print(f"[ensemble] Load {len(CATSA_CKPTS)} CatSA + CORE trên {device} ...", flush=True)
    catsa_models = [_load_catsa_model(d, data, device) for _, d in CATSA_CKPTS]
    core_model = _load_core_model(CORE_CKPT[1], n_items, device)

    combos = _make_combos(names)
    print(f"[ensemble] {len(combos)} tổ hợp × {len(REPEAT_DELTAS)} delta — đánh giá trên VAL ...", flush=True)
    val_res, n_val = _eval_combos(val_loader, catsa_models, core_model, combos, REPEAT_DELTAS, n_items, device)

    ranked = sorted(val_res.items(), key=lambda kv: kv[1]["mrr@20"], reverse=True)
    print(f"\n[ensemble] VAL ({n_val} mẫu) — top 15 theo mrr@20:")
    for c, m in ranked[:15]:
        print(f"  {m['mrr@20']:.4f}  hr@20={m['hr@20']:.4f}  {c}")

    # Đánh giá TEST: top-5 combo theo val + các single làm mốc
    top_combos = [c for c, _ in ranked[:5]]
    singles = [f"{nm}|d0" for nm in names]
    chosen = list(dict.fromkeys(top_combos + singles))
    combo_map = {nm: w for nm, w in combos}
    test_combos = []
    for c in chosen:
        nm, dl = c.rsplit("|d", 1)
        test_combos.append((nm, combo_map[nm]))
    test_deltas = sorted({float(c.rsplit("|d", 1)[1]) for c in chosen})

    print(f"\n[ensemble] Đánh giá TEST với {len(test_combos)} combo × delta {test_deltas} ...", flush=True)
    test_res, n_test = _eval_combos(
        test_loader, catsa_models, core_model,
        list(dict.fromkeys(test_combos)), test_deltas, n_items, device,
    )

    print(f"\n[ensemble] KẾT QUẢ TEST ({n_test} mẫu):")
    test_ranked = sorted(test_res.items(), key=lambda kv: kv[1]["mrr@20"], reverse=True)
    for c, m in test_ranked:
        mark = " ★" if c in chosen[:5] else ""
        print(f"  mrr@20={m['mrr@20']:.4f}  hr@20={m['hr@20']:.4f}  ndcg@20={m['ndcg@20']:.4f}  {c}{mark}")

    best_val_combo = ranked[0][0]
    print(f"\n[ensemble] Combo chọn theo VAL: {best_val_combo}")
    if best_val_combo in test_res:
        m = test_res[best_val_combo]
        print(f"[ensemble] → TEST chính thức: mrr@20={m['mrr@20']:.4f} | hr@20={m['hr@20']:.4f} | ndcg@20={m['ndcg@20']:.4f}")

    out = {
        "muc_tieu": "Bản B — ensemble không ràng buộc, chọn combo trên val",
        "val_top15": {c: v for c, v in ranked[:15]},
        "test": {c: v for c, v in test_ranked},
        "combo_chon_theo_val": best_val_combo,
    }
    out_path = Path("checkpoints/ensemble_eval_result.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    print(f"[ensemble] Đã ghi {out_path}")


if __name__ == "__main__":
    main()
