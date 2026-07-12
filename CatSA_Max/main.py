"""CatSA_Max — tái hiện kết quả ensemble thuần CatSA (test mrr@20 ≈ 0.3895)
từ A-Z bằng MỘT lệnh duy nhất:

    python CatSA_Max/main.py

Quy trình tự động:
  1. Train tuần tự 6 thành viên (tất cả đều là kiến trúc CatSA++ v2, khác
     nhau ở công tắc config/seed) → checkpoints/CatSA_Max/<tên>/
     - Tự RESUME: thành viên nào đã có KẾT QUẢ TEST thì bỏ qua (chạy lại
       lệnh sau khi bị ngắt sẽ tiếp tục từ chỗ dở).
  2. Ensemble bucket-routing: chọn trọng số riêng cho từng nhóm độ dài
     phiên (1-3 / 4-7 / 8+) trên VAL, cộng repeat-boost, đánh giá TEST.
  3. Ghi kết quả cuối vào CatSA_Max/result.yaml.

Tùy chọn:
    --retrain      train lại tất cả kể cả đã xong
    --skip-train   bỏ qua bước train (dùng checkpoint có sẵn), chỉ ensemble

Thành viên (xem MEMBERS bên dưới):
    v2              CatSA++ v2 base (InfoNCE)
    proto           + prototype loss (cl_type both)
    multi           + length gate + multi-interest readout
    len_gate        + length-aware gate
    len_gate_seed43 = len_gate, seed 43 (đa dạng thuần trọng số)
    cat_intent      + category-intent + repeat-boost học được

LƯU Ý trung thực khi công bố: nhánh sequential của CatSA++ v2 dùng
transformer encoder mượn từ CORE (đã ghi trong docstring encoder) — khai
báo trong Method; repeat-boost + bucket-routing là thành phần inference.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from common import finish_wandb, init_wandb, load_config, setup_logger  # noqa: E402
from tienxuly import load_processed  # noqa: E402
from CatSA import train_model  # noqa: E402
from CatSA.dataset import make_loader  # noqa: E402
from CatSA.ensemble_eval import _load_catsa_model  # noqa: E402
from CatSA.ensemble_eval6 import (  # noqa: E402
    BUCKETS, _bucket_label, _eval_per_bucket, _eval_routed, _variants,
)

REF_RUN = "baseline/catsa_plus_v2_len_gate.yaml"
CKPT_ROOT = Path("checkpoints/CatSA_Max")

# Định nghĩa 6 thành viên = override trên config tham chiếu (len_gate).
# Config tham chiếu đã có: length_aware_gate=true, cl_type=both,
# lambda_proto=0.05, use_multi_interest=false, use_cat_intent=false,
# use_repeat_boost=false, seed=42.
MEMBERS: dict[str, dict[str, dict]] = {
    "v2": {
        "model": {"length_aware_gate": False},
        "training": {"cl_type": "infonce", "lambda_proto": 0.0},
    },
    "proto": {"model": {"length_aware_gate": False}, "training": {}},
    "multi": {"model": {"use_multi_interest": True}, "training": {}},
    "len_gate": {"model": {}, "training": {}},
    "len_gate_seed43": {"model": {}, "training": {"seed": 43}},
    "cat_intent": {
        "model": {"use_cat_intent": True, "use_repeat_boost": True},
        "training": {},
    },
}
# Vai trò trong ensemble (khớp cấu trúc phase 7 — ensemble_eval7_pure)
GROUP_OLD = ["v2", "proto", "multi", "len_gate"]
GROUP_A = "cat_intent"
GROUP_REST = ["len_gate_seed43"]


def _member_cfg(name: str):
    cfg = copy.deepcopy(load_config("config", catsa_run=REF_RUN, catsa_suite="retailrocket"))
    for k, v in MEMBERS[name]["model"].items():
        setattr(cfg.model, k, v)
    for k, v in MEMBERS[name]["training"].items():
        setattr(cfg.training, k, v)
    cfg.training.save_dir = str(CKPT_ROOT / name)
    cfg.project.filename_mode = "custom"
    cfg.project.custom_filename = f"CatSA_Max/{name}.log"
    cfg.version = f"catsa_max_{name}"
    return cfg


def _member_done(name: str) -> bool:
    info = CKPT_ROOT / name / "info.yaml"
    if not info.exists():
        return False
    d = yaml.safe_load(open(info, encoding="utf-8"))
    return "test_metrics" in d


def _train_members(retrain: bool) -> None:
    for i, name in enumerate(MEMBERS, 1):
        if not retrain and _member_done(name):
            print(f"[CatSA_Max] ({i}/{len(MEMBERS)}) {name}: ĐÃ XONG — bỏ qua", flush=True)
            continue
        print(f"[CatSA_Max] ({i}/{len(MEMBERS)}) train {name} ...", flush=True)
        cfg = _member_cfg(name)
        log = setup_logger(cfg.logging, cfg.project)
        log.info("=== CatSA_Max member: %s ===", name)
        init_wandb(cfg)
        try:
            data = load_processed(cfg.data)
            _, test_metrics = train_model(data, cfg)
            print(f"[CatSA_Max] {name} xong: {test_metrics}", flush=True)
        finally:
            finish_wandb()


def _run_ensemble() -> dict:
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

    print(f"[CatSA_Max] Load 6 thành viên trên {device} ...", flush=True)
    old_models = [_load_catsa_model(str(CKPT_ROOT / n), data, device) for n in GROUP_OLD]
    model_a = _load_catsa_model(str(CKPT_ROOT / GROUP_A), data, device)
    rest_models = [("catsa", _load_catsa_model(str(CKPT_ROOT / n), data, device)) for n in GROUP_REST]

    variants = _variants()
    keys = [f"old{wo:g}|A{wa:g}|rest{wr:g}|rep{dr:g}" for wo, wa, wr, dr in variants]
    key2var = dict(zip(keys, variants))

    print(f"[CatSA_Max] {len(variants)} biến thể × 3 bucket — chọn trọng số trên VAL ...", flush=True)
    val_res, n_val = _eval_per_bucket(
        loader_for(data["val_sessions"]), old_models, model_a, rest_models, variants, n_items, device,
    )
    bucket_weights = {}
    for lb, res in val_res.items():
        best = max(res, key=lambda c: res[c]["mrr@20"])
        bucket_weights[lb] = key2var[best]
        print(f"  bucket {lb} (n={n_val[lb]}): {best} — val mrr@20={res[best]['mrr@20']:.4f}", flush=True)

    print("[CatSA_Max] Đánh giá TEST ...", flush=True)
    test_metrics, n_test = _eval_routed(
        loader_for(data["test_sessions"]), old_models, model_a, rest_models, bucket_weights, n_items, device,
    )
    print(f"\n[CatSA_Max] ===== KẾT QUẢ CUỐI (TEST, {n_test} mẫu) =====")
    print("  " + " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    result = {
        "mo_ta": "CatSA_Max — ensemble thuần CatSA 6 thành viên + repeat-boost + bucket-routing",
        "thanh_vien": {n: str(CKPT_ROOT / n) for n in MEMBERS},
        "bucket_weights (old, A, rest, repeat_delta)": {lb: list(w) for lb, w in bucket_weights.items()},
        "test_metrics": test_metrics,
    }
    out = Path("CatSA_Max/result.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    print(f"[CatSA_Max] Đã ghi {out}")
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CatSA_Max — train 6 thành viên + ensemble, 1 lệnh")
    parser.add_argument("--retrain", action="store_true", help="train lại tất cả kể cả đã xong")
    parser.add_argument("--skip-train", action="store_true", help="chỉ chạy ensemble từ checkpoint có sẵn")
    args = parser.parse_args()

    if not args.skip_train:
        _train_members(args.retrain)
    missing = [n for n in MEMBERS if not (CKPT_ROOT / n / "best_model.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu checkpoint thành viên: {missing} — chạy lại không có --skip-train")
    _run_ensemble()


if __name__ == "__main__":
    main()
