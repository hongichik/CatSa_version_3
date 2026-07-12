"""Multi-seed runner (finding T4, CatSA_Correctness_Synthesis) — chạy 1 file
version qua N seed, tổng hợp mean/std, đánh dấu rõ single-seed KHÔNG được
báo cáo đơn lẻ.

Chạy:
    python CatSA/multi_seed.py --run baseline/catsa_plus_v2_len_gate.yaml --seeds 42,43,44,45,46
    python CatSA/multi_seed.py --run baseline/catsa_plus_v2_len_gate.yaml --n-seeds 5 --base-seed 42

Kết quả: in bảng mean±std ra console + ghi
    checkpoints/<save_dir gốc>/multi_seed_summary.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import yaml  # noqa: E402

from common import load_config, setup_logger, init_wandb, finish_wandb  # noqa: E402
from tienxuly import load_processed  # noqa: E402
from CatSA import train_model  # noqa: E402


def _run_single_seed(config_dir: str, catsa_run: str, suite: str, seed: int) -> dict[str, float]:
    cfg = load_config(config_dir, catsa_run=catsa_run, catsa_suite=suite)
    cfg = copy.deepcopy(cfg)
    cfg.training.seed = seed

    # Tách output theo seed — tránh nhiều seed ghi đè cùng 1 checkpoint/log
    if cfg.training.save_dir:
        cfg.training.save_dir = f"{cfg.training.save_dir}_seed{seed}"
    if cfg.project.filename_mode == "custom" and cfg.project.custom_filename:
        stem, ext = os.path.splitext(cfg.project.custom_filename)
        cfg.project.custom_filename = f"{stem}_seed{seed}{ext}"

    log = setup_logger(cfg.logging, cfg.project)
    log.info("=== Multi-seed run: seed=%d ===", seed)
    init_wandb(cfg)
    try:
        data = load_processed(cfg.data)
        _, test_metrics = train_model(data, cfg)
        return test_metrics
    finally:
        finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy CatSA qua nhiều seed và tổng hợp mean/std")
    parser.add_argument("--config", default="config")
    parser.add_argument("--run", required=True, metavar="FILE")
    parser.add_argument("--suite", default="retailrocket", choices=["retailrocket", "diginetica"])
    parser.add_argument("--seeds", default=None, help="Danh sách seed, vd 42,43,44,45,46")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [args.base_seed + i for i in range(args.n_seeds)]

    print(f"[multi_seed] Chạy {args.run} qua {len(seeds)} seed: {seeds}")
    all_metrics: list[dict[str, float]] = []
    for seed in seeds:
        print(f"[multi_seed] --- seed={seed} ---")
        metrics = _run_single_seed(args.config, args.run, args.suite, seed)
        all_metrics.append(metrics)
        print(f"[multi_seed] seed={seed}: {metrics}")

    keys = sorted(all_metrics[0].keys())
    summary: dict[str, dict[str, float]] = {}
    print("\n[multi_seed] === TỔNG HỢP (mean ± std, KHÔNG dùng số 1 seed đơn lẻ để báo cáo) ===")
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[k] = {"mean": mean, "std": std, "n_seeds": len(vals), "values": vals}
        print(f"  {k}: {mean:.4f} ± {std:.4f}  (n={len(vals)})")

    out_dir = Path("checkpoints") / "multi_seed_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.run).stem}_seeds_{'_'.join(map(str, seeds))}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "run": args.run,
                "suite": args.suite,
                "seeds": seeds,
                "single_seed_non_reportable": True,
                "summary": summary,
            },
            f, allow_unicode=True, sort_keys=False,
        )
    print(f"[multi_seed] Đã ghi: {out_path}")


if __name__ == "__main__":
    main()
