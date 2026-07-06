"""Entrypoint RIÊNG của bước huấn luyện CatSA — chạy độc lập, không qua main tập trung.

Cách chạy (từ bất kỳ đâu, kết quả luôn ghi về thư mục gốc dự án):
    python CatSA/main.py                        # chạy LẦN LƯỢT mọi file trong select.yaml
    python CatSA/main.py --suite diginetica           # chạy select.yaml trong config/catsa/diginetica/
    python CatSA/main.py --suite diginetica --run catsa_v1.yaml
    python CatSA/main.py --config config_khac   # dùng cây cấu hình khác

Yêu cầu: đã chạy tiền xử lý trước đó (python tienxuly/main.py) và khai báo
đúng thư mục dữ liệu trong section `data` của từng file version CatSA.
Xem CatSA/README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Cho phép chạy trực tiếp file này từ bất kỳ đâu (tương tự tienxuly/main.py)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from common import (  # noqa: E402
    load_config,
    dump_config,
    list_catsa_runs,
    setup_logger,
    init_wandb,
    finish_wandb,
)
from tienxuly import load_processed  # noqa: E402
from CatSA import train_model  # noqa: E402


def _resolve_runs(config_dir: str, run_arg: str | None, suite: str | None) -> list[str]:
    """Xác định danh sách file version sẽ chạy."""
    if run_arg:
        return [run_arg]
    runs = list_catsa_runs(config_dir, suite=suite)
    if not runs:
        label = f"catsa/{suite}/" if suite else "catsa/"
        raise FileNotFoundError(
            f"Không tìm thấy config/{label}select.yaml hoặc run: rỗng trong {config_dir}"
        )
    return runs


def _run_one(config_dir: str, catsa_run: str, suite: str | None = None) -> dict[str, float]:
    """Train + eval một file version CatSA; trả về test metrics."""
    cfg = load_config(config_dir, catsa_run=catsa_run, catsa_suite=suite)
    log = setup_logger(cfg.logging, cfg.project)
    log.info("%s", dump_config(cfg))
    init_wandb(cfg)

    try:
        try:
            data = load_processed(cfg.data)
        except FileNotFoundError as e:
            log.error("%s", e)
            log.error(
                "Chạy tiền xử lý (nếu chưa có data) rồi chỉnh data.data_dir "
                "trong %s — hiện tại: %s",
                catsa_run,
                cfg.data.data_dir,
            )
            raise

        log.info(
            "Dữ liệu [%s]: |I|=%d, |C|=%d, train=%d phiên",
            cfg.data.data_dir,
            data["n_items"],
            data["n_cats"],
            len(data["train_sessions"]),
        )
        _, test_metrics = train_model(data, cfg)
        log.info(
            "[CatSA] Hoàn tất %s. Test metrics: %s",
            catsa_run,
            " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()),
        )
        return test_metrics
    finally:
        finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá CatSA")
    parser.add_argument(
        "--config",
        default="config",
        help="Thư mục chứa cây cấu hình YAML (mặc định: config)",
    )
    parser.add_argument(
        "--run",
        default=None,
        metavar="FILE",
        help="Chỉ chạy một file version (vd catsa_v1.yaml). Bỏ qua => chạy hết select.yaml",
    )
    parser.add_argument(
        "--suite",
        default=None,
        choices=["diginetica"],
        help="Chạy select.yaml trong config/catsa/<suite>/ (mặc định: config/catsa/select.yaml)",
    )
    args = parser.parse_args()

    try:
        runs = _resolve_runs(args.config, args.run, args.suite)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        sys.exit(1)

    if len(runs) > 1:
        print(f"[CatSA] Sẽ chạy lần lượt {len(runs)} cấu hình: {', '.join(runs)}")

    failed: list[str] = []
    for i, catsa_run in enumerate(runs, start=1):
        if len(runs) > 1:
            print(f"\n[CatSA] ({i}/{len(runs)}) Bắt đầu: {catsa_run}")
        try:
            _run_one(args.config, catsa_run, suite=args.suite)
        except FileNotFoundError:
            failed.append(catsa_run)
        except Exception:
            print(f"[CatSA] Lỗi khi chạy {catsa_run}:", file=sys.stderr)
            import traceback
            traceback.print_exc()
            failed.append(catsa_run)

    if failed:
        print(f"\n[CatSA] Thất bại {len(failed)}/{len(runs)}: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    if len(runs) > 1:
        print(f"\n[CatSA] Hoàn tất tất cả {len(runs)} cấu hình.")


if __name__ == "__main__":
    main()
