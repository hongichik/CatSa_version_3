"""Entrypoint huấn luyện CORE — dùng dữ liệu đã tiền xử lý trong data/.

Cách chạy (từ thư mục gốc dự án):
    python CORE/main.py                              # config/core/retailrocket/select.yaml
    python CORE/main.py --suite diginetica --run core_trm.yaml
    python CORE/main.py --suite retailrocket --run core_trm_retailrocket.yaml

Cấu hình: config/core/ (tương tự config/catsa/).
Paper: https://github.com/RUCAIBox/CORE
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from common import (  # noqa: E402
    load_core_config,
    dump_core_config,
    list_core_runs,
    setup_logger,
    init_wandb,
    finish_wandb,
)
from tienxuly import load_processed  # noqa: E402
from CORE import train_model  # noqa: E402


def _resolve_runs(config_dir: str, run_arg: str | None, suite: str | None) -> list[str]:
    if run_arg:
        return [run_arg]
    runs = list_core_runs(config_dir, suite=suite)
    if not runs:
        label = f"core/{suite}/" if suite else "core/"
        raise FileNotFoundError(
            f"Không tìm thấy config/{label}select.yaml hoặc run: rỗng trong {config_dir}"
        )
    return runs


def _run_one(config_dir: str, core_run: str, suite: str | None = None) -> dict[str, float]:
    cfg = load_core_config(config_dir, core_run=core_run, core_suite=suite)
    log = setup_logger(cfg.logging, cfg.project)
    log.info("%s", dump_core_config(cfg))
    init_wandb(cfg)

    try:
        try:
            data = load_processed(cfg.data)
        except FileNotFoundError as e:
            log.error("%s", e)
            log.error(
                "Chạy tiền xử lý trước (python tienxuly/main.py) rồi chỉnh data.data_dir "
                "trong %s — hiện tại: %s",
                core_run,
                cfg.data.data_dir,
            )
            raise

        log.info(
            "Dữ liệu [%s]: |I|=%d, train=%d phiên",
            cfg.data.data_dir,
            data["n_items"],
            len(data["train_sessions"]),
        )
        _, test_metrics = train_model(data, cfg)
        log.info(
            "[CORE] Hoàn tất %s. Test metrics: %s",
            core_run,
            " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()),
        )
        return test_metrics
    finally:
        finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá CORE")
    parser.add_argument("--config", default="config", help="Thư mục cấu hình YAML")
    parser.add_argument(
        "--run",
        default=None,
        metavar="FILE",
        help="Chỉ chạy một file version (vd core_trm.yaml)",
    )
    parser.add_argument(
        "--suite",
        default="retailrocket",
        choices=["retailrocket", "diginetica", "tmall", "yoochoose"],
        help="Chạy select.yaml trong config/core/<suite>/ (mặc định: retailrocket)",
    )
    args = parser.parse_args()

    try:
        runs = _resolve_runs(args.config, args.run, args.suite)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        sys.exit(1)

    if len(runs) > 1:
        print(f"[CORE] Sẽ chạy lần lượt {len(runs)} cấu hình: {', '.join(runs)}")

    failed: list[str] = []
    for i, core_run in enumerate(runs, start=1):
        if len(runs) > 1:
            print(f"\n[CORE] ({i}/{len(runs)}) Bắt đầu: {core_run}")
        try:
            _run_one(args.config, core_run, suite=args.suite)
        except FileNotFoundError:
            failed.append(core_run)
        except Exception:
            print(f"[CORE] Lỗi khi chạy {core_run}:", file=sys.stderr)
            import traceback
            traceback.print_exc()
            failed.append(core_run)

    if failed:
        print(f"\n[CORE] Thất bại {len(failed)}/{len(runs)}: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    if len(runs) > 1:
        print(f"\n[CORE] Hoàn tất tất cả {len(runs)} cấu hình.")


if __name__ == "__main__":
    main()
