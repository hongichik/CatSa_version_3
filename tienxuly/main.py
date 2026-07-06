"""Entrypoint RIÊNG của bước tiền xử lý — chạy độc lập, không qua main tập trung.

Cách chạy (từ bất kỳ đâu, kết quả luôn ghi về thư mục gốc dự án):
    python tienxuly/main.py                              # chạy LẦN LƯỢT select.yaml
    python tienxuly/main.py --run retailrocket_2_5.yaml  # một cấu hình
    python tienxuly/main.py --config config_khac

Đầu ra: <preprocess.output_dir>/train.txt, val.txt, test.txt, lookup_tables.pkl
Xem tienxuly/README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from common import load_config, list_preprocess_runs, setup_logger  # noqa: E402
from tienxuly import download_dataset, preprocess  # noqa: E402


def _resolve_runs(config_dir: str, run_arg: str | None, suite: str | None) -> list[str]:
    if run_arg:
        return [run_arg]
    runs = list_preprocess_runs(config_dir, suite=suite)
    if not runs:
        label = f"tienxuly/{suite}/" if suite else "tienxuly/"
        raise FileNotFoundError(
            f"Không tìm thấy config/{label}select.yaml hoặc run: rỗng"
        )
    return runs


def _run_one(config_dir: str, preprocess_run: str, raw_dir: Path) -> dict:
    cfg = load_config(config_dir, preprocess_run=preprocess_run)
    log = setup_logger(cfg.logging, cfg.project, to_file=False)
    log.info(
        "[tienxuly] %s → dataset=%s | output_dir=%s | len mode=%s [%d..%s] prefix≤%d | require_category=%s",
        preprocess_run,
        cfg.dataset.name,
        cfg.preprocess.output_dir,
        cfg.preprocess.session_length_mode,
        cfg.preprocess.min_session_length,
        cfg.preprocess.max_session_length,
        cfg.preprocess.max_prefix_length,
        cfg.preprocess.require_item_category,
    )
    data = preprocess(raw_dir, cfg.preprocess, cfg.dataset.name)
    log.info(
        "[tienxuly] Hoàn tất %s: |I|=%d, |C|=%d, train/val/test = %d/%d/%d phiên",
        preprocess_run,
        data["n_items"],
        data["n_cats"],
        len(data["train_sessions"]),
        len(data["val_sessions"]),
        len(data["test_sessions"]),
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu cho CatSA")
    parser.add_argument("--config", default="config",
                        help="Thư mục chứa cây cấu hình YAML (mặc định: config)")
    parser.add_argument(
        "--run",
        default=None,
        metavar="FILE",
        help="Chỉ chạy một file version (vd retailrocket_2_5.yaml hoặc diginetica_item_hon_5.yaml)",
    )
    parser.add_argument(
        "--suite",
        default="retailrocket",
        choices=["retailrocket", "diginetica"],
        help="Chạy select.yaml trong config/tienxuly/<suite>/ (mặc định: retailrocket)",
    )
    args = parser.parse_args()

    try:
        runs = _resolve_runs(args.config, args.run, args.suite)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        sys.exit(1)

    if len(runs) > 1:
        print(f"[tienxuly] Sẽ chạy lần lượt {len(runs)} cấu hình: {', '.join(runs)}")

    cached_handle: str | None = None
    raw_dir: Path | None = None
    failed: list[str] = []
    for i, preprocess_run in enumerate(runs, start=1):
        if len(runs) > 1:
            print(f"\n[tienxuly] ({i}/{len(runs)}) Bắt đầu: {preprocess_run}")
        try:
            cfg = load_config(
                args.config,
                preprocess_run=preprocess_run,
                catsa_suite=args.suite,
            )
            handle = (
                cfg.dataset.kagglehub_handle
                if cfg.dataset.source == "kagglehub"
                else cfg.dataset.local_path
            )
            if raw_dir is None or handle != cached_handle:
                raw_dir = download_dataset(cfg.dataset)
                cached_handle = handle
            _run_one(args.config, preprocess_run, raw_dir)
        except Exception:
            import traceback
            print(f"[tienxuly] Lỗi khi chạy {preprocess_run}:", file=sys.stderr)
            traceback.print_exc()
            failed.append(preprocess_run)

    if failed:
        print(f"\n[tienxuly] Thất bại {len(failed)}/{len(runs)}: {', '.join(failed)}",
              file=sys.stderr)
        sys.exit(1)
    if len(runs) > 1:
        print(f"\n[tienxuly] Hoàn tất tất cả {len(runs)} cấu hình.")


if __name__ == "__main__":
    main()
