"""Entrypoint MSGIFSR (WSDM 2022) — chạy trên dữ liệu đã tiền xử lý của demo2.

Cách chạy (từ thư mục gốc dự án):
    python MSGIFSR/main.py                                 # config/msgifsr/retailrocket/select.yaml
    python MSGIFSR/main.py --suite diginetica
    python MSGIFSR/main.py --suite retailrocket --run msgifsr_retailrocket.yaml
    python MSGIFSR/main.py --run msgifsr_toy.yaml          # smoke test nhanh

Pipeline:
  1. Chuyển data/<dir>/*.txt → MSGIFSR_repo/datasets/<name>/ (comma-separated + num_items.txt)
  2. Huấn luyện MSGIFSR (GNN + DGL), early stopping trên val
  3. Đánh giá full-ranking trên test, log vào Log/<dataset>/

Cấu hình: config/msgifsr/<suite>/ (tương tự config/core/).
Paper/code gốc: https://github.com/SpaceLearner/SessionRec-pytorch
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from common.config import DataConfig, LoggingConfig, ProjectConfig  # noqa: E402
from common.logger import setup_logger  # noqa: E402
from tienxuly import load_processed  # noqa: E402
from MSGIFSR.adapter import demo2_to_msgifsr  # noqa: E402
from MSGIFSR.train import train_model  # noqa: E402

MSGIFSR_REPO = ROOT / "MSGIFSR_repo"
DATASETS_ROOT = MSGIFSR_REPO / "datasets"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_run_yaml(suite_dir: Path, run: str) -> Path:
    """Tìm file run: cho phép đường dẫn con (main/x.yaml) hoặc basename."""
    target = run if run.endswith((".yaml", ".yml")) else f"{run}.yaml"
    direct = suite_dir / target
    if direct.is_file():
        return direct
    hits = sorted(p for p in suite_dir.rglob(Path(target).name) if p.is_file())
    if not hits:
        raise FileNotFoundError(f"Không tìm thấy config '{target}' trong {suite_dir}")
    if len(hits) > 1:
        raise ValueError(
            f"Config '{Path(target).name}' trùng tên: {hits} — dùng đường dẫn "
            f"(vd main/{Path(target).name})"
        )
    return hits[0]


def _resolve_runs(config_dir: str, run_arg: str | None, suite: str) -> list[str]:
    if run_arg:
        return [run_arg]
    select = Path(config_dir) / "msgifsr" / suite / "select.yaml"
    if not select.exists():
        raise FileNotFoundError(f"Không tìm thấy {select}")
    sel = _load_yaml(select)
    runs = [r for r in sel.get("run", []) or [] if r and not str(r).strip().startswith("#")]
    if not runs:
        raise FileNotFoundError(f"'run:' rỗng trong {select}")
    return runs


def _load_run_config(config_dir: str, suite: str, run: str) -> dict:
    run_path = _find_run_yaml(Path(config_dir) / "msgifsr" / suite, run)
    cfg = _load_yaml(run_path)
    cfg["_logging"] = _load_yaml(Path(config_dir) / "common" / "logging.yaml").get("logging", {})
    return cfg


def _build_logging(cfg: dict) -> LoggingConfig:
    lg = cfg.get("_logging", {})
    return LoggingConfig(dir=lg.get("dir", "Log"), level=lg.get("level", "INFO"), console=lg.get("console", True))


def _build_project(cfg: dict) -> ProjectConfig:
    proj = cfg.get("project", {})
    return ProjectConfig(
        name=proj.get("name", "retailrocket"),
        filename_mode=proj.get("filename_mode", "auto"),
        custom_filename=proj.get("custom_filename", ""),
    )


def _build_data(cfg: dict) -> DataConfig:
    d = cfg.get("data", {})
    return DataConfig(
        data_dir=d["data_dir"],
        train_file=d.get("train_file", "train.txt"),
        val_file=d.get("val_file", "val.txt"),
        test_file=d.get("test_file", "test.txt"),
        lookup_file=d.get("lookup_file", "lookup_tables.pkl"),
    )


def _run_one(config_dir: str, suite: str, run: str) -> dict[str, float]:
    cfg = _load_run_config(config_dir, suite, run)
    log = setup_logger(_build_logging(cfg), _build_project(cfg))
    log.info("=" * 68)
    log.info("[MSGIFSR] Bắt đầu run=%s  suite=%s", run, suite)
    log.info("[MSGIFSR] Cấu hình: %s", cfg)

    try:
        data_cfg = _build_data(cfg)
        data = load_processed(data_cfg)
        log.info(
            "[MSGIFSR] Dữ liệu [%s]: |I|=%d, train=%d phiên",
            data_cfg.data_dir, data["n_items"], len(data["train_sessions"]),
        )

        link_cfg = cfg.get("msgifsr", {})
        dataset_name = link_cfg.get("dataset_name", cfg["project"]["name"])
        dataset_dir = demo2_to_msgifsr(
            data,
            dataset_name,
            DATASETS_ROOT,
            reuse=link_cfg.get("reuse_converted", True),
            logger=log,
        )

        _, test_metrics = train_model(dataset_dir, cfg)
        log.info(
            "[MSGIFSR] Hoàn tất %s. Test: %s",
            run,
            " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()),
        )
        return test_metrics
    except FileNotFoundError:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện & đánh giá MSGIFSR")
    parser.add_argument("--config", default="config")
    parser.add_argument("--run", default=None, metavar="FILE")
    parser.add_argument("--suite", default="retailrocket", choices=["retailrocket", "diginetica"])
    args = parser.parse_args()

    try:
        runs = _resolve_runs(args.config, args.run, args.suite)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        sys.exit(1)

    if len(runs) > 1:
        print(f"[MSGIFSR] Sẽ chạy lần lượt {len(runs)} cấu hình: {', '.join(runs)}")

    failed: list[str] = []
    for i, run in enumerate(runs, start=1):
        if len(runs) > 1:
            print(f"\n[MSGIFSR] ({i}/{len(runs)}) Bắt đầu: {run}")
        try:
            _run_one(args.config, args.suite, run)
        except Exception:
            print(f"[MSGIFSR] Lỗi khi chạy {run}:", file=sys.stderr)
            import traceback
            traceback.print_exc()
            failed.append(run)

    if failed:
        print(f"\n[MSGIFSR] Thất bại {len(failed)}/{len(runs)}: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
