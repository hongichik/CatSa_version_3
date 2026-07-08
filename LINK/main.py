"""Entrypoint LINK (SIGIR 2025) — chạy trên dữ liệu đã tiền xử lý của demo2.

Cách chạy (từ thư mục gốc dự án):
    python LINK/main.py                                 # config/link/retailrocket/select.yaml
    python LINK/main.py --suite diginetica
    python LINK/main.py --suite retailrocket --run link_retailrocket.yaml
    python LINK/main.py --run link_toy.yaml            # smoke test nhanh

Pipeline (giữ nguyên code gốc trong LINK_repo/, chỉ bọc theo cấu trúc demo2):
  1. Chuyển data/<dataset>/*.txt  →  RecBole atomic files (LINK/adapter.py)
  2. Huấn luyện teacher core_trm  →  saved_models_for_embedding/.../dense_matrix.npy
  3. Huấn luyện + đánh giá LINK (closed-form) trên teacher matrix
  4. Trích metric (Recall@K = HR@K, MRR@K) và ghi vào Log/<dataset>/

Cấu hình: config/link/<suite>/ (tương tự config/core/).
Paper/code gốc: https://github.com/jin530/LINK
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from common.config import DataConfig, LoggingConfig, ProjectConfig  # noqa: E402
from common.logger import setup_logger  # noqa: E402
from tienxuly import load_processed  # noqa: E402
from LINK.adapter import demo2_to_recbole  # noqa: E402

LINK_REPO = ROOT / "LINK_repo"


# --------------------------------------------------------------------------- #
# Đọc cấu hình (kiểu demo2: config/link/<suite>/<run>.yaml + select.yaml)
# --------------------------------------------------------------------------- #
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
    select = Path(config_dir) / "link" / suite / "select.yaml"
    if not select.exists():
        raise FileNotFoundError(f"Không tìm thấy {select}")
    sel = _load_yaml(select)
    runs = sel.get("run", []) or []
    runs = [r for r in runs if r and not str(r).strip().startswith("#")]
    if not runs:
        raise FileNotFoundError(f"'run:' rỗng trong {select}")
    return runs


def _load_run_config(config_dir: str, suite: str, run: str) -> dict:
    run_path = _find_run_yaml(Path(config_dir) / "link" / suite, run)
    cfg = _load_yaml(run_path)
    logging_cfg = _load_yaml(Path(config_dir) / "common" / "logging.yaml").get(
        "logging", {}
    )
    cfg["_logging"] = logging_cfg
    return cfg


def _build_logging(cfg: dict) -> LoggingConfig:
    lg = cfg.get("_logging", {})
    return LoggingConfig(
        dir=lg.get("dir", "Log"),
        level=lg.get("level", "INFO"),
        console=lg.get("console", True),
    )


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


# --------------------------------------------------------------------------- #
# Chạy một lệnh con của LINK_repo, tee output ra logger, trả về toàn bộ text
# --------------------------------------------------------------------------- #
def _run_subprocess(cmd: list[str], env: dict, log, tag: str) -> str:
    log.info("[%s] $ %s", tag, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(LINK_REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        log.info("[%s] %s", tag, line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"[{tag}] tiến trình lỗi (mã {proc.returncode})")
    return "\n".join(lines)


_METRIC_RE = re.compile(
    r"'(recall|mrr|ndcg|hit|precision)@(\d+)'\s*:\s*(?:np\.float64\()?([0-9.]+)"
)


def _parse_metrics(text: str) -> dict[str, float]:
    """Lấy metric từ dòng 'test result' cuối cùng của RecBole."""
    idx = text.rfind("test result")
    segment = text[idx:] if idx >= 0 else text
    metrics: dict[str, float] = {}
    for name, k, val in _METRIC_RE.findall(segment):
        metrics[f"{name}@{k}"] = float(val)
    return metrics


# --------------------------------------------------------------------------- #
def _run_one(config_dir: str, suite: str, run: str) -> dict[str, float]:
    cfg = _load_run_config(config_dir, suite, run)
    logging_cfg = _build_logging(cfg)
    project = _build_project(cfg)
    data_cfg = _build_data(cfg)
    link = cfg.get("link", {})
    dataset_name = link.get("dataset_name", project.name)

    log = setup_logger(logging_cfg, project)
    log.info("=" * 68)
    log.info("[LINK] Bắt đầu run=%s  suite=%s  dataset=%s", run, suite, dataset_name)
    log.info("[LINK] Cấu hình: %s", cfg)

    # 1) Load dữ liệu demo2
    data = load_processed(data_cfg)
    n_items = data["n_items"]
    log.info(
        "[LINK] Dữ liệu [%s]: |I|=%d, train=%d phiên",
        data_cfg.data_dir,
        n_items,
        len(data["train_sessions"]),
    )
    approx_gb = (n_items * n_items * 4) / 1e9
    if approx_gb > 20:
        log.warning(
            "[LINK] CẢNH BÁO: n_items=%d ⇒ ma trận đặc ~%.1f GB/bản. "
            "LINK dựng nhiều bản → có thể thiếu RAM. Cân nhắc dataset nhỏ hơn.",
            n_items,
            approx_gb,
        )

    # 2) Sinh dataset RecBole
    reuse = bool(link.get("reuse_converted", True))
    demo2_to_recbole(data, dataset_name, LINK_REPO / "dataset", reuse=reuse, logger=log)

    # Môi trường cho tiến trình con
    env = dict(os.environ)
    gpu_id = str(link.get("gpu_id", 0))

    # 3) Huấn luyện teacher core_trm (sinh dense_matrix.npy)
    teacher_model = link.get("teacher_model", "core_trm")
    teacher_npy = (
        LINK_REPO
        / "saved_models_for_embedding"
        / f"linear_teacher_{dataset_name}_{teacher_model}"
        / "dense_matrix.npy"
    )
    if link.get("reuse_teacher", True) and teacher_npy.exists():
        log.info("[LINK] Dùng lại teacher matrix có sẵn: %s", teacher_npy)
    else:
        teacher_cmd = [
            sys.executable, "main.py",
            "--model", teacher_model,
            "--dataset", dataset_name,
            "--config2", "props/gpu0_1024.yaml",
            "--gpu_id", gpu_id,
            "--epochs", str(link.get("teacher_epochs", 20)),
            "--train_batch_size", str(link.get("train_batch_size", 1024)),
            "--eval_batch_size", str(link.get("eval_batch_size", 512)),
            "--learning_rate", str(link.get("learning_rate", 0.001)),
            "--stopping_step", str(link.get("stopping_step", 5)),
        ]
        _run_subprocess(teacher_cmd, env, log, tag="teacher")
        if not teacher_npy.exists():
            raise RuntimeError(f"Không tìm thấy teacher matrix sau khi train: {teacher_npy}")

    # 4) Huấn luyện + đánh giá LINK
    rel_teacher = teacher_npy.relative_to(LINK_REPO)
    link_cmd = [
        sys.executable, "main.py",
        "--model", "link",
        "--dataset", dataset_name,
        "--gpu_id", gpu_id,
        "--teacher_path", str(rel_teacher),
        "--reg", str(link.get("reg", 10.0)),
        "--reg_teacher", str(link.get("reg_teacher", 100)),
        "--predict_weight", str(link.get("predict_weight", 4.0)),
        "--slis_alpha", str(link.get("slis_alpha", 0.9)),
        "--teacher_normalize", str(link.get("teacher_normalize", True)),
        "--teacher_temperature", str(link.get("teacher_temperature", 0.1)),
    ]
    out = _run_subprocess(link_cmd, env, log, tag="link")

    metrics = _parse_metrics(out)
    if not metrics:
        log.warning("[LINK] Không trích được metric từ output.")
    else:
        # Recall@K = HR@K cho next-item (chỉ 1 item đúng)
        pretty = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        log.info("[LINK] Hoàn tất %s. Test metrics: %s", run, pretty)
        for k in ("recall@20", "mrr@20"):
            if k in metrics:
                alias = "hr@20" if k == "recall@20" else k
                log.info("[LINK]   → %s = %.4f", alias, metrics[k])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện & đánh giá LINK (SIGIR 2025)")
    parser.add_argument("--config", default="config", help="Thư mục cấu hình YAML")
    parser.add_argument("--run", default=None, metavar="FILE",
                        help="Chỉ chạy một file version (vd link_retailrocket.yaml)")
    parser.add_argument("--suite", default="retailrocket",
                        choices=["retailrocket", "diginetica"],
                        help="Chạy select.yaml trong config/link/<suite>/")
    args = parser.parse_args()

    try:
        runs = _resolve_runs(args.config, args.run, args.suite)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        sys.exit(1)

    if len(runs) > 1:
        print(f"[LINK] Sẽ chạy lần lượt {len(runs)} cấu hình: {', '.join(runs)}")

    failed: list[str] = []
    for i, run in enumerate(runs, start=1):
        if len(runs) > 1:
            print(f"\n[LINK] ({i}/{len(runs)}) Bắt đầu: {run}")
        try:
            _run_one(args.config, args.suite, run)
        except Exception:
            print(f"[LINK] Lỗi khi chạy {run}:", file=sys.stderr)
            import traceback
            traceback.print_exc()
            failed.append(run)

    if failed:
        print(f"\n[LINK] Thất bại {len(failed)}/{len(runs)}: {', '.join(failed)}",
              file=sys.stderr)
        sys.exit(1)
    if len(runs) > 1:
        print(f"\n[LINK] Hoàn tất tất cả {len(runs)} cấu hình.")


if __name__ == "__main__":
    main()
