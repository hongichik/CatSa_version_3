"""Tracker Weights & Biases — ghi log thí nghiệm lên wandb song song với file log.

Cấu hình trong config/common/wandb.yaml. Thiết kế "fail-safe": nếu wandb gặp
lỗi (không có mạng, API key sai...) thì chỉ ghi WARNING vào log rồi tiếp tục
chạy bình thường — tracking không được phép làm chết training.

Cách dùng:
    init_wandb(cfg)                 # gọi 1 lần trong main.py sau setup_logger
    log_metrics({"train/loss": x})  # gọi ở bất kỳ đâu (no-op nếu wandb tắt)
    finish_wandb()                  # gọi cuối main.py
"""

from __future__ import annotations

import dataclasses

from .config import Config, CoreConfig
from .logger import get_log_path, get_logger

_run = None  # wandb Run đang hoạt động (None = tắt hoặc init thất bại)


def init_wandb(cfg: Config | CoreConfig) -> None:
    """Đăng nhập và khởi tạo run wandb theo cấu hình; đẩy toàn bộ config lên run."""
    global _run
    log = get_logger()

    if not cfg.wandb.enabled:
        log.info("wandb: tắt (wandb.enabled=false)")
        return

    try:
        import wandb

        if cfg.wandb.api_key:
            wandb.login(key=cfg.wandb.api_key, relogin=True, verify=True)

        # Tên run mặc định = tên file log để dễ đối chiếu run <-> file log
        log_path = get_log_path()
        run_name = cfg.wandb.run_name or (log_path.stem if log_path else None)

        # Đẩy toàn bộ cấu hình lên wandb (trừ api_key — thông tin bí mật)
        cfg_dict = dataclasses.asdict(cfg)
        cfg_dict["wandb"].pop("api_key", None)

        _run = wandb.init(
            # Không khai báo wandb.project => dùng tên dự án (project.name)
            project=cfg.wandb.project or cfg.project.name,
            entity=cfg.wandb.entity or None,
            name=run_name,
            mode=cfg.wandb.mode,
            config=cfg_dict,
        )
        log.info("wandb: run '%s' — %s", _run.name, _run.url or "(offline)")
    except Exception as e:  # noqa: BLE001 — tracking lỗi không được chặn training
        _run = None
        log.warning("wandb: khởi tạo thất bại (%s) — tiếp tục chạy không có wandb", e)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Ghi một dict metric lên wandb (no-op nếu wandb tắt/lỗi)."""
    if _run is None:
        return
    try:
        _run.log(metrics, step=step)
    except Exception as e:  # noqa: BLE001
        get_logger().warning("wandb: log thất bại (%s)", e)


def set_summary(metrics: dict) -> None:
    """Ghi metric tổng kết (ví dụ kết quả test cuối) vào summary của run."""
    if _run is None:
        return
    try:
        for k, v in metrics.items():
            _run.summary[k] = v
    except Exception as e:  # noqa: BLE001
        get_logger().warning("wandb: set summary thất bại (%s)", e)


def finish_wandb() -> None:
    """Kết thúc run wandb (flush toàn bộ dữ liệu còn lại)."""
    global _run
    if _run is None:
        return
    try:
        _run.finish()
    except Exception as e:  # noqa: BLE001
        get_logger().warning("wandb: finish thất bại (%s)", e)
    _run = None
