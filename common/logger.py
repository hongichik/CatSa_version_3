"""Logger dùng chung cho toàn pipeline CatSA.

Log ghi vào <logging.dir>/<project.name>/ (ví dụ Log/retailrocket/):
- thư mục gốc lấy từ config/common/logging.yaml (dùng chung mọi dự án);
- tên dự án + cách đặt tên file lấy từ config riêng của dự án
  (ví dụ config/catsa/project.yaml):
    * không khai báo => "auto": tên file dạng stt-ngày-tháng-năm-giờ.log,
      ví dụ 001-04-07-2026-08.log (stt tự tăng trong thư mục log của dự án)
    * khai báo custom_filename => dùng đúng tên đó
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from .config import LoggingConfig, ProjectConfig

_LOGGER_NAME = "catsa"
_AUTO_PATTERN = re.compile(r"^(\d+)-\d{2}-\d{2}-\d{4}-\d{2}\.log$")
_current_log_path: Path | None = None


def _next_stt(log_dir: Path) -> int:
    """Tìm số thứ tự tiếp theo dựa trên các file log auto đã có trong thư mục."""
    max_stt = 0
    for f in log_dir.glob("*.log"):
        m = _AUTO_PATTERN.match(f.name)
        if m:
            max_stt = max(max_stt, int(m.group(1)))
    return max_stt + 1


def _auto_filename(log_dir: Path) -> str:
    now = datetime.now()
    stt = _next_stt(log_dir)
    # stt-ngày-tháng-năm-giờ.log
    return f"{stt:03d}-{now:%d-%m-%Y-%H}.log"


def setup_logger(
    cfg: LoggingConfig, project: ProjectConfig, to_file: bool = True
) -> logging.Logger:
    """Khởi tạo logger; log ghi vào <cfg.dir>/<project.name>/ theo quy tắc của dự án.

    to_file=False: KHÔNG ghi file, chỉ in console — dùng cho các bước phụ trợ
    không cần lưu vết (ví dụ tiền xử lý dữ liệu).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, cfg.level, logging.INFO))
    logger.handlers.clear()  # tránh gắn handler trùng khi gọi lại

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    global _current_log_path

    if to_file:
        log_dir = Path(cfg.dir) / project.name
        log_dir.mkdir(parents=True, exist_ok=True)
        if project.filename_mode == "custom":
            filename = project.custom_filename
        else:
            filename = _auto_filename(log_dir)
        log_path = log_dir / filename
        _current_log_path = log_path

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        if cfg.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(fmt)
            logger.addHandler(console_handler)
        logger.info("Log được ghi tại: %s", log_path.resolve())
    else:
        _current_log_path = None
        # Không ghi file thì luôn in console (nếu không sẽ chẳng thấy gì)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    """Lấy logger đã setup (dùng trong các module con)."""
    return logging.getLogger(_LOGGER_NAME)


def get_log_path() -> Path | None:
    """Đường dẫn file log hiện tại (dùng đặt tên run wandb trùng tên file log)."""
    return _current_log_path
