"""Tải dataset theo cấu hình trong config/tienxuly/*/dataset.yaml.

- source = "kagglehub": tải qua thư viện kagglehub.
- source = "local"    : dùng thư mục dữ liệu có sẵn trên máy.

RetailRocket : events.csv, item_properties_*.csv, category_tree.csv
Diginetica   : train-item-views.csv, product-categories.csv, products.csv
"""

from __future__ import annotations

from pathlib import Path

from common.config import DatasetConfig
from common.logger import get_logger


def download_dataset(cfg: DatasetConfig) -> Path:
    """Tải (hoặc định vị) dataset thô, trả về Path thư mục chứa dữ liệu."""
    log = get_logger()

    if cfg.source == "kagglehub":
        import kagglehub

        log.info("Tải dataset từ Kaggle: %s (kagglehub)", cfg.kagglehub_handle)
        path = Path(kagglehub.dataset_download(cfg.kagglehub_handle))
        log.info("Dataset đã sẵn sàng tại: %s", path)
    elif cfg.source == "local":
        path = Path(cfg.local_path)
        log.info("Dùng dataset local tại: %s", path)
        if not path.exists():
            raise FileNotFoundError(
                f"Thư mục dataset local không tồn tại: {path} "
                f"(kiểm tra dataset.local_path trong config/tienxuly/dataset.yaml)"
            )
    else:
        raise ValueError(f"<source> không hợp lệ: {cfg.source} (chỉ hỗ trợ kagglehub | local)")

    return path
