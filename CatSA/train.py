"""Giai đoạn 4 + 6 — Training loop end-to-end.

- use_cl = false (section training trong config/catsa/catsa_v<N>.yaml):
  chỉ L_rec → biến thể A2 (Module 1 only).
- use_cl = true : CatSA đầy đủ — mỗi batch sinh phiên biến thể (Module 2),
  forward 2 lần (z_s và z_s~), multi-task loss L = L_rec + lambda * L_CL-session.

Early stopping theo primary_metric (mặc định HR@20) trên validation.

Checkpoint được PHÂN CẤP để biết nó là của cái gì:
    <checkpoint_dir>/<tên dự án>/<version cấu hình>/<tên run>/best_model.pt
    ví dụ: checkpoints/catsa/catsa_v1/003-04-07-2026-09/best_model.pt
(tên run = tên file log => đối chiếu 1-1 với file log trong Log/<dự án>/).
Cạnh best_model.pt luôn có info.yaml mô tả checkpoint: dự án, version, run,
epoch tốt nhất, metrics validation/test và toàn bộ cấu hình lúc train.
"""

from __future__ import annotations

import dataclasses
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam

from common.config import Config
from common.logger import get_log_path, get_logger
from common.tracker import log_metrics, set_summary

from .augment import CatSAAugmenter
from .dataset import make_loader
from .evaluate import evaluate_model
from .losses import session_level_infonce
from .model import CatSAEncoder


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn randomness để tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_checkpoint_dir(cfg: Config) -> Path:
    """Thư mục lưu model tốt nhất.

    - training.save_dir chỉ định → dùng đúng thư mục đó.
    - save_dir trống → mặc định theo version: <gốc>/<dự án>/<version>/<run>/.
    """
    if cfg.training.save_dir:
        ckpt_dir = Path(cfg.training.save_dir)
    else:
        log_path = get_log_path()
        run_name = log_path.stem if log_path else datetime.now().strftime("run-%Y%m%d-%H%M%S")
        ckpt_dir = Path(cfg.training.checkpoint_dir) / cfg.project.name / cfg.version / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _write_checkpoint_info(info_path: Path, cfg: Config, extra: dict) -> None:
    """Ghi info.yaml cạnh checkpoint — mô tả checkpoint này là của cái gì."""
    log_path = get_log_path()
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict["wandb"].pop("api_key", None)  # không lưu thông tin bí mật

    info = {
        "mo_ta": "Checkpoint model tốt nhất (theo validation) của CatSA",
        "du_an": cfg.project.name,
        "version_cau_hinh": cfg.version,
        "run": log_path.stem if log_path else None,
        "file_log": str(log_path) if log_path else None,
        "luu_luc": datetime.now().isoformat(timespec="seconds"),
        **extra,
        "cau_hinh": cfg_dict,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, allow_unicode=True, sort_keys=False)


def train_model(data: dict, cfg: Config) -> tuple[CatSAEncoder, dict[str, float]]:
    """Train CatSA (hoặc A2) trên dữ liệu đã tiền xử lý; trả về (model, test metrics)."""
    log = get_logger()
    tr = cfg.training
    set_seed(tr.seed)
    device = resolve_device(tr.device)
    log.info("Thiết bị: %s | use_cl=%s (%s)", device, tr.use_cl,
             "CatSA đầy đủ" if tr.use_cl else "A2 - Module 1 only")

    item2cat = data["item2cat"]
    cat_parent = data["cat_parent"] if cfg.model.use_taxonomy else None

    # Module 2 chỉ dùng khi bật contrastive learning
    augmenter = CatSAAugmenter(item2cat, data["cat2items"], cat_parent, cfg.augment) \
        if tr.use_cl else None
    if augmenter is not None:
        log.info("Augmentation strategies: %s | eta_aug=%.2f, k_min=%d",
                 augmenter.strategies, cfg.augment.eta_aug, cfg.augment.k_min)

    max_prefix = int(data.get("max_prefix_length", 50))
    log.info("max_prefix_length=%d (giới hạn lịch sử khi tạo mẫu sliding window)", max_prefix)

    train_loader = make_loader(
        data["train_sessions"], item2cat, cat_parent, cfg.model.use_taxonomy,
        tr.batch_size, shuffle=True, num_workers=tr.num_workers, augmenter=augmenter,
        max_prefix_length=max_prefix,
    )
    val_loader = make_loader(
        data["val_sessions"], item2cat, cat_parent, cfg.model.use_taxonomy,
        tr.batch_size, shuffle=False, num_workers=tr.num_workers,
        max_prefix_length=max_prefix,
    )
    test_loader = make_loader(
        data["test_sessions"], item2cat, cat_parent, cfg.model.use_taxonomy,
        tr.batch_size, shuffle=False, num_workers=tr.num_workers,
        max_prefix_length=max_prefix,
    )
    log.info("Số mẫu train/val/test (sliding window): %d / %d / %d",
             len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    model = CatSAEncoder(
        n_items=data["n_items"], n_cats=data["n_cats"],
        d=cfg.model.embedding_dim, n_layers=cfg.model.num_layers,
        use_taxonomy=cfg.model.use_taxonomy, dropout=cfg.model.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=tr.learning_rate, weight_decay=tr.weight_decay)

    # Checkpoint phân cấp: <gốc>/<dự án>/<version>/<run>/ + info.yaml mô tả
    ckpt_dir = _make_checkpoint_dir(cfg)
    ckpt_path = ckpt_dir / "best_model.pt"
    info_path = ckpt_dir / "info.yaml"
    log.info("Checkpoint của run này: %s", ckpt_dir.resolve())

    primary = cfg.evaluation.primary_metric
    best_metric = -1.0
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, tr.max_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = total_rec = total_cl = 0.0
        n_seen = 0

        for batch_orig, batch_aug, targets in train_loader:
            batch_orig = batch_orig.to(device)
            targets = targets.to(device)

            z_s = model(batch_orig)                        # (B, d)
            logits = model.scores(z_s)                     # (B, |I|)
            loss_rec = F.cross_entropy(logits, targets)    # L_rec — chỉ phiên gốc

            if batch_aug is not None:
                batch_aug = batch_aug.to(device)
                z_s_tilde = model(batch_aug)               # forward lần 2 cho s~
                loss_cl = session_level_infonce(z_s, z_s_tilde, tau=tr.tau)
                loss = loss_rec + tr.lambda_cl * loss_cl   # multi-task objective
            else:
                loss_cl = torch.tensor(0.0)
                loss = loss_rec

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tr.grad_clip)
            optimizer.step()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_rec += loss_rec.item() * bs
            total_cl += loss_cl.item() * bs
            n_seen += bs

        log.info("Epoch %d: total=%.4f, rec=%.4f, cl=%.4f (%.1fs)",
                 epoch, total_loss / n_seen, total_rec / n_seen,
                 total_cl / n_seen, time.time() - t0)

        # Validation mỗi epoch + early stopping
        val_metrics = evaluate_model(model, val_loader, device, cfg.evaluation.top_k)
        log.info("  Val: %s", " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        # Đẩy metrics của epoch lên wandb (no-op nếu wandb tắt)
        log_metrics(
            {
                "train/loss_total": total_loss / n_seen,
                "train/loss_rec": total_rec / n_seen,
                "train/loss_cl": total_cl / n_seen,
                **{f"val/{k}": v for k, v in val_metrics.items()},
            },
            step=epoch,
        )

        if val_metrics[primary] > best_metric:
            best_metric = val_metrics[primary]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            _write_checkpoint_info(info_path, cfg, {
                "epoch_tot_nhat": best_epoch,
                "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            })
            log.info("  → Model tốt nhất mới (%s=%.4f), lưu %s", primary, best_metric, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= tr.patience:
                log.info("Early stopping tại epoch %d (%s không cải thiện %d epoch)",
                         epoch, primary, tr.patience)
                break

    # Đánh giá test với model tốt nhất
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, device, cfg.evaluation.top_k)
    log.info("KẾT QUẢ TEST (best model): %s",
             " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    set_summary({f"test/{k}": v for k, v in test_metrics.items()})

    # Cập nhật info.yaml lần cuối: thêm kết quả test của checkpoint này
    _write_checkpoint_info(info_path, cfg, {
        "epoch_tot_nhat": best_epoch,
        "val_metrics_tot_nhat": {primary: round(best_metric, 6)},
        "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
    })
    return model, test_metrics
