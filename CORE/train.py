"""Training loop CORE — dùng dữ liệu đã tiền xử lý trong data/."""

from __future__ import annotations

import dataclasses
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam

from common.config import CoreConfig
from common.logger import get_log_path, get_logger
from common.tracker import log_metrics, set_summary

from .dataset import make_loader, pad_batch
from .evaluate import evaluate_model
from .models import COREave, COREtrm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(cfg: CoreConfig, n_items: int) -> torch.nn.Module:
    m = cfg.core_model
    common = dict(
        n_items=n_items,
        embedding_size=m.embedding_size,
        sess_dropout=m.sess_dropout,
        item_dropout=m.item_dropout,
        temperature=m.temperature,
    )
    if m.type == "ave":
        return COREave(**common)
    if m.type == "trm":
        return COREtrm(
            **common,
            max_seq_length=m.max_seq_length,
            n_layers=m.n_layers,
            n_heads=m.n_heads,
            inner_size=m.inner_size,
            hidden_dropout_prob=m.hidden_dropout_prob,
            attn_dropout_prob=m.attn_dropout_prob,
            hidden_act=m.hidden_act,
            layer_norm_eps=m.layer_norm_eps,
            initializer_range=m.initializer_range,
        )
    raise ValueError(f"core_model.type phải là 'ave' hoặc 'trm', nhận: {m.type}")


def _make_checkpoint_dir(cfg: CoreConfig) -> Path:
    tr = cfg.core_training
    if tr.save_dir:
        ckpt_dir = Path(tr.save_dir)
    else:
        log_path = get_log_path()
        run_name = log_path.stem if log_path else datetime.now().strftime("run-%Y%m%d-%H%M%S")
        ckpt_dir = Path(tr.checkpoint_dir) / cfg.project.name / cfg.version / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _write_checkpoint_info(info_path: Path, cfg: CoreConfig, extra: dict) -> None:
    log_path = get_log_path()
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict["wandb"].pop("api_key", None)
    info = {
        "mo_ta": "Checkpoint model tốt nhất (theo validation) của CORE",
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


def train_model(data: dict, cfg: CoreConfig) -> tuple[torch.nn.Module, dict[str, float]]:
    log = get_logger()
    tr = cfg.core_training
    set_seed(tr.seed)
    device = resolve_device(tr.device)
    log.info("Thiết bị: %s | model=%s", device, cfg.core_model.type)

    max_prefix = int(data.get("max_prefix_length", cfg.core_model.max_seq_length))
    log.info("max_prefix_length=%d", max_prefix)

    train_loader = make_loader(
        data["train_sessions"], tr.batch_size, shuffle=True,
        num_workers=tr.num_workers, max_prefix_length=max_prefix,
    )
    val_loader = make_loader(
        data["val_sessions"], tr.batch_size, shuffle=False,
        num_workers=tr.num_workers, max_prefix_length=max_prefix,
    )
    test_loader = make_loader(
        data["test_sessions"], tr.batch_size, shuffle=False,
        num_workers=tr.num_workers, max_prefix_length=max_prefix,
    )
    log.info(
        "Số mẫu train/val/test (sliding window): %d / %d / %d",
        len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset),
    )

    model = build_model(cfg, data["n_items"]).to(device)
    optimizer = Adam(model.parameters(), lr=tr.learning_rate, weight_decay=tr.weight_decay)

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
        total_loss = 0.0
        n_seen = 0

        for seqs, targets in train_loader:
            item_seq = pad_batch(seqs, device)
            targets_t = torch.tensor(targets, dtype=torch.long, device=device)
            loss = model.loss(item_seq, targets_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tr.grad_clip)
            optimizer.step()

            bs = targets_t.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

        log.info("Epoch %d: loss=%.4f (%.1fs)", epoch, total_loss / n_seen, time.time() - t0)

        val_metrics = evaluate_model(model, val_loader, device, cfg.evaluation.top_k)
        log.info("  Val: %s", " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
        log_metrics(
            {"train/loss": total_loss / n_seen, **{f"val/{k}": v for k, v in val_metrics.items()}},
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

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, device, cfg.evaluation.top_k)
    log.info("KẾT QUẢ TEST (best model): %s",
             " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    set_summary({f"test/{k}": v for k, v in test_metrics.items()})

    _write_checkpoint_info(info_path, cfg, {
        "epoch_tot_nhat": best_epoch,
        "val_metrics_tot_nhat": {primary: round(best_metric, 6)},
        "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
    })
    return model, test_metrics
