"""Training loop MSGIFSR — dùng dữ liệu đã tiền xử lý trong data/."""

from __future__ import annotations

import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

ROOT = Path(__file__).resolve().parent.parent
MSGIFSR_REPO = ROOT / "MSGIFSR_repo"
if str(MSGIFSR_REPO) not in sys.path:
    sys.path.insert(0, str(MSGIFSR_REPO))

from src.models import MSGIFSR  # noqa: E402
from src.utils.data.collate import (  # noqa: E402
    collate_fn_factory_ccs,
    seq_to_ccs_graph,
)

from common.logger import get_log_path, get_logger  # noqa: E402

from .dataset import load_msgifsr_sessions, make_loaders  # noqa: E402
from .evaluate import evaluate_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(cfg: dict, n_items: int, dataset_dir: Path, device: torch.device):
    m = cfg["msgifsr_model"]
    return MSGIFSR(
        n_items,
        str(dataset_dir),
        m.get("embedding_dim", 256),
        m.get("num_layers", 1),
        dropout=m.get("feat_drop", 0.1),
        reducer=m.get("reducer", "mean"),
        order=m.get("order", 1),
        norm=m.get("norm", True),
        extra=m.get("extra", True),
        fusion=m.get("fusion", True),
        device=device,
    )


def _make_checkpoint_dir(cfg: dict) -> Path:
    tr = cfg["msgifsr_training"]
    if tr.get("save_dir"):
        ckpt_dir = Path(tr["save_dir"])
    else:
        log_path = get_log_path()
        run_name = log_path.stem if log_path else datetime.now().strftime("run-%Y%m%d-%H%M%S")
        ckpt_dir = Path(tr.get("checkpoint_dir", "checkpoints")) / cfg["project"]["name"] / "msgifsr" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _write_checkpoint_info(info_path: Path, cfg: dict, extra: dict) -> None:
    log_path = get_log_path()
    payload = {"config": cfg, "log_file": str(log_path) if log_path else None, **extra}
    with open(info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def train_model(dataset_dir: Path, cfg: dict) -> tuple[dict[str, float], dict[str, float]]:
    log = get_logger()
    tr = cfg["msgifsr_training"]
    ev = cfg["evaluation"]
    mcfg = cfg["msgifsr_model"]

    set_seed(tr.get("seed", 42))
    device = resolve_device(tr.get("device", "auto"))

    train_sessions, val_sessions, test_sessions, n_items = load_msgifsr_sessions(dataset_dir)
    log.info(
        "MSGIFSR [%s]: |I|=%d, train=%d val=%d test=%d mẫu (ước lượng)",
        dataset_dir,
        n_items,
        sum(max(0, len(s) - 1) for s in train_sessions),
        sum(max(0, len(s) - 1) for s in val_sessions),
        sum(max(0, len(s) - 1) for s in test_sessions),
    )

    order = mcfg.get("order", 1)
    collate_fn = collate_fn_factory_ccs((seq_to_ccs_graph,), order=order)
    train_loader, val_loader, test_loader = make_loaders(
        train_sessions,
        val_sessions,
        test_sessions,
        collate_fn,
        batch_size=tr.get("batch_size", 512),
        num_workers=tr.get("num_workers", 0),
    )

    model = build_model(cfg, n_items, dataset_dir, device).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("MSGIFSR model: order=%d, layers=%d, dim=%d, params=%d",
             order, mcfg.get("num_layers", 1), mcfg.get("embedding_dim", 256), n_params)

    optimizer = Adam(
        model.parameters(),
        lr=tr.get("learning_rate", 1e-3),
        weight_decay=tr.get("weight_decay", 1e-4),
    )
    scheduler = StepLR(optimizer, step_size=tr.get("lr_step_size", 3), gamma=tr.get("lr_gamma", 0.1))

    ckpt_dir = _make_checkpoint_dir(cfg)
    best_path = ckpt_dir / "best_model.pt"
    primary = ev.get("primary_metric", "mrr@20")
    top_k = ev.get("top_k", [10, 20])

    best_val = -1.0
    bad_epochs = 0
    patience = tr.get("patience", 3)
    max_epochs = tr.get("max_epochs", 30)
    log_interval = tr.get("log_interval", 100)

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            inputs, labels = batch
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(*inputs)
            loss = F.nll_loss(scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            if batch_idx % log_interval == 0:
                log.info("  Epoch %d batch %d: loss=%.4f", epoch, batch_idx, running_loss / log_interval)
                running_loss = 0.0

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, top_k)
        val_score = val_metrics.get(primary, val_metrics.get("mrr@20", 0.0))
        log.info(
            "  Epoch %d (%.1fs) Val: %s",
            epoch,
            time.time() - t0,
            " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
        )

        if val_score > best_val:
            best_val = val_score
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            log.info("  → Model tốt nhất mới (%s=%.4f)", primary, val_score)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("  Early stopping sau %d epoch không cải thiện.", patience)
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    else:
        log.warning("Không có checkpoint — dùng weights epoch cuối.")

    test_metrics = evaluate_model(model, test_loader, device, top_k)
    log.info(
        "  Test: %s",
        " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()),
    )

    _write_checkpoint_info(
        ckpt_dir / "info.yaml",
        cfg,
        {"best_val": best_val, "test_metrics": test_metrics},
    )
    return val_metrics, test_metrics
