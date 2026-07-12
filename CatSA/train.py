"""Giai đoạn 4 + 6 — Training loop end-to-end."""

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

from .augment import CatSAAugmenter, resolve_unk_cat_idx
from .aux_heads import AuxiliaryHeads
from .dataset import make_loader
from .evaluate import evaluate_by_length_buckets, evaluate_dual_length, evaluate_model
from .losses import (
    auxiliary_category_loss,
    auxiliary_parent_loss,
    category_prototype_loss,
    session_level_infonce,
)
from .model import build_encoder


def _model_scores(model, z_s: torch.Tensor, batch=None) -> torch.Tensor:
    """Gọi scores(); truyền batch nếu encoder hỗ trợ."""
    import inspect
    sig = inspect.signature(model.scores)
    if batch is not None and len(sig.parameters) >= 2:
        return model.scores(z_s, batch)
    return model.scores(z_s)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=True + benchmark=True: cuDNN vẫn autotune nhưng chỉ trong
    # tập thuật toán deterministic (finding T2 vẫn giữ được reproducibility).
    # Lưu ý: session graph có kích thước thay đổi theo batch nên benchmark có
    # thể phải autotune lại nhiều lần — lợi ích tốc độ phụ thuộc mức đa dạng
    # kích thước batch trong dữ liệu thực tế.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id: int) -> None:
    """Seed riêng cho mỗi DataLoader worker (finding T2) — nếu không, mọi
    worker kế thừa cùng 1 state random từ process cha, phá vỡ tái lập khi
    num_workers > 0."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_checkpoint_dir(cfg: Config) -> Path:
    if cfg.training.save_dir:
        ckpt_dir = Path(cfg.training.save_dir)
    else:
        log_path = get_log_path()
        run_name = log_path.stem if log_path else datetime.now().strftime("run-%Y%m%d-%H%M%S")
        ckpt_dir = Path(cfg.training.checkpoint_dir) / cfg.project.name / cfg.version / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _write_checkpoint_info(info_path: Path, cfg: Config, extra: dict) -> None:
    log_path = get_log_path()
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict["wandb"].pop("api_key", None)
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


def _save_checkpoint(path: Path, model, aux_heads) -> None:
    torch.save({
        "model": model.state_dict(),
        "aux_heads": aux_heads.state_dict() if aux_heads is not None else None,
    }, path)


def _load_checkpoint(path: Path, model, aux_heads, device) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        if aux_heads is not None and state.get("aux_heads"):
            aux_heads.load_state_dict(state["aux_heads"])
    else:
        model.load_state_dict(state)


def _compute_cl(
    tr, batch_aug, model, z_s, targets, item2cat, cat_parent, device,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=device)
    if tr.cl_type in ("infonce", "both"):
        if batch_aug is None:
            raise ValueError("cl_type infonce/both cần augmentation (use_cl=true)")
        batch_aug = batch_aug.to(device, non_blocking=True)
        z_tilde = model(batch_aug)
        loss = loss + tr.lambda_cl * session_level_infonce(z_s, z_tilde, tau=tr.tau)
    if tr.cl_type in ("prototype", "both"):
        loss = loss + tr.lambda_proto * category_prototype_loss(
            z_s, targets, item2cat, tau=tr.tau, cat_parent=cat_parent,
        )
    return loss


def run_length_dual_eval(data: dict, cfg: Config) -> dict[str, float]:
    """Đánh giá kết hợp 2 checkpoint (short/long) với routing theo len(prefix)."""
    log = get_logger()
    tr = cfg.training
    ev = cfg.evaluation
    device = resolve_device(tr.device)

    ckpt_short = Path(ev.checkpoint_short)
    ckpt_long = Path(ev.checkpoint_long)
    if not ckpt_short.is_file():
        raise FileNotFoundError(f"checkpoint_short không tồn tại: {ckpt_short}")
    if not ckpt_long.is_file():
        raise FileNotFoundError(f"checkpoint_long không tồn tại: {ckpt_long}")

    log.info(
        "Length-dual eval: threshold=%d | short=%s | long=%s",
        ev.length_threshold, ckpt_short, ckpt_long,
    )

    item2cat = data["item2cat"]
    cat_parent = data["cat_parent"] if cfg.model.use_taxonomy else None
    max_prefix = int(data.get("max_prefix_length", 50))

    val_loader = make_loader(
        data["val_sessions"], item2cat, cat_parent, cfg.model.use_taxonomy,
        tr.batch_size, shuffle=False, num_workers=tr.num_workers,
        max_prefix_length=max_prefix, add_star_node=cfg.model.use_star_node,
    )
    test_loader = make_loader(
        data["test_sessions"], item2cat, cat_parent, cfg.model.use_taxonomy,
        tr.batch_size, shuffle=False, num_workers=tr.num_workers,
        max_prefix_length=max_prefix, add_star_node=cfg.model.use_star_node,
    )

    model_short = build_encoder(
        cfg.model, data["n_items"], data["n_cats"],
        item2cat=item2cat, cat2items=data.get("cat2items"),
        cat_parent=cat_parent,
    ).to(device)
    model_long = build_encoder(
        cfg.model, data["n_items"], data["n_cats"],
        item2cat=item2cat, cat2items=data.get("cat2items"),
        cat_parent=cat_parent,
    ).to(device)

    _load_checkpoint(ckpt_short, model_short, None, device)
    _load_checkpoint(ckpt_long, model_long, None, device)

    val_metrics = evaluate_dual_length(
        model_short, model_long, val_loader, device, ev.top_k, ev.length_threshold,
    )
    log.info(
        "Val (dual): %s | short=%d long=%d",
        " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if "@" in k and not k.startswith("short_") and not k.startswith("long_")),
        int(val_metrics.get("n_short", 0)),
        int(val_metrics.get("n_long", 0)),
    )

    test_metrics = evaluate_dual_length(
        model_short, model_long, test_loader, device, ev.top_k, ev.length_threshold,
    )
    log.info(
        "TEST (dual): %s | short=%d long=%d",
        " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items() if "@" in k and not k.startswith("short_") and not k.startswith("long_")),
        int(test_metrics.get("n_short", 0)),
        int(test_metrics.get("n_long", 0)),
    )
    log.info(
        "  Short bucket: %s",
        " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items() if k.startswith("short_")),
    )
    log.info(
        "  Long bucket: %s",
        " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items() if k.startswith("long_")),
    )
    set_summary({f"test/{k}": v for k, v in test_metrics.items()})
    return test_metrics


def train_model(data: dict, cfg: Config):
    if cfg.evaluation.mode == "length_dual":
        return None, run_length_dual_eval(data, cfg)

    log = get_logger()
    tr = cfg.training
    mc = cfg.model
    set_seed(tr.seed)
    device = resolve_device(tr.device)

    log.info(
        "Thiết bị: %s | encoder=%s | fusion=%s | use_cl=%s | cl_type=%s | aux_cat=%s | aux_parent=%s",
        device, mc.encoder_type, mc.fusion_type, tr.use_cl, tr.cl_type,
        tr.aux_cat, tr.aux_parent,
    )

    item2cat = data["item2cat"]
    cat_parent = data["cat_parent"] if cfg.model.use_taxonomy else None

    unk_cat_idx = resolve_unk_cat_idx(data)
    augmenter = CatSAAugmenter(
        item2cat, data["cat2items"], cat_parent, cfg.augment,
        unk_cat_idx=unk_cat_idx, seed=tr.seed,
    ) if tr.use_cl and tr.cl_type in ("infonce", "both") else None

    max_prefix = int(data.get("max_prefix_length", 50))
    pl_min, pl_max = tr.prefix_len_min, tr.prefix_len_max
    if pl_min > 0 or pl_max > 0:
        log.info("Lọc mẫu theo len(prefix): min=%s max=%s", pl_min or "—", pl_max or "—")

    train_loader = make_loader(
        data["train_sessions"], item2cat, cat_parent, cfg.model.use_taxonomy,
        tr.batch_size, shuffle=True, num_workers=tr.num_workers, augmenter=augmenter,
        max_prefix_length=max_prefix,
        prefix_len_min=pl_min, prefix_len_max=pl_max,
        add_star_node=mc.use_star_node,
        drop_last=True, worker_init_fn=_worker_init_fn,
    )
    val_loader = make_loader(
        data["val_sessions"], item2cat, cat_parent, cfg.model.use_taxonomy,
        tr.batch_size, shuffle=False, num_workers=tr.num_workers,
        max_prefix_length=max_prefix,
        prefix_len_min=pl_min, prefix_len_max=pl_max,
        add_star_node=mc.use_star_node,
    )
    test_loader = make_loader(
        data["test_sessions"], item2cat, cat_parent, cfg.model.use_taxonomy,
        tr.batch_size, shuffle=False, num_workers=tr.num_workers,
        max_prefix_length=max_prefix,
        prefix_len_min=pl_min, prefix_len_max=pl_max,
        add_star_node=mc.use_star_node,
    )

    model = build_encoder(
        cfg.model, data["n_items"], data["n_cats"],
        item2cat=item2cat, cat2items=data.get("cat2items"),
        cat_parent=cat_parent,
    ).to(device)

    use_aux = tr.aux_cat or tr.aux_parent
    aux_heads = AuxiliaryHeads(
        cfg.model.embedding_dim, data["n_cats"], tr.aux_cat, tr.aux_parent,
    ).to(device) if use_aux else None

    params = list(model.parameters())
    if aux_heads is not None:
        params += list(aux_heads.parameters())
    optimizer = Adam(params, lr=tr.learning_rate, weight_decay=tr.weight_decay)

    ckpt_dir = _make_checkpoint_dir(cfg)
    ckpt_path = ckpt_dir / "best_model.pt"
    info_path = ckpt_dir / "info.yaml"

    primary = cfg.evaluation.primary_metric
    best_metric = -1.0
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, tr.max_epochs + 1):
        model.train()
        if aux_heads is not None:
            aux_heads.train()
        t0 = time.time()
        total_loss = total_rec = total_cl = total_aux = 0.0
        n_seen = 0
        if augmenter is not None:
            augmenter.reset_stats()

        for batch_orig, batch_aug, targets in train_loader:
            batch_orig = batch_orig.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            z_s = model(batch_orig)
            loss_rec = F.cross_entropy(_model_scores(model, z_s, batch_orig), targets)

            loss_cl = torch.tensor(0.0, device=device)
            if tr.use_cl:
                loss_cl = _compute_cl(
                    tr, batch_aug, model, z_s, targets, item2cat, cat_parent, device,
                )

            loss_aux = torch.tensor(0.0, device=device)
            if aux_heads is not None:
                aux_out = aux_heads(z_s)
                if "cat" in aux_out:
                    loss_aux = loss_aux + tr.lambda_aux_cat * auxiliary_category_loss(
                        aux_out["cat"], targets, item2cat,
                    )
                if "parent" in aux_out and cat_parent:
                    loss_aux = loss_aux + tr.lambda_aux_parent * auxiliary_parent_loss(
                        aux_out["parent"], targets, item2cat, cat_parent,
                    )

            loss = loss_rec + loss_cl + loss_aux
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=tr.grad_clip)
            optimizer.step()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_rec += loss_rec.item() * bs
            total_cl += loss_cl.item() * bs
            total_aux += loss_aux.item() * bs
            n_seen += bs

        cl_str = f"{total_cl / n_seen:.4f}" if tr.use_cl else "n/a (use_cl=false)"
        log.info(
            "Epoch %d: total=%.4f, rec=%.4f, cl=%s, aux=%.4f (%.1fs)",
            epoch, total_loss / n_seen, total_rec / n_seen,
            cl_str, total_aux / n_seen, time.time() - t0,
        )
        if augmenter is not None:
            log.info(
                "  Augment no-op rate: %.4f (%d/%d lần gọi không tạo thay đổi thật)",
                augmenter.noop_rate(), augmenter.n_noop, augmenter.n_calls,
            )

        val_metrics = evaluate_model(model, val_loader, device, cfg.evaluation.top_k)
        log.info("  Val: %s", " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
        log_metrics({
            "train/loss_total": total_loss / n_seen,
            "train/loss_rec": total_rec / n_seen,
            **({"train/loss_cl": total_cl / n_seen} if tr.use_cl else {}),
            "train/loss_aux": total_aux / n_seen,
            **({"train/augment_noop_rate": augmenter.noop_rate()} if augmenter is not None else {}),
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }, step=epoch)

        if val_metrics[primary] > best_metric:
            best_metric = val_metrics[primary]
            best_epoch = epoch
            patience_count = 0
            _save_checkpoint(ckpt_path, model, aux_heads)
            _write_checkpoint_info(info_path, cfg, {
                "epoch_tot_nhat": best_epoch,
                "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            })
            log.info("  → Model tốt nhất mới (%s=%.4f)", primary, best_metric)
        else:
            patience_count += 1
            if patience_count >= tr.patience:
                log.info("Early stopping tại epoch %d", epoch)
                break

    _load_checkpoint(ckpt_path, model, aux_heads, device)
    test_metrics = evaluate_model(model, test_loader, device, cfg.evaluation.top_k)
    log.info("KẾT QUẢ TEST: %s", " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    set_summary({f"test/{k}": v for k, v in test_metrics.items()})

    # Sub-population analysis theo độ dài phiên (finding T6) — luôn báo cáo
    # kèm kết quả tổng, vì đây là phân tích cốt lõi của CatSA (overall SBR):
    # phiên ngắn/dài có hiệu năng rất khác nhau (xem phân tích len_dual).
    bucket_metrics: dict[str, float] = {}
    try:
        bucket_metrics = evaluate_by_length_buckets(
            model, test_loader, device, cfg.evaluation.top_k,
            buckets=[(1, 3), (4, 7), (8, None)],
        )
        log.info(
            "  Sub-population theo độ dài phiên: %s",
            " | ".join(f"{k}={v:.4f}" for k, v in bucket_metrics.items()),
        )
    except ValueError:
        log.info("  Bỏ qua sub-population analysis (batch thiếu session_lists)")

    _write_checkpoint_info(info_path, cfg, {
        "epoch_tot_nhat": best_epoch,
        "val_metrics_tot_nhat": {primary: round(best_metric, 6)},
        "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
        "test_metrics_by_length_bucket": {k: round(v, 6) for k, v in bucket_metrics.items()},
    })
    return model, test_metrics
