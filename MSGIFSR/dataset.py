"""Dataset sliding-window cho MSGIFSR — đọc file đã chuyển từ demo2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, SequentialSampler

ROOT = Path(__file__).resolve().parent.parent
MSGIFSR_REPO = ROOT / "MSGIFSR_repo"
if str(MSGIFSR_REPO) not in sys.path:
    sys.path.insert(0, str(MSGIFSR_REPO))

from src.utils.data.dataset import AugmentedDataset  # noqa: E402


def read_sessions(path: Path) -> list[list[int]]:
    """Đọc phiên từ file (comma hoặc space separated)."""
    sessions: list[list[int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sep = "," if "," in line else " "
            sessions.append([int(x) for x in line.split(sep)])
    return sessions


def load_msgifsr_sessions(dataset_dir: Path) -> tuple[list, list, list, int]:
    """Trả về train_sessions, val_sessions, test_sessions, n_items."""
    train = read_sessions(dataset_dir / "train.txt")
    val = read_sessions(dataset_dir / "val.txt")
    test = read_sessions(dataset_dir / "test.txt")
    with open(dataset_dir / "num_items.txt", encoding="utf-8") as f:
        n_items = int(f.readline().strip())
    return train, val, test, n_items


def make_loaders(
    train_sessions: list,
    val_sessions: list,
    test_sessions: list,
    collate_fn,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train dùng SequentialSampler (giữ thứ tự — theo khuyến nghị paper gốc)."""
    train_set = AugmentedDataset(train_sessions)
    val_set = AugmentedDataset(val_sessions)
    test_set = AugmentedDataset(test_sessions)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=SequentialSampler(train_set),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
