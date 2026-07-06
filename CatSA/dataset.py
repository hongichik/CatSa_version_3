"""Giai đoạn 4 — Dataset và DataLoader cho next-item prediction.

Mỗi phiên s = [i_1, ..., i_n] được expand thành n-1 mẫu kiểu sliding window:
    ([i_1], i_2), ([i_1, i_2], i_3), ..., ([i_1..i_{n-1}], i_n)
(đây là "sequence augmentation" chuẩn của SBR — khác với augmentation Module 2).

collate_fn xây heterogeneous graph cho từng sub-session rồi gom bằng
Batch.from_data_list; nếu có augmenter (CatSA đầy đủ) thì đồng thời sinh
phiên biến thể và batch graph tương ứng cho contrastive learning.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .augment import CatSAAugmenter
from .graph import sessions_to_batch


def expand_session_samples(
    sessions: list[list[int]],
    max_prefix_length: int = 0,
) -> list[tuple[list[int], int]]:
    """Sliding window; prefix dài hơn max_prefix_length chỉ lấy phần cuối (giống test_all)."""
    samples: list[tuple[list[int], int]] = []
    for s in sessions:
        for t in range(1, len(s)):
            prefix = s[:t]
            if max_prefix_length > 0 and len(prefix) > max_prefix_length:
                prefix = prefix[-max_prefix_length:]
            samples.append((prefix, s[t]))
    return samples


class SessionDataset(Dataset):
    def __init__(self, sessions: list[list[int]], max_prefix_length: int = 0):
        self.samples = expand_session_samples(sessions, max_prefix_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]


class GraphCollator:
    """Gom một batch (sub_session, target) thành (Batch gốc, Batch augmented, targets)."""

    def __init__(
        self,
        item2cat: dict[int, int],
        cat_parent: dict[int, int] | None,
        use_taxonomy: bool,
        augmenter: CatSAAugmenter | None = None,
    ):
        self.item2cat = item2cat
        self.cat_parent = cat_parent
        self.use_taxonomy = use_taxonomy
        self.augmenter = augmenter  # None → chỉ L_rec (biến thể A2)

    def __call__(self, batch: list[tuple[list[int], int]]):
        sessions = [s for s, _ in batch]
        targets = torch.tensor([t for _, t in batch], dtype=torch.long)

        batch_orig = sessions_to_batch(
            sessions, self.item2cat, self.cat_parent, self.use_taxonomy
        )

        batch_aug = None
        if self.augmenter is not None:
            sessions_aug = [self.augmenter(s) for s in sessions]
            batch_aug = sessions_to_batch(
                sessions_aug, self.item2cat, self.cat_parent, self.use_taxonomy
            )

        return batch_orig, batch_aug, targets


def make_loader(
    sessions: list[list[int]],
    item2cat: dict[int, int],
    cat_parent: dict[int, int] | None,
    use_taxonomy: bool,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    augmenter: CatSAAugmenter | None = None,
    max_prefix_length: int = 0,
) -> DataLoader:
    return DataLoader(
        SessionDataset(sessions, max_prefix_length=max_prefix_length),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=GraphCollator(item2cat, cat_parent, use_taxonomy, augmenter),
    )
