"""Giai đoạn 4 — Dataset và DataLoader cho next-item prediction.

Mỗi phiên s = [i_1, ..., i_n] được expand thành n-1 mẫu kiểu sliding window:
    ([i_1], i_2), ([i_1, i_2], i_3), ..., ([i_1..i_{n-1}], i_n)
(đây là "sequence augmentation" chuẩn của SBR — khác với augmentation Module 2).

collate_fn xây heterogeneous graph cho từng sub-session rồi gom bằng
Batch.from_data_list; nếu có augmenter (CatSA đầy đủ) thì đồng thời sinh
phiên biến thể và batch graph tương ứng cho contrastive learning.
"""

from __future__ import annotations

from typing import Callable

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
    def __init__(
        self,
        sessions: list[list[int]],
        max_prefix_length: int = 0,
        prefix_len_min: int = 0,
        prefix_len_max: int = 0,
    ):
        """prefix_len_min/max: lọc mẫu theo len(prefix); 0 = không lọc."""
        self.samples = expand_session_samples(sessions, max_prefix_length)
        if prefix_len_min > 0 or prefix_len_max > 0:
            filtered: list[tuple[list[int], int]] = []
            for prefix, target in self.samples:
                pl = len(prefix)
                if prefix_len_min > 0 and pl < prefix_len_min:
                    continue
                if prefix_len_max > 0 and pl > prefix_len_max:
                    continue
                filtered.append((prefix, target))
            self.samples = filtered

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
        add_star_node: bool = False,
    ):
        self.item2cat = item2cat
        self.cat_parent = cat_parent
        self.use_taxonomy = use_taxonomy
        self.augmenter = augmenter  # None → chỉ L_rec (biến thể A2)
        self.add_star_node = add_star_node

    def __call__(self, batch: list[tuple[list[int], int]]):
        sessions = [s for s, _ in batch]
        targets = torch.tensor([t for _, t in batch], dtype=torch.long)

        batch_orig = sessions_to_batch(
            sessions, self.item2cat, self.cat_parent, self.use_taxonomy,
            add_star_node=self.add_star_node,
        )
        batch_orig.session_lists = sessions

        batch_aug = None
        if self.augmenter is not None:
            sessions_aug = [self.augmenter(s) for s in sessions]
            batch_aug = sessions_to_batch(
                sessions_aug, self.item2cat, self.cat_parent, self.use_taxonomy,
                add_star_node=self.add_star_node,
            )
            batch_aug.session_lists = sessions_aug

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
    prefix_len_min: int = 0,
    prefix_len_max: int = 0,
    add_star_node: bool = False,
    drop_last: bool = False,
    worker_init_fn: Callable[[int], None] | None = None,
) -> DataLoader:
    """drop_last: bỏ batch cuối nếu < batch_size (finding L3/T3) — dùng cho
    train_loader để tránh batch cỡ 1 lọt vào session_level_infonce (B>=2
    guard). KHÔNG dùng cho val/test loader — eval phải duyệt hết mọi mẫu."""
    return DataLoader(
        SessionDataset(
            sessions,
            max_prefix_length=max_prefix_length,
            prefix_len_min=prefix_len_min,
            prefix_len_max=prefix_len_max,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=GraphCollator(item2cat, cat_parent, use_taxonomy, augmenter, add_star_node),
        drop_last=drop_last,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        # Xây graph (sessions_to_batch) trong collate_fn tốn CPU nặng; với
        # num_workers=0 việc này chặn main thread và để GPU rảnh (idle) trong
        # lúc chờ. pin_memory chỉ có tác dụng khi num_workers > 0, không ảnh
        # hưởng tới kết quả/reproducibility.
        #
        # KHÔNG bật persistent_workers / prefetch_factor cao: SessionDataset
        # giữ toàn bộ sample (~18.8M tuple với train=yoochoose) trong 1 list
        # Python — fork worker qua multiprocessing bị "COW breakage" (mỗi
        # worker tự tăng refcount trên hầu hết object khi shuffle=True truy
        # cập ngẫu nhiên toàn bộ list), mỗi worker RES phồng lên xấp xỉ bằng
        # cả dataset gốc. persistent_workers=True từng làm pool train +
        # pool val cùng tồn tại một lúc → OOM-kill giữa epoch (không có
        # traceback vì SIGKILL không bắt được).
        pin_memory=num_workers > 0,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
