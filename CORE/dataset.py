"""Dataset CORE — đọc phiên từ data/ (train.txt, val.txt, test.txt)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def pad_batch(seqs: list[list[int]], device: torch.device) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return out


def expand_session_samples(
    sessions: list[list[int]],
    max_prefix_length: int = 0,
) -> list[tuple[list[int], int]]:
    """Sliding window giống CatSA / CORE gốc."""
    samples: list[tuple[list[int], int]] = []
    for s in sessions:
        for t in range(1, len(s)):
            prefix = s[:t]
            if max_prefix_length > 0 and len(prefix) > max_prefix_length:
                prefix = prefix[-max_prefix_length:]
            samples.append((prefix, s[t]))
    return samples


def _to_model_ids(item_id: int) -> int:
    """Chuyển id 0-indexed trong data/ sang id embedding (1..n, 0=pad)."""
    return item_id + 1


class CoreSessionDataset(Dataset):
    def __init__(self, sessions: list[list[int]], max_prefix_length: int = 0):
        self.samples = expand_session_samples(sessions, max_prefix_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        prefix, target = self.samples[idx]
        seq = [_to_model_ids(i) for i in prefix]
        return seq, _to_model_ids(target)


def _collate(batch: list[tuple[list[int], int]]) -> tuple[list[list[int]], list[int]]:
    seqs, targets = zip(*batch)
    return list(seqs), list(targets)


def make_loader(
    sessions: list[list[int]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_prefix_length: int = 0,
) -> DataLoader:
    return DataLoader(
        CoreSessionDataset(sessions, max_prefix_length),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
    )
