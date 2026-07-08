"""Giai đoạn 5 — Module 2: category-guided augmentation.

Ba chiến lược sinh phiên biến thể s~ từ phiên gốc s (chạy hoàn toàn ở mức
input data, không đụng đến encoder hay loss):
    - same    : same-leaf substitution — thay item bằng item CÙNG leaf category
    - sibling : sibling-leaf substitution — thay bằng item của category ANH EM
                (cùng parent trong taxonomy); fallback sang same-leaf nếu
                category không có sibling
    - hybrid  : same-leaf substitution + random crop

Cơ chế fallback k_min: nếu category có ít hơn k_min ứng viên thì KHÔNG thay
item đó (giữ nguyên) — tránh substitution kém chất lượng với category quá nhỏ.

Item không có danh mục thật (gán UNK lúc tiền xử lý) cũng KHÔNG được thay —
giữ nguyên item gốc.
"""

from __future__ import annotations

import random
from collections import defaultdict

from common.config import AugmentConfig

# Khớp tienxuly.preprocess._UNK_CAT_RAW — dùng khi suy unk_cat_idx từ cat_id_map cũ
_UNK_CAT_RAW = -1


def resolve_unk_cat_idx(data: dict) -> int | None:
    """Chỉ số category UNK trong lookup (None nếu dataset không có item không danh mục)."""
    if "unk_cat_idx" in data:
        return data["unk_cat_idx"]
    cat_id_map = data.get("cat_id_map") or {}
    return cat_id_map.get(_UNK_CAT_RAW)


def compute_siblings(cat_parent: dict[int, int]) -> dict[int, list[int]]:
    """Precompute bảng siblings (làm 1 lần, reuse): cat → các cat cùng parent."""
    parent_to_children: dict[int, list[int]] = defaultdict(list)
    for cat, parent in cat_parent.items():
        parent_to_children[parent].append(cat)

    siblings: dict[int, list[int]] = {}
    for cat, parent in cat_parent.items():
        siblings[cat] = [c for c in parent_to_children[parent] if c != cat]
    return siblings


class CatSAAugmenter:
    """Hàm augment chung: mỗi lần gọi chọn ngẫu nhiên 1 chiến lược."""

    def __init__(
        self,
        item2cat: dict[int, int],
        cat2items: dict[int, list[int]],
        cat_parent: dict[int, int] | None,
        cfg: AugmentConfig,
        unk_cat_idx: int | None = None,
    ):
        self.item2cat = item2cat
        self.cat2items = cat2items
        self.cfg = cfg
        self.unk_cat_idx = unk_cat_idx

        # Dataset không có taxonomy → loại chiến lược sibling
        self.siblings = compute_siblings(cat_parent) if cat_parent else None
        self.strategies = list(cfg.strategies)
        if self.siblings is None and "sibling" in self.strategies:
            self.strategies = [s for s in self.strategies if s != "sibling"]

        self.weights = cfg.strategy_weights or None
        if self.weights and len(self.weights) != len(self.strategies):
            self.weights = None  # số weight không khớp → dùng xác suất đều

    def _can_substitute_item(self, item_id: int) -> bool:
        """Item không có category thật (UNK) hoặc thiếu mapping → không thay."""
        cat = self.item2cat.get(item_id)
        if cat is None:
            return False
        if self.unk_cat_idx is not None and cat == self.unk_cat_idx:
            return False
        return True

    def _can_substitute_cat(self, cat_idx: int) -> bool:
        if self.unk_cat_idx is not None and cat_idx == self.unk_cat_idx:
            return False
        return True

    # ------------------------------------------------------------------
    # Ba chiến lược
    # ------------------------------------------------------------------

    def same_leaf(self, session: list[int]) -> list[int]:
        """Thay k = max(1, round(eta_aug*n)) item bằng item cùng leaf category."""
        n = len(session)
        k = max(1, round(self.cfg.eta_aug * n))
        positions = random.sample(range(n), min(k, n))

        augmented = list(session)
        for pos in positions:
            i_j = session[pos]
            if not self._can_substitute_item(i_j):
                continue
            cat_j = self.item2cat[i_j]
            candidates = self.cat2items.get(cat_j, [])
            if len(candidates) < self.cfg.k_min:
                continue  # fallback k_min: category quá nhỏ, giữ nguyên
            valid = [c for c in candidates if c != i_j]
            if valid:
                augmented[pos] = random.choice(valid)
        return augmented

    def sibling_leaf(self, session: list[int]) -> list[int]:
        """Thay item bằng item thuộc sibling category (cùng parent)."""
        n = len(session)
        k = max(1, round(self.cfg.eta_aug * n))
        positions = random.sample(range(n), min(k, n))

        augmented = list(session)
        for pos in positions:
            i_j = session[pos]
            if not self._can_substitute_item(i_j):
                continue
            cat_j = self.item2cat[i_j]
            sibling_cats = self.siblings.get(cat_j, []) if self.siblings else []

            if not sibling_cats:
                # Không có sibling → fallback sang same-leaf cho vị trí này
                candidates = self.cat2items.get(cat_j, [])
                valid = [c for c in candidates if c != i_j]
                if len(valid) < self.cfg.k_min:
                    continue
                augmented[pos] = random.choice(valid)
            else:
                valid_sib_cats = [c for c in sibling_cats if self._can_substitute_cat(c)]
                if not valid_sib_cats:
                    continue
                sib_cat = random.choice(valid_sib_cats)
                sib_candidates = self.cat2items.get(sib_cat, [])
                if len(sib_candidates) < self.cfg.k_min:
                    continue
                augmented[pos] = random.choice(sib_candidates)
        return augmented

    def hybrid(self, session: list[int]) -> list[int]:
        """Same-leaf substitution (giữ intent) rồi random crop (đa dạng cấu trúc)."""
        augmented = self.same_leaf(session)
        n = len(augmented)
        crop_len = max(1, int(n * self.cfg.eta_crop))
        start = random.randint(0, n - crop_len)
        return augmented[start: start + crop_len]

    # ------------------------------------------------------------------
    # Hàm augment chung
    # ------------------------------------------------------------------

    def __call__(self, session: list[int]) -> list[int]:
        """Sinh phiên biến thể: chọn ngẫu nhiên 1 chiến lược rồi áp dụng."""
        if self.weights:
            strategy = random.choices(self.strategies, weights=self.weights)[0]
        else:
            strategy = random.choice(self.strategies)

        if strategy == "same":
            return self.same_leaf(session)
        if strategy == "sibling":
            return self.sibling_leaf(session)
        if strategy == "hybrid":
            return self.hybrid(session)
        raise ValueError(f"Chiến lược augmentation không hợp lệ: {strategy}")
