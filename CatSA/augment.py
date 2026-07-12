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
from common.logger import get_logger

# Khớp tienxuly.preprocess._UNK_CAT_RAW — dùng khi suy unk_cat_idx từ cat_id_map cũ
_UNK_CAT_RAW = -1

# Độ dài tối thiểu của phiên sau crop (hybrid) — dưới ngưỡng này BỎ crop
# (xem CatSA_Correctness_Synthesis finding A1: crop_len có thể ra 1 với
# phiên ngắn, tạo positive pair gần như vô nghĩa cho session-level CL)
MIN_SESSION_LEN = 2


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
        seed: int | None = None,
    ):
        # RNG per-instance (finding A4) — dùng random module TOÀN CỤC không
        # tái lập được khi DataLoader chạy nhiều worker (mỗi worker process
        # có state random riêng, không seed theo seed thí nghiệm). Instance
        # này nhận seed rõ ràng, độc lập với global random.
        self._rng = random.Random(seed)

        if not (0 < cfg.eta_aug <= 1):
            raise ValueError(f"augment.eta_aug phải trong (0, 1], nhận: {cfg.eta_aug}")
        if not (0 < cfg.eta_crop <= 1):
            raise ValueError(f"augment.eta_crop phải trong (0, 1], nhận: {cfg.eta_crop}")

        self.item2cat = item2cat
        self.cat2items = cat2items
        self.cfg = cfg
        self.unk_cat_idx = unk_cat_idx

        # Dataset không có taxonomy → loại chiến lược sibling
        self.siblings = compute_siblings(cat_parent) if cat_parent else None
        self.strategies = list(cfg.strategies)
        if self.siblings is None and "sibling" in self.strategies:
            self.strategies = [s for s in self.strategies if s != "sibling"]

        # Weights ánh xạ theo TÊN chiến lược (finding A5) — không theo vị trí,
        # để không lệch thứ tự khi 1 chiến lược (vd sibling) bị loại. Nếu có
        # chiến lược bị loại, re-normalize trọng số các chiến lược còn lại
        # thay vì âm thầm rơi về đều, kèm log cảnh báo.
        self.weights: list[float] | None = None
        if cfg.strategy_weights:
            if len(cfg.strategy_weights) != len(cfg.strategies):
                get_logger().warning(
                    "augment.strategy_weights (%d phần tử) không khớp "
                    "augment.strategies (%d phần tử) — dùng xác suất đều.",
                    len(cfg.strategy_weights), len(cfg.strategies),
                )
            else:
                name_to_weight = dict(zip(cfg.strategies, cfg.strategy_weights))
                kept = {s: name_to_weight[s] for s in self.strategies if s in name_to_weight}
                total = sum(kept.values())
                if len(kept) < len(cfg.strategies):
                    dropped = set(cfg.strategies) - set(kept)
                    get_logger().warning(
                        "augment: chiến lược %s bị loại (dataset không có "
                        "taxonomy) — re-normalize trọng số còn lại: %s",
                        sorted(dropped),
                        {s: round(w / total, 4) for s, w in kept.items()} if total else kept,
                    )
                if total > 0:
                    self.weights = [kept[s] / total for s in self.strategies]

        # Thống kê no-op augmentation (finding A2) — dùng để chẩn đoán A7
        # (L_CL không giảm vì positive pair thực chất là (z_s, z_s))
        self.n_calls = 0
        self.n_noop = 0

    def noop_rate(self) -> float:
        """Tỷ lệ lần gọi augment mà phiên biến thể == phiên gốc (không đổi gì)."""
        return self.n_noop / self.n_calls if self.n_calls else 0.0

    def reset_stats(self) -> None:
        self.n_calls = 0
        self.n_noop = 0

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
        positions = self._rng.sample(range(n), min(k, n))

        augmented = list(session)
        for pos in positions:
            i_j = session[pos]
            if not self._can_substitute_item(i_j):
                continue
            cat_j = self.item2cat[i_j]
            candidates = self.cat2items.get(cat_j, [])
            # k_min luôn tính trên pool THỰC SỰ bốc ra (đã loại i_j) — nhất
            # quán với sibling_leaf (finding A3), tránh lệch ngưỡng 1 đơn vị
            valid = [c for c in candidates if c != i_j]
            if len(valid) < self.cfg.k_min:
                continue  # fallback k_min: category quá nhỏ, giữ nguyên
            augmented[pos] = self._rng.choice(valid)
        return augmented

    def sibling_leaf(self, session: list[int]) -> list[int]:
        """Thay item bằng item thuộc sibling category (cùng parent)."""
        n = len(session)
        k = max(1, round(self.cfg.eta_aug * n))
        positions = self._rng.sample(range(n), min(k, n))

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
                augmented[pos] = self._rng.choice(valid)
            else:
                valid_sib_cats = [c for c in sibling_cats if self._can_substitute_cat(c)]
                if not valid_sib_cats:
                    continue
                # Gộp ứng viên của TẤT CẢ sibling category (finding A6) —
                # trước đây chỉ bốc 1 sibling rồi bỏ cuộc nếu category đó
                # < k_min, dù sibling khác có thể đủ ứng viên.
                sib_candidates = [
                    it for c in valid_sib_cats for it in self.cat2items.get(c, [])
                ]
                if len(sib_candidates) < self.cfg.k_min:
                    continue
                augmented[pos] = self._rng.choice(sib_candidates)
        return augmented

    def hybrid(self, session: list[int]) -> list[int]:
        """Same-leaf substitution (giữ intent) rồi random crop (đa dạng cấu trúc).

        Guard (finding A1): nếu crop sẽ để lại < MIN_SESSION_LEN item, BỎ crop
        (chỉ trả về kết quả same-leaf) — tránh phiên biến thể độ dài 1, vốn
        cho positive pair gần như vô nghĩa trong session-level InfoNCE.
        """
        augmented = self.same_leaf(session)
        n = len(augmented)
        crop_len = max(1, int(n * self.cfg.eta_crop))
        if crop_len < MIN_SESSION_LEN:
            return augmented
        start = self._rng.randint(0, n - crop_len)
        return augmented[start: start + crop_len]

    # ------------------------------------------------------------------
    # Hàm augment chung
    # ------------------------------------------------------------------

    def _apply(self, strategy: str, session: list[int]) -> list[int]:
        if strategy == "same":
            return self.same_leaf(session)
        if strategy == "sibling":
            return self.sibling_leaf(session)
        if strategy == "hybrid":
            return self.hybrid(session)
        raise ValueError(f"Chiến lược augmentation không hợp lệ: {strategy}")

    def __call__(self, session: list[int]) -> list[int]:
        """Sinh phiên biến thể: chọn ngẫu nhiên 1 chiến lược rồi áp dụng.

        Guard (finding A2): nếu kết quả trùng y hệt phiên gốc (mọi vị trí bị
        k_min-skip), thử lần lượt các chiến lược còn lại trước khi chấp nhận
        no-op — tránh positive pair (z_s, z_s) trùng khít làm CL signal ≈ 0.
        Luôn đếm thống kê no-op thực tế (self.n_calls/self.n_noop) để chẩn
        đoán A7 (L_CL không giảm).
        """
        self.n_calls += 1
        if self.weights:
            first = self._rng.choices(self.strategies, weights=self.weights)[0]
        else:
            first = self._rng.choice(self.strategies)

        augmented = self._apply(first, session)
        if augmented != session:
            return augmented

        # No-op ở lần thử đầu — thử các chiến lược còn lại (mỗi cái 1 lần)
        for strategy in self.strategies:
            if strategy == first:
                continue
            augmented = self._apply(strategy, session)
            if augmented != session:
                return augmented

        # Không chiến lược nào tạo được thay đổi thật (mọi category trong
        # phiên đều < k_min) — chấp nhận no-op, chỉ ghi nhận thống kê.
        self.n_noop += 1
        return augmented
