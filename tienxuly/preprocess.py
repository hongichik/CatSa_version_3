"""Tiền xử lý RetailRocket / Diginetica → sessions train/val/test + lookup tables (Giai đoạn 1).

Đầu vào :
    - RetailRocket: events.csv, item_properties_part*.csv, category_tree.csv
    - Diginetica  : train-item-views.csv, product-categories.csv (flat category, không taxonomy)

Đầu ra  : các file trong <preprocess.output_dir>, TÊN FILE cấu hình được
trong config/tienxuly/preprocess.yaml (train_file, val_file, test_file, lookup_file):
    - train.txt / val.txt / test.txt : mỗi dòng một phiên
      dạng "item_1 item_2 ... item_n" (item đã index hóa 0..|I|-1)
    - lookup_tables.pkl : dict chứa
        item2cat   : dict[item_idx -> cat_idx]        (leaf category của mỗi item)
        cat2items  : dict[cat_idx -> list[item_idx]]  (lookup ngược cho Module 2)
        cat_parent : dict[cat_idx -> parent_idx]      (taxonomy 1 cấp, sibling-leaf)
        n_items, n_cats : kích thước vocabulary (cat gồm cả leaf và parent)
        item_id_map / cat_id_map : mapping id thô -> index (để convert ngược)

Tuân thủ 3 lưu ý kỹ thuật của tài liệu:
    1. Item id thô được remap thành index liên tiếp 0..|I|-1.
    2. Category cũng remap thành index 0..|C|-1 (leaf + parent chung không gian).
    3. Chống data leakage: vocabulary và cat2items CHỈ xây từ item của train + val;
       item chỉ xuất hiện trong test bị loại khỏi phiên test.
"""

from __future__ import annotations

import csv
import datetime
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common.config import DataConfig, PreprocessConfig
from common.logger import get_logger


# ---------------------------------------------------------------------------
# Đọc dữ liệu thô
# ---------------------------------------------------------------------------

def _load_events(raw_dir: Path, event_types: list[str]) -> pd.DataFrame:
    """Đọc events.csv, lọc theo loại sự kiện, sắp theo (visitor, thời gian)."""
    log = get_logger()
    events = pd.read_csv(raw_dir / "events.csv")
    log.info("events.csv: %d dòng, %d visitor", len(events), events["visitorid"].nunique())
    events = events[events["event"].isin(event_types)]
    log.info("Sau lọc event_types=%s: %d dòng", event_types, len(events))
    return events.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)


def _load_item2cat_raw(raw_dir: Path) -> dict[int, int]:
    """Xây item2cat (id thô) từ item_properties: lọc property == 'categoryid',
    giữ giá trị mới nhất theo timestamp cho mỗi item."""
    log = get_logger()
    parts = sorted(raw_dir.glob("item_properties_part*.csv"))
    if not parts:
        raise FileNotFoundError(f"Không tìm thấy item_properties_part*.csv trong {raw_dir}")
    frames = []
    for p in parts:
        df = pd.read_csv(p)
        frames.append(df[df["property"] == "categoryid"])
    props = pd.concat(frames, ignore_index=True)
    # Mỗi item có thể đổi category theo thời gian — lấy bản ghi mới nhất
    props = props.sort_values("timestamp").drop_duplicates("itemid", keep="last")
    item2cat = dict(zip(props["itemid"].astype(int), props["value"].astype(int)))
    log.info("item2cat (thô): %d item có category", len(item2cat))
    return item2cat


def _load_cat_parent_raw(raw_dir: Path) -> dict[int, int]:
    """Xây cat_parent (id thô) từ category_tree.csv; category gốc (parent NaN) bị bỏ."""
    log = get_logger()
    tree = pd.read_csv(raw_dir / "category_tree.csv")
    tree = tree.dropna(subset=["parentid"])
    cat_parent = dict(zip(tree["categoryid"].astype(int), tree["parentid"].astype(int)))
    log.info("cat_parent (thô): %d quan hệ child->parent", len(cat_parent))
    return cat_parent


def _load_diginetica_sessions(raw_dir: Path) -> list[tuple[int, list[int]]]:
    """Đọc train-item-views.csv — phiên đã có sẵn theo sessionId."""
    log = get_logger()
    path = raw_dir / "train-item-views.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy train-item-views.csv trong {raw_dir}")

    by_sess: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sid = row["sessionId"]
            item = int(row["itemId"])
            tf = int(row["timeframe"])
            date = row["eventdate"]
            by_sess[sid].append((tf, item, date))

    sessions: list[tuple[int, list[int]]] = []
    for clicks in by_sess.values():
        clicks.sort(key=lambda x: x[0])
        items = [c[1] for c in clicks]
        ts = int(datetime.datetime.strptime(clicks[0][2], "%Y-%m-%d").timestamp())
        sessions.append((ts, items))

    log.info("train-item-views.csv: %d phiên thô", len(sessions))
    return sessions


def _load_diginetica_item2cat(raw_dir: Path) -> dict[int, int]:
    """Xây item2cat từ product-categories.csv (Diginetica — flat category)."""
    log = get_logger()
    path = raw_dir / "product-categories.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy product-categories.csv trong {raw_dir}")
    item2cat: dict[int, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            item2cat[int(row["itemId"])] = int(row["categoryId"])
    log.info("product-categories.csv: %d item có category", len(item2cat))
    return item2cat


# ---------------------------------------------------------------------------
# Sessionize và lọc
# ---------------------------------------------------------------------------

def _is_same_day(t1: int, t2: int) -> bool:
    """Hai timestamp (ms hoặc s) có cùng ngày lịch không."""
    if t1 >= 1e10:
        t1 //= 1000
    if t2 >= 1e10:
        t2 //= 1000
    return (
        datetime.datetime.fromtimestamp(t1).date()
        == datetime.datetime.fromtimestamp(t2).date()
    )


def _sessionize_by_gap(events: pd.DataFrame, gap_seconds: int) -> list[tuple[int, list[int]]]:
    """Cắt chuỗi sự kiện của mỗi visitor thành các phiên theo khoảng cách thời gian.

    Trả về list (timestamp_bắt_đầu, [item_id thô, ...]) — timestamp dùng để
    chia train/val/test theo thời gian.
    """
    sessions: list[tuple[int, list[int]]] = []
    gap_ms = gap_seconds * 1000  # timestamp RetailRocket ở đơn vị millisecond

    for _, grp in events.groupby("visitorid", sort=False):
        ts = grp["timestamp"].to_numpy()
        items = grp["itemid"].to_numpy()
        start = 0
        for i in range(1, len(ts)):
            if ts[i] - ts[i - 1] > gap_ms:
                sessions.append((int(ts[start]), [int(x) for x in items[start:i]]))
                start = i
        sessions.append((int(ts[start]), [int(x) for x in items[start:]]))
    return sessions


def _sessionize_same_day(events: pd.DataFrame) -> list[tuple[int, list[int]]]:
    """Cắt phiên khi hai click liên tiếp không cùng ngày lịch (protocol test_all)."""
    sessions: list[tuple[int, list[int]]] = []
    for _, grp in events.groupby("visitorid", sort=False):
        ts = grp["timestamp"].to_numpy()
        items = grp["itemid"].to_numpy()
        if len(ts) == 0:
            continue
        start = 0
        for i in range(1, len(ts)):
            if not _is_same_day(int(ts[i - 1]), int(ts[i])):
                sessions.append((int(ts[start]), [int(x) for x in items[start:i]]))
                start = i
        sessions.append((int(ts[start]), [int(x) for x in items[start:]]))
    return sessions


def _sessionize(events: pd.DataFrame, cfg: PreprocessConfig) -> list[tuple[int, list[int]]]:
    if cfg.session_same_day:
        return _sessionize_same_day(events)
    return _sessionize_by_gap(events, cfg.session_gap_seconds)


def _dedup_consecutive(session: list[int]) -> list[int]:
    """Bỏ các click lặp liên tiếp cùng một item (chuẩn tiền xử lý SBR)."""
    out = [session[0]]
    for it in session[1:]:
        if it != out[-1]:
            out.append(it)
    return out


# Id category giả cho item không có danh mục (chỉ dùng khi require_item_category=false)
_UNK_CAT_RAW = -1


def _session_length_ok(length: int, cfg: PreprocessConfig) -> bool:
    """Kiểm tra độ dài phiên sau khi đã clean/dedup."""
    if length < cfg.min_session_length:
        return False
    if cfg.session_length_mode == "filter" and length > cfg.max_session_length:
        return False
    return True


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------

def _write_sessions(path: Path, sessions: list[list[int]]) -> None:
    """Ghi file phiên: mỗi dòng một phiên dạng 'item_1 item_2 ... item_n'."""
    with open(path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(" ".join(str(it) for it in s) + "\n")


def _read_sessions(path: Path) -> list[list[int]]:
    """Đọc file phiên (ngược với _write_sessions)."""
    sessions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append([int(x) for x in line.split()])
    return sessions


def _process_sessions(
    sessions: list[tuple[int, list[int]]],
    item2cat_raw: dict[int, int],
    cat_parent_raw: dict[int, int],
    cfg: PreprocessConfig,
) -> dict:
    """Lọc, chia train/val/test, remap vocabulary và ghi file (dùng chung mọi dataset)."""
    log = get_logger()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Tổng số phiên thô: %d", len(sessions))

    # Đếm support của item trên toàn bộ phiên
    support: dict[int, int] = defaultdict(int)
    for _, s in sessions:
        for it in s:
            support[it] += 1

    len_desc = (
        f"{cfg.min_session_length}..{cfg.max_session_length}"
        if cfg.session_length_mode == "filter"
        else f">={cfg.min_session_length} (giữ nguyên, prefix≤{cfg.max_prefix_length} lúc train)"
    )
    log.info(
        "Lọc phiên: support>=%d, require_category=%s, dedup=%s, độ dài %s, mode=%s",
        cfg.min_item_support,
        cfg.require_item_category,
        cfg.dedup_consecutive,
        len_desc,
        cfg.session_length_mode,
    )

    def _item_ok(it: int) -> bool:
        if support[it] < cfg.min_item_support:
            return False
        if cfg.require_item_category and it not in item2cat_raw:
            return False
        return True

    def _clean(s: list[int]) -> list[int]:
        s = [it for it in s if _item_ok(it)]
        if not s:
            return []
        if cfg.dedup_consecutive:
            s = _dedup_consecutive(s)
        return s

    cleaned = []
    for ts, s in sessions:
        s2 = _clean(s)
        if _session_length_ok(len(s2), cfg):
            cleaned.append((ts, s2))
    log.info("Số phiên sau lọc: %d", len(cleaned))

    # Giới hạn số phiên nếu chạy thử (max_sessions > 0)
    if cfg.max_sessions > 0 and len(cleaned) > cfg.max_sessions:
        cleaned = cleaned[: cfg.max_sessions]
        log.info("Giới hạn còn %d phiên (max_sessions, chạy thử)", len(cleaned))

    # Chia train/val/test THEO THỜI GIAN: phiên mới nhất vào test
    cleaned.sort(key=lambda x: x[0])
    n = len(cleaned)
    n_test = int(n * cfg.test_ratio)
    n_val = int(n * cfg.val_ratio)
    train_raw = [s for _, s in cleaned[: n - n_val - n_test]]
    val_raw = [s for _, s in cleaned[n - n_val - n_test: n - n_test]]
    test_raw = [s for _, s in cleaned[n - n_test:]]
    log.info("Chia theo thời gian: train=%d, val=%d, test=%d", len(train_raw), len(val_raw), len(test_raw))

    # --- Chống leakage: vocabulary chỉ từ train + val ---
    vocab_items = sorted({it for s in train_raw + val_raw for it in s})
    item_id_map = {raw: idx for idx, raw in enumerate(vocab_items)}
    log.info("Vocabulary |I| = %d item (chỉ từ train+val)", len(vocab_items))

    # Không gian category chung cho leaf + parent (parent dùng chung embedding)
    leaf_cats = sorted({item2cat_raw[it] for it in vocab_items if it in item2cat_raw})
    if not cfg.require_item_category and any(it not in item2cat_raw for it in vocab_items):
        leaf_cats = sorted(set(leaf_cats) | {_UNK_CAT_RAW})
    parent_cats = sorted({
        cat_parent_raw[c] for c in leaf_cats
        if c != _UNK_CAT_RAW and c in cat_parent_raw
    })
    all_cats = sorted(set(leaf_cats) | set(parent_cats))
    cat_id_map = {raw: idx for idx, raw in enumerate(all_cats)}
    log.info("|C| = %d category (leaf=%d, parent xuất hiện=%d, unk=%s)",
             len(all_cats), len(leaf_cats), len(parent_cats),
             _UNK_CAT_RAW in cat_id_map)

    # Lookup tables trên index
    item2cat: dict[int, int] = {}
    for it in vocab_items:
        item_idx = item_id_map[it]
        if it in item2cat_raw:
            item2cat[item_idx] = cat_id_map[item2cat_raw[it]]
        else:
            item2cat[item_idx] = cat_id_map[_UNK_CAT_RAW]
    cat2items: dict[int, list[int]] = defaultdict(list)
    for item_idx, cat_idx in item2cat.items():
        cat2items[cat_idx].append(item_idx)
    cat2items = dict(cat2items)
    cat_parent = {
        cat_id_map[c]: cat_id_map[cat_parent_raw[c]]
        for c in leaf_cats
        if c != _UNK_CAT_RAW and c in cat_parent_raw
    }

    # Remap phiên sang index; phiên test loại item ngoài vocab (cold item)
    def _remap(sess_list: list[list[int]]) -> list[list[int]]:
        out = []
        for s in sess_list:
            s2 = [item_id_map[it] for it in s if it in item_id_map]
            if _session_length_ok(len(s2), cfg):
                out.append(s2)
        return out

    train_sessions = _remap(train_raw)
    val_sessions = _remap(val_raw)
    test_sessions = _remap(test_raw)
    log.info("Phiên sau remap: train=%d, val=%d, test=%d",
             len(train_sessions), len(val_sessions), len(test_sessions))

    data = {
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "test_sessions": test_sessions,
        "item2cat": item2cat,
        "cat2items": cat2items,
        "cat_parent": cat_parent,
        "n_items": len(vocab_items),
        "n_cats": len(all_cats),
        "item_id_map": item_id_map,
        "cat_id_map": cat_id_map,
        "max_prefix_length": cfg.max_prefix_length,
        "session_length_mode": cfg.session_length_mode,
    }

    _sanity_checks(data, cfg.require_item_category)

    # Ghi 3 file phiên riêng biệt + 1 file lookup (tên file theo config)
    _write_sessions(out_dir / cfg.train_file, train_sessions)
    _write_sessions(out_dir / cfg.val_file, val_sessions)
    _write_sessions(out_dir / cfg.test_file, test_sessions)
    lookup = {k: v for k, v in data.items()
              if k not in ("train_sessions", "val_sessions", "test_sessions")}
    with open(out_dir / cfg.lookup_file, "wb") as f:
        pickle.dump(lookup, f)

    log.info("Đã lưu kết quả tiền xử lý vào %s:", out_dir.resolve())
    log.info("  %-20s %d phiên", cfg.train_file, len(train_sessions))
    log.info("  %-20s %d phiên", cfg.val_file, len(val_sessions))
    log.info("  %-20s %d phiên", cfg.test_file, len(test_sessions))
    log.info("  %-20s lookup tables (|I|=%d, |C|=%d)", cfg.lookup_file,
             data["n_items"], data["n_cats"])
    return data


def preprocess(raw_dir: Path, cfg: PreprocessConfig, dataset_name: str = "retailrocket") -> dict:
    """Chạy toàn bộ tiền xử lý, lưu train/val/test + lookup, trả về dict dữ liệu."""
    log = get_logger()
    name = dataset_name.strip().lower()

    if name == "diginetica":
        log.info("Dataset Diginetica — phiên có sẵn trong train-item-views.csv")
        sessions = _load_diginetica_sessions(raw_dir)
        item2cat_raw = _load_diginetica_item2cat(raw_dir)
        return _process_sessions(sessions, item2cat_raw, {}, cfg)

    if name != "retailrocket":
        raise ValueError(f"Dataset không hỗ trợ tiền xử lý: {dataset_name}")

    events = _load_events(raw_dir, cfg.event_types)
    item2cat_raw = _load_item2cat_raw(raw_dir)
    cat_parent_raw = _load_cat_parent_raw(raw_dir)

    if cfg.session_same_day:
        log.info("Sessionize (cùng ngày lịch)...")
    else:
        log.info("Sessionize (gap=%ds)...", cfg.session_gap_seconds)
    sessions = _sessionize(events, cfg)
    return _process_sessions(sessions, item2cat_raw, cat_parent_raw, cfg)


def _sanity_checks(data: dict, require_item_category: bool) -> None:
    """Bốn sanity check cuối Giai đoạn 1 theo tài liệu hướng dẫn."""
    log = get_logger()
    item2cat, cat2items, cat_parent = data["item2cat"], data["cat2items"], data["cat_parent"]

    # CHECK 1 — mọi item trong train đều có mapping category (kể cả UNK nếu cho phép)
    train_items = {it for s in data["train_sessions"] for it in s}
    assert all(it in item2cat for it in train_items), "CHECK 1 FAIL: item thiếu category"

    # CHECK 2 — cat2items đảo chiều đúng với item2cat
    for cat, items in cat2items.items():
        for it in items:
            assert item2cat[it] == cat, "CHECK 2 FAIL: cat2items không nhất quán"

    # CHECK 3 — phân bố kích thước category
    sizes = sorted(len(v) for v in cat2items.values())
    small = sum(1 for s in sizes if s < 5)
    log.info("CHECK 3 — |C_leaf|=%d | size min=%d, max=%d, median=%d | #cat size<5: %d",
             len(cat2items), sizes[0], sizes[-1], sizes[len(sizes) // 2], small)

    # CHECK 4 — taxonomy không tự tham chiếu
    for child, parent in cat_parent.items():
        assert child != parent, "CHECK 4 FAIL: category tự làm parent của chính nó"

    log.info("Sanity check Giai đoạn 1: PASS (4/4)")


def load_processed(cfg: DataConfig | PreprocessConfig) -> dict:
    """Load dữ liệu đã tiền xử lý: 3 file phiên + lookup.

    CatSA train dùng cfg.data (section `data` trong catsa_vX.yaml).
    Có thể truyền PreprocessConfig để tương thích ngược.
    """
    if isinstance(cfg, DataConfig):
        out_dir = Path(cfg.data_dir)
        train_file, val_file, test_file, lookup_file = (
            cfg.train_file, cfg.val_file, cfg.test_file, cfg.lookup_file,
        )
    else:
        out_dir = Path(cfg.output_dir)
        train_file, val_file, test_file, lookup_file = (
            cfg.train_file, cfg.val_file, cfg.test_file, cfg.lookup_file,
        )
    paths = {
        "train_sessions": out_dir / train_file,
        "val_sessions": out_dir / val_file,
        "test_sessions": out_dir / test_file,
        "lookup": out_dir / lookup_file,
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu dữ liệu tiền xử lý: {', '.join(missing)} — "
            f"chạy tienxuly rồi kiểm tra data.data_dir trong config/catsa/<version>.yaml"
        )

    with open(paths["lookup"], "rb") as f:
        data = pickle.load(f)
    data["train_sessions"] = _read_sessions(paths["train_sessions"])
    data["val_sessions"] = _read_sessions(paths["val_sessions"])
    data["test_sessions"] = _read_sessions(paths["test_sessions"])
    return data
