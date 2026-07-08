"""Chuyển dữ liệu demo2 sang định dạng MSGIFSR.

demo2: train.txt / val.txt / test.txt — mỗi dòng một phiên, item 0-indexed, cách nhau
bằng khoảng trắng.

MSGIFSR gốc: train.txt / test.txt — item cách nhau bằng dấu phẩy, kèm num_items.txt.
Wrapper này sinh thêm val.txt để early-stopping giống CatSA/CORE.
"""

from __future__ import annotations

from pathlib import Path


def _write_sessions(path: Path, sessions: list[list[int]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in sessions:
            if len(s) < 2:
                continue
            f.write(",".join(map(str, s)) + "\n")
            n += 1
    return n


def demo2_to_msgifsr(
    data: dict,
    dataset_name: str,
    out_root: str | Path,
    reuse: bool = True,
    logger=None,
) -> Path:
    """Sinh thư mục dataset MSGIFSR từ dict `load_processed`."""
    out_dir = Path(out_root) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "train": out_dir / "train.txt",
        "val": out_dir / "val.txt",
        "test": out_dir / "test.txt",
        "num_items": out_dir / "num_items.txt",
    }

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    if reuse and all(p.exists() for p in files.values()):
        _log(f"[adapter] Dùng lại dataset MSGIFSR có sẵn tại {out_dir}")
        return out_dir

    n_train = _write_sessions(files["train"], data["train_sessions"])
    n_val = _write_sessions(files["val"], data["val_sessions"])
    n_test = _write_sessions(files["test"], data["test_sessions"])
    files["num_items"].write_text(f"{data['n_items']}\n", encoding="utf-8")

    _log(
        f"[adapter] Đã sinh dataset MSGIFSR '{dataset_name}' tại {out_dir}: "
        f"train={n_train}, val={n_val}, test={n_test}, n_items={data['n_items']}"
    )
    return out_dir
