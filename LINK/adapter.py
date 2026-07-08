"""Chuyển dữ liệu đã tiền xử lý của demo2 sang định dạng RecBole mà LINK cần.

demo2 lưu mỗi phiên trên một dòng (item id 0-indexed, cách nhau bằng khoảng trắng)
trong train.txt / val.txt / test.txt. LINK (dựa trên RecBole) cần:

- 3 file atomic (benchmark): <name>.train.inter / .valid.inter / .test.inter
  Cột: session_id:token, item_id_list:token_seq, item_id:token
  Mỗi dòng là một mẫu (prefix, target) — mở rộng sliding-window giống CatSA/CORE.

- 1 file phiên đầy đủ: <name>.train.session  (SLIS/LINK dùng để dựng ma trận tuyến tính)
  Cột: session_id:token, item_id_list:token_seq  (mỗi dòng = 1 phiên train đầy đủ)

Item id ghi ra dạng token (chuỗi số). RecBole tự remap token → id nội bộ (0 = PAD),
nên item id 0 của demo2 không xung đột.
"""

from __future__ import annotations

from pathlib import Path


def _write_inter(path: Path, sessions: list[list[int]]) -> int:
    """Ghi file .inter (mở rộng prefix→target). Trả về số mẫu."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
        row_id = 0
        for s in sessions:
            if len(s) < 2:
                continue
            for t in range(1, len(s)):
                prefix = s[:t]
                target = s[t]
                f.write(f"{row_id}\t{' '.join(map(str, prefix))}\t{target}\n")
                row_id += 1
                n += 1
    return n


def _write_session(path: Path, sessions: list[list[int]]) -> int:
    """Ghi file .session (phiên đầy đủ, chỉ train). Trả về số phiên."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("session_id:token\titem_id_list:token_seq\n")
        sid = 0
        for s in sessions:
            if len(s) < 2:
                continue
            f.write(f"{sid}\t{' '.join(map(str, s))}\n")
            sid += 1
            n += 1
    return n


def demo2_to_recbole(
    data: dict,
    dataset_name: str,
    out_root: str | Path,
    reuse: bool = True,
    logger=None,
) -> Path:
    """Sinh dataset RecBole cho LINK từ dict `load_processed`.

    Args:
        data: dict có train_sessions / val_sessions / test_sessions.
        dataset_name: tên dataset RecBole (thư mục con trong out_root).
        out_root: thư mục gốc chứa dataset RecBole (thường LINK_repo/dataset).
        reuse: nếu True và đã có đủ 4 file thì bỏ qua (không sinh lại).
        logger: logger tuỳ chọn để in tiến độ.

    Returns:
        Đường dẫn thư mục dataset RecBole đã tạo.
    """
    out_dir = Path(out_root) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "train_inter": out_dir / f"{dataset_name}.train.inter",
        "valid_inter": out_dir / f"{dataset_name}.valid.inter",
        "test_inter": out_dir / f"{dataset_name}.test.inter",
        "train_session": out_dir / f"{dataset_name}.train.session",
    }

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    if reuse and all(p.exists() for p in files.values()):
        _log(f"[adapter] Dùng lại dataset RecBole có sẵn tại {out_dir}")
        return out_dir

    n_train = _write_inter(files["train_inter"], data["train_sessions"])
    n_valid = _write_inter(files["valid_inter"], data["val_sessions"])
    n_test = _write_inter(files["test_inter"], data["test_sessions"])
    n_sess = _write_session(files["train_session"], data["train_sessions"])

    _log(
        f"[adapter] Đã sinh dataset RecBole '{dataset_name}' tại {out_dir}: "
        f"train={n_train} mẫu, valid={n_valid}, test={n_test}, "
        f"train.session={n_sess} phiên"
    )
    return out_dir
