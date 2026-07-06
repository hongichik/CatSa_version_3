"""Theo dõi process đang chạy ngầm — dùng bởi main.py và runner.py.

Mọi process nền (runner + job) được ghi vào RUN/state/active.yaml.
main.py --run  : liệt kê process còn sống
main.py --kill : dừng theo tên job hoặc PID
"""

from __future__ import annotations

import os
import signal
from datetime import datetime
from pathlib import Path

import yaml

RUN_DIR = Path(__file__).resolve().parent
STATE_DIR = RUN_DIR / "state"
REGISTRY_FILE = STATE_DIR / "active.yaml"


def _load_registry() -> list[dict]:
    if not REGISTRY_FILE.exists():
        return []
    with open(REGISTRY_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("processes") or []


def _save_registry(processes: list[dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump({"processes": processes}, f, allow_unicode=True, sort_keys=False)


def is_alive(pid: int) -> bool:
    """Kiểm tra process còn sống thật (không tính zombie)."""
    if pid <= 0:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            state = f.read().split(")", 1)[1].split()[0]
        if state == "Z":
            return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def register(pid: int, name: str, command: str, log_file: str | Path,
             kind: str = "job") -> None:
    """Ghi process mới vào registry."""
    processes = _load_registry()
    processes.append({
        "pid": pid,
        "name": name,
        "kind": kind,          # "runner" | "job"
        "command": command,
        "log_file": str(log_file),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_registry(processes)


def cleanup_dead() -> list[dict]:
    """Loại process đã chết khỏi registry, trả về danh sách còn sống."""
    alive = [p for p in _load_registry() if is_alive(p["pid"])]
    _save_registry(alive)
    return alive


def list_running() -> list[dict]:
    """Danh sách process còn sống (tự dọn registry trước)."""
    return cleanup_dead()


def kill_process(pid: int) -> bool:
    """Gửi SIGTERM tới PID. Trả về True nếu gửi được."""
    if not is_alive(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False


def kill_by_target(target: str) -> list[tuple[int, str, bool]]:
    """Dừng process theo PID hoặc tên job/runner.

    target: số PID (vd "1804141") hoặc tên (vd "train_catsa", "runner")
    Trả về list (pid, name, success).
    """
    processes = list_running()
    results: list[tuple[int, str, bool]] = []

    if target.isdigit():
        pid = int(target)
        match = next((p for p in processes if p["pid"] == pid), None)
        name = match["name"] if match else str(pid)
        ok = kill_process(pid)
        results.append((pid, name, ok))
    else:
        matched = [p for p in processes if p["name"] == target]
        if not matched:
            return results
        for p in matched:
            ok = kill_process(p["pid"])
            results.append((p["pid"], p["name"], ok))

    cleanup_dead()
    return results
