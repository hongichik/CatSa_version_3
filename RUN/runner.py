"""Code chạy ngầm của RUN/ — scheduler + launcher cho các job trong jobs.yaml.

KHÔNG gọi trực tiếp file này — dùng main.py ở thư mục gốc:
    python main.py            # nạp RUN/jobs.yaml, chạy các job enabled (ngầm)
    python main.py --list     # chỉ xem danh sách job

Cách hoạt động:
    - main.py spawn runner.py thành process NỀN (detached) rồi thoát ngay.
    - runner.py chờ đến thời điểm `when` của từng job, đến giờ thì spawn job
      thành process nền riêng (job không chết theo runner/terminal).
    - Output mỗi job → RUN/logs/<name>-<thời điểm>.log
    - Nhật ký của chính runner → RUN/logs/runner.log
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parent
sys.path.insert(0, str(ROOT))          # import RUN.processes khi chạy nền

from RUN.processes import register
ROOT = RUN_DIR.parent          # thư mục gốc dự án — cwd khi chạy các job
LOGS_DIR = RUN_DIR / "logs"
JOBS_FILE = RUN_DIR / "jobs.yaml"


def load_jobs() -> list[dict]:
    """Đọc danh sách job từ jobs.yaml (cả job đang tắt, để hiển thị --list)."""
    if not JOBS_FILE.exists():
        raise FileNotFoundError(f"Thiếu file cấu hình job: {JOBS_FILE}")
    with open(JOBS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    jobs = data.get("jobs") or []
    for j in jobs:
        if not j.get("name") or not j.get("command"):
            raise ValueError(f"Job thiếu 'name' hoặc 'command': {j}")
    return jobs


def parse_when(when) -> datetime:
    """Đổi giá trị `when` thành thời điểm chạy cụ thể.

    - "now"                  → ngay bây giờ
    - "HH:MM" / "HH:MM:SS"   → hôm nay lúc giờ đó; nếu đã qua thì ngày mai
    - "YYYY-MM-DD HH:MM[:SS]"→ đúng thời điểm đó
    """
    now = datetime.now()
    s = str(when).strip()
    if s.lower() == "now":
        return now

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(s, fmt).time()
            run_at = datetime.combine(now.date(), t)
            if run_at <= now:
                run_at += timedelta(days=1)  # giờ hôm nay đã qua → ngày mai
            return run_at
        except ValueError:
            pass

    raise ValueError(
        f"Giá trị when không hợp lệ: {when!r} "
        f"(chấp nhận: now | HH:MM | HH:MM:SS | YYYY-MM-DD HH:MM[:SS])"
    )


def _log(msg: str) -> None:
    """In nhật ký runner ra stdout — main.py đã redirect stdout của runner
    vào RUN/logs/runner.log nên không tự ghi file (tránh trùng dòng)."""
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}", flush=True)


def _is_alive(pid: int) -> bool:
    """Kiểm tra process còn chạy thật (không tính zombie)."""
    if pid <= 0:
        return False
    # Reap con trực tiếp nếu đã exit (tránh kẹt zombie)
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
    except ChildProcessError:
        pass
    except ProcessLookupError:
        return False
    # Linux: zombie vẫn trả về tồn tại với kill(0) — đọc state từ /proc
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            state = f.read().split(")", 1)[1].split()[0]
        if state == "Z":
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            return False
    except OSError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_for_pid(pid: int, name: str) -> None:
    """Chờ job con kết thúc và reap (tránh zombie block job tiếp theo)."""
    _log(f"Chờ job '{name}' (pid={pid}) kết thúc...")
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        while _is_alive(pid):
            time.sleep(5)
    _log(f"Job '{name}' (pid={pid}) đã kết thúc.")


def spawn_detached(command: str, log_file: Path) -> int:
    """Chạy lệnh thành process NỀN, sống độc lập (không chết theo terminal/runner).

    Trả về PID của process.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab") as out:
        proc = subprocess.Popen(
            shlex.split(command),
            cwd=ROOT,                    # lệnh luôn tính từ thư mục gốc dự án
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,      # tách session → không nhận SIGHUP
        )
    return proc.pid


def main() -> None:
    jobs = [j for j in load_jobs() if j.get("enabled", False)]
    if not jobs:
        _log("Không có job nào enabled — thoát.")
        return

    # Tính thời điểm chạy; giữ thứ tự khai báo trong yaml khi cùng `when`
    schedule = sorted(
        enumerate((parse_when(j.get("when", "now")), j) for j in jobs),
        key=lambda x: (x[1][0], x[0]),
    )
    _log(f"Runner khởi động (pid={os.getpid()}) — {len(schedule)} job:")
    for _, (run_at, j) in schedule:
        after = j.get("after")
        suffix = f" (sau '{after}')" if after else ""
        _log(f"  - {j['name']}: '{j['command']}' lúc {run_at:%Y-%m-%d %H:%M:%S}{suffix}")

    spawned: dict[str, int] = {}
    for idx, (run_at, j) in schedule:
        wait = (run_at - datetime.now()).total_seconds()
        if wait > 0:
            _log(f"Chờ {wait:.0f}s đến giờ chạy job '{j['name']}'...")
            time.sleep(wait)

        after = j.get("after")
        if after:
            dep = str(after).strip()
            if dep not in spawned:
                raise ValueError(
                    f"Job '{j['name']}': after='{dep}' — job phụ thuộc chưa được chạy "
                    f"(kiểm tra thứ tự trong jobs.yaml)"
                )
            _wait_for_pid(spawned[dep], dep)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = LOGS_DIR / f"{j['name']}-{stamp}.log"
        pid = spawn_detached(j["command"], log_file)
        spawned[j["name"]] = pid
        register(pid, j["name"], j["command"], log_file, kind="job")
        _log(f"ĐÃ CHẠY job '{j['name']}' (pid={pid}) — output: {log_file}")

    _log("Runner xong: mọi job đã được khởi chạy (job cuối vẫn có thể đang chạy nền).")


if __name__ == "__main__":
    main()
