"""Lệnh gốc quản lý CODE CHẠY — mọi job đều chạy NGẦM (nền).

    python main.py              # nạp RUN/jobs.yaml, khởi chạy các job enabled
    python main.py --list       # xem job đã khai báo trong RUN/jobs.yaml
    python main.py --run        # xem process đang chạy ngầm (PID, log, lệnh)
    python main.py --kill=NAME    # dừng theo tên job (vd train_catsa) hoặc runner
    python main.py --kill=PID     # dừng theo PID (vd 1804141)

Cấu hình job (lệnh gì, chạy lúc mấy giờ) nằm trong RUN/jobs.yaml;
code chạy ngầm nằm trong RUN/runner.py — xem RUN/README.md.

Gọi main.py xong là có thể thoát terminal: runner + job vẫn chạy tiếp.
Output từng job: RUN/logs/<tên job>-<thời điểm>.log
Nhật ký runner : RUN/logs/runner.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from RUN.processes import kill_by_target, list_running, register  # noqa: E402
from RUN.runner import LOGS_DIR, load_jobs, parse_when, spawn_detached  # noqa: E402


def _print_running() -> None:
    """In danh sách process đang chạy ngầm."""
    processes = list_running()
    if not processes:
        print("Không có process nào đang chạy ngầm.")
        return
    print(f"{'LOẠI':<8} {'PID':<8} {'TÊN':<16} {'BẮT ĐẦU':<20} LOG / LỆNH")
    for p in processes:
        print(f"{p.get('kind', 'job'):<8} {p['pid']:<8} {p['name']:<16} "
              f"{p.get('started_at', ''):<20} {p.get('log_file', '')}")
        print(f"{'':8} {'':8} {'':16} {'':20} {p.get('command', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quản lý chạy job ngầm (RUN/)")
    parser.add_argument("--list", action="store_true",
                        help="Liệt kê job trong RUN/jobs.yaml (cấu hình), không chạy")
    parser.add_argument("--run", action="store_true",
                        help="Hiển thị process đang chạy ngầm (PID, log, lệnh)")
    parser.add_argument("--kill", metavar="NAME|PID",
                        help="Dừng process theo tên job/runner hoặc PID")
    args = parser.parse_args()

    if args.run:
        _print_running()
        return

    if args.kill is not None:
        results = kill_by_target(args.kill)
        if not results:
            print(f"Không tìm thấy process đang chạy: {args.kill!r}")
            print("Xem danh sách: python main.py --run")
            sys.exit(1)
        for pid, name, ok in results:
            if ok:
                print(f"Đã gửi tín hiệu dừng → {name} (pid={pid})")
            else:
                print(f"Không dừng được (đã chết?): {name} (pid={pid})")
        return

    jobs = load_jobs()
    print(f"{'JOB':<16} {'ENABLED':<8} {'WHEN':<22} COMMAND")
    for j in jobs:
        when = str(j.get("when", "now"))
        run_at = parse_when(when)
        print(f"{j['name']:<16} {str(j.get('enabled', False)):<8} "
              f"{when} ({run_at:%d-%m %H:%M:%S})".ljust(48) + f" {j['command']}")

    if args.list:
        return

    enabled = [j for j in jobs if j.get("enabled", False)]
    if not enabled:
        print("\nKhông có job nào enabled trong RUN/jobs.yaml — không chạy gì.")
        return

    # Spawn runner thành process NỀN rồi thoát ngay — runner lo phần hẹn giờ
    runner_cmd = f"{sys.executable} RUN/runner.py"
    runner_log = LOGS_DIR / "runner.log"
    pid = spawn_detached(runner_cmd, runner_log)
    register(pid, "runner", runner_cmd, runner_log, kind="runner")
    print(f"\nĐã khởi chạy runner ngầm (pid={pid}) cho {len(enabled)} job enabled.")
    print(f"Theo dõi:  tail -f RUN/logs/runner.log")
    print(f"Xem đang chạy: python main.py --run")
    print(f"Dừng job:      python main.py --kill=<tên hoặc pid>")
    print("Có thể thoát terminal — job vẫn chạy tiếp.")


if __name__ == "__main__":
    main()
