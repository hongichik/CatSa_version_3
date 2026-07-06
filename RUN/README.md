# RUN — Quản lý code chạy (chạy ngầm + hẹn giờ)

Thư mục này quản lý việc **chạy các lệnh của dự án dưới nền (background)**,
có thể hẹn giờ chạy. Lệnh gốc để phát lệnh nằm ở thư mục gốc: `main.py`.

## Cấu trúc

```
main.py            # (ở thư mục GỐC) lệnh phát: đọc jobs.yaml, khởi chạy ngầm
RUN/
  jobs.yaml        # CẤU HÌNH: job nào, lệnh gì, chạy lúc mấy giờ
  runner.py        # code chạy ngầm: hẹn giờ + spawn job nền (không gọi trực tiếp)
  logs/            # output của từng job + nhật ký runner (tự tạo)
```

## Cách dùng

```bash
python main.py              # chạy mọi job enabled trong RUN/jobs.yaml (ngầm)
python main.py --list       # xem job đã khai báo (cấu hình), không chạy
python main.py --run         # xem process đang chạy ngầm (PID, log, lệnh)
python main.py --kill=NAME   # dừng theo tên (vd train_catsa, runner)
python main.py --kill=PID    # dừng theo PID (vd 1804141)

tail -f RUN/logs/runner.log                 # theo dõi runner
tail -f RUN/logs/train_catsa-<thời điểm>.log  # theo dõi output một job
```

Gọi `python main.py` xong là **thoát terminal được ngay** — runner và các job
đều là process nền độc lập, không chết theo terminal.

## Cấu hình job (RUN/jobs.yaml)

```yaml
jobs:
  - name: train_catsa           # tên job (đặt tên file log output)
    command: python CatSA/main.py   # lệnh chạy, tính từ thư mục gốc dự án
    enabled: true               # false = bỏ qua job này
    when: now                   # thời gian chạy, xem bên dưới
```

Giá trị `when`:

| Giá trị | Ý nghĩa |
|---|---|
| `now` | chạy ngay khi gọi `python main.py` |
| `"HH:MM"` hoặc `"HH:MM:SS"` | hôm nay lúc giờ đó; nếu đã qua giờ → ngày mai |
| `"YYYY-MM-DD HH:MM[:SS]"` | chạy đúng thời điểm đó |

Nhiều job được chạy theo thứ tự thời gian; mỗi job là một process nền riêng
với file output riêng trong `RUN/logs/`.

## Dừng job đang chạy

Vì job chạy ngầm, dùng `main.py --run` để xem PID rồi `--kill` để dừng:

```bash
python main.py --run              # xem danh sách đang chạy
python main.py --kill=train_catsa   # dừng theo tên job
python main.py --kill=runner        # dừng runner (job chưa đến giờ sẽ không chạy)
python main.py --kill=1804141       # dừng theo PID
```

Registry process lưu tại `RUN/state/active.yaml` (tự dọn process đã chết).
