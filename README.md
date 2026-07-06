# CatSA — Category-Enhanced Session-based Recommendation

Cài đặt theo tài liệu "Hướng dẫn xây dựng CatSA" (6 giai đoạn), gồm ba thành phần:

- **Module 1** — Category-Enhanced Session Graph: heterogeneous graph (item / category / parent) + RGCN-style encoder + soft-attention readout (SR-GNN).
- **Module 2** — Category-Structure-Guided Augmentation: same-leaf, sibling-leaf, hybrid.
- **Session-level InfoNCE Loss**: multi-task `L = L_rec + λ·L_CL-session`.

## Cấu trúc thư mục

```
config/             # TOÀN BỘ cấu hình (YAML, có chú thích từng mục để làm gì)
  common/           #   cấu hình dùng chung cho mọi dự án
    logging.yaml    #     log: thư mục gốc, mức log, console
    wandb.yaml      #     Weights & Biases: api_key, project, run name, online/offline
  tienxuly/         #   cấu hình tiền xử lý
    dataset.yaml    #     nguồn dữ liệu (kagglehub / local)
    preprocess.yaml #     tham số xử lý (session gap, lọc, chia train/val/test)
  catsa/            #   cấu hình dự án catsa - CHỈ 2 file
    select.yaml     #     chọn version cấu hình sẽ chạy (run: catsa_v1.yaml)
    catsa_v1.yaml   #     TOÀN BỘ cấu hình version 1: project, model (Module 1),
                    #     augment (Module 2), training, evaluation
main.py             # lệnh gốc QUẢN LÝ CHẠY: đọc RUN/jobs.yaml, chạy job ngầm
RUN/                # quản lý code chạy - xem RUN/README.md
  jobs.yaml         #   cấu hình job: lệnh gì, chạy lúc mấy giờ
  runner.py         #   code chạy ngầm (hẹn giờ + spawn nền)
  logs/             #   output từng job + nhật ký runner (tự tạo)
common/             # dùng chung (KHÔNG chạy trực tiếp)
  config.py         #   parser các file YAML trong config/
  logger.py         #   logger ghi ra Log/ (tên file stt-ngày-tháng-năm-giờ.log)
  tracker.py        #   Weights & Biases tracker
tienxuly/           # Giai đoạn 1: CHẠY RIÊNG - xem tienxuly/README.md
  main.py           #   entrypoint: python tienxuly/main.py
  download.py       #   tải dataset qua kagglehub hoặc dùng dữ liệu local
  preprocess.py     #   sessionize, lọc, chia train/val/test, lookup tables
CatSA/              # Giai đoạn 2-6: CHẠY RIÊNG - xem CatSA/README.md
  main.py           #   entrypoint: python CatSA/main.py
  graph.py          #   Giai đoạn 2: session → heterogeneous graph (HeteroData)
  model.py          #   Giai đoạn 3: encoder RGCN-style + soft-attention readout
  dataset.py        #   Giai đoạn 4: Dataset sliding-window + collate graph batch
  evaluate.py       #   Giai đoạn 4: full-ranking HR@K / NDCG@K / MRR@K
  augment.py        #   Giai đoạn 5: Module 2 (same / sibling / hybrid)
  losses.py         #   Giai đoạn 6: session-level InfoNCE (symmetric)
  train.py          #   Giai đoạn 4+6: training loop, early stopping, checkpoint
Log/                # file log (tự tạo), phân cấp Log/<dự án>/
checkpoints/        # checkpoint (tự tạo), phân cấp <dự án>/<version>/<run>/
                    #   mỗi run: best_model.pt + info.yaml (mô tả của cái gì)
data/               # kết quả tiền xử lý (tự tạo): train.txt, val.txt,
                    #   test.txt (mỗi dòng 1 phiên) + lookup_tables.pkl
                    #   (tên file chỉnh trong config/tienxuly/preprocess.yaml)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Dataset mặc định là RetailRocket, tải tự động qua `kagglehub`
(`retailrocket/ecommerce-dataset`) — cần đăng nhập Kaggle lần đầu
(`~/.kaggle/kaggle.json` hoặc biến môi trường `KAGGLE_USERNAME`/`KAGGLE_KEY`).

## Chạy

Không có main tập trung — **mỗi thư mục chạy riêng**, theo thứ tự:

```bash
# Bước 1: tải dataset + tiền xử lý (chi tiết: tienxuly/README.md)
python tienxuly/main.py

# Bước 2: huấn luyện + đánh giá CatSA (chi tiết: CatSA/README.md)
python CatSA/main.py

# Cả hai đều nhận --config để dùng cây cấu hình khác:
python CatSA/main.py --config config_khac
```

Chạy từ thư mục nào cũng được — mọi kết quả (log, data, checkpoint) luôn
ghi về thư mục gốc dự án.

**Chạy ngầm / hẹn giờ** (xem `RUN/README.md`): khai báo job trong
`RUN/jobs.yaml` (lệnh + thời gian chạy, ví dụ `when: "23:30"`), rồi:

```bash
python main.py              # khởi chạy các job enabled dưới nền, thoát terminal được
python main.py --list       # xem job đã khai báo trong RUN/jobs.yaml
python main.py --run        # xem process đang chạy ngầm
python main.py --kill=NAME  # dừng job/runner theo tên hoặc PID
```

## Cấu hình (cây thư mục config/)

Cấu hình dạng YAML, chia thư mục con theo từng phần của hệ thống
(`common/`, `tienxuly/`, `catsa/`), mỗi tham số có chú thích. Loader
**tự quét đệ quy mọi file YAML** trong cây `config/` và gộp các section lại.
Mỗi section chỉ được khai báo ở một file (trùng sẽ báo lỗi).

**Cơ chế version cho dự án**: thư mục nào có `select.yaml` thì chỉ file
được khai trong `run:` được nạp, các file còn lại bị bỏ qua. Muốn thêm
version mới: copy `catsa_v1.yaml` → `catsa_v2.yaml`, chỉnh tham số, rồi đổi
`run: catsa_v2.yaml` trong `select.yaml` — không phải sửa code.

Một số điểm chính:

- **Logging**: `dir` trong `config/common/logging.yaml` là thư mục GỐC;
  log của mỗi dự án ghi vào `<dir>/<tên dự án>/` (ví dụ `Log/catsa/`).
  Tên dự án và cách đặt tên file log nằm trong section `project` của file
  version đang chạy (`config/catsa/catsa_v1.yaml`): không khai báo gì = `auto` (tên
  `stt-ngày-tháng-năm-giờ.log`, ví dụ `001-04-07-2026-08.log`);
  khai báo `custom_filename` = đổi sang tên đó.
- **Weights & Biases**: bật/tắt qua `enabled` trong `config/common/wandb.yaml`;
  loss + metrics mỗi epoch và kết quả test được đẩy lên wandb song song với
  file log, tên run mặc định trùng tên file log để dễ đối chiếu.
  CẢNH BÁO: file này chứa `api_key` — không commit lên Git công khai.
- **Dataset**: `source: kagglehub` (tải tự động) hoặc `local` (chỉ đến
  thư mục đã có sẵn qua `local_path`).
- **Lưu model tốt nhất**: luôn lưu `best_model.pt` + `info.yaml`. Vị trí qua
  `training.save_dir`: để trống = mặc định theo version
  (`checkpoints/<dự án>/<version>/<run>/`); chỉ định đường dẫn = lưu đúng đó.
- **Ablation A2** (Module 1 only, không CL): đặt `use_cl: false` trong
  section `training` của `config/catsa/catsa_v1.yaml`.
- **Chạy thử nhanh**: đặt `max_sessions` (ví dụ 5000) trong
  `config/tienxuly/preprocess.yaml` để giới hạn dữ liệu.
- Hyperparameter mặc định theo tài liệu: `d=100`, `L=2`, `batch=100`, `lr=1e-3`,
  `λ=0.1`, `τ=0.5`, `η_aug=0.3`, `k_min=5`, seed 42.
# CatSa_version_3
# CatSa_version_3
