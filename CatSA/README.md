# CatSA — Huấn luyện & Đánh giá (Giai đoạn 2-6)

Module chạy **độc lập**, cài đặt thuật toán CatSA đầy đủ theo tài liệu hướng dẫn:

- **Module 1** — heterogeneous session graph (`graph.py`) + RGCN-style encoder
  với soft-attention readout (`model.py`)
- **Module 2** — category-guided augmentation: same-leaf / sibling-leaf / hybrid
  (`augment.py`)
- **Session-level InfoNCE** (`losses.py`) — multi-task `L = L_rec + λ·L_CL`

## Chạy

```bash
# BƯỚC TRƯỚC (bắt buộc): tiền xử lý dữ liệu
python tienxuly/main.py

# Huấn luyện + đánh giá — mặc định config/catsa/retailrocket/select.yaml
python CatSA/main.py

# Diginetica
python CatSA/main.py --suite diginetica

# Chỉ một cấu hình
python CatSA/main.py --run catsa_2_5.yaml
python CatSA/main.py --suite diginetica --run catsa_tune_a2.yaml

python CatSA/main.py --config config_khac   # dùng cây cấu hình khác
```

Danh sách chạy khai báo trong `config/catsa/retailrocket/select.yaml` hoặc
`config/catsa/diginetica/select.yaml`.

Chạy từ bất kỳ thư mục nào cũng được — kết quả (log, checkpoint) luôn ghi về
thư mục gốc dự án. Nếu chưa tiền xử lý, chương trình báo lỗi kèm hướng dẫn.

## Cấu hình liên quan (trong cây `config/`)

Mỗi thí nghiệm CatSA nằm trong **một file version** riêng:
`config/catsa/retailrocket/catsa_v1.yaml`, ... hoặc `config/catsa/diginetica/`.
Thêm file mới rồi liệt kê trong `select.yaml` của suite tương ứng.

| Section | Ý nghĩa |
|---|---|
| `project` | tên dataset (`retailrocket` / `diginetica` → `Log/<dataset>/`), cách đặt tên file log |
| `data` | **thư mục dữ liệu đã tiền xử lý** (`data_dir` + tên file train/val/test/lookup) — mỗi version có thể trỏ dataset khác |
| `model` | embedding dim, số lớp, encoder_type, fusion_type, taxonomy, dropout |
| `augment` | chiến lược, `eta_aug`, `eta_crop`, `k_min` |
| `training` | `use_cl`, `lambda_cl`, `tau`, batch, lr, epochs, seed, vị trí lưu model |
| `evaluation` | top-K, metric chính cho early stopping |

Ngoài ra: `config/common/wandb.yaml` để bật/tắt ghi log lên Weights & Biases.

## Kết quả sau khi chạy

- **Model tốt nhất** (theo `primary_metric` trên validation) luôn được lưu:
  `best_model.pt` + `info.yaml` (mô tả: dự án, version, run, epoch, metrics,
  toàn bộ cấu hình lúc train). Vị trí: `training.save_dir` nếu chỉ định,
  mặc định `checkpoints/<dự án>/<version>/<tên run>/`.
- **Log**: `Log/<dự án>/stt-ngày-tháng-năm-giờ.log` (+ wandb nếu bật).
- **Kết quả test** (HR@K, NDCG@K, MRR@K full-ranking) in cuối log và ghi vào
  `info.yaml`.

## Encoder & loss (cấu hình nghiên cứu)

Xem `config/catsa/retailrocket/catsa_research.yaml` — mẫu đầy đủ các tùy chọn.

**`model.encoder_type`**: `rgcn` | `concat` | `dual_path` | `hgt` | `retrieval` | `transition` | `soft_cat`

**`model.fusion_type`** (dual_path, transition): `cross_attn` | `gate` | `sum`

**`training.cl_type`** (khi `use_cl: true`): `infonce` | `prototype` | `both`

**`training.aux_cat` / `aux_parent`**: auxiliary task dự đoán category/parent item tiếp theo.

Ví dụ chạy biến thể dual-path:

```bash
python CatSA/main.py --run catsa_enc_dual_path.yaml
```

## Các biến thể thí nghiệm

- **CatSA đầy đủ** (mặc định): `use_cl: true`.
- **A2 — Module 1 only** (ablation): `use_cl: false` — chỉ train `L_rec`,
  không augmentation, không contrastive learning.
