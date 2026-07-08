# LINK (SIGIR 2025) — tích hợp vào demo2

Baseline **LINK**: *Linear Item-Item Models with Neural Knowledge for Session-based
Recommendation* (Choi et al., SIGIR 2025). Code gốc dựa trên **RecBole** nằm trong
`LINK_repo/` (đã xoá `.git`). Thư mục `LINK/` này bọc code gốc lại theo đúng cấu trúc
demo2 (giống `CORE/`): đọc `config/link/<suite>/`, dùng dữ liệu đã tiền xử lý trong
`data/`, ghi log vào `Log/<dataset>/`.

## Cách chạy

```bash
# Smoke test nhanh (toy, n_items=60) — kiểm tra pipeline
python LINK/main.py --run link_toy.yaml

# RetailRocket (bản 4_7, n_items≈47k — chạy được với RAM ~62GB)
python LINK/main.py --suite retailrocket

# Diginetica
python LINK/main.py --suite diginetica

# Chỉ định 1 config cụ thể
python LINK/main.py --suite retailrocket --run link_retailrocket.yaml
```

## Pipeline (3 bước, tự động)

1. **Adapter** (`LINK/adapter.py`): chuyển `data/<dir>/{train,val,test}.txt`
   (phiên, item 0-indexed cách nhau bởi khoảng trắng) → RecBole atomic files trong
   `LINK_repo/dataset/<dataset_name>/`:
   - `*.train/valid/test.inter` — mẫu (prefix, target) mở rộng sliding-window
   - `*.train.session` — phiên train đầy đủ (SLIS/LINK dùng dựng ma trận tuyến tính)
2. **Teacher** `core_trm` (neural, RecBole): huấn luyện rồi trích ma trận tri thức
   `saved_models_for_embedding/linear_teacher_<name>_core_trm/dense_matrix.npy`.
3. **LINK** (closed-form): kết hợp SLIS + tri thức teacher → đánh giá full-ranking.

Metric in ra: `recall@K` (= HR@K cho next-item) và `mrr@K`, ghi vào `Log/<dataset>/`.

## Cấu hình `config/link/<suite>/<run>.yaml`

Các khoá quan trọng trong section `link:` (siêu tham số lấy theo `run_link.sh` gốc):

| Khoá | Ý nghĩa |
|------|---------|
| `dataset_name` | tên dataset RecBole sinh trong `LINK_repo/dataset/` |
| `reuse_converted` / `reuse_teacher` | dùng lại atomic files / teacher matrix nếu đã có |
| `teacher_epochs`, `train_batch_size`, `learning_rate`, `stopping_step` | huấn luyện teacher |
| `reg`, `reg_teacher`, `predict_weight`, `slis_alpha`, `teacher_temperature` | siêu tham số LINK |
| `gpu_id` | GPU dùng cho teacher |

## ⚠️ Lưu ý bộ nhớ (quan trọng)

LINK dựng **ma trận đặc `(n_items × n_items)` float32** và nghịch đảo nó:

| Dataset | n_items | ~RAM/ma trận |
|---------|---------|--------------|
| `data/diginetica` | 44.459 | ~7.9 GB |
| `data/retailrocket_4_7` | 47.090 | ~8.9 GB |
| `data/retailrocket_item_hon_5` (đầy đủ) | 79.649 | ~25 GB |

Vì pipeline giữ nhiều bản ma trận + workspace nghịch đảo, bản `item_hon_5` (~25GB×nhiều)
**dễ OOM** trên máy 62GB. Do đó `select.yaml` mặc định dùng `retailrocket_4_7`.
Muốn chạy bản đầy đủ, dùng `--run link_retailrocket_full.yaml` khi máy đủ RAM.

## Thay đổi so với repo gốc

- Thêm shim NumPy 2.0 ở đầu `LINK_repo/main.py` (`np.float_`/`np.complex_`/`np.unicode_`)
  để tương thích RecBole 1.2.0.
- `LINK_repo/props/overall.yaml`: thêm `train_neg_sample_args: ~` (RecBole ≥1.2 yêu cầu
  khi `loss_type: CE`).
- Viết lại `make_teacher_matrix_for_link` trong `LINK_repo/main.py` để trích teacher
  matrix bằng cách dựng trực tiếp phiên đơn-item (cách deepcopy DataLoader cũ không
  còn khớp sampler RecBole 1.2).

## Phụ thuộc

`recbole==1.2.0`, `torch`, `entmax`, `scipy`, `scikit-learn`, `numpy`, `pandas`
(đã có sẵn trong môi trường; `entmax` được cài thêm).
