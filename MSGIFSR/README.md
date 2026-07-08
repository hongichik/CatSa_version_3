# MSGIFSR (WSDM 2022) — tích hợp vào demo2

Baseline **MSGIFSR**: *Learning Multi-granularity User Intent Unit for Session-based
Recommendation*. Code gốc (DGL + PyTorch) nằm trong `MSGIFSR_repo/` (đã xoá `.git`).
Thư mục `MSGIFSR/` bọc theo cấu trúc demo2: đọc `config/msgifsr/<suite>/`, dùng dữ liệu
`data/`, ghi log `Log/<dataset>/`.

## Cách chạy

```bash
# Smoke test nhanh (toy, n_items=60)
python MSGIFSR/main.py --run msgifsr_toy.yaml

# RetailRocket item_hon_5 (cùng bộ CatSA — ~79k item, KHÔNG bị OOM kiểu LINK)
python MSGIFSR/main.py --suite retailrocket

# Diginetica
python MSGIFSR/main.py --suite diginetica
```

## Pipeline

1. **Adapter** (`MSGIFSR/adapter.py`): `data/<dir>/*.txt` → `MSGIFSR_repo/datasets/<name>/`
   (comma-separated + `num_items.txt`, kèm `val.txt` cho early stopping).
2. **Train** (`MSGIFSR/train.py`): GNN sliding-window, early stopping trên val.
3. **Eval** (`MSGIFSR/evaluate.py`): full-ranking HR@K, NDCG@K, MRR@K trên test.

## Cấu hình `config/msgifsr/<suite>/<run>.yaml`

| Section | Ý nghĩa |
|---------|---------|
| `project` | Tên log (`Log/<name>/`) |
| `data` | Đường dẫn `data/<dataset>/` đã tiền xử lý |
| `msgifsr` | `dataset_name`, `reuse_converted` |
| `msgifsr_model` | `embedding_dim`, `num_layers`, `order`, `extra`, `fusion`... |
| `msgifsr_training` | `batch_size`, `learning_rate`, `max_epochs`, `patience`... |
| `evaluation` | `top_k`, `primary_metric` |

## So với LINK

MSGIFSR dùng **embedding + GNN** (bộ nhớ tuyến tính theo `n_items`), **không** tạo ma trận
`n_items²` như LINK → chạy được trên `retailrocket_item_hon_5` (~79k item) trong RAM 62GB.

## Phụ thuộc

- `dgl` (cài: `pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html`)
- `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`

## Thay đổi so với repo gốc

- Không sửa logic model trong `MSGIFSR_repo/`.
- Wrapper đọc dữ liệu demo2, thêm `val.txt`, early stopping trên val (gốc chỉ train/test).
- Đánh giá full-ranking giống CatSA/CORE (HR@K, NDCG@K, MRR@K).
