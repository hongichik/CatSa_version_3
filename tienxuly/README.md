# tienxuly — Tải dataset & Tiền xử lý (Giai đoạn 1 của CatSA)

Module chạy **độc lập**, chuyển dataset thô thành sessions + lookup tables cho CatSA.

## Chạy

```bash
# Chạy LẦN LƯỢT config/tienxuly/retailrocket/select.yaml (mặc định)
python tienxuly/main.py

# Diginetica
python tienxuly/main.py --suite diginetica

# Chỉ một phiên bản
python tienxuly/main.py --run retailrocket_2_5.yaml
```

## Cấu hình

| File | Ý nghĩa |
|---|---|
| `config/tienxuly/retailrocket/dataset.yaml` | nguồn RetailRocket |
| `config/tienxuly/retailrocket/select.yaml` | danh sách preprocess RR |
| `config/tienxuly/diginetica/` | tương tự cho Diginetica |

### Tham số mới trong mỗi file `preprocess`

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `session_length_mode` | `truncate` | Phiên dài hơn `max_session_length` bị **cắt** lấy phần cuối (chuẩn SBR) |
| | `filter` | Chỉ **giữ** phiên có độ dài trong `[min_session_length, max_session_length]` |
| `require_item_category` | `true` | Chỉ giữ item có category trong `item_properties` |
| | `false` | Giữ cả item không category → gán category **UNK** trong lookup |

### Các phiên bản có sẵn

| File | Output | Mô tả |
|---|---|---|
| `retailrocket_2_5.yaml` | `data/retailrocket_2_5` | Phiên độ dài 2–5 |
| `retailrocket_3_6.yaml` | `data/retailrocket_3_6` | Phiên 3–6 |
| `retailrocket_4_7.yaml` | `data/retailrocket_4_7` | Phiên 4–7 |
| `retailrocket_2_7.yaml` | `data/retailrocket_2_7` | Phiên 2–7 |
| `retailrocket_category.yaml` | `data/retailrocket_category` | Chỉ item có category, truncate 2–50 |
| `retailrocket_no_category.yaml` | `data/retailrocket_no_category` | Cả item không category |
| `retailrocket_item_hon_5.yaml` | `data/retailrocket_item_hon_5` | Chuẩn cũ (support≥5, truncate) |

CatSA train trỏ `data.data_dir` trong `config/catsa/<version>.yaml` tới thư mục tương ứng.

## Đầu ra

Trong `<preprocess.output_dir>`: `train.txt`, `val.txt`, `test.txt`, `lookup_tables.pkl`.
