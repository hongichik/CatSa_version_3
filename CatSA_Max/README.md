# CatSA_Max

Gói tái hiện **từ A-Z** kết quả ensemble thuần CatSA trên RetailRocket
(`data/retailrocket_item_hon_5`, full test 81,372 mẫu per-prefix).

## Chạy (1 lệnh duy nhất)

```bash
python CatSA_Max/main.py
```

Lệnh này tự động:
1. **Train tuần tự 6 thành viên** (~1h/model trên 1 GPU, tổng ~6h) vào
   `checkpoints/CatSA_Max/<tên>/`. Log từng model: `Log/retailrocket/CatSA_Max/<tên>.log`.
2. **Ensemble bucket-routing**: chọn trọng số riêng cho từng nhóm độ dài
   phiên (1-3 / 4-7 / 8+) trên tập VAL + repeat-boost, rồi đánh giá TEST.
3. Ghi kết quả cuối: `CatSA_Max/result.yaml`.

**Tự resume**: nếu bị ngắt giữa chừng, chạy lại đúng lệnh trên — thành viên
đã train xong (có KẾT QUẢ TEST) được bỏ qua tự động.

Tùy chọn:
- `--skip-train` — chỉ chạy bước ensemble từ checkpoint có sẵn (~12 phút)
- `--retrain` — ép train lại tất cả

## 6 thành viên (đều là kiến trúc CatSA++ v2: Module 1 + Module 2 + InfoNCE)

| Tên | Khác biệt so với base | Loại khác biệt |
|---|---|---|
| v2 | — (cl_type=infonce) | mốc |
| proto | + prototype loss (cl_type=both, λ_proto=0.05) | hàm loss |
| multi | + length gate + multi-interest readout | kiến trúc |
| len_gate | + length-aware gate | kiến trúc |
| len_gate_seed43 | = len_gate, seed 43 | thuần trọng số |
| cat_intent | + category-intent + repeat-boost học được | kiến trúc |

Định nghĩa chính xác từng thành viên: dict `MEMBERS` trong `main.py`
(override trên config tham chiếu `config/catsa/retailrocket/baseline/catsa_plus_v2_len_gate.yaml`).

## Kết quả kỳ vọng (đã kiểm chứng 2026-07-10, seed 42/43)

| Hệ | Test mrr@20 |
|---|---|
| **CatSA_Max (ensemble 6 CatSA)** | **≈ 0.3895** |
| Control: ensemble CORE-only (2 seed, cùng điều kiện) | 0.3835 |
| Thành viên đơn tốt nhất | ≈ 0.370 |

Do PyTorch Geometric dùng scatter/atomic ops không deterministic 100% trên
GPU, train lại từ đầu có thể lệch ±0.001–0.002 ở từng thành viên; kết quả
ensemble thường ổn định hơn.

## Ghi chú trung thực khi công bố

- Nhánh sequential của CatSA++ v2 dùng **transformer encoder mượn từ CORE**
  (Hou et al. 2022) — phải khai báo trong Method, không claim novelty phần này.
- **Repeat-boost** (cộng δ vào logit item đã xuất hiện trong prefix — tiền lệ
  RepeatNet) và **bucket-routing** (trọng số chọn trên val theo độ dài phiên)
  là thành phần inference — mô tả rõ trong paper.
- Đây là kết quả **ensemble**; kết quả single-model để làm bảng chính:
  CatSA v3_full + repeat-aware δ=6 ≈ 0.3815 (ngang CORE cùng điều kiện).
- Số liệu 1 seed/thành viên — cần multi-seed + Wilcoxon trước khi nộp bài
  (xem `CatSA/multi_seed.py`).

## Phụ thuộc

Gói này tái dùng thư viện của repo: `CatSA/` (encoder, train, eval, ensemble),
`common/` (config, logger), `tienxuly/` (load dữ liệu). Chạy từ thư mục gốc repo.
