# CORE (tích hợp demo2)

Baseline [CORE (SIGIR'22)](https://github.com/RUCAIBox/CORE) — chạy trên dữ liệu đã tiền xử lý trong `data/` (cùng pipeline với CatSA).

## Cách chạy

```bash
# Tiền xử lý (nếu chưa có data/)
python tienxuly/main.py --suite diginetica

# Train CORE
python CORE/main.py --suite diginetica              # trm + ave
python CORE/main.py --suite diginetica --run core_trm.yaml
python CORE/main.py --run core_trm_retailrocket.yaml
```

## Cấu hình

Giống CatSA, trong `config/core/`:

- `config/core/select.yaml` — RetailRocket mặc định
- `config/core/diginetica/select.yaml` — Diginetica
- Mỗi file version gồm: `project`, `data`, `core_model`, `core_training`, `evaluation`

Log: `Log/CORE/<custom_filename>`  
Checkpoint: `checkpoints/CORE/...`

## Khác bản gốc

- Không dùng RecBole — model port sang PyTorch thuần, đọc `train.txt/val.txt/test.txt`
- Dùng chung tiền xử lý `tienxuly/` và `common/` (logging, wandb, config loader)

## Paper

```
@inproceedings{hou2022core,
  title={CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space},
  author={Yupeng Hou and Binbin Hu and Zhiqiang Zhang and Wayne Xin Zhao},
  booktitle={SIGIR},
  year={2022}
}
```
