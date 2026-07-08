# RetailRocket — layout config & log

## Config (`config/*/retailrocket/`)

```
config/catsa/retailrocket/
  select.yaml
  baseline/       # v1–v4, v2_mg_core
  sweep/          # eta / dim / batch
  encoder/        # enc_concat, dual_path, hgt, ...
  ablation_len/   # 2_5, 4_7, category, ...
  research/

config/core/retailrocket/main/
config/link/retailrocket/{main,toy}/
config/msgifsr/retailrocket/{main,toy}/
```

Chạy:

```bash
python CatSA/main.py --run baseline/catsa_v1.yaml
python CatSA/main.py --run catsa_v1.yaml          # basename cũng được
python LINK/main.py --run main/link_retailrocket.yaml
python MSGIFSR/main.py --run main/msgifsr_retailrocket.yaml
```

## Log (`Log/retailrocket/`)

Song song với config — `project.custom_filename` dùng đường dẫn tương đối:

```
Log/retailrocket/
  CatSA/{baseline,sweep,encoder,ablation_len}/
  CORE/main/
  LINK/{main,toy}/
  MSGIFSR/{main,toy}/
```
