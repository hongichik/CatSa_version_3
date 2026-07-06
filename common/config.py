"""Đọc cấu hình YAML từ cây thư mục config/ thành object Python có kiểu đúng.

Cấu hình được tách thành các thư mục con theo từng phần của hệ thống:
    config/common/logging.yaml       — logging (dùng chung)
    config/common/wandb.yaml         — Weights & Biases (tracking online)
    config/tienxuly/dataset.yaml     — nguồn dữ liệu (tienxuly/download.py)
    config/tienxuly/preprocess.yaml  — tham số tiền xử lý (tienxuly/preprocess.py)
    config/catsa/select.yaml         — CHỌN version cấu hình CatSA sẽ chạy
    config/catsa/catsa_v1.yaml       — toàn bộ cấu hình CatSA version 1
                                       (project, data, model, augment, training, evaluation)

Loader QUÉT ĐỆ QUY mọi file *.yaml / *.yml trong cây config/ rồi gộp các
section cấp cao nhất (logging, dataset, preprocess, model, ...) lại — vì vậy
về sau thêm cấu hình mới chỉ cần thả file YAML vào thư mục con phù hợp,
không phải sửa code load. Mỗi section chỉ được khai báo ở đúng MỘT file
(khai báo trùng sẽ báo lỗi để tránh cấu hình ghi đè ngầm).

CƠ CHẾ CHỌN VERSION: thư mục con nào có file select.yaml thì CHỈ các file
được khai báo trong `run:` mới được nạp (một file hoặc danh sách). CatSA/main.py
chạy lần lượt toàn bộ danh sách; tienxuly chỉ cần config chung nên dùng phần tử
đầu tiên trong list khi load_config() không chỉ định catsa_run.

Mọi tham số đều nằm trong YAML — code chỉ đọc, không hard-code.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LoggingConfig:
    dir: str                # thư mục GỐC; log thực tế ghi vào <dir>/<project.name>/
    level: str
    console: bool


@dataclass
class ProjectConfig:
    name: str               # tên dự án — log ghi vào <logging.dir>/<name>/
    filename_mode: str      # "auto" | "custom" (không khai báo => auto)
    custom_filename: str    # chỉ dùng khi filename_mode = "custom"


@dataclass
class WandbConfig:
    enabled: bool
    api_key: str
    project: str
    entity: str
    run_name: str
    mode: str               # "online" | "offline"


@dataclass
class DatasetConfig:
    source: str             # "kagglehub" | "local"
    kagglehub_handle: str
    local_path: str
    name: str


@dataclass
class PreprocessConfig:
    output_dir: str
    train_file: str         # file phiên train (mỗi dòng: "item_1 item_2 ...")
    val_file: str           # file phiên validation
    test_file: str          # file phiên test
    lookup_file: str        # pickle lookup tables + metadata
    event_types: list[str]
    # true: cắt phiên khi sang ngày lịch khác; false: dùng session_gap_seconds.
    session_same_day: bool
    session_gap_seconds: int
    # true: bỏ click liên tiếp trùng item trong cùng phiên.
    dedup_consecutive: bool
    min_item_support: int
    min_session_length: int
    # filter: chỉ GIỮ phiên có độ dài trong [min_session_length, max_session_length].
    max_session_length: int
    # keep: giữ nguyên phiên (không cắt); filter: lọc phiên theo độ dài.
    session_length_mode: str
    # Giới hạn prefix khi train/eval (sliding window); 0 = không giới hạn.
    # test_all: 50 — chỉ cắt prefix, không cắt phiên.
    max_prefix_length: int
    # true: chỉ giữ item có category trong item_properties.
    # false: giữ cả item không có category (gán category UNK trong lookup).
    require_item_category: bool
    val_ratio: float
    test_ratio: float
    max_sessions: int


@dataclass
class DataConfig:
    """Đường dẫn dữ liệu ĐÃ TIỀN XỬ LÝ — chỉ dùng khi train CatSA (đọc file)."""
    data_dir: str
    train_file: str
    val_file: str
    test_file: str
    lookup_file: str


@dataclass
class ModelConfig:
    embedding_dim: int
    num_layers: int
    use_taxonomy: bool
    dropout: float


@dataclass
class AugmentConfig:
    strategies: list[str]
    strategy_weights: list[float]
    eta_aug: float
    eta_crop: float
    k_min: int


@dataclass
class TrainingConfig:
    use_cl: bool
    lambda_cl: float
    tau: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    grad_clip: float
    seed: int
    device: str
    num_workers: int
    checkpoint_dir: str     # thư mục GỐC (dùng khi save_dir trống)
    save_dir: str           # "" = mặc định theo version; khác "" = lưu đúng chỗ này


@dataclass
class EvaluationConfig:
    top_k: list[int]
    primary_metric: str


@dataclass
class CoreModelConfig:
    type: str               # ave | trm
    embedding_size: int
    inner_size: int
    n_layers: int
    n_heads: int
    hidden_dropout_prob: float
    attn_dropout_prob: float
    hidden_act: str
    layer_norm_eps: float
    initializer_range: float
    sess_dropout: float
    item_dropout: float
    temperature: float
    max_seq_length: int


@dataclass
class CoreTrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    grad_clip: float
    seed: int
    device: str
    num_workers: int
    checkpoint_dir: str
    save_dir: str


@dataclass
class CoreConfig:
    logging: LoggingConfig
    project: ProjectConfig
    wandb: WandbConfig
    data: DataConfig
    core_model: CoreModelConfig
    core_training: CoreTrainingConfig
    evaluation: EvaluationConfig
    config_dir: str = field(default="config")
    version: str = field(default="default")


@dataclass
class Config:
    logging: LoggingConfig
    project: ProjectConfig
    wandb: WandbConfig
    dataset: DatasetConfig
    preprocess: PreprocessConfig
    data: DataConfig
    model: ModelConfig
    augment: AugmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    config_dir: str = field(default="config")
    # Version cấu hình đang chạy = tên file (không đuôi) mà select.yaml chọn,
    # ví dụ "catsa_v1". Dùng để phân cấp thư mục checkpoint/kết quả theo version.
    version: str = field(default="default")


# Tên file "chọn version" trong một thư mục con cấu hình
SELECT_FILE = "select.yaml"


def _normalize_run_filename(name: str) -> str:
    """Chuẩn hóa tên file version: catsa_v1 -> catsa_v1.yaml."""
    name = str(name).strip()
    if not name:
        raise ValueError("Tên file version không được để trống")
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    return name


def _parse_select_runs(sel: dict) -> list[str]:
    """Đọc khóa run từ select.yaml — hỗ trợ một file hoặc danh sách."""
    run = sel.get("run")
    if run is None:
        return []
    if isinstance(run, str):
        name = _normalize_run_filename(run)
        return [name]
    if isinstance(run, list):
        runs = [_normalize_run_filename(r) for r in run]
        if not runs:
            raise ValueError("select.yaml: run: [] rỗng")
        return runs
    raise ValueError(
        f"select.yaml: run phải là string hoặc list, nhận {type(run).__name__}"
    )


def list_catsa_runs(
    config_dir: str | Path = "config",
    suite: str | None = None,
) -> list[str]:
    """Trả về danh sách file version CatSA trong select.yaml."""
    config_dir = Path(config_dir)
    if suite:
        select_path = config_dir / "catsa" / suite / SELECT_FILE
    else:
        select_path = config_dir / "catsa" / SELECT_FILE
    if not select_path.exists():
        return []
    with open(select_path, encoding="utf-8") as f:
        sel = yaml.safe_load(f) or {}
    return _parse_select_runs(sel)


def list_core_runs(
    config_dir: str | Path = "config",
    suite: str | None = None,
) -> list[str]:
    """Trả về danh sách file version CORE trong select.yaml."""
    config_dir = Path(config_dir)
    if suite:
        select_path = config_dir / "core" / suite / SELECT_FILE
    else:
        select_path = config_dir / "core" / SELECT_FILE
    if not select_path.exists():
        return []
    with open(select_path, encoding="utf-8") as f:
        sel = yaml.safe_load(f) or {}
    return _parse_select_runs(sel)


def list_preprocess_runs(
    config_dir: str | Path = "config",
    suite: str | None = None,
) -> list[str]:
    """Trả về danh sách file version tiền xử lý trong select.yaml.

    suite=None  → config/tienxuly/select.yaml (RetailRocket, mặc định)
    suite='diginetica' → config/tienxuly/diginetica/select.yaml
    """
    config_dir = Path(config_dir)
    if suite:
        select_path = config_dir / "tienxuly" / suite / SELECT_FILE
    else:
        select_path = config_dir / "tienxuly" / SELECT_FILE
    if not select_path.exists():
        return []
    with open(select_path, encoding="utf-8") as f:
        sel = yaml.safe_load(f) or {}
    return _parse_select_runs(sel)


# File YAML dùng chung trong thư mục có select.yaml (KHÔNG phải version)
SHARED_CONFIG_FILES = frozenset({"dataset.yaml"})

# Section cho phép file version ghi đè file dùng chung (vd dataset.yaml)
_OVERRIDE_SECTIONS = frozenset({"dataset", "preprocess"})


def _core_root(config_dir: Path) -> Path:
    return config_dir / "core"


def _is_core_select_dir(d: Path, config_dir: Path) -> bool:
    root = _core_root(config_dir)
    return d == root or d.parent == root


def find_core_yaml(
    config_dir: Path,
    run_name: str,
    suite: str | None = None,
) -> Path:
    """Tìm file CORE trong config/core/ (suite=None: root; suite: subfolder)."""
    root = _core_root(config_dir)
    target = _normalize_run_filename(run_name)
    if suite:
        search = root / suite
        chosen = search / target
        if not chosen.exists():
            raise FileNotFoundError(
                f"Không tìm thấy CORE config '{target}' trong {search}"
            )
        return chosen
    direct = root / target
    if direct.exists():
        return direct
    hits = sorted(p for p in root.rglob(target) if p.is_file())
    if not hits:
        raise FileNotFoundError(
            f"Không tìm thấy CORE config '{target}' trong {root}"
        )
    if len(hits) > 1:
        raise ValueError(
            f"CORE config '{target}' trùng tên: {hits} — dùng --suite để chọn"
        )
    return hits[0]


def _catsa_root(config_dir: Path) -> Path:
    return config_dir / "catsa"


def _is_catsa_select_dir(d: Path, config_dir: Path) -> bool:
    root = _catsa_root(config_dir)
    return d == root or d.parent == root


def find_catsa_yaml(
    config_dir: Path,
    run_name: str,
    suite: str | None = None,
) -> Path:
    """Tìm file CatSA trong config/catsa/ (suite=None: root; suite: subfolder)."""
    root = _catsa_root(config_dir)
    target = _normalize_run_filename(run_name)
    if suite:
        search = root / suite
        chosen = search / target
        if not chosen.exists():
            raise FileNotFoundError(
                f"Không tìm thấy CatSA config '{target}' trong {search}"
            )
        return chosen
    direct = root / target
    if direct.exists():
        return direct
    hits = sorted(p for p in root.rglob(target) if p.is_file())
    if not hits:
        raise FileNotFoundError(
            f"Không tìm thấy CatSA config '{target}' trong {root}"
        )
    if len(hits) > 1:
        raise ValueError(
            f"CatSA config '{target}' trùng tên: {hits} — dùng --suite để chọn"
        )
    return hits[0]


def _tienxuly_root(config_dir: Path) -> Path:
    return config_dir / "tienxuly"


def _is_tienxuly_select_dir(d: Path, config_dir: Path) -> bool:
    root = _tienxuly_root(config_dir)
    return d == root or d.parent == root


def find_preprocess_yaml(config_dir: Path, run_name: str) -> Path:
    """Tìm file preprocess trong config/tienxuly/ (kể cả subfolder diginetica/)."""
    root = _tienxuly_root(config_dir)
    target = _normalize_run_filename(run_name)
    hits = sorted(p for p in root.rglob(target) if p.is_file())
    if not hits:
        raise FileNotFoundError(
            f"Không tìm thấy preprocess config '{target}' trong {root}"
        )
    if len(hits) > 1:
        raise ValueError(f"Preprocess config '{target}' trùng tên: {hits}")
    return hits[0]


def _collect_config_files(
    config_dir: Path,
    catsa_run: str | None = None,
    preprocess_run: str | None = None,
    catsa_suite: str | None = None,
    core_run: str | None = None,
    core_suite: str | None = None,
    core_only: bool = False,
) -> list[Path]:
    """Liệt kê các file YAML sẽ được nạp, tôn trọng cơ chế select.yaml.

    Thư mục có select.yaml:
    - Nạp thêm file YAML dùng chung (vd dataset.yaml) không nằm trong run.
    - catsa_run / preprocess_run: chỉ nạp file version tương ứng trong thư mục đó.
    - Không chỉ định => nạp phần tử đầu trong run (load_config đơn lẻ).
    """
    files: list[Path] = []
    dirs = [config_dir] + sorted(p for p in config_dir.rglob("*") if p.is_dir())
    for d in dirs:
        yamls = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix in (".yaml", ".yml"))
        select_path = d / SELECT_FILE
        if select_path in yamls:
            with open(select_path, encoding="utf-8") as f:
                sel = yaml.safe_load(f) or {}
            runs = _parse_select_runs(sel)
            if not runs:
                raise ValueError(f"{select_path} thiếu khóa 'run:' (file hoặc danh sách)")
            run_names = {_normalize_run_filename(r) for r in runs}

            if _is_core_select_dir(d, config_dir):
                if not core_only:
                    continue
                if core_run is not None:
                    run_path = find_core_yaml(config_dir, core_run, core_suite)
                    if d != run_path.parent:
                        continue
                    chosen_run = core_run
                elif core_suite:
                    if d != _core_root(config_dir) / core_suite:
                        continue
                    chosen_run = None
                elif d != _core_root(config_dir):
                    continue
                else:
                    chosen_run = None
            elif _is_catsa_select_dir(d, config_dir):
                if core_only:
                    continue
                if catsa_run is not None:
                    run_path = find_catsa_yaml(config_dir, catsa_run, catsa_suite)
                    if d != run_path.parent:
                        continue
                    chosen_run = catsa_run
                elif catsa_suite:
                    if d != _catsa_root(config_dir) / catsa_suite:
                        continue
                    chosen_run = None
                elif d != _catsa_root(config_dir):
                    continue
                else:
                    chosen_run = None
            elif _is_tienxuly_select_dir(d, config_dir):
                if core_only:
                    continue
                if preprocess_run is not None:
                    run_path = find_preprocess_yaml(config_dir, preprocess_run)
                    if d != run_path.parent:
                        continue
                    chosen_run = preprocess_run
                elif d != _tienxuly_root(config_dir):
                    continue
                else:
                    chosen_run = None
            else:
                continue

            shared = [p for p in yamls if p.name in SHARED_CONFIG_FILES]
            files.extend(shared)

            if chosen_run is not None:
                if _is_core_select_dir(d, config_dir):
                    chosen = find_core_yaml(config_dir, chosen_run, core_suite)
                    if d != chosen.parent:
                        continue
                elif _is_catsa_select_dir(d, config_dir):
                    chosen = find_catsa_yaml(config_dir, chosen_run, catsa_suite)
                    if d != chosen.parent:
                        continue
                else:
                    target = _normalize_run_filename(chosen_run)
                    if target not in run_names:
                        raise FileNotFoundError(
                            f"{select_path}: '{target}' không có trong run: {sorted(run_names)}"
                        )
                    chosen = find_preprocess_yaml(config_dir, chosen_run)
            else:
                chosen = d / runs[0]
            if not chosen.exists():
                raise FileNotFoundError(
                    f"{select_path} chọn '{chosen.name}' nhưng file không tồn tại trong {d}"
                )
            files.append(chosen)
        else:
            files.extend(yamls)
    return files


def _merge_yaml_tree(
    config_dir: Path,
    catsa_run: str | None = None,
    preprocess_run: str | None = None,
    catsa_suite: str | None = None,
    core_run: str | None = None,
    core_suite: str | None = None,
    core_only: bool = False,
) -> tuple[dict, dict[str, Path]]:
    """Quét cây config/, gộp các section cấp cao nhất từ các file được nạp.

    Trả về (merged, origin) với origin[section] = file khai báo section đó
    (dùng cho thông báo lỗi khi khai báo trùng).
    """
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục cấu hình: {config_dir}")

    files = _collect_config_files(
        config_dir, catsa_run=catsa_run, preprocess_run=preprocess_run,
        catsa_suite=catsa_suite, core_run=core_run, core_suite=core_suite,
        core_only=core_only,
    )
    if not files:
        raise FileNotFoundError(f"Không có file YAML nào trong {config_dir}")

    merged: dict = {}
    origin: dict[str, Path] = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"File cấu hình không hợp lệ (phải là mapping YAML): {path}")
        for section, content in data.items():
            if section in merged and section not in _OVERRIDE_SECTIONS:
                raise ValueError(
                    f"Section '{section}' bị khai báo trùng trong {path} "
                    f"(đã có ở {origin[section]}) — mỗi section chỉ được ở một file"
                )
            merged[section] = content
            origin[section] = path
    return merged, origin


def _build_project(proj: dict) -> ProjectConfig:
    """Dựng ProjectConfig với quy tắc đặt tên file log của dự án:

    - Không khai báo gì            → filename_mode = "auto"
    - Chỉ khai báo custom_filename → hiểu là có đổi tên → "custom"
    - Khai báo filename_mode tường minh → dùng đúng giá trị đó
    """
    name = str(proj.get("name") or "").strip()
    if not name:
        raise ValueError("project.name không được để trống (config .../project.yaml)")

    custom_filename = str(proj.get("custom_filename") or "").strip()
    mode = str(proj.get("filename_mode") or "").strip().lower()
    if not mode:
        mode = "custom" if custom_filename else "auto"
    if mode == "custom" and not custom_filename:
        raise ValueError(
            "project.filename_mode = custom nhưng thiếu project.custom_filename"
        )
    return ProjectConfig(name=name, filename_mode=mode, custom_filename=custom_filename)


def _build_data_config(merged: dict) -> DataConfig:
    """Dựng DataConfig: ưu tiên section `data` trong file version CatSA;
    nếu thiếu thì lấy từ preprocess (tương thích ngược)."""
    if "data" in merged:
        d = merged["data"]
        data_dir = str(d.get("data_dir") or "").strip()
        if not data_dir:
            raise ValueError("data.data_dir không được để trống (config/catsa/<version>.yaml)")
        return DataConfig(
            data_dir=data_dir,
            train_file=str(d.get("train_file", "train.txt")),
            val_file=str(d.get("val_file", "val.txt")),
            test_file=str(d.get("test_file", "test.txt")),
            lookup_file=str(d.get("lookup_file", "lookup_tables.pkl")),
        )
    pre = merged["preprocess"]
    return DataConfig(
        data_dir=str(pre.get("output_dir", "data/processed")),
        train_file=str(pre.get("train_file", "train.txt")),
        val_file=str(pre.get("val_file", "val.txt")),
        test_file=str(pre.get("test_file", "test.txt")),
        lookup_file=str(pre.get("lookup_file", "lookup_tables.pkl")),
    )


def _build_preprocess_config(pre: dict) -> PreprocessConfig:
    mode = str(pre.get("session_length_mode", "keep")).strip().lower()
    if mode not in ("keep", "filter"):
        raise ValueError(
            f"preprocess.session_length_mode phải là 'keep' hoặc 'filter', nhận: {mode}"
        )
    return PreprocessConfig(
        output_dir=str(pre.get("output_dir", "data/processed")),
        train_file=str(pre.get("train_file", "train.txt")),
        val_file=str(pre.get("val_file", "val.txt")),
        test_file=str(pre.get("test_file", "test.txt")),
        lookup_file=str(pre.get("lookup_file", "lookup_tables.pkl")),
        event_types=[str(e) for e in pre.get("event_types", ["view"])],
        session_same_day=bool(pre.get("session_same_day", False)),
        session_gap_seconds=int(pre.get("session_gap_seconds", 1800)),
        dedup_consecutive=bool(pre.get("dedup_consecutive", True)),
        min_item_support=int(pre.get("min_item_support", 5)),
        min_session_length=int(pre.get("min_session_length", 2)),
        max_session_length=int(pre.get("max_session_length", 50)),
        session_length_mode=mode,
        max_prefix_length=int(pre.get("max_prefix_length", 50)),
        require_item_category=bool(pre.get("require_item_category", True)),
        val_ratio=float(pre.get("val_ratio", 0.1)),
        test_ratio=float(pre.get("test_ratio", 0.1)),
        max_sessions=int(pre.get("max_sessions", 0)),
    )


def load_config(
    config_dir: str | Path = "config",
    catsa_run: str | None = None,
    preprocess_run: str | None = None,
    catsa_suite: str | None = None,
) -> Config:
    """Đọc YAML trong cây config/ và trả về Config đầy đủ.

    catsa_run: file version CatSA (vd catsa_v1.yaml).
    preprocess_run: file version tiền xử lý (vd retailrocket_2_5.yaml).
    catsa_suite: subfolder CatSA (vd diginetica) — dùng với --suite diginetica.
    Bỏ trống => phần tử đầu trong select.yaml của từng thư mục tương ứng.
    """
    config_dir = Path(config_dir)
    merged, origin = _merge_yaml_tree(
        config_dir, catsa_run=catsa_run, preprocess_run=preprocess_run,
        catsa_suite=catsa_suite,
    )

    required = ["logging", "project", "dataset", "preprocess", "model", "augment", "training", "evaluation"]
    missing = [s for s in required if s not in merged]
    if missing:
        raise KeyError(
            f"Thiếu section {missing} trong cây cấu hình {config_dir} "
            f"(kiểm tra các file YAML trong thư mục con)"
        )

    log_raw = merged["logging"]
    proj = merged["project"]
    # wandb là section TÙY CHỌN — thiếu file thì mặc định tắt
    wb = merged.get("wandb", {})
    ds = merged["dataset"]
    pre = merged["preprocess"]
    model = merged["model"]
    aug = merged["augment"]
    train = merged["training"]
    ev = merged["evaluation"]

    return Config(
        logging=LoggingConfig(
            dir=str(log_raw.get("dir", "Log")),
            level=str(log_raw.get("level", "INFO")).upper(),
            console=bool(log_raw.get("console", True)),
        ),
        project=_build_project(proj),
        wandb=WandbConfig(
            enabled=bool(wb.get("enabled", False)),
            api_key=str(wb.get("api_key") or ""),
            project=str(wb.get("project") or ""),
            entity=str(wb.get("entity") or ""),
            run_name=str(wb.get("run_name") or ""),
            mode=str(wb.get("mode", "online")).lower(),
        ),
        dataset=DatasetConfig(
            source=str(ds.get("source", "kagglehub")),
            kagglehub_handle=str(ds.get("kagglehub_handle", "retailrocket/ecommerce-dataset")),
            local_path=str(ds.get("local_path", "data/raw")),
            name=str(ds.get("name", "retailrocket")),
        ),
        preprocess=_build_preprocess_config(pre),
        data=_build_data_config(merged),
        model=ModelConfig(
            embedding_dim=int(model.get("embedding_dim", 100)),
            num_layers=int(model.get("num_layers", 2)),
            use_taxonomy=bool(model.get("use_taxonomy", True)),
            dropout=float(model.get("dropout", 0.1)),
        ),
        augment=AugmentConfig(
            strategies=[str(s) for s in aug.get("strategies", ["same", "sibling", "hybrid"])],
            strategy_weights=[float(w) for w in aug.get("strategy_weights", [])],
            eta_aug=float(aug.get("eta_aug", 0.3)),
            eta_crop=float(aug.get("eta_crop", 0.75)),
            k_min=int(aug.get("k_min", 5)),
        ),
        training=TrainingConfig(
            use_cl=bool(train.get("use_cl", True)),
            lambda_cl=float(train.get("lambda_cl", 0.1)),
            tau=float(train.get("tau", 0.5)),
            batch_size=int(train.get("batch_size", 100)),
            learning_rate=float(train.get("learning_rate", 0.001)),
            weight_decay=float(train.get("weight_decay", 0.00001)),
            max_epochs=int(train.get("max_epochs", 30)),
            patience=int(train.get("patience", 5)),
            grad_clip=float(train.get("grad_clip", 5.0)),
            seed=int(train.get("seed", 42)),
            device=str(train.get("device", "auto")),
            num_workers=int(train.get("num_workers", 0)),
            checkpoint_dir=str(train.get("checkpoint_dir", "checkpoints")),
            save_dir=str(train.get("save_dir") or ""),
        ),
        evaluation=EvaluationConfig(
            top_k=[int(k) for k in ev.get("top_k", [10, 20])],
            primary_metric=str(ev.get("primary_metric", "hr@20")).lower(),
        ),
        config_dir=str(config_dir),
        # Section "project" nằm trong file version được chọn (vd catsa_v1.yaml)
        # → tên file đó chính là version cấu hình đang chạy
        version=origin["project"].stem,
    )


def _build_core_model_config(model: dict) -> CoreModelConfig:
    model_type = str(model.get("type", "trm")).strip().lower()
    if model_type not in ("ave", "trm"):
        raise ValueError(f"core_model.type phải là 'ave' hoặc 'trm', nhận: {model_type}")
    return CoreModelConfig(
        type=model_type,
        embedding_size=int(model.get("embedding_size", 100)),
        inner_size=int(model.get("inner_size", 256)),
        n_layers=int(model.get("n_layers", 2)),
        n_heads=int(model.get("n_heads", 2)),
        hidden_dropout_prob=float(model.get("hidden_dropout_prob", 0.5)),
        attn_dropout_prob=float(model.get("attn_dropout_prob", 0.5)),
        hidden_act=str(model.get("hidden_act", "gelu")),
        layer_norm_eps=float(model.get("layer_norm_eps", 1e-12)),
        initializer_range=float(model.get("initializer_range", 0.02)),
        sess_dropout=float(model.get("sess_dropout", 0.2)),
        item_dropout=float(model.get("item_dropout", 0.2)),
        temperature=float(model.get("temperature", 0.07)),
        max_seq_length=int(model.get("max_seq_length", 50)),
    )


def _build_core_training_config(train: dict) -> CoreTrainingConfig:
    return CoreTrainingConfig(
        batch_size=int(train.get("batch_size", 256)),
        learning_rate=float(train.get("learning_rate", 0.001)),
        weight_decay=float(train.get("weight_decay", 0.00001)),
        max_epochs=int(train.get("max_epochs", 30)),
        patience=int(train.get("patience", 5)),
        grad_clip=float(train.get("grad_clip", 5.0)),
        seed=int(train.get("seed", 42)),
        device=str(train.get("device", "auto")),
        num_workers=int(train.get("num_workers", 0)),
        checkpoint_dir=str(train.get("checkpoint_dir", "checkpoints")),
        save_dir=str(train.get("save_dir") or ""),
    )


def load_core_config(
    config_dir: str | Path = "config",
    core_run: str | None = None,
    core_suite: str | None = None,
) -> CoreConfig:
    """Đọc cấu hình CORE từ config/core/ (không nạp catsa/tienxuly)."""
    config_dir = Path(config_dir)
    merged, origin = _merge_yaml_tree(
        config_dir, core_run=core_run, core_suite=core_suite, core_only=True,
    )

    required = ["logging", "project", "data", "core_model", "core_training", "evaluation"]
    missing = [s for s in required if s not in merged]
    if missing:
        raise KeyError(
            f"Thiếu section {missing} trong cấu hình CORE {config_dir} "
            f"(kiểm tra config/core/<version>.yaml)"
        )

    log_raw = merged["logging"]
    proj = merged["project"]
    wb = merged.get("wandb", {})
    ev = merged["evaluation"]

    return CoreConfig(
        logging=LoggingConfig(
            dir=str(log_raw.get("dir", "Log")),
            level=str(log_raw.get("level", "INFO")).upper(),
            console=bool(log_raw.get("console", True)),
        ),
        project=_build_project(proj),
        wandb=WandbConfig(
            enabled=bool(wb.get("enabled", False)),
            api_key=str(wb.get("api_key") or ""),
            project=str(wb.get("project") or ""),
            entity=str(wb.get("entity") or ""),
            run_name=str(wb.get("run_name") or ""),
            mode=str(wb.get("mode", "online")).lower(),
        ),
        data=_build_data_config(merged),
        core_model=_build_core_model_config(merged["core_model"]),
        core_training=_build_core_training_config(merged["core_training"]),
        evaluation=EvaluationConfig(
            top_k=[int(k) for k in ev.get("top_k", [10, 20])],
            primary_metric=str(ev.get("primary_metric", "mrr@20")).lower(),
        ),
        config_dir=str(config_dir),
        version=origin["project"].stem,
    )


def dump_config(cfg: Config) -> str:
    """Xuất toàn bộ cấu hình thành chuỗi YAML dễ đọc — dùng ghi vào đầu file log
    mỗi lần chạy, để xem lại log là biết run đó dùng thông số gì.

    api_key của wandb được che (thông tin bí mật, không đưa vào log).
    """
    cfg_dict = dataclasses.asdict(cfg)
    if cfg_dict.get("wandb", {}).get("api_key"):
        cfg_dict["wandb"]["api_key"] = "***da-che***"

    body = yaml.safe_dump(cfg_dict, allow_unicode=True, sort_keys=False)
    sep = "=" * 68
    return (
        f"\n{sep}\n"
        f"CẤU HÌNH LÚC CHẠY (config_dir: {cfg.config_dir} | version: {cfg.version})\n"
        f"{sep}\n"
        f"{body}"
        f"{sep}"
    )


def dump_core_config(cfg: CoreConfig) -> str:
    """Xuất cấu hình CORE ra chuỗi YAML (ghi vào log)."""
    cfg_dict = dataclasses.asdict(cfg)
    if cfg_dict.get("wandb", {}).get("api_key"):
        cfg_dict["wandb"]["api_key"] = "***da-che***"
    body = yaml.safe_dump(cfg_dict, allow_unicode=True, sort_keys=False)
    sep = "=" * 68
    return (
        f"\n{sep}\n"
        f"CẤU HÌNH CORE (config_dir: {cfg.config_dir} | version: {cfg.version})\n"
        f"{sep}\n"
        f"{body}"
        f"{sep}"
    )
