"""Factory chọn encoder theo config."""

from __future__ import annotations

from common.config import ModelConfig

from .common import ENCODER_TYPES, FUSION_TYPES
from .concat import ConcatEncoder
from .dual_path import DualPathEncoder
from .hgt import HGTEncoder
from .mg_core import MGCoreEncoder
from .retrieval import RetrievalEncoder
from .rgcn import RGCNEncoder
from .soft_cat import SoftCatEncoder
from .transition import TransitionEncoder

_ENCODER_MAP = {
    "rgcn": RGCNEncoder,
    "concat": ConcatEncoder,
    "dual_path": DualPathEncoder,
    "hgt": HGTEncoder,
    "retrieval": RetrievalEncoder,
    "transition": TransitionEncoder,
    "soft_cat": SoftCatEncoder,
    "mg_core": MGCoreEncoder,
}


def build_encoder(
    cfg: ModelConfig,
    n_items: int,
    n_cats: int,
    item2cat: dict[int, int] | None = None,
    cat2items: dict[int, list[int]] | None = None,
):
    enc = cfg.encoder_type
    if enc not in ENCODER_TYPES:
        raise ValueError(f"encoder_type không hợp lệ: {enc!r}. Chọn một trong {sorted(ENCODER_TYPES)}")
    if cfg.fusion_type not in FUSION_TYPES:
        raise ValueError(f"fusion_type không hợp lệ: {cfg.fusion_type!r}")

    kwargs = dict(
        n_items=n_items,
        n_cats=n_cats,
        d=cfg.embedding_dim,
        n_layers=cfg.num_layers,
        use_taxonomy=cfg.use_taxonomy,
        dropout=cfg.dropout,
        fusion_type=cfg.fusion_type,
        n_heads=cfg.n_heads,
        retrieval_topk=cfg.retrieval_topk,
    )
    if enc in ("concat", "transition", "retrieval", "mg_core"):
        if item2cat is None:
            raise ValueError(f"encoder {enc} cần item2cat")
        kwargs["item2cat"] = item2cat
    if enc in ("retrieval", "mg_core"):
        if cat2items is None:
            raise ValueError(f"encoder {enc} cần cat2items")
        kwargs["cat2items"] = cat2items

    for key in (
        "max_seq_length", "trm_layers", "trm_inner_size", "temperature",
        "sess_dropout", "item_dropout", "extra_rerank", "extra_beta",
    ):
        if hasattr(cfg, key):
            kwargs[key] = getattr(cfg, key)

    return _ENCODER_MAP[enc](**kwargs)
