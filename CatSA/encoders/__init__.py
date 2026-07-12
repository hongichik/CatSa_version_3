"""Factory chọn encoder theo config."""

from __future__ import annotations

from common.config import ModelConfig

from .common import ENCODER_TYPES, FUSION_TYPES
from .catsa_plus import CatSAPlusEncoder
from .catsa_plus_v2 import CatSAPlusV2Encoder

_ENCODER_MAP = {
    "catsa_plus": CatSAPlusEncoder,
    "catsa_plus_v2": CatSAPlusV2Encoder,
}


def build_encoder(
    cfg: ModelConfig,
    n_items: int,
    n_cats: int,
    item2cat: dict[int, int] | None = None,
    cat2items: dict[int, list[int]] | None = None,
    cat_parent: dict[int, int] | None = None,
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
    if enc == "catsa_plus":
        if getattr(cfg, "use_error_aux", False) and item2cat is None:
            raise ValueError("catsa_plus use_error_aux cần item2cat")
        if getattr(cfg, "post_process", False) and (item2cat is None or cat2items is None):
            raise ValueError("catsa_plus post_process cần item2cat và cat2items")
        if item2cat is not None:
            kwargs["item2cat"] = item2cat
        if cat2items is not None:
            kwargs["cat2items"] = cat2items
        if cat_parent is not None:
            kwargs["cat_parent"] = cat_parent
    if enc == "catsa_plus_v2":
        if getattr(cfg, "use_cat_bias", False) and item2cat is None:
            raise ValueError("catsa_plus_v2 use_cat_bias cần item2cat")
        if getattr(cfg, "use_cat_intent", False) and item2cat is None:
            raise ValueError("catsa_plus_v2 use_cat_intent cần item2cat")
        if item2cat is not None:
            kwargs["item2cat"] = item2cat

    for key in (
        "max_seq_length", "trm_layers", "trm_inner_size", "temperature",
        "sess_dropout", "item_dropout", "extra_rerank", "extra_beta",
        "trm_dropout", "use_seq_trm",
        "use_error_aux", "error_aux_alpha",
        "post_process", "post_same_boost", "post_sib_boost", "post_other_penalty",
        "use_module1", "path_fusion", "dual_score_beta", "learn_score_beta",
        "trm_residual_gamma", "use_cat_bias",
        "length_aware_gate", "length_gate_max_len", "use_star_node",
        "use_multi_interest", "n_interests",
        "use_cat_intent", "cat_intent_beta", "cat_intent_layers", "cat_intent_conf_gate",
        "use_repeat_boost", "repeat_boost_init",
    ):
        if hasattr(cfg, key):
            kwargs[key] = getattr(cfg, key)

    return _ENCODER_MAP[enc](**kwargs)
