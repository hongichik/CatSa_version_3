"""Giai đoạn 3 — Factory encoder (các encoder đời v1 đã loại bỏ)."""

from __future__ import annotations

from common.config import ModelConfig

from .encoders import build_encoder

__all__ = ["build_encoder"]
