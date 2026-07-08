"""Giai đoạn 3 — Factory encoder + tương thích ngược CatSAEncoder."""

from __future__ import annotations

from common.config import ModelConfig

from .encoders import build_encoder
from .encoders.rgcn import RGCNEncoder

# Alias tương thích code cũ
CatSAEncoder = RGCNEncoder

__all__ = ["CatSAEncoder", "RGCNEncoder", "build_encoder"]
