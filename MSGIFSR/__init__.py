"""Tích hợp baseline MSGIFSR (WSDM 2022) vào pipeline demo2."""

from .adapter import demo2_to_msgifsr
from .train import train_model

__all__ = ["demo2_to_msgifsr", "train_model"]
