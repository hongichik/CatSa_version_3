"""Tích hợp baseline LINK (SIGIR 2025) vào pipeline demo2.

Giữ nguyên code gốc trong LINK_repo/ (dựa trên RecBole); module này chỉ bọc lại
theo cấu trúc demo2: đọc config/link/<suite>/, chuyển dữ liệu data/<dataset>/*.txt
sang định dạng RecBole, chạy pipeline teacher→LINK và ghi log về Log/<dataset>/.
"""

from .adapter import demo2_to_recbole

__all__ = ["demo2_to_recbole"]
