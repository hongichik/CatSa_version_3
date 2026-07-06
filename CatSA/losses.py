"""Giai đoạn 6 — Session-level InfoNCE loss (symmetric, kiểu SimCLR).

Cặp dương  : (z_s[i], z_s_tilde[i]) — phiên gốc và biến thể của CHÍNH nó.
Cặp âm     : (z_s[i], z_s_tilde[j]) với j != i — biến thể của phiên KHÁC trong batch.
Symmetric  : cross-entropy theo cả hàng và cột rồi lấy trung bình
             (gradient ổn định hơn, đối xứng giữa (s, s~) và (s~, s)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def session_level_infonce(
    z_s: torch.Tensor, z_s_tilde: torch.Tensor, tau: float = 0.5
) -> torch.Tensor:
    """
    z_s        : (B, d) — embedding của phiên gốc
    z_s_tilde  : (B, d) — embedding của phiên augmented tương ứng
    tau        : temperature (nhỏ → phân biệt sắc nét, gradient mạnh hơn)
    Trả về     : scalar loss
    """
    B = z_s.size(0)

    # Chuẩn hoá L2 → dot product = cosine similarity, không phụ thuộc độ dài vector
    z_s = F.normalize(z_s, dim=1)
    z_s_tilde = F.normalize(z_s_tilde, dim=1)

    # sim[i, j] = cos(z_s[i], z_s_tilde[j]); đường chéo là cặp dương
    sim = torch.mm(z_s, z_s_tilde.T) / tau  # (B, B)

    # InfoNCE = cross-entropy với target là vị trí cặp dương (đường chéo)
    targets = torch.arange(B, device=z_s.device)
    loss_row = F.cross_entropy(sim, targets)      # theo hàng
    loss_col = F.cross_entropy(sim.T, targets)    # theo cột (đối xứng)

    return (loss_row + loss_col) / 2
