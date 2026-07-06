"""CORE-trm — session encoder bằng Transformer + attention pooling."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from CORE.layers import TransformerEncoder
from CORE.models.core_ave import COREave


class TransNet(nn.Module):
    def __init__(
        self,
        max_seq_length: int,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fn = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _attention_mask(item_seq: torch.Tensor) -> torch.Tensor:
        attention_mask = item_seq.ne(0)
        extended = attention_mask.unsqueeze(1).unsqueeze(2)
        extended = torch.tril(extended.expand(-1, -1, item_seq.size(-1), -1))
        return torch.where(extended, 0.0, -1e4)

    def forward(self, item_seq: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        mask = item_seq.gt(0)
        position_ids = torch.arange(item_seq.size(1), device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        input_emb = item_emb + self.position_embedding(position_ids)
        input_emb = self.dropout(self.layer_norm(input_emb))

        trm_output = self.trm_encoder(
            input_emb, self._attention_mask(item_seq), output_all_encoded_layers=True,
        )
        output = trm_output[-1]

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, torch.tensor(-9e15, device=item_seq.device))
        return torch.softmax(alpha, dim=1, dtype=torch.float)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class COREtrm(COREave):
    def __init__(
        self,
        n_items: int,
        embedding_size: int,
        sess_dropout: float,
        item_dropout: float,
        temperature: float,
        max_seq_length: int,
        n_layers: int,
        n_heads: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        initializer_range: float,
    ) -> None:
        super().__init__(n_items, embedding_size, sess_dropout, item_dropout, temperature)
        self.net = TransNet(
            max_seq_length=max_seq_length,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=embedding_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
        )

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        alpha = self.net(item_seq, x)
        seq_output = torch.sum(alpha * x, dim=1)
        return F.normalize(seq_output, dim=-1)
