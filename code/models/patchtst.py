from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PatchTSTConfig:
    seq_len: int
    pred_len: int
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    channel_independence: bool = True
    use_instance_norm: bool = True
    task: str = "forecast"  # "forecast" or "reconstruct"


class InstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, C, L]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        return x_norm, mean, std


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, C, N, patch_len]
        return self.proj(patches)  # [B, C, N, d_model]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        return x + self.pe[:, : x.size(1)]


class PatchTST(nn.Module):
    def __init__(self, config: PatchTSTConfig, in_channels: int):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.instance_norm = InstanceNorm() if config.use_instance_norm else None

        self.patch_len = config.patch_len
        self.stride = config.stride

        self.patch_embed = PatchEmbedding(config.patch_len, config.d_model)
        self.positional = PositionalEncoding(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        if config.task == "forecast":
            self.head = nn.Linear(config.d_model, config.pred_len)
        else:
            self.head = nn.Linear(config.d_model, config.patch_len)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, N, patch_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        x = x.transpose(1, 2)  # [B, C, L]
        if self.instance_norm is not None:
            x, mean, std = self.instance_norm(x)
        else:
            mean = std = None

        patches = self._patchify(x)  # [B, C, N, patch_len]
        b, c, n, p = patches.shape
        tokens = self.patch_embed(patches)  # [B, C, N, D]

        if self.config.channel_independence:
            tokens = tokens.reshape(b * c, n, -1)
        else:
            tokens = tokens.reshape(b, c * n, -1)

        tokens = self.positional(tokens)
        enc = self.encoder(tokens)

        if self.config.task == "forecast":
            out = self.head(enc)  # [B*C, N, pred_len] or [B, C*N, pred_len]
            if self.config.channel_independence:
                out = out.view(b, c, n, -1).mean(dim=2)  # [B, C, pred_len]
            else:
                out = out.view(b, c, n, -1).mean(dim=2)
        else:
            out = self.head(enc)  # [B*C, N, patch_len]
            if self.config.channel_independence:
                out = out.view(b, c, n, -1)
            else:
                out = out.view(b, c, n, -1)

        if self.instance_norm is not None and mean is not None and std is not None:
            if self.config.task == "forecast":
                out = out * std + mean
            else:
                out = out

        if self.config.task == "forecast":
            return out.transpose(1, 2)  # [B, pred_len, C]
        return out  # [B, C, N, patch_len]
