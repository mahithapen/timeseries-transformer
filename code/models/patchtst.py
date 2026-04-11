from __future__ import annotations

import math
from dataclasses import dataclass

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
    use_instance_norm: bool = True
    task: str = "forecast"
    mask_ratio: float = 0.4

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / (self.stdev + self.eps)
            if self.affine: x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

class PatchTST(nn.Module):
    def __init__(self, config: PatchTSTConfig, in_channels: int):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.revin = RevIN(in_channels) if config.use_instance_norm else None
        
        # Fixed: Correct math for number of patches using unfold
        self.num_patches = (config.seq_len - config.patch_len) // config.stride + 1
        
        self.patch_embed = nn.Linear(config.patch_len, config.d_model)
        self.positional = PositionalEncoding(config.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(config.d_model * self.num_patches, config.pred_len)
        )
        self.reconstruction_head = nn.Linear(config.d_model, config.patch_len)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        x = x.permute(0, 2, 1).reshape(bsz * channels, seq_len)
        return x.unfold(dimension=-1, size=self.config.patch_len, step=self.config.stride)

    def _encode_patches(self, patches: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.patch_embed(patches)
        if mask is not None:
            mask_token = self.mask_token.expand(z.size(0), z.size(1), -1)
            z = torch.where(mask.unsqueeze(-1), mask_token, z)
        z = self.positional(z)
        return self.encoder(z)

    def _random_patch_mask(self, batch_size: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        mask = torch.rand(batch_size, self.num_patches, device=device) < mask_ratio
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            random_index = torch.randint(self.num_patches, (int(empty_rows.sum().item()),), device=device)
            mask[empty_rows, random_index] = True
        return mask

    def forward_pretrain(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.revin:
            x = self.revin(x, 'norm')

        patches = self._patchify(x)
        mask = self._random_patch_mask(
            batch_size=patches.size(0),
            mask_ratio=self.config.mask_ratio if mask_ratio is None else mask_ratio,
            device=patches.device,
        )
        encoded = self._encode_patches(patches, mask=mask)
        reconstructed = self.reconstruction_head(encoded)

        bsz = x.size(0)
        channels = x.size(2)
        reconstructed = reconstructed.reshape(bsz, channels, self.num_patches, self.config.patch_len)
        target = patches.reshape(bsz, channels, self.num_patches, self.config.patch_len)
        mask = mask.reshape(bsz, channels, self.num_patches)
        return reconstructed, target, mask

    def forward(self, x: torch.Tensor):
        if self.revin:
            x = self.revin(x, 'norm')

        bsz, _, channels = x.shape
        patches = self._patchify(x)
        encoded = self._encode_patches(patches)
        forecast = self.forecast_head(encoded)
        forecast = forecast.reshape(bsz, channels, -1).permute(0, 2, 1)

        if self.revin:
            forecast = self.revin(forecast, 'denorm')
        return forecast
