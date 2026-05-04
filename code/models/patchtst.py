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
    hierarchical_patching: bool = False
    hierarchical_levels: int = 2
    hierarchical_merge_factor: int = 2
    # Mutually exclusive with hierarchical_patching: pyramid uses window attention (+ alternating shift).
    swin_like_patching: bool = False
    swin_window_size: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    attn_dropout: float = 0.0
    fc_dropout: float = 0.1
    head_dropout: float = 0.0
    use_instance_norm: bool = True
    revin_affine: bool = False
    task: str = "forecast"
    mask_ratio: float = 0.4
    padding_patch: str | None = "end"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = True):
        super().__init__()
        if learnable:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.uniform_(self.pe, -0.02, 0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BatchNorm1dTokens(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = BatchNorm1dTokens(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = BatchNorm1dTokens(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm_attn(x + self.dropout_attn(attn_out))
        ff_out = self.ff(x)
        x = self.norm_ffn(x + self.dropout_ffn(ff_out))
        return x


class PatchTSTEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            PatchTSTEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(n_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class WindowAttention1D(nn.Module):
    """Self-attention independently within each 1D patch-token window (batch_first)."""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, window_size: int, shift_size: int) -> torch.Tensor:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        batch_size, seq_len, _ = x.shape
        if seq_len % window_size != 0:
            raise RuntimeError(f"seq_len {seq_len} not divisible by window_size {window_size}")
        if shift_size:
            x = torch.roll(x, shifts=-shift_size, dims=1)
        num_windows = seq_len // window_size
        xw = x.reshape(batch_size, num_windows, window_size, x.size(-1)).reshape(
            batch_size * num_windows, window_size, x.size(-1)
        )
        out, _ = self.attn(xw, xw, xw, need_weights=False)
        x = out.reshape(batch_size, num_windows * window_size, x.size(-1))
        if shift_size:
            x = torch.roll(x, shifts=shift_size, dims=1)
        return x


class SwinEncoderLayer1D(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float,
        shift_size: int,
    ) -> None:
        super().__init__()
        self.shift_size = shift_size
        self.window_attn = WindowAttention1D(d_model, n_heads, attn_dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = BatchNorm1dTokens(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = BatchNorm1dTokens(d_model)

    def forward(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        shift = self.shift_size if window_size > 1 else 0
        if shift and shift >= window_size:
            shift = shift % window_size
        attn_out = self.window_attn(x, window_size, shift)
        x = self.norm_attn(x + self.dropout_attn(attn_out))
        ff_out = self.ff(x)
        x = self.norm_ffn(x + self.dropout_ffn(ff_out))
        return x


class SwinTransformerStage1D(nn.Module):
    """Swin-style stack on patch-token sequences: window attention with alternating cyclic shift along time."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        window_size: int,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.window_size_cfg = max(1, window_size)
        base_shift = self.window_size_cfg // 2
        self.blocks = nn.ModuleList()
        for layer_index in range(n_layers):
            use_shift = base_shift > 0 and (layer_index % 2 == 1)
            self.blocks.append(
                SwinEncoderLayer1D(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    shift_size=base_shift if use_shift else 0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_orig, dim = x.shape
        if seq_orig == 0:
            return x
        ws = min(self.window_size_cfg, seq_orig)
        ws = max(ws, 1)
        pad_len = (ws - seq_orig % ws) % ws
        if pad_len:
            pad = x[:, -1:, :].expand(batch_size, pad_len, dim)
            x = torch.cat([x, pad], dim=1)
        for block in self.blocks:
            x = block(x, window_size=ws)
        return x[:, :seq_orig, :]


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
        self.revin = RevIN(in_channels, affine=config.revin_affine) if config.use_instance_norm else None
        self.hierarchical_patching = config.hierarchical_patching
        self.swin_like_patching = config.swin_like_patching
        if self.hierarchical_patching and self.swin_like_patching:
            raise ValueError("Use at most one of hierarchical_patching and swin_like_patching.")
        self.hierarchical_levels = max(1, config.hierarchical_levels)
        self.hierarchical_merge_factor = config.hierarchical_merge_factor
        if self.hierarchical_merge_factor < 2:
            raise ValueError("hierarchical_merge_factor must be at least 2")

        self.pad_layer = None
        self.num_patches = (config.seq_len - config.patch_len) // config.stride + 1
        if config.padding_patch == "end":
            self.pad_layer = nn.ReplicationPad1d((0, config.stride))
            self.num_patches += 1
        
        self.patch_embed = nn.Linear(config.patch_len, config.d_model)
        self.positional = PositionalEncoding(config.d_model, max_len=self.num_patches, learnable=True)
        self.input_dropout = nn.Dropout(config.dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        if self.hierarchical_patching or self.swin_like_patching:
            if self.swin_like_patching:
                self.encoders = nn.ModuleList(self._build_swin_stage() for _ in range(self.hierarchical_levels))
            else:
                self.encoders = nn.ModuleList(self._build_encoder() for _ in range(self.hierarchical_levels))
            self.merge_layers = nn.ModuleList(
                nn.Linear(config.d_model * self.hierarchical_merge_factor, config.d_model)
                for _ in range(self.hierarchical_levels - 1)
            )
            self.fusion_norm = nn.LayerNorm(config.d_model)
        else:
            self.encoder = self._build_encoder()
            self.merge_layers = nn.ModuleList()
            self.fusion_norm = nn.Identity()

        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(config.fc_dropout),
            nn.Linear(config.d_model * self.num_patches, config.pred_len),
            nn.Dropout(config.head_dropout),
        )
        self.reconstruction_head = nn.Linear(config.d_model, config.patch_len)

    def _build_encoder(self) -> PatchTSTEncoder:
        return PatchTSTEncoder(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            attn_dropout=self.config.attn_dropout,
        )

    def _build_swin_stage(self) -> SwinTransformerStage1D:
        return SwinTransformerStage1D(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            n_layers=self.config.n_layers,
            window_size=self.config.swin_window_size,
            dropout=self.config.dropout,
            attn_dropout=self.config.attn_dropout,
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        x = x.permute(0, 2, 1).reshape(bsz * channels, seq_len)
        if self.pad_layer is not None:
            x = self.pad_layer(x)
        return x.unfold(dimension=-1, size=self.config.patch_len, step=self.config.stride)

    def _encode_patches(self, patches: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.patch_embed(patches)
        if mask is not None:
            mask_token = self.mask_token.expand(z.size(0), z.size(1), -1)
            z = torch.where(mask.unsqueeze(-1), mask_token, z)
        if not self.hierarchical_patching and not self.swin_like_patching:
            z = self.input_dropout(self.positional(z))
            return self.encoder(z)

        return self._pyramid_encode(z)

    def _pyramid_encode(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        current = self.encoders[0](self.input_dropout(self.positional(tokens)))
        outputs.append(current)

        for level, merge_layer in enumerate(self.merge_layers, start=1):
            current = self._merge_patch_tokens(current, merge_layer)
            current = self.encoders[level](self.input_dropout(self.positional(current)))
            outputs.append(current)

        fused = outputs[0]
        scale = self.hierarchical_merge_factor
        for level, coarse in enumerate(outputs[1:], start=1):
            repeat = scale ** level
            upsampled = coarse.repeat_interleave(repeat, dim=1)[:, : self.num_patches]
            fused = fused + upsampled

        return self.fusion_norm(fused / len(outputs))

    def _merge_patch_tokens(self, tokens: torch.Tensor, merge_layer: nn.Linear) -> torch.Tensor:
        batch_size, num_tokens, dim = tokens.shape
        remainder = num_tokens % self.hierarchical_merge_factor
        if remainder != 0:
            pad_count = self.hierarchical_merge_factor - remainder
            pad_tokens = tokens[:, -1:, :].expand(batch_size, pad_count, dim)
            tokens = torch.cat([tokens, pad_tokens], dim=1)

        merged = tokens.reshape(batch_size, -1, self.hierarchical_merge_factor * dim)
        return merge_layer(merged)

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
