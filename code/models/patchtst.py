import torch
import torch.nn as nn


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
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x

    def _get_statistics(self, x):
        # x: [B, L, C]
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


class PatchTST(nn.Module):
    def __init__(self, config, in_channels: int):
        super().__init__()
        self.config = config
        self.in_channels = in_channels

        # 1. RevIN Implementation
        self.revin = RevIN(in_channels) if config.use_instance_norm else None

        # Calculate number of patches
        self.num_patches = int(
            (config.seq_len - config.patch_len) / config.stride + 2)

        self.patch_embed = nn.Linear(config.patch_len, config.d_model)
        self.positional = PositionalEncoding(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers)

        # 2. Flatten Head Implementation
        if config.task == "forecast":
            self.head = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(config.d_model * self.num_patches, config.pred_len)
            )

    def forward(self, x: torch.Tensor):
        # x: [B, L, C]
        if self.revin:
            x = self.revin(x, 'norm')

        # 3. Channel Independence Reshape
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)  # [B*C, L, 1]

        # Patching
        x = x.unfold(dimension=1, size=self.config.patch_len,
                     step=self.config.stride)
        # x: [B*C, num_patches, patch_len]

        # Backbone
        z = self.patch_embed(x)
        z = self.positional(z)
        z = self.encoder(z)  # [B*C, num_patches, d_model]

        # Head
        z = self.head(z)  # [B*C, pred_len]

        # Reshape back and Denormalize
        z = z.reshape(B, C, -1).permute(0, 2, 1)  # [B, pred_len, C]
        if self.revin:
            z = self.revin(z, 'denorm')

        return z
