from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    in_channels: int
    scaler: StandardScaler | None
    series_length: int


def load_time_series(data_path: str | Path) -> np.ndarray:
    path = Path(data_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError(f"No numeric columns found in {path}")
        series = numeric.to_numpy(dtype=np.float32, copy=True)
    elif suffix in {".npy", ".npz"}:
        loaded = np.load(path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = list(loaded.keys())
            if not keys:
                raise ValueError(f"No arrays found in {path}")
            series = loaded[keys[0]]
        else:
            series = loaded
        series = np.asarray(series, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2:
        raise ValueError(f"Expected a 2D time series array, got shape {series.shape}")
    if not np.isfinite(series).all():
        raise ValueError(f"Found non-finite values in {path}")

    return series.astype(np.float32, copy=False)


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        seq_len: int,
        pred_len: int,
    ) -> None:
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len

        if len(self.series) < self.seq_len + self.pred_len:
            raise ValueError(
                f"Split is too small for seq_len={seq_len} and pred_len={pred_len}. "
                f"Need at least {self.seq_len + self.pred_len} timesteps, got {len(self.series)}."
            )

    def __len__(self) -> int:
        return len(self.series) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_start = idx
        seq_end = seq_start + self.seq_len
        pred_end = seq_end + self.pred_len
        x = self.series[seq_start:seq_end]
        y = self.series[seq_end:pred_end]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class ContextWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int) -> None:
        self.series = series
        self.seq_len = seq_len

        if len(self.series) < self.seq_len:
            raise ValueError(
                f"Split is too small for seq_len={seq_len}. Need at least {self.seq_len} timesteps, "
                f"got {len(self.series)}."
            )

    def __len__(self) -> int:
        return len(self.series) - self.seq_len + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.series[idx : idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32)


def _compute_split_points(length: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    train_end = int(length * (1.0 - val_ratio - test_ratio))
    val_end = int(length * (1.0 - test_ratio))
    return train_end, val_end


def _compute_split_borders(
    length: int,
    seq_len: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[int], list[int]]:
    train_end, val_end = _compute_split_points(length, val_ratio, test_ratio)
    border1s = [0, train_end - seq_len, val_end - seq_len]
    border2s = [train_end, val_end, length]
    return border1s, border2s


def build_datasets(
    data_path: str | Path,
    seq_len: int,
    pred_len: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    scale: bool = True,
) -> DatasetBundle:
    series = load_time_series(data_path)
    total_length = len(series)
    border1s, border2s = _compute_split_borders(total_length, seq_len, val_ratio, test_ratio)
    train_start, val_start, test_start = border1s
    train_end, val_end, test_end = border2s

    min_required = seq_len + pred_len
    if train_end < min_required:
        raise ValueError(
            f"Training split is too small for seq_len={seq_len} and pred_len={pred_len}. "
            f"Need at least {min_required} timesteps before the train boundary, got {train_end}."
        )
    if val_start < 0 or test_start < 0:
        raise ValueError(
            f"Validation/test splits are too small for seq_len={seq_len}. "
            "Reduce seq_len or increase the dataset length."
        )

    scaler: StandardScaler | None = None
    processed = series
    if scale:
        scaler = StandardScaler()
        scaler.fit(series[:train_end])
        processed = scaler.transform(series).astype(np.float32)

    train = ForecastWindowDataset(
        processed[train_start:train_end],
        seq_len=seq_len,
        pred_len=pred_len,
    )
    val = ForecastWindowDataset(
        processed[val_start:val_end],
        seq_len=seq_len,
        pred_len=pred_len,
    )
    test = ForecastWindowDataset(
        processed[test_start:test_end],
        seq_len=seq_len,
        pred_len=pred_len,
    )

    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        in_channels=processed.shape[1],
        scaler=scaler,
        series_length=total_length,
    )


def build_pretrain_dataset(
    data_path: str | Path,
    seq_len: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    scale: bool = True,
) -> tuple[ContextWindowDataset, int]:
    series = load_time_series(data_path)
    border1s, border2s = _compute_split_borders(len(series), seq_len, val_ratio, test_ratio)
    train_start = border1s[0]
    train_end = border2s[0]

    if train_end < seq_len:
        raise ValueError(
            f"Training split is too small for seq_len={seq_len}. Need at least {seq_len} timesteps."
        )

    if scale:
        scaler = StandardScaler()
        scaler.fit(series[:train_end])
        series = scaler.transform(series).astype(np.float32)

    dataset = ContextWindowDataset(series[train_start:train_end], seq_len=seq_len)
    return dataset, series.shape[1]
