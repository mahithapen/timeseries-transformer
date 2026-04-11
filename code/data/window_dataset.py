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
        target_start: int,
        target_end: int,
    ) -> None:
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_start = target_start
        self.target_end = target_end

        if self.target_end <= self.target_start:
            raise ValueError(
                f"Empty split for seq_len={seq_len}, pred_len={pred_len}, "
                f"target range=({target_start}, {target_end})"
            )

    def __len__(self) -> int:
        return self.target_end - self.target_start

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.target_start + idx
        x = self.series[t - self.seq_len : t]
        y = self.series[t : t + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class ContextWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, target_start: int, target_end: int) -> None:
        self.series = series
        self.seq_len = seq_len
        self.target_start = target_start
        self.target_end = target_end

        if self.target_end <= self.target_start:
            raise ValueError(
                f"Empty split for seq_len={seq_len}, target range=({target_start}, {target_end})"
            )

    def __len__(self) -> int:
        return self.target_end - self.target_start

    def __getitem__(self, idx: int) -> torch.Tensor:
        t = self.target_start + idx
        x = self.series[t - self.seq_len : t]
        return torch.tensor(x, dtype=torch.float32)


def _compute_split_points(length: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    train_end = int(length * (1.0 - val_ratio - test_ratio))
    val_end = int(length * (1.0 - test_ratio))
    return train_end, val_end


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
    train_end, val_end = _compute_split_points(total_length, val_ratio, test_ratio)

    min_required = seq_len + pred_len + 1
    if train_end < min_required:
        raise ValueError(
            f"Training split is too small for seq_len={seq_len} and pred_len={pred_len}. "
            f"Need at least {min_required} timesteps before the train boundary, got {train_end}."
        )

    scaler: StandardScaler | None = None
    processed = series
    if scale:
        scaler = StandardScaler()
        scaler.fit(series[:train_end])
        processed = scaler.transform(series).astype(np.float32)

    train = ForecastWindowDataset(
        processed,
        seq_len=seq_len,
        pred_len=pred_len,
        target_start=seq_len,
        target_end=train_end - pred_len + 1,
    )
    val = ForecastWindowDataset(
        processed,
        seq_len=seq_len,
        pred_len=pred_len,
        target_start=train_end,
        target_end=val_end - pred_len + 1,
    )
    test = ForecastWindowDataset(
        processed,
        seq_len=seq_len,
        pred_len=pred_len,
        target_start=val_end,
        target_end=total_length - pred_len + 1,
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
    train_end, _ = _compute_split_points(len(series), val_ratio, test_ratio)

    if train_end <= seq_len:
        raise ValueError(
            f"Training split is too small for seq_len={seq_len}. Need more than {seq_len} timesteps."
        )

    if scale:
        scaler = StandardScaler()
        scaler.fit(series[:train_end])
        series = scaler.transform(series).astype(np.float32)

    dataset = ContextWindowDataset(
        series,
        seq_len=seq_len,
        target_start=seq_len,
        target_end=train_end + 1,
    )
    return dataset, series.shape[1]
