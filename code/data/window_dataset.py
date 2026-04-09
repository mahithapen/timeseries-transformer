from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int):
        # series: [T, C]
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.series) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
