import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeSeriesLoader(Dataset):
    """
    Data loader for CSV time-series datasets (e.g., Jena Weather, Electricity).
    Handles scaling, train/val/test splits, and multi-channel formatting.
    """
    def __init__(self, root_path, data_path, flag='train', size=None, 
                 target='T (degC)', scale=True):
        # size [seq_len, pred_len]
        self.seq_len = size[0]
        self.pred_len = size[1]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 1. Handle Timestamp Column
        # The Kaggle Jena file uses "Date Time"
        cols = list(df_raw.columns)
        if 'Date Time' in cols:
            df_raw = df_raw.drop(columns=['Date Time'])
        elif 'date' in cols:
            df_raw = df_raw.drop(columns=['date'])
        
        # 2. Re-order to ensure target is the last column if needed
        # PatchTST typically processes all channels independently
        df_data = df_raw.copy()

        # 3. Calculate Splits (Standard 70/10/20 split)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 4. Scaling
        if self.scale:
            # Fit scaler only on training data to prevent data leakage
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        # Sliding window indices
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

class WindowDataset(Dataset):
    """
    Fallback loader for raw numpy arrays (Synthetic Data).
    """
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)