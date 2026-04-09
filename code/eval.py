from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.window_dataset import WindowDataset
from models.patchtst import PatchTST, PatchTSTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST evaluation")
    parser.add_argument("--data", type=str, required=False, default="", help="Path to .npy series")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data:
        series = np.load(args.data)
    else:
        series = np.random.randn(2000, 8).astype(np.float32)

    dataset = WindowDataset(series, args.seq_len, args.pred_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    config = PatchTSTConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        task="forecast",
    )

    model = PatchTST(config, in_channels=series.shape[1]).to(args.device)
    model.eval()

    mae = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    total_mae = 0.0
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)
            pred = model(x)
            total_mae += mae(pred, y).item()
            total_mse += mse(pred, y).item()
            count += 1

    print(f"MAE: {total_mae / max(1, count):.4f}")
    print(f"MSE: {total_mse / max(1, count):.4f}")


if __name__ == "__main__":
    main()
