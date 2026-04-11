from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import os
from torch.utils.data import DataLoader

# Updated imports to use the TimeSeriesLoader for CSV support
from data.window_dataset import TimeSeriesLoader
from models.patchtst import PatchTST, PatchTSTConfig
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST training")
    parser.add_argument("--data", type=str, required=False, default="", help="Path to .csv or .npy series")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Path to the data directory relative to this script
    root_path = './data/'

    if args.data and args.data.endswith('.csv'):
        print(f"Loading CSV dataset: {args.data}")
        # Initialize the CSV loader specifically for datasets like Jena Weather
        dataset = TimeSeriesLoader(
            root_path=root_path,
            data_path=args.data,
            flag='train',
            size=[args.seq_len, args.pred_len],
            scale=True
        )
        # Determine in_channels from the features in the processed dataframe
        in_channels = dataset.data_x.shape[-1]
    else:
        # Fallback to synthetic data if no CSV is provided
        print("Using synthetic data for training.")
        series = np.random.randn(2000, 8).astype(np.float32)
        in_channels = 8
        # You would need a separate WindowDataset class if using raw numpy arrays
        from data.window_dataset import WindowDataset
        dataset = WindowDataset(series, args.seq_len, args.pred_len)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

    model = PatchTST(config, in_channels=in_channels).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    print(f"Starting training on {args.device} for {args.epochs} epochs...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()