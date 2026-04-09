from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.window_dataset import WindowDataset
from models.patchtst import PatchTST, PatchTSTConfig
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST training")
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.data:
        series = np.load(args.data)
    else:
        # Placeholder synthetic data for quick sanity checks
        series = np.random.randn(2000, 8).astype(np.float32)

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

    model = PatchTST(config, in_channels=series.shape[1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

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
