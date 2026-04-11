from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from data.window_dataset import build_datasets
from models.patchtst import PatchTST, PatchTSTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved training checkpoint")
    parser.add_argument("--data", type=str, default="", help="Optional override for the dataset path")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = PatchTSTConfig(**checkpoint["config"])
    data_path = args.data or checkpoint["data_path"]

    bundle = build_datasets(
        data_path=data_path,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        val_ratio=checkpoint["val_ratio"],
        test_ratio=checkpoint["test_ratio"],
        scale=checkpoint["scale"],
    )

    dataset = bundle.val if args.split == "val" else bundle.test
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = PatchTST(config, in_channels=checkpoint["in_channels"]).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
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

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"MAE: {total_mae / max(1, count):.4f}")
    print(f"MSE: {total_mse / max(1, count):.4f}")


if __name__ == "__main__":
    main()
