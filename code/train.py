from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.window_dataset import build_datasets, build_pretrain_dataset
from models.patchtst import PatchTST, PatchTSTConfig
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST training")
    parser.add_argument("--data", type=str, required=True, help="Path to a .csv, .npy, or .npz series")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--hierarchical-patching",
        action="store_true",
        help="Enable multi-scale hierarchical patching with progressive token merging.",
    )
    parser.add_argument(
        "--hierarchical-levels",
        type=int,
        default=2,
        help="Number of hierarchical encoder levels when hierarchical patching is enabled.",
    )
    parser.add_argument(
        "--hierarchical-merge-factor",
        type=int,
        default=2,
        help="How many adjacent patch tokens to merge between hierarchical levels.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fc-dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument(
        "--padding-patch",
        type=str,
        choices=["end", "none"],
        default="end",
        help="Patch padding strategy. Use 'end' to match the paper's extra trailing patch.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--pretrain-mask-ratio", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience on validation loss")
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["type3", "none"],
        default="type3",
        help="Learning rate scheduler for supervised training",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--no-scale", action="store_true", help="Disable train-split normalization")
    parser.add_argument(
        "--revin-affine",
        action="store_true",
        help="Enable affine parameters in RevIN. Disabled by default to better match the paper recipe.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/patchtst_best.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PatchTSTConfig:
    return PatchTSTConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        patch_len=args.patch_len,
        stride=args.stride,
        hierarchical_patching=args.hierarchical_patching,
        hierarchical_levels=args.hierarchical_levels,
        hierarchical_merge_factor=args.hierarchical_merge_factor,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        fc_dropout=args.fc_dropout,
        head_dropout=args.head_dropout,
        task="forecast",
        mask_ratio=args.pretrain_mask_ratio,
        revin_affine=args.revin_affine,
        padding_patch=None if args.padding_patch == "none" else args.padding_patch,
    )


def evaluate_forecast(model: PatchTST, loader: DataLoader, device: str, criterion: torch.nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
    return total_loss / max(1, len(loader))


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.num_bad_epochs = 0

    def step(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, base_lr: float, schedule: str) -> float:
    if schedule == "none":
        lr = base_lr
    elif schedule == "type3":
        lr = base_lr if epoch < 3 else base_lr * (0.9 ** (epoch - 3))
    else:
        raise ValueError(f"Unsupported scheduler: {schedule}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def run_pretraining(
    model: PatchTST,
    loader: DataLoader,
    epochs: int,
    device: str,
    lr: float,
    mask_ratio: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x in loader:
            x = x.to(device)
            reconstructed, target, mask = model.forward_pretrain(x, mask_ratio=mask_ratio)
            squared_error = (reconstructed - target) ** 2
            masked_error = squared_error * mask.unsqueeze(-1).float()
            loss = masked_error.sum() / mask.sum().clamp_min(1).float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Pretrain {epoch + 1}/{epochs} - masked patch loss: {avg_loss:.4f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    scale = not args.no_scale
    data_path = Path(args.data)
    bundle = build_datasets(
        data_path=data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        scale=scale,
    )

    train_loader = DataLoader(bundle.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(bundle.val, batch_size=args.batch_size, shuffle=False)

    config = build_config(args)
    model = PatchTST(config, in_channels=bundle.in_channels).to(args.device)

    if args.pretrain_epochs > 0:
        pretrain_dataset, _ = build_pretrain_dataset(
            data_path=data_path,
            seq_len=args.seq_len,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            scale=scale,
        )
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Starting masked-patch pretraining for {args.pretrain_epochs} epochs...")
        run_pretraining(
            model=model,
            loader=pretrain_loader,
            epochs=args.pretrain_epochs,
            device=args.device,
            lr=args.pretrain_lr,
            mask_ratio=args.pretrain_mask_ratio,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    best_val_loss = float("inf")
    early_stopper = EarlyStopping(args.patience)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting supervised training on {args.device} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        current_lr = adjust_learning_rate(optimizer, epoch + 1, args.lr, args.scheduler)
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_loss = evaluate_forecast(model, val_loader, args.device, criterion)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - lr: {current_lr:.6g} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "in_channels": bundle.in_channels,
                    "data_path": str(data_path),
                    "val_ratio": args.val_ratio,
                    "test_ratio": args.test_ratio,
                    "scale": scale,
                    "best_val_loss": best_val_loss,
                    "pretrain_epochs": args.pretrain_epochs,
                    "pretrain_mask_ratio": args.pretrain_mask_ratio,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        if early_stopper.step(val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break


if __name__ == "__main__":
    main()
