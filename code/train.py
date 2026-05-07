from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.window_dataset import build_datasets
from models.dlinear import DLinear
from models.patchtst import PatchTST, PatchTSTConfig
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time Series Forecasting Training")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["patchtst", "dlinear"],
        default="patchtst",
        help="Model architecture to train",
    )
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
    parser.add_argument("--hierarchical-levels", type=int, default=2)
    parser.add_argument("--hierarchical-merge-factor", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--fc-dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument(
        "--padding-patch",
        type=str,
        choices=["end", "none"],
        default="end",
        help="Patch padding strategy. DLinear ignores this option.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience on validation loss",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["type3", "none"],
        default="type3",
        help="Learning rate scheduler for supervised training",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Disable early stopping and run the full requested number of epochs.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--no-scale", action="store_true", help="Disable train-split normalization")
    parser.add_argument(
        "--revin-affine",
        action="store_true",
        help="Enable affine parameters in RevIN.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile before training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training mid-run from the checkpoint.",
    )
    return parser.parse_args()


def build_patchtst_config(args: argparse.Namespace) -> PatchTSTConfig:
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
        attn_dropout=args.attn_dropout,
        fc_dropout=args.fc_dropout,
        head_dropout=args.head_dropout,
        revin_affine=args.revin_affine,
        padding_patch=None if args.padding_patch == "none" else args.padding_patch,
    )


def base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def evaluate_forecast(
    model: torch.nn.Module, loader: DataLoader, device: str, criterion: torch.nn.Module
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            total_loss += criterion(pred, y.to(device)).item()
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


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, base_lr: float, schedule: str
) -> float:
    if schedule == "none":
        lr = base_lr
    elif schedule == "type3":
        lr = base_lr if epoch < 3 else base_lr * (0.9 ** (epoch - 3))
    else:
        raise ValueError(f"Unsupported scheduler: {schedule}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    config_dict: dict[str, Any],
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    save_dict = {
        "model_state_dict": base_model(model).state_dict(),
        "config": config_dict,
        **metadata,
    }
    if optimizer is not None:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(save_dict, checkpoint_path)


def train_forecast(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    scheduler: str,
    patience: int,
    early_stopping_enabled: bool,
    checkpoint_path: Path,
    config_dict: dict[str, Any],
    metadata: dict[str, Any],
    resume: bool,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    stopper = EarlyStopping(patience) if early_stopping_enabled else None
    best_val_loss = float("inf")
    start_epoch = 0

    if resume and checkpoint_path.exists():
        print(f"Resuming supervised training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = {
            key.replace("_orig_mod.", ""): value
            for key, value in checkpoint["model_state_dict"].items()
        }
        base_model(model).load_state_dict(state_dict)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if stopper is not None:
            stopper.num_bad_epochs = checkpoint.get("num_bad_epochs", 0)
            stopper.best_loss = best_val_loss

    if start_epoch >= epochs:
        print(f"Supervised training already completed up to epoch {epochs}.")
        return best_val_loss

    print(f"Starting supervised training on {device} from epoch {start_epoch} to {epochs}...")
    for epoch in range(start_epoch, epochs):
        current_lr = adjust_learning_rate(optimizer, epoch + 1, lr, scheduler)
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_loss = evaluate_forecast(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch + 1}/{epochs} - lr: {current_lr:.6g} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metadata["best_val_loss"] = best_val_loss
            metadata["epoch"] = epoch
            metadata["num_bad_epochs"] = stopper.num_bad_epochs if stopper else 0
            save_checkpoint(checkpoint_path, model, config_dict, metadata, optimizer)
            print(f"Saved checkpoint to {checkpoint_path}")

        if stopper is not None and stopper.step(val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return best_val_loss


def build_model(
    args: argparse.Namespace, in_channels: int
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.model_type == "dlinear":
        model = DLinear(seq_len=args.seq_len, pred_len=args.pred_len, channels=in_channels)
        return model, {
            "model_type": "dlinear",
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
        }

    config = build_patchtst_config(args)
    config_dict = asdict(config)
    config_dict["model_type"] = "patchtst"
    return PatchTST(config, in_channels=in_channels), config_dict


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = build_datasets(
        data_path=Path(args.data),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        scale=not args.no_scale,
    )
    train_loader = DataLoader(bundle.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(bundle.val, batch_size=args.batch_size, shuffle=False)

    model, config_dict = build_model(args, bundle.in_channels)
    model = model.to(args.device)
    if args.compile:
        model = torch.compile(model)

    metadata: dict[str, Any] = {
        "in_channels": bundle.in_channels,
        "data_path": str(Path(args.data)),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "scale": not args.no_scale,
        "best_val_loss": float("inf"),
        "training_stage": "supervised",
        "seed": args.seed,
    }

    train_forecast(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        scheduler=args.scheduler,
        patience=args.patience,
        early_stopping_enabled=not args.disable_early_stopping,
        checkpoint_path=checkpoint_path,
        config_dict=config_dict,
        metadata=metadata,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
