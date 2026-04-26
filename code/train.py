from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    parser.add_argument("--hierarchical-levels", type=int, default=2)
    parser.add_argument("--hierarchical-merge-factor", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--fc-dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument(
        "--padding-patch",
        type=str,
        choices=["end", "none"],
        default="end",
        help="Patch padding strategy. Use 'end' to match the paper's extra trailing patch.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Supervised epochs from scratch")
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--pretrain-mask-ratio", type=float, default=0.4)
    parser.add_argument("--linear-probe-epochs", type=int, default=0)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience on validation loss")
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
        help="Enable affine parameters in RevIN. Disabled by default to better match the paper recipe.",
    )
    parser.add_argument(
        "--pretrain-only",
        action="store_true",
        help="Run masked patch pretraining only and save a pretrained checkpoint.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="",
        help="Optional pretrained checkpoint to initialize the encoder before downstream training.",
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
        attn_dropout=args.attn_dropout,
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


def freeze_for_linear_probe(model: PatchTST) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.forecast_head.parameters():
        parameter.requires_grad = True


def unfreeze_all(model: PatchTST) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def load_pretrained_backbone(model: PatchTST, checkpoint_path: Path, device: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_state = checkpoint["model_state_dict"]
    current_state = model.state_dict()

    filtered_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, tensor in pretrained_state.items():
        if name.startswith("forecast_head"):
            skipped.append(name)
            continue
        if name in current_state and current_state[name].shape == tensor.shape:
            filtered_state[name] = tensor
        else:
            skipped.append(name)

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded pretrained backbone from {checkpoint_path}")
    if skipped:
        print(f"Skipped {len(skipped)} parameter(s) due to head or shape mismatch.")
    if missing:
        print(f"Missing parameters after load: {len(missing)}")
    if unexpected:
        print(f"Unexpected parameters after load: {len(unexpected)}")
    return checkpoint


def save_checkpoint(
    checkpoint_path: Path,
    model: PatchTST,
    config: PatchTSTConfig,
    metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            **metadata,
        },
        checkpoint_path,
    )


def run_supervised_phase(
    *,
    model: PatchTST,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    scheduler: str,
    patience: int,
    early_stopping_enabled: bool,
    checkpoint_path: Path,
    config: PatchTSTConfig,
    metadata: dict[str, Any],
    phase_name: str,
    best_val_loss: float,
) -> float:
    if epochs <= 0:
        return best_val_loss

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
    criterion = torch.nn.MSELoss()
    stopper = EarlyStopping(patience) if early_stopping_enabled else None

    print(f"Starting {phase_name} on {device} for {epochs} epochs...")
    for epoch in range(epochs):
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
            f"{phase_name} {epoch + 1}/{epochs} - lr: {current_lr:.6g} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metadata["best_val_loss"] = best_val_loss
            metadata["last_phase"] = phase_name
            save_checkpoint(checkpoint_path, model, config, metadata)
            print(f"Saved checkpoint to {checkpoint_path}")

        if stopper is not None and stopper.step(val_loss):
            print(f"Early stopping triggered during {phase_name} after {epoch + 1} epochs.")
            break

    return best_val_loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    scale = not args.no_scale
    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    pretrained_checkpoint_path = Path(args.pretrained_checkpoint) if args.pretrained_checkpoint else None

    if args.pretrain_only:
        if args.pretrain_epochs <= 0:
            raise ValueError("--pretrain-only requires --pretrain-epochs > 0")

        pretrain_dataset, in_channels = build_pretrain_dataset(
            data_path=data_path,
            seq_len=args.seq_len,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            scale=scale,
        )
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True)
        model = PatchTST(config, in_channels=in_channels).to(args.device)

        print(f"Starting masked-patch pretraining for {args.pretrain_epochs} epochs...")
        run_pretraining(
            model=model,
            loader=pretrain_loader,
            epochs=args.pretrain_epochs,
            device=args.device,
            lr=args.pretrain_lr,
            mask_ratio=args.pretrain_mask_ratio,
        )
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            config=config,
            metadata={
                "in_channels": in_channels,
                "data_path": str(data_path),
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "scale": scale,
                "best_val_loss": float("inf"),
                "training_stage": "pretrain_only",
                "pretrain_epochs": args.pretrain_epochs,
                "pretrain_mask_ratio": args.pretrain_mask_ratio,
                "seed": args.seed,
            },
        )
        print(f"Saved pretrained checkpoint to {checkpoint_path}")
        return

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
    model = PatchTST(config, in_channels=bundle.in_channels).to(args.device)

    loaded_pretrain_metadata: dict[str, Any] | None = None
    if pretrained_checkpoint_path is not None:
        loaded_pretrain_metadata = load_pretrained_backbone(model, pretrained_checkpoint_path, args.device)

    if args.pretrain_epochs > 0 and pretrained_checkpoint_path is None:
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

    metadata: dict[str, Any] = {
        "in_channels": bundle.in_channels,
        "data_path": str(data_path),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "scale": scale,
        "best_val_loss": float("inf"),
        "training_stage": "supervised",
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_mask_ratio": args.pretrain_mask_ratio,
        "seed": args.seed,
        "pretrained_checkpoint": str(pretrained_checkpoint_path) if pretrained_checkpoint_path else "",
        "linear_probe_epochs": args.linear_probe_epochs,
        "finetune_epochs": args.finetune_epochs,
    }
    if loaded_pretrain_metadata is not None:
        metadata["pretrained_source_data"] = loaded_pretrain_metadata.get("data_path", "")
        metadata["pretrained_source_stage"] = loaded_pretrain_metadata.get("training_stage", "")

    best_val_loss = float("inf")
    early_stopping_enabled = not args.disable_early_stopping

    if args.linear_probe_epochs > 0:
        freeze_for_linear_probe(model)
        metadata["training_stage"] = "linear_probe"
        best_val_loss = run_supervised_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            epochs=args.linear_probe_epochs,
            lr=args.probe_lr,
            scheduler=args.scheduler,
            patience=args.patience,
            early_stopping_enabled=early_stopping_enabled,
            checkpoint_path=checkpoint_path,
            config=config,
            metadata=metadata,
            phase_name="linear_probe",
            best_val_loss=best_val_loss,
        )

    finetune_epochs = args.finetune_epochs
    if finetune_epochs > 0 or (args.linear_probe_epochs == 0 and (pretrained_checkpoint_path is not None or args.pretrain_epochs > 0)):
        unfreeze_all(model)
        metadata["training_stage"] = "finetune"
        best_val_loss = run_supervised_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            epochs=finetune_epochs if finetune_epochs > 0 else args.epochs,
            lr=args.finetune_lr if finetune_epochs > 0 else args.lr,
            scheduler=args.scheduler,
            patience=args.patience,
            early_stopping_enabled=early_stopping_enabled,
            checkpoint_path=checkpoint_path,
            config=config,
            metadata=metadata,
            phase_name="finetune",
            best_val_loss=best_val_loss,
        )
    elif args.linear_probe_epochs == 0:
        unfreeze_all(model)
        metadata["training_stage"] = "supervised"
        best_val_loss = run_supervised_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            scheduler=args.scheduler,
            patience=args.patience,
            early_stopping_enabled=early_stopping_enabled,
            checkpoint_path=checkpoint_path,
            config=config,
            metadata=metadata,
            phase_name="supervised",
            best_val_loss=best_val_loss,
        )


if __name__ == "__main__":
    main()
