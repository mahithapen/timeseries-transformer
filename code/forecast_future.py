from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models.dlinear import DLinear
from models.patchtst import PatchTST, PatchTSTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast timesteps beyond the end of a time-series CSV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved training checkpoint")
    parser.add_argument("--data", type=str, default="", help="Optional override for the dataset path")
    parser.add_argument("--output", type=str, default="results/future_forecast.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def infer_future_dates(frame: pd.DataFrame, pred_len: int) -> list[str]:
    if "date" not in frame.columns:
        return [str(step) for step in range(1, pred_len + 1)]

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().all():
        return [str(step) for step in range(1, pred_len + 1)]

    dates = dates.dropna()
    last_date = dates.iloc[-1]
    if len(dates) >= 2:
        inferred_freq = pd.infer_freq(dates)
        offset = pd.tseries.frequencies.to_offset(inferred_freq) if inferred_freq else dates.iloc[-1] - dates.iloc[-2]
    else:
        offset = pd.Timedelta(days=1)

    return [(last_date + offset * step).strftime("%Y-%m-%d") for step in range(1, pred_len + 1)]


def build_model(checkpoint: dict, device: str) -> torch.nn.Module:
    config = checkpoint["config"]
    model_type = config.get("model_type", "patchtst")

    if model_type == "dlinear":
        return DLinear(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            channels=checkpoint["in_channels"],
        ).to(device)

    config_fields = {field.name for field in fields(PatchTSTConfig)}
    patchtst_config = PatchTSTConfig(
        **{key: value for key, value in config.items() if key in config_fields}
    )
    return PatchTST(patchtst_config, in_channels=checkpoint["in_channels"]).to(device)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    data_path = args.data or checkpoint["data_path"]

    frame = pd.read_csv(data_path)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError(f"No numeric columns found in {data_path}")

    series = numeric.to_numpy(dtype=np.float32, copy=True)
    train_end = int(len(series) * (1.0 - checkpoint["val_ratio"] - checkpoint["test_ratio"]))
    scaler = None
    processed = series
    if checkpoint["scale"]:
        scaler = StandardScaler()
        scaler.fit(series[:train_end])
        processed = scaler.transform(series).astype(np.float32)

    if len(processed) < config["seq_len"]:
        raise ValueError(f"Need at least seq_len={config['seq_len']} rows, got {len(processed)}")

    x = torch.tensor(processed[-config["seq_len"] :], dtype=torch.float32).unsqueeze(0).to(args.device)
    model = build_model(checkpoint, args.device)
    state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    if scaler is not None:
        pred = scaler.inverse_transform(pred)

    future_dates = infer_future_dates(frame, config["pred_len"])
    rows = []
    for horizon_idx in range(config["pred_len"]):
        for channel_idx, feature_name in enumerate(numeric.columns):
            rows.append(
                {
                    "date": future_dates[horizon_idx],
                    "horizon_step": horizon_idx + 1,
                    "feature": feature_name,
                    "prediction": pred[horizon_idx, channel_idx],
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output.to_csv(output_path, index=False)
    print(f"Saved future forecast to {output_path}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
