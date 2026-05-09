# Code

This folder contains the training, evaluation, forecasting, data-loading, and model code for the time-series forecasting experiments.

## Setup

Run commands from the repository root:

```bash
source venv/bin/activate
```

If you are starting from a fresh environment:

```bash
pip install -r code/requirements.txt
```

## Training

The main training script supports `dlinear` and `patchtst`. PatchTST can be run either normally or with hierarchical patching.

### DLinear

```bash
python code/train.py \
  --model-type dlinear \
  --data data/weather.csv \
  --seq-len 336 \
  --pred-len 96 \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --checkpoint checkpoints/weather_supervised_dlinear_seq336_pred96.pt
```

### PatchTST/42

```bash
python code/train.py \
  --model-type patchtst \
  --data data/weather.csv \
  --seq-len 336 \
  --pred-len 96 \
  --patch-len 16 \
  --stride 8 \
  --padding-patch end \
  --d-model 128 \
  --n-heads 16 \
  --n-layers 3 \
  --d-ff 256 \
  --dropout 0.2 \
  --fc-dropout 0.2 \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --checkpoint checkpoints/weather_supervised_patchtst42_seq336_pred96.pt
```

### PatchTST/64

```bash
python code/train.py \
  --model-type patchtst \
  --data data/weather.csv \
  --seq-len 512 \
  --pred-len 96 \
  --patch-len 16 \
  --stride 8 \
  --padding-patch end \
  --d-model 128 \
  --n-heads 16 \
  --n-layers 3 \
  --d-ff 256 \
  --dropout 0.2 \
  --fc-dropout 0.2 \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --checkpoint checkpoints/weather_supervised_patchtst64_seq512_pred96.pt
```

### Hierarchical PatchTST

Add these flags to a PatchTST command:

```bash
--hierarchical-patching \
--hierarchical-levels 2 \
--hierarchical-merge-factor 2
```

Example:

```bash
python code/train.py \
  --model-type patchtst \
  --data data/weather.csv \
  --seq-len 336 \
  --pred-len 96 \
  --patch-len 16 \
  --stride 8 \
  --padding-patch end \
  --hierarchical-patching \
  --hierarchical-levels 2 \
  --hierarchical-merge-factor 2 \
  --d-model 128 \
  --n-heads 16 \
  --n-layers 3 \
  --d-ff 256 \
  --dropout 0.2 \
  --fc-dropout 0.2 \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --checkpoint checkpoints/weather_hierarchical_patchtst42_seq336_pred96.pt
```

Use `--resume` to continue from an existing checkpoint.

## Evaluation

Evaluate a checkpoint on the validation or test split:

```bash
python code/eval.py \
  --checkpoint checkpoints/weather_hierarchical_patchtst42_seq336_pred96.pt \
  --data data/weather.csv \
  --split test \
  --batch-size 128
```

`--data` is optional if the checkpoint was trained with the same dataset path you want to evaluate.

## Future Forecasting

Generate predictions beyond the end of a CSV:

```bash
python code/forecast_future.py \
  --checkpoint checkpoints/weather_hierarchical_patchtst42_seq336_pred96.pt \
  --data data/weather.csv \
  --output results/weather_future_forecast.csv
```

The output CSV contains one row per forecast horizon step and feature.

## Colab Workflow

Use `code/Final_Colab.ipynb` for the full Colab workflow. The notebook runs:

- DLinear supervised
- PatchTST/42 supervised
- PatchTST/64 supervised
- hierarchical PatchTST/42
- hierarchical PatchTST/64

It saves checkpoints and summary metrics under the configured Drive paths.
