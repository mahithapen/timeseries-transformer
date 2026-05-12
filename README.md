# PatchTST Re-implementation for Long-Term Time Series Forecasting

**Authors:** Mahitha Penmetsa (`msp259`), Aayush Agnihotri (`aa2328`), Cindy Liang (`cl2329`), Peter Bidoshi (`pjb294`)

## Introduction

This repository contains our CS 4782 final project re-implementation of **"A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers"** (ICLR 2023). The paper introduces **PatchTST**, a transformer architecture that improves long-term time series forecasting via patching and channel independence, enabling longer look-back windows with lower compute.

## Chosen Result

We run supervised forecasting experiments on the **Electricity, Traffic, and Weather** datasets, comparing DLinear with PatchTST/42 and PatchTST/64 style variants with a prediction length of 96. We compare the results to the DLinear model.
<img width="1006" height="488" alt="image" src="https://github.com/user-attachments/assets/04c4e8fe-502c-4c54-9d64-195bf7704774" />



## GitHub Contents

- `code/`: Re-implementation code and configs
  - `models/`: Holds the code for PatchTST and dLinear
  - `Final_Colab.ipynb`: Jupyter notebook used to train and evaluate our model to generate our results
  - `eval.py`: Python file used to evaluate our model
  - `train.py`: Python file used to train our model
  - `generate_forecast.py`: Python file used to make future predictions on values that extend beyond the dataset (Ithaca weather predicitons for our poster)
  - `generate_forecast.py`: Loads in the time series data, makes the train/val/test split, builds the sliding window datasets
- `data/`: Dataset download instructions and metadata
- `results/`: Output tables, plots, and logs
- `poster/`: Final poster PDF
- `report/`: Final 2-page report PDF
- `docs/`: Internal notes and checklists

## Re-implementation Details

Time series input is first processed through preprocess_data.py to generate a train/test split and generate sliding windows. We implemented our supervised PatchTTST by first using Reversible Instance Normalization, treating each input feature independently (channel independence), and apply patching (grouping local timesteps to a singular token) with a patch length of 16 and a stride of 8. Learnable positional embeddings are added, and this input is then fed into a Transformer encoder with 3 encoder layers. Each layer uses multi-head self-attention with 16 attention heads, followed by residual connection and batch normalization, then a feed-forward network, followed by another residual connection and batch normalization. Finally, we perform forecasting with a linear head. To evaluate our model using MSE and MAE. We train for 100 epochs with early stopping, Adam optimizer, and a learning rate scheduler. We predict for the next 96 time steps and use look-back windows of length 336 and 512.

For our HPatch addition, we use 2 Transformer encoder layers. Adjacent tokens from the output of the first layer were merged and projected back to the original embedding dimension before being processed by a second encoder level. The outputs of the first and second layers are combined and forecasting is again performed with a linear head.

## Reproduction Steps

1. Create and activate a Python environment
2. Install dependencies from `code/requirements.txt`
3. Download datasets listed in `data/README.md`

### Training
#### DLinear

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

#### PatchTST/42

```bash
python code/train.py \
  --model-type patchtst \
  --data data/weather.csv \
  --seq-len 336 \
  --pred-len 96 \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --checkpoint checkpoints/weather_supervised_patchtst42_seq336_pred96.pt
```

#### PatchTST/64

```bash
python code/train.py \
  --model-type patchtst \
  --data data/weather.csv \
  --seq-len 512 \
  --pred-len 96 \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --checkpoint checkpoints/weather_supervised_patchtst64_seq512_pred96.pt
```

#### Hierarchical PatchTST

Add this flag to a PatchTST command:

```bash
--hierarchical-patching
```

Use `--resume` to continue from an existing checkpoint.

### Evaluation

Evaluate a checkpoint on the validation or test split:

```bash
python code/eval.py \
  --checkpoint checkpoints/weather_hierarchical_patchtst42_seq336_pred96.pt \
  --data data/weather.csv \
  --split test \
  --batch-size 128
```

`--data` is optional if the checkpoint was trained with the same dataset path you want to evaluate.

### Future Forecasting

Generate predictions beyond the end of a CSV:

```bash
python code/generate_forecast.py \
  --checkpoint checkpoints/weather_hierarchical_patchtst42_seq336_pred96.pt \
  --data data/weather.csv \
  --output results/weather_future_forecast.csv
```

The output CSV contains one row per forecast horizon step and feature.

### Colab Workflow

Use `code/Final_Colab.ipynb` for the full Colab workflow. The notebook runs:

- DLinear supervised
- PatchTST/42 supervised
- PatchTST/64 supervised
- hierarchical PatchTST/42
- hierarchical PatchTST/64

It saves checkpoints and summary metrics under the configured Drive paths.

We trained on NVIDIA A100 GPU.

## Results / Insights
<img width="448" height="311" alt="image" src="https://github.com/user-attachments/assets/0be41687-45d9-4045-a01e-02085b497454" />
We showed that the PatchTST and HPatch model outperforms dLinear on all three datasets. We report MSE and MAE values comparable to the original paper, even beating it in some cases, such as an MSE of 0.1468 for our PatchTST/64 compared to 0.1490 in the paper. However, we fail to see significant gains from implementing hierarchical patching (HPatch), which we attribute to potential strong local periodicity that may already exist in the current datasets, and its true benefits could be revealed in contexts with greater long-range dependencies. 

The expected end result of using this GitHub repo is a trained forecasting model checkpoint, test-set MSE/MAE metrics, and optional future forecasts saved as CSV files. 

## Conclusion

Transformers can be optimized for time series forecasting through patching and channel independence. Patching allows us to compute more informative attention calculations due to the preservation of local semantic information within a token. Linear models like dLinear reach a plateau as lookback windows increase; the time and space complexity reduction in patching allows us to scale the lookback window to improve accuracy. By isolating variables through channel independence, the model was able to learn distinct attention patterns for the diverse behaviors in the datasets, while preventing noise leakage between channels. However, we learned how hard it may be to add extensions to the study due to varying contexts in datasets, hyperparameters to tune, and design choices to make when implementing hierarchical patching. 

## References

[1] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. _A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers_. ICLR 2023.

[2] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? arXiv preprint arXiv:2205.13504, 2022.

[3] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with Auto-Correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, 2021.


## Acknowledgements

This project was completed as part of **CS 4782** at Cornell University.
