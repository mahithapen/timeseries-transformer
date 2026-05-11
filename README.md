# PatchTST Re-implementation for Long-Term Time Series Forecasting

**Authors:** Mahitha Penmetsa (`msp259`), Aayush Agnihotri (`aa2328`), Cindy Liang (`cl2329`), Peter Bidoshi (`pjb294`)

## Introduction

This repository contains our CS 4782 final project re-implementation of **"A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers"** (ICLR 2023). The paper introduces **PatchTST**, a transformer architecture that improves long-term time series forecasting via patching and channel independence, enabling longer look-back windows with lower compute.

## Chosen Result

We run supervised forecasting experiments on the **Electricity, Traffic, and Weather** datasets, comparing DLinear with PatchTST/42 and PatchTST/64 style variants with a prediction length of 96. We compare the results to the DLinear model.
<img width="400" height="293" alt="image" src="https://github.com/user-attachments/assets/164531d0-04ad-4e9e-a302-00e37a636efa" />

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
4. Run training scripts in `code/`

We trained on [INSERT GPU] for approximately [X] hours to get to 100 epochs.

## Results / Insights

Results will be added to `results/` and summarized here once experiments complete.

## Conclusion

We will summarize key takeaways after the experiments and analysis are complete.

## References

- Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. _A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers_. ICLR 2023.
- Electricity Load Diagrams 2011-2014 (UCI ML Repository)
- PeMS Traffic Dataset
- Jena Weather Dataset

## Acknowledgements

This project was completed as part of **CS 4782** at Cornell University.
