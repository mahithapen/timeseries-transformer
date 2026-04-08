# PatchTST Re-implementation for Long-Term Time Series Forecasting

**Authors:** Mahitha Penmetsa (`msp259`), Aayush Agnihotri (`aa2328`), Cindy Liang (`cl2329`), Peter Bidoshi (`pjb294`)

## Introduction
This repository contains our CS 4782 final project re-implementation of **"A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers"** (ICLR 2023). The paper introduces **PatchTST**, a transformer architecture that improves long-term time series forecasting via patching and channel independence, enabling longer look-back windows with lower compute.

## Chosen Result
We aim to reproduce the **supervised and self-supervised** results reported in **Tables 3 and 4** of the paper on the **Electricity, Traffic, and Weather** datasets. These results highlight PatchTST's performance gains over prior baselines for long-term forecasting and representation learning.

## GitHub Contents
- `code/`: Re-implementation code and configs
- `data/`: Dataset download instructions and metadata
- `results/`: Output tables, plots, and logs
- `poster/`: Final poster PDF
- `report/`: Final 2-page report PDF
- `docs/`: Internal notes and checklists

## Re-implementation Details
- **Framework:** PyTorch
- **Key ideas:** channel independence, instance normalization, patching (length `P`, stride `S`), learnable positional encodings, transformer encoder, MLP head
- **Tasks:**
  - Supervised forecasting (MSE/MAE)
  - Self-supervised reconstruction with 40% patch masking
- **Resource-aware changes:** reduce attention heads (16 -> 4) and training epochs (100 -> 20)

## Reproduction Steps
1. Create and activate a Python environment
2. Install dependencies from `code/requirements.txt` (to be added)
3. Download datasets listed in `data/README.md`
4. Run training scripts in `code/` (commands to be added)

## Results / Insights
Results will be added to `results/` and summarized here once experiments complete.

## Conclusion
We will summarize key takeaways after the experiments and analysis are complete.

## References
- Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers*. ICLR 2023.
- Electricity Load Diagrams 2011-2014 (UCI ML Repository)
- PeMS Traffic Dataset
- Jena Weather Dataset

## Acknowledgements
This project was completed as part of **CS 4782** at Cornell University.
