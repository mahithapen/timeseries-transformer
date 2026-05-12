# Data

This project uses the following datasets referenced by the PatchTST paper:

- Electricity Load Diagrams 2011-2014 (UCI ML Repository): save as `data/electricity.csv`
  - https://drive.google.com/drive/folders/1ZhaQwLYcnhT5zEEZhBTo03-jDwl7bt7v
- PeMS Traffic Dataset (Caltrans): save as `data/traffic.csv`
  - https://drive.google.com/file/d/1U3BZ3Wvuvd9HVAx5Nl3bHYG9rsh5-yZX/view?usp=drive_link
- Jena Weather Dataset (MPI-BGC): save as `data/weather.csv`
  - https://drive.google.com/file/d/1Tc7GeVN7DLEl-RAs-JVwG9yFMf--S8dy/view?usp=drive_link

## Download Notes
- Download each dataset from the linked source and place the CSV files directly in this `data/` directory using the filenames above.
- The training scripts read numeric columns from each CSV. Non-numeric columns, such as timestamps, are ignored during model training and evaluation.
- `weather_small.csv` is a small sample file included for quick local smoke tests.
- `ithaca_weather.csv` was used for the future forecasting example shown in the poster.

## Expected Layout

After downloading the full benchmark datasets, the directory should look like:

```text
data/
  electricity.csv
  traffic.csv
  weather.csv
  weather_small.csv
  ithaca_weather.csv
  README.md
```
