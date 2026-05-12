# Results

`results.csv` contains the final test-set MAE and MSE values from our DLinear, PatchTST, and hierarchical PatchTST runs on Weather, Electricity, and Traffic.

The log files contain the Colab cell outputs from model training:

- `weather_logs`
- `electricity_logs`
- `traffic_logs`

The checkpoint paths in `results.csv` reflect the original Google Drive paths used during Colab training. To reproduce locally, train with the commands in the root `README.md`; checkpoints will be written under the local `checkpoints/` directory.
