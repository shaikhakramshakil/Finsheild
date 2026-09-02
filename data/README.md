# Data layout

```
data/
  raw/                # gitignored — raw CSV (creditcard.csv) fetched via scripts/download_dataset.py or Colab notebook
  processed/          # gitignored — train.csv, val.csv, test.csv + scaler.joblib generated after preprocessing
  sample_creditcard.csv  # optional small checked-in sample (not required; tests use data/raw if present)
```

- Raw dataset is NOT committed. Acquire via `python scripts/download_dataset.py --help` or `notebooks/colab/01_dataset.ipynb`.
- Processed splits are generated locally/Colab: `data/processed/train.csv` (70%), `val.csv` (15%), `test.csv` (15%) with `scaler.joblib`.
- Synthetic fallback: `python scripts/download_dataset.py --synthetic` creates a lightweight CSV for CI without Kaggle creds.
