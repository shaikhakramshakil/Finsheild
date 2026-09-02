#!/usr/bin/env python3
"""Download Finsheild dataset — KaggleCreditCardFraud + synthetic fallback."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def synthetic_df(n: int = 10000, fraud_ratio: float = 0.0172, seed: int = 42) -> pd.DataFrame:
    """Generate lightweight synthetic fallback matching Expected Columns."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n_fraud = max(2, int(n * fraud_ratio))
    n_legit = n - n_fraud
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    # Time ~ uniform 0-172800 (2 days), Amount lognormal, V1-V28 ~ normal
    time_legit = rng.uniform(0, 172800, size=n_legit)
    time_fraud = rng.uniform(0, 172800, size=n_fraud)
    data = {}
    for c in cols:
        if c == "Time":
            data[c] = np.concatenate([time_legit, time_fraud])
        elif c == "Amount":
            # legit: mean ~70, fraud slightly higher
            amt_legit = rng.lognormal(mean=3.5, sigma=1.0, size=n_legit)
            amt_fraud = rng.lognormal(mean=4.0, sigma=1.2, size=n_fraud)
            data[c] = np.concatenate([amt_legit, amt_fraud])
        elif c == "Class":
            data[c] = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
        else:
            # PCA features ~ N(0,1), fraud slightly shifted
            legit = rng.normal(0, 1, size=n_legit)
            fraud = rng.normal(0.2, 1.2, size=n_fraud)
            data[c] = np.concatenate([legit, fraud])
    df = pd.DataFrame(data)
    # shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def download_kaggle(output: Path, force: bool = False) -> bool:
    """Try kagglehub -> opendatasets -> kaggle CLI. Return True if succeeded."""
    if output.exists() and not force:
        logger.info("Raw already exists at %s — skipping download (use --force to overwrite)", output)
        return True
    output.parent.mkdir(parents=True, exist_ok=True)

    # 1) kagglehub
    try:
        import kagglehub

        logger.info("Trying kagglehub download (mlg-ulb/creditcardfraud)...")
        dl_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        # kagglehub downloads to cache; locate csv
        dl_path = Path(dl_path)
        for cand in [dl_path / "creditcard.csv", dl_path / "creditcardfraud" / "creditcard.csv"]:
            if cand.exists():
                cand.rename(output) if not output.exists() else cand.replace(output)
                logger.info("Downloaded via kagglehub to %s", output)
                return True
        # fallback: search recursively
        for p in dl_path.rglob("*.csv"):
            p.replace(output)
            logger.info("Downloaded via kagglehub (found %s) to %s", p, output)
            return True
        logger.warning("kagglehub downloaded but creditcard.csv not found in %s", dl_path)
    except Exception as e:
        logger.warning("kagglehub failed: %s", e)

    # 2) opendatasets
    try:
        import opendatasets as od

        logger.info("Trying opendatasets download...")
        tmp = output.parent / "_tmp_kaggle"
        tmp.mkdir(parents=True, exist_ok=True)
        od.download("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", data_dir=str(tmp))
        for p in tmp.rglob("creditcard.csv"):
            p.replace(output)
            logger.info("Downloaded via opendatasets to %s", output)
            return True
        logger.warning("opendatasets downloaded but csv not found under %s", tmp)
    except Exception as e:
        logger.warning("opendatasets failed: %s", e)

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Finsheild dataset")
    parser.add_argument("--output", default="data/raw/creditcard.csv", help="Output CSV path")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic fallback dataset")
    parser.add_argument("--n", type=int, default=10000, help="Synthetic rows (if --synthetic)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    parser.add_argument("--dry-run", action="store_true", help="Print acquisition path without downloading")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.output)

    if args.dry_run:
        print("Acquisition path:")
        print("  1. KaggleHub: pip install kagglehub && kagglehub.dataset_download('mlg-ulb/creditcardfraud')")
        print("     Requires KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json (see https://www.kaggle.com/settings)")
        print("  2. Opendatasets: pip install opendatasets && opendatasets.download('https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud')")
        print("  3. Kaggle CLI: pip install kaggle && kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip")
        print("  4. Colab fallback URL (if dataset is made available via direct link): see docs/dataset.md and notebooks/colab/01_dataset.ipynb")
        print(f"  Output: {out}")
        print("  Synthetic fallback: python scripts/download_dataset.py --synthetic")
        sys.exit(0)

    if args.synthetic:
        df = synthetic_df(n=args.n)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Synthetic dataset written to {out} — shape {df.shape}, fraud {int(df['Class'].sum())}")
        sys.exit(0)

    ok = download_kaggle(out, force=args.force)
    if ok and out.exists():
        print(f"Dataset ready at {out} — {out.stat().st_size} bytes")
        sys.exit(0)

    print("Kaggle download failed or credentials missing. Options:")
    print("  - Set Kaggle credentials: export KAGGLE_USERNAME=... KAGGLE_KEY=...  or place ~/.kaggle/kaggle.json")
    print("  - Run synthetic fallback: python scripts/download_dataset.py --synthetic")
    print("  - See docs/dataset.md and notebooks/colab/01_dataset.ipynb for Colab instructions")
    sys.exit(2)


if __name__ == "__main__":
    main()
