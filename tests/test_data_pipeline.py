"""Tests for Phase 1 dataset pipeline — at least 6 checks."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from finsheild.data.loader import EXPECTED_COLUMNS, load_raw, validate_schema
from finsheild.data.preprocessing import FraudPreprocessor
from finsheild.data.splits import make_splits

RAW = Path("data/raw/creditcard.csv")


@pytest.fixture(scope="module")
def df():
    assert RAW.exists(), f"Raw dataset missing at {RAW}; run: python scripts/download_dataset.py --synthetic"
    return load_raw(RAW)


def test_loader_returns_expected_columns(df):
    assert list(df.columns) == EXPECTED_COLUMNS or set(df.columns) == set(EXPECTED_COLUMNS)


def test_loader_class_column_present_and_binary(df):
    assert "Class" in df.columns
    vals = set(df["Class"].unique())
    assert vals.issubset({0, 1}), f"Unexpected class values {vals}"


def test_validate_schema_raises_on_missing_column(df):
    bad = df.drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_schema(bad)


def test_splits_disjoint(df):
    # Check disjoint by construction using a unique row id (handles duplicate values in real ULB)
    df_with_id = df.copy()
    df_with_id["_row_id"] = range(len(df_with_id))
    train, val, test = make_splits(df_with_id, test_size=0.15, val_size=0.15, random_state=42)
    assert len(train) + len(val) + len(test) == len(df)
    assert len(set(train["_row_id"]) & set(val["_row_id"])) == 0, "Train and val overlap"
    assert len(set(train["_row_id"]) & set(test["_row_id"])) == 0, "Train and test overlap"
    assert len(set(val["_row_id"]) & set(test["_row_id"])) == 0, "Val and test overlap"
    # Also ensure no duplicated ids across splits
    assert len(train["_row_id"]) + len(val["_row_id"]) + len(test["_row_id"]) == len(df)


def test_stratification_preserved(df):
    train, val, test = make_splits(df, test_size=0.15, val_size=0.15, random_state=42)
    # overall fraud rate
    overall_rate = df["Class"].mean()
    for name, split in [("train", train), ("val", val), ("test", test)]:
        rate = split["Class"].mean()
        # Allow ±1% absolute tolerance for small dataset
        assert abs(rate - overall_rate) < 0.01, f"{name} rate {rate:.4f} differs from overall {overall_rate:.4f}"


def test_no_leakage_scaler_fitted_only_on_train(df):
    train, val, test = make_splits(df, test_size=0.15, val_size=0.15, random_state=42)
    pre = FraudPreprocessor(scale_features=["Amount", "Time"])
    train_t = pre.fit_transform_train(train)
    val_t = pre.transform(val)
    test_t = pre.transform(test)

    # Scaler should NOT be fitted on full data
    import numpy as np

    full_scaler = StandardScaler().fit(df[["Amount", "Time"]])
    assert not np.allclose(pre.scaler.mean_, full_scaler.mean_), "Scaler appears fitted on full data (leakage)"
    # Train transformed mean approx 0
    assert abs(train_t["Amount"].mean()) < 0.01
    assert abs(train_t["Time"].mean()) < 0.01


def test_time_not_leaked_without_documentation(df):
    # Ensure docs note time-based split alternative
    doc = Path("docs/dataset.md")
    assert doc.exists(), "docs/dataset.md missing"
    text = doc.read_text()
    assert "Time" in text, "docs must discuss Time feature"
    assert "time-based" in text.lower() or "temporal" in text.lower(), "docs must mention time-based split alternative"


def test_download_script_help_and_dry_run():
    help_out = subprocess.check_output([sys.executable, "scripts/download_dataset.py", "--help"], text=True)
    assert "synthetic" in help_out.lower()
    dry = subprocess.check_output([sys.executable, "scripts/download_dataset.py", "--dry-run"], text=True)
    assert "kaggle" in dry.lower()


def test_metrics_and_report_exist():
    metrics_path = Path("evaluation/metrics.json")
    report_path = Path("evaluation/reports/dataset_report.md")
    assert metrics_path.exists(), "metrics.json missing — run pipeline to generate"
    assert report_path.exists(), "dataset_report.md missing"
    import json

    data = json.loads(metrics_path.read_text())
    assert "dataset_shape" in data or "dataset_rows" in data
    assert "class_counts" in data
    assert "split_sizes" in data
