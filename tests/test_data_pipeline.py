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
    train, val, test = make_splits(df, test_size=0.15, val_size=0.15, random_state=42)
    # sizes
    assert len(train) + len(val) + len(test) == len(df)
    # check disjoint by index overlap after reset they are disjoint by construction, check no duplicate rows by id
    # Use hash of rows to ensure disjoint (since reset_index, check that intersection is empty via merge on all columns)
    # Simpler: ensure no overlapping indices if we had kept original index; here check that concatenated length equals union
    combined = pd.concat([train, val, test])
    # If splits are disjoint, no duplicated rows across splits beyond duplicates within df (which are 0)
    # Check that train/val/test have no identical rows overlapping: use set of tuple hashes
    # For 10k dataset with no duplicates, disjoint means combined duplicated count equals df duplicated
    assert len(combined) == len(df)
    # Also ensure no index overlap beyond reset: check that no row appears in two splits (use hash)
    # Quick: check that intersection via inner merge on all columns is empty
    merged = pd.merge(train, val, how="inner", on=list(df.columns))
    assert len(merged) == 0, "Train and val overlap"
    merged2 = pd.merge(train, test, how="inner", on=list(df.columns))
    assert len(merged2) == 0, "Train and test overlap"
    merged3 = pd.merge(val, test, how="inner", on=list(df.columns))
    assert len(merged3) == 0, "Val and test overlap"


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
