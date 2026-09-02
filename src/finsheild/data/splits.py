"""Train/validation/test splits with stratification and leakage notes."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

TARGET_COL = "Class"


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")
    if not (0 < test_size < 1) or not (0 < val_size < 1):
        raise ValueError("test_size and val_size must be in (0,1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    strat = df[target_col] if stratify else None
    if stratify:
        min_class_count = df[target_col].value_counts().min()
        if min_class_count < 2:
            raise ValueError(
                f"Stratified split requires at least 2 samples per class, got min {min_class_count}"
            )

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )
    val_proportion = val_size / (1 - test_size)
    strat2 = train_val[target_col] if stratify else None
    train, val = train_test_split(
        train_val, test_size=val_proportion, random_state=random_state, stratify=strat2
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    logger.info(
        "Splits: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%) — total %d",
        len(train),
        len(train) / len(df) * 100,
        len(val),
        len(val) / len(df) * 100,
        len(test),
        len(test) / len(df) * 100,
        len(df),
    )
    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str | Path = "data/processed",
    fmt: str = "csv",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        if fmt == "parquet":
            p = out_dir / f"{name}.parquet"
            frame.to_parquet(p, index=False)
        else:
            p = out_dir / f"{name}.csv"
            frame.to_csv(p, index=False)
        paths[name] = p
        logger.info("Saved %s split to %s (%d rows)", name, p, len(frame))
    return paths
