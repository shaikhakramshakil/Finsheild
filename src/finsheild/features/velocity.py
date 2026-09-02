"""Velocity features — counts and amounts in rolling time windows.

Leakage rule
------------
For transaction at time ``t``, the "5-minute velocity" is the number of
prior transactions for the same account with ``ts ∈ (t − 300s, t)`` (a
strict past interval). The current row is excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_window_counts(tx: pd.DataFrame,
                            windows_seconds: tuple[int, ...],
                            time_col: str = "ts",
                            group_col: str = "account_id") -> pd.DataFrame:
    """Return per-transaction velocity counts/amounts at each window width.

    Output columns: ``vel_count_{W}s``, ``vel_amount_{W}s`` for each W.
    Excludes the current row from each window.
    """
    out = pd.DataFrame({"txn_id": tx["txn_id"].values})
    if len(tx) == 0:
        for w in windows_seconds:
            out[f"vel_count_{w}s"] = np.array([], dtype="int32")
            out[f"vel_amount_{w}s"] = np.array([], dtype="float32")
        return out

    df = tx[["txn_id", time_col, "amount", group_col]].copy()
    df["_orig_idx"] = np.arange(len(df))
    df = df.sort_values([group_col, time_col, "txn_id"]).reset_index(drop=True)

    for w in windows_seconds:
        counts_sorted = np.zeros(len(df), dtype="int32")
        amounts_sorted = np.zeros(len(df), dtype="float64")
        # ts dtype is datetime64[us], so w in seconds → w*1e6 microseconds
        w_us = int(w) * 1_000_000
        for gid, idx_block in df.groupby(group_col, sort=False).groups.items():
            block = df.loc[idx_block]
            ts = block[time_col].astype("int64").to_numpy()
            amt = block["amount"].to_numpy(dtype="float64")
            n = len(block)
            j = 0
            for i in range(n):
                while j < i and (ts[i] - ts[j]) >= w_us:
                    j += 1
                if i == 0:
                    counts_sorted[block.index[i]] = 0
                    amounts_sorted[block.index[i]] = 0.0
                else:
                    if j < i:
                        counts_sorted[block.index[i]] = i - j
                        amounts_sorted[block.index[i]] = amt[j:i].sum()
                    else:
                        counts_sorted[block.index[i]] = 0
                        amounts_sorted[block.index[i]] = 0.0
        # counts_sorted is indexed by sorted position; remap to original tx order
        counts_orig = np.zeros(len(df), dtype="int32")
        amounts_orig = np.zeros(len(df), dtype="float64")
        orig_idx = df["_orig_idx"].to_numpy()
        for sorted_pos in range(len(df)):
            counts_orig[orig_idx[sorted_pos]] = counts_sorted[sorted_pos]
            amounts_orig[orig_idx[sorted_pos]] = amounts_sorted[sorted_pos]
        out[f"vel_count_{w}s"] = counts_orig.astype("int32")
        out[f"vel_amount_{w}s"] = amounts_orig.astype("float32")

    return out


def build_velocity_features(tx: pd.DataFrame,
                              windows_seconds: tuple[int, ...] = (300, 3600, 86_400),
                              high_value_threshold: float = 1000.0) -> pd.DataFrame:
    """Return per-transaction velocity features.

    Output columns: ``vel_count_{W}s``, ``vel_amount_{W}s`` for each W,
    plus ``vel_high_value_count_3600s`` (count of prior high-value txns in
    the last hour).
    """
    base = _rolling_window_counts(tx, windows_seconds)

    # High-value velocity over the 1h window.
    high_value = tx[tx["amount"] > high_value_threshold]
    if len(high_value):
        hv_counts = _rolling_window_counts(
            high_value, (3600,), time_col="ts", group_col="account_id")
        hv_counts = hv_counts.rename(
            columns={"vel_count_3600s": "vel_high_value_count_3600s"})
        base = base.merge(hv_counts[["txn_id", "vel_high_value_count_3600s"]],
                           on="txn_id", how="left")
        base["vel_high_value_count_3600s"] = (
            base["vel_high_value_count_3600s"].fillna(0).astype("int32"))
    else:
        base["vel_high_value_count_3600s"] = np.int32(0)

    return base


def feature_columns(windows_seconds: tuple[int, ...]) -> list[str]:
    cols = []
    for w in windows_seconds:
        cols.append(f"vel_count_{w}s")
        cols.append(f"vel_amount_{w}s")
    cols.append("vel_high_value_count_3600s")
    return cols