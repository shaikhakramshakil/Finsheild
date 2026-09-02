"""Device features — known-vs-new device and account-sharing counts.

Leakage rule
------------
For transaction at time ``t``:
* ``is_new_device`` is True if the (account_id, device_id) pair has no
  prior transaction in ``account_devices`` with ``first_used_ts < t`` and
  the account is not linked to this device in the link table at all.
* ``device_account_count`` is the number of distinct accounts currently
  using the same device_id, computed from the entire link table (link
  table is a static reference, not a time series).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_device_features(tx: pd.DataFrame,
                           account_devices: pd.DataFrame) -> pd.DataFrame:
    """Return per-transaction device features.

    Output columns: ``txn_id, is_new_device, device_account_count,
    device_is_shared, is_primary_device_for_account``.
    """
    # 1) device_account_count: how many accounts share each device
    dev_account_count = account_devices.groupby("device_id")["account_id"] \
        .nunique().rename("device_account_count").reset_index()
    dev_account_count["device_account_count"] = \
        dev_account_count["device_account_count"].astype("int32")

    # 2) is_primary_device_for_account: per (account, device) whether the
    #    device is the primary for that account.
    primary_map = account_devices[account_devices["is_primary"]][
        ["account_id", "device_id"]].assign(is_primary_device_for_account=1)
    primary_map["is_primary_device_for_account"] = primary_map[
        "is_primary_device_for_account"].astype("int8")

    # 3) is_new_device: at time t, has the (account, device) pair been used
    #    before? Check against the account_devices link table (which is a
    #    static snapshot) — but only the row's first_used_ts, NOT t.
    #    Rule: if the link table contains (account_id, device_id) with
    #    first_used_ts < t → known; else → new.
    df = tx[["txn_id", "account_id", "device_id", "ts"]].copy()
    df = df.merge(dev_account_count, on="device_id", how="left")
    df = df.merge(primary_map, on=["account_id", "device_id"], how="left")
    df["is_primary_device_for_account"] = (
        df["is_primary_device_for_account"].fillna(0).astype("int8"))
    df["device_account_count"] = (
        df["device_account_count"].fillna(1).astype("int32"))
    df["device_is_shared"] = (df["device_account_count"] >= 2).astype("int8")

    # For is_new_device, check the account_devices link table:
    # the device must exist in the link AND first_used_ts < ts
    ad = account_devices[["account_id", "device_id", "first_used_ts"]] \
        .rename(columns={"first_used_ts": "ad_first_used_ts"})
    df = df.merge(ad, on=["account_id", "device_id"], how="left")
    # If no link row at all, the device is unknown — but per Phase 4 FK
    # integrity it always exists in the link. So is_new_device = 1 iff
    # the link's first_used_ts is strictly AFTER the txn ts.
    df["is_new_device"] = (df["ad_first_used_ts"] > df["ts"]).astype("int8")
    # Default: if no link row at all (shouldn't happen), treat as new
    df.loc[df["ad_first_used_ts"].isna(), "is_new_device"] = 1
    df = df.drop(columns=["ad_first_used_ts"])

    return df[["txn_id", "is_new_device", "device_account_count",
               "device_is_shared", "is_primary_device_for_account"]]


FEATURE_COLUMNS = ["is_new_device", "device_account_count", "device_is_shared",
                    "is_primary_device_for_account"]