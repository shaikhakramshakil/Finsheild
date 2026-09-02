"""Behavioral profile per user.

Each :class:`BehavioralProfile` aggregates a user's historical transactions
(no leakage — all rows with ``ts < build time`` are used; callers build from
a snapshot that already excludes future rows, so a simple groupby suffices).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BehavioralProfile:
    """Per-user behavioral summary.

    Attributes
    ----------
    user_id:
        Owner of the profile.
    mean_amount:
        Mean transaction amount.
    std_amount:
        Sample standard deviation of amount (0 when ``txn_count < 2``).
    median_amount:
        Median transaction amount.
    txn_count:
        Total number of historical transactions.
    hour_histogram:
        Length-24 array of transaction counts per hour (0-23).
    common_merchants:
        Top-3 most frequent merchant_ids (fewer if user has <3 distinct).
    known_devices:
        Set of device_ids seen in history.
    known_locations:
        Set of location_ids seen in history.
    avg_daily_frequency:
        Average transactions per day (``txn_count / n_days``).
    """

    user_id: int
    mean_amount: float
    std_amount: float
    median_amount: float
    txn_count: int
    hour_histogram: np.ndarray = field(repr=False)
    common_merchants: list = field(default_factory=list)
    known_devices: set = field(default_factory=set)
    known_locations: set = field(default_factory=set)
    avg_daily_frequency: float = 0.0


def _hour_histogram(hours: pd.Series) -> np.ndarray:
    hist = np.zeros(24, dtype=np.int64)
    if len(hours) == 0:
        return hist
    counts = hours.value_counts()
    for hr, cnt in counts.items():
        try:
            h = int(hr)
        except Exception:
            continue
        if 0 <= h < 24:
            hist[h] = int(cnt)
    return hist


def build_profiles(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
) -> dict[int, BehavioralProfile]:
    """Build :class:`BehavioralProfile` for each user.

    Parameters
    ----------
    transactions:
        Must contain ``account_id``, ``amount``, ``ts``, ``merchant_id``,
        ``device_id``, ``location_id``. Extra columns are ignored.
    accounts:
        Must contain ``account_id``, ``user_id``. Maps each transaction's
        account to its owning user.

    Returns
    -------
    Dict mapping ``user_id -> BehavioralProfile``.

    Notes
    -----
    All history passed in is used (caller is responsible for ensuring
    ``ts < profile build time`` if a temporal cut-off is required).
    This function itself does a plain groupby — no temporal filtering.
    """
    if transactions is None or len(transactions) == 0:
        return {}
    if accounts is None or len(accounts) == 0:
        return {}
    if "account_id" not in transactions.columns or "user_id" not in accounts.columns:
        return {}

    # Map account -> user
    acct_to_user = accounts.set_index("account_id")["user_id"].to_dict()

    tx = transactions.copy()
    tx["user_id"] = tx["account_id"].map(acct_to_user)
    # drop rows where account has no user mapping
    tx = tx.dropna(subset=["user_id"])
    if len(tx) == 0:
        return {}

    # normalize user_id to int if possible
    try:
        tx["user_id"] = tx["user_id"].astype(int)
    except Exception:
        pass

    # ensure ts is datetime for hour + daily frequency
    if "ts" in tx.columns:
        try:
            tx["_ts"] = pd.to_datetime(tx["ts"])
        except Exception:
            tx["_ts"] = tx["ts"]
    else:
        tx["_ts"] = pd.NaT

    # fallback hour extraction: use _ts else try hour column
    if "_ts" in tx.columns and pd.api.types.is_datetime64_any_dtype(tx["_ts"]):
        tx["_hour"] = tx["_ts"].dt.hour
    elif "hour" in tx.columns:
        tx["_hour"] = tx["hour"].astype(int)
    else:
        tx["_hour"] = 0

    profiles: dict[int, BehavioralProfile] = {}

    grouped = tx.groupby("user_id", sort=False)

    for user_id, grp in grouped:
        amounts = grp["amount"].astype(float) if "amount" in grp.columns else pd.Series(dtype=float)
        txn_count = int(len(grp))

        if txn_count == 0:
            mean_amount = 0.0
            std_amount = 0.0
            median_amount = 0.0
        else:
            mean_amount = float(amounts.mean())
            median_amount = float(amounts.median()) if len(amounts) else 0.0
            if txn_count >= 2:
                std_val = float(amounts.std(ddof=1))
                std_amount = 0.0 if np.isnan(std_val) else std_val
            else:
                std_amount = 0.0

        hist = _hour_histogram(grp["_hour"])

        # top-3 merchants
        if "merchant_id" in grp.columns and txn_count > 0:
            common_merchants = grp["merchant_id"].value_counts().head(3).index.tolist()
            # ensure python ints
            common_merchants = [int(x) if isinstance(x, (np.integer,)) else x for x in common_merchants]
        else:
            common_merchants = []

        if "device_id" in grp.columns:
            known_devices = set(grp["device_id"].dropna().tolist())
            # normalize int types
            known_devices = {int(x) if isinstance(x, np.integer) else x for x in known_devices}
        else:
            known_devices = set()

        if "location_id" in grp.columns:
            known_locations = set(grp["location_id"].dropna().tolist())
            known_locations = {int(x) if isinstance(x, np.integer) else x for x in known_locations}
        else:
            known_locations = set()

        # avg daily frequency: txn_count / n_days
        try:
            if "_ts" in grp.columns and pd.api.types.is_datetime64_any_dtype(grp["_ts"]):
                dates = grp["_ts"].dt.floor("D")
                if len(dates) > 0 and dates.notna().any():
                    n_days = int((dates.max() - dates.min()).days) + 1
                    n_days = max(1, n_days)
                else:
                    n_days = 1
            else:
                n_days = 1
        except Exception:
            n_days = 1

        avg_daily_frequency = float(txn_count) / float(n_days) if n_days else float(txn_count)

        # ensure user_id is int key
        try:
            uid = int(user_id)
        except Exception:
            uid = user_id

        profiles[uid] = BehavioralProfile(
            user_id=uid,
            mean_amount=mean_amount,
            std_amount=std_amount,
            median_amount=median_amount,
            txn_count=txn_count,
            hour_histogram=hist,
            common_merchants=list(common_merchants),
            known_devices=set(known_devices),
            known_locations=set(known_locations),
            avg_daily_frequency=avg_daily_frequency,
        )

    return profiles
