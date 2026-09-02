"""Scoring a single transaction against a :class:`BehavioralProfile`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _get_field(row: Any, key: str, default: Any = None) -> Any:
    """Fetch ``key`` from dict / Series / object."""
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        if isinstance(row, pd.Series):
            return row.get(key, default)
        # mapping-like
        if hasattr(row, "__getitem__"):
            try:
                return row[key]
            except Exception:
                pass
        return getattr(row, key, default)
    except Exception:
        return default


def _txn_hour(row: Any) -> int | None:
    # explicit hour field takes precedence
    v = _get_field(row, "hour", None)
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    ts = _get_field(row, "ts", None)
    if ts is not None:
        try:
            ts_parsed = pd.to_datetime(ts)
            if pd.notna(ts_parsed):
                return int(ts_parsed.hour)
        except Exception:
            pass
    # fallback: hour_histogram index? no
    return None


def score_transaction(
    txn_row: Any,
    profile: Any | None,
    daily_count: int | None = None,
) -> dict:
    """Score a single transaction against a profile.

    Parameters
    ----------
    txn_row:
        Mapping / Series with at least ``amount``, ``merchant_id``,
        ``device_id``, ``location_id``, ``ts``/``hour``. Extra keys ignored.
        May also contain ``daily_count`` / ``todays_count`` for frequency
        anomaly detection.
    profile:
        :class:`BehavioralProfile` or ``None`` (unknown user).
    daily_count:
        Optional explicit count of transactions *including* this one for the
        transaction's day. If ``None``, the function looks for
        ``daily_count``/``todays_count``/``day_count`` inside ``txn_row``.
        If still ``None``, ``frequency_anomaly`` is ``False``.

    Returns
    -------
    Dict with keys:
    ``amount_zscore`` (float),
    ``is_unusual_hour`` (bool),
    ``is_new_merchant`` (bool),
    ``is_new_device`` (bool),
    ``is_new_location`` (bool),
    ``frequency_anomaly`` (bool).
    """
    amount = _get_field(txn_row, "amount", 0.0)
    try:
        amount_f = float(amount)
    except Exception:
        amount_f = 0.0

    merchant_id = _get_field(txn_row, "merchant_id", None)
    device_id = _get_field(txn_row, "device_id", None)
    location_id = _get_field(txn_row, "location_id", None)

    # Resolve daily_count from explicit param or row fields
    if daily_count is None:
        for key in ("daily_count", "todays_count", "today_count", "day_count", "day_txn_count"):
            v = _get_field(txn_row, key, None)
            if v is not None:
                try:
                    daily_count = int(v)
                    break
                except Exception:
                    continue

    if profile is None:
        # Unknown user -> everything is anomalous except frequency (needs avg)
        return {
            "amount_zscore": 0.0,
            "is_unusual_hour": True,
            "is_new_merchant": True,
            "is_new_device": True,
            "is_new_location": True,
            "frequency_anomaly": False,
        }

    # amount_zscore
    std = getattr(profile, "std_amount", 0.0) or 0.0
    mean = getattr(profile, "mean_amount", 0.0) or 0.0
    try:
        std_f = float(std)
        mean_f = float(mean)
    except Exception:
        std_f = 0.0
        mean_f = 0.0

    if std_f > 0:
        amount_zscore = float((amount_f - mean_f) / std_f)
    else:
        amount_zscore = 0.0
    # guard NaN/inf
    if not np.isfinite(amount_zscore):
        amount_zscore = 0.0

    # is_unusual_hour: hour not in top 50% (top 12) of histogram.
    # Interpretation: the 12 most frequent hours are "usual". If a user's
    # history is concentrated in a few hours, hours with zero count are always
    # unusual (otherwise ties among zeros would make them seem usual).
    hour = _txn_hour(txn_row)
    hist = getattr(profile, "hour_histogram", None)
    if hour is None or hist is None:
        is_unusual_hour = False
    else:
        try:
            hist_arr = np.asarray(hist, dtype=np.int64)
            if hist_arr.shape != (24,):
                tmp = np.zeros(24, dtype=np.int64)
                upto = min(len(hist_arr), 24)
                tmp[:upto] = hist_arr[:upto]
                hist_arr = tmp
            if hist_arr.sum() == 0:
                is_unusual_hour = True
            elif hist_arr[int(hour)] == 0:
                # never seen this hour before -> unusual regardless of top-12 tie
                is_unusual_hour = True
            else:
                order = np.lexsort((np.arange(24), -hist_arr))
                top_hours = set(int(x) for x in order[:12])
                is_unusual_hour = bool(int(hour) not in top_hours)
        except Exception:
            is_unusual_hour = False

    # is_new_merchant: not in common_merchants (top-3)
    common = getattr(profile, "common_merchants", []) or []
    try:
        common_set = set(common)
    except Exception:
        common_set = set()
    if merchant_id is None:
        is_new_merchant = False
    else:
        is_new_merchant = bool(merchant_id not in common_set)

    # is_new_device / is_new_location
    known_devices = getattr(profile, "known_devices", set()) or set()
    known_locations = getattr(profile, "known_locations", set()) or set()
    try:
        known_devices = set(known_devices)
    except Exception:
        known_devices = set()
    try:
        known_locations = set(known_locations)
    except Exception:
        known_locations = set()

    is_new_device = bool(device_id is not None and device_id not in known_devices)
    is_new_location = bool(location_id is not None and location_id not in known_locations)

    # frequency_anomaly: daily_count > 2 * avg_daily_frequency
    avg_freq = getattr(profile, "avg_daily_frequency", 0.0) or 0.0
    try:
        avg_f = float(avg_freq)
    except Exception:
        avg_f = 0.0

    if daily_count is None or avg_f <= 0:
        # if daily_count not provided, we cannot claim anomaly
        # special case: if we can infer daily_count as 1 and avg is very small, still false
        frequency_anomaly = False
    else:
        try:
            dc = int(daily_count)
            frequency_anomaly = bool(dc > 2 * avg_f)
        except Exception:
            frequency_anomaly = False

    return {
        "amount_zscore": float(amount_zscore),
        "is_unusual_hour": bool(is_unusual_hour),
        "is_new_merchant": bool(is_new_merchant),
        "is_new_device": bool(is_new_device),
        "is_new_location": bool(is_new_location),
        "frequency_anomaly": bool(frequency_anomaly),
    }
