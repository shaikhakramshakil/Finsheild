"""Link-table generators.

Currently only ``account_devices`` — the many-to-many relationship between
accounts and devices. This is the foundation for:

* new-device detection (Phase 5+)
* device sharing / mule rings (Phase 8/9)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finsheild.synthetic_env.config import SyntheticEnvConfig
from finsheild.synthetic_env.rng import make_rng


def generate_account_devices(config: SyntheticEnvConfig,
                              accounts: pd.DataFrame,
                              devices: pd.DataFrame) -> pd.DataFrame:
    """Generate ``account_devices`` link rows.

    Columns: ``account_id, device_id, first_used_ts, last_used_ts,
    is_primary``.

    Each account gets 1-3 devices. The primary device is the one used
    earliest; a small share of accounts (the mule ring anchor) shares a
    device with several other accounts — see ``DEVICE_SHARE_FRACTION``.
    """
    n_accounts = len(accounts)
    n_devices = len(devices)
    if n_accounts == 0 or n_devices == 0:
        return pd.DataFrame(columns=[
            "account_id", "device_id", "first_used_ts", "last_used_ts",
            "is_primary"])

    rng = make_rng(config.seed, "account_devices")
    rows = []

    # Most accounts: 1 device. A minority: 2-3 devices. A small pool:
    # accounts share devices with one or two other accounts.
    device_pool_size = max(8, n_devices // 50)
    shared_device_pool = rng.choice(n_devices, size=device_pool_size,
                                    replace=False) + 1  # device_id starts at 1

    for i, acc_row in accounts.iterrows():
        acc_id = int(acc_row["account_id"])
        n_acc_devices = int(rng.integers(1, 4))  # 1, 2 or 3 devices
        chosen = []
        # 1) primary device — random
        chosen.append(int(rng.integers(1, n_devices + 1)))
        # 2) optional secondary / tertiary
        for _ in range(n_acc_devices - 1):
            cand = int(rng.integers(1, n_devices + 1))
            if cand not in chosen:
                chosen.append(cand)
        # 3) with small probability, replace one with a shared device
        if rng.random() < 0.04 and n_acc_devices > 1 and len(shared_device_pool) > 0:
            shared = int(rng.choice(shared_device_pool))
            if shared not in chosen:
                chosen[-1] = shared

        # Timestamps: each device has a first_used within first 60 days,
        # and a last_used within the env's tx window.
        first_used_offsets = rng.integers(0, 60 * 86_400, size=n_acc_devices)
        last_used_offsets = rng.integers(0, config.total_seconds,
                                          size=n_acc_devices)
        first_used_ts = pd.Timestamp(config.start_ts) \
            - pd.to_timedelta(first_used_offsets, unit="s")
        last_used_ts = pd.Timestamp(config.start_ts) \
            + pd.to_timedelta(last_used_offsets, unit="s")

        for k, dev_id in enumerate(chosen):
            rows.append({
                "account_id": acc_id,
                "device_id": dev_id,
                "first_used_ts": first_used_ts[k],
                "last_used_ts": last_used_ts[k],
                "is_primary": k == 0,
            })

    return pd.DataFrame(rows)