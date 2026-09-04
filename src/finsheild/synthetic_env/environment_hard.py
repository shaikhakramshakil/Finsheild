"""Hard-overlap variant of the synthetic environment.

This module reuses the same entity generators (``generate_users``,
``generate_accounts``, …) from :mod:`finsheild.synthetic_env.entities`
to keep the underlying reference tables identical, then mixes the
existing background transaction distribution with the weak-signal
fraud scenarios from :mod:`finsheild.synthetic_env.scenarios_hard`.

The output is identical in schema to the easy synthetic environment
(seven tables, same columns, same dtypes), so feature engineering and
downstream ML code work unchanged. The only thing that changes is the
fraud distribution.

The :func:`generate_hard_overlap_environment` function is the public
entry point.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from finsheild.synthetic_env.config import SyntheticEnvConfig
from finsheild.synthetic_env.entities import (
    generate_accounts,
    generate_devices,
    generate_locations,
    generate_merchants,
    generate_users,
)
from finsheild.synthetic_env.environment import SyntheticEnvironment
from finsheild.synthetic_env.links import generate_account_devices
from finsheild.synthetic_env.rng import make_random, make_rng
from finsheild.synthetic_env.scenarios_hard import (
    HardFraud,
    build_hard_overlap_events,
)


def _build_background(
    config: SyntheticEnvConfig,
    accounts: pd.DataFrame,
    users: pd.DataFrame,
    devices: pd.DataFrame,
    merchants: pd.DataFrame,
    locations: pd.DataFrame,
    account_devices: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    rng = make_rng(config.seed, "hard_overlap_bg")
    rnd = make_random(config.seed, "hard_overlap_bg_random")
    start = pd.Timestamp(config.start_ts)
    end = start + pd.Timedelta(days=config.time_span_days)
    total_seconds = config.total_seconds

    account_status = accounts.set_index("account_id")["status"]
    active_ids = account_status[account_status == "active"].index.to_numpy()
    if len(active_ids) == 0:
        active_ids = accounts["account_id"].to_numpy()
    acc_choices = rng.choice(active_ids, size=n, replace=True)

    acc_dev_map = account_devices.groupby("account_id")["device_id"].apply(
        lambda s: s.to_numpy()).to_dict()
    dev_choices = np.empty(n, dtype=np.int64)
    for i, acc_id in enumerate(acc_choices):
        devs = acc_dev_map.get(int(acc_id))
        if devs is None or len(devs) == 0:
            dev_choices[i] = int(rng.integers(1, len(devices) + 1))
        else:
            dev_choices[i] = int(rng.choice(devs))

    # Legit merchants occasionally visit high-risk too (overlap)
    merchant_probs = np.full(len(merchants), 0.7)
    mask = merchants["risk_band"].to_numpy() == "high"
    merchant_probs[mask] = 0.3
    merchant_probs = merchant_probs / merchant_probs.sum()
    merchant_choices = rng.choice(merchants["merchant_id"].to_numpy(),
                                   size=n, replace=True, p=merchant_probs)

    user_country = users.set_index("user_id")["home_country"]
    location_country = locations.set_index("location_id")["country"]
    country_locs = {}
    for loc_id, country in location_country.items():
        country_locs.setdefault(country, []).append(int(loc_id))
    acc_user_map = accounts.set_index("account_id")["user_id"]
    loc_choices = np.empty(n, dtype=np.int64)
    for i, acc_id in enumerate(acc_choices):
        user_id = acc_user_map.get(int(acc_id))
        home = user_country.get(user_id, "US")
        pool = country_locs.get(home, country_locs["US"])
        # Legitimate users travel sometimes: 70% home, 30% anywhere
        if rng.random() < 0.7:
            loc_choices[i] = int(rng.choice(pool))
        else:
            loc_choices[i] = int(rng.choice(locations["location_id"].to_numpy()))

    # Timestamp distribution: same hourly modulation (legit users do 2-5 AM rarely)
    base_offsets = rng.integers(0, total_seconds, size=n)
    hour_weights = np.array([
        0.4, 0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.6, 1.0, 1.2, 1.4, 1.5,
        1.6, 1.4, 1.3, 1.4, 1.5, 1.6, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hour_choices = rng.choice(24, size=n, p=hour_weights)
    base_ts = start + pd.to_timedelta(base_offsets, unit="s")
    sampled_hour_ts = base_ts.floor("D") + pd.to_timedelta(
        hour_choices * 3600 + rng.integers(0, 3600, size=n), unit="s")
    sampled_hour_ts = np.where(pd.to_datetime(sampled_hour_ts) > end, end,
                                 sampled_hour_ts)

    amounts = np.round(rng.lognormal(mean=3.6, sigma=0.95, size=n), 2)

    return pd.DataFrame({
        "txn_id": np.arange(1, n + 1, dtype=np.int64),
        "account_id": acc_choices.astype(np.int64),
        "device_id": dev_choices.astype(np.int64),
        "merchant_id": merchant_choices.astype(np.int64),
        "location_id": loc_choices.astype(np.int64),
        "ts": pd.to_datetime(sampled_hour_ts),
        "amount": amounts.astype(float),
        "currency": "USD",
        "channel": rnd.choices(["online", "pos", "atm", "mobile"], k=n),
        "status": "settled",
        "scenario_tag": "background",
        "label_fraud": 0,
        "scenario_extra": "",
    })


def _fraud_to_df(events: List[HardFraud]) -> pd.DataFrame:
    rows = []
    for e in events:
        d = e.__dict__.copy()
        d["scenario_extra"] = str(d.pop("extra"))
        rows.append(d)
    return pd.DataFrame(rows)


def generate_hard_overlap_environment(
    config: SyntheticEnvConfig | None = None,
    n_per_scenario: int = 30,
) -> SyntheticEnvironment:
    """Generate the hard-overlap variant of the synthetic environment.

    Background uses the *same* easy-environment distribution so that
    legitimate behaviour is unchanged. Fraud rows are produced by the
    weak-signal scenarios. The total fraud count is roughly
    ``n_per_scenario * 5``; combined with the background count, the
    resulting fraud prevalence is ~1% when ``config.n_transactions`` is
    around 8 000–10 000.
    """
    if config is None:
        config = SyntheticEnvConfig.ci()

    users = generate_users(config)
    locations = generate_locations(config)
    devices = generate_devices(config)
    merchants = generate_merchants(config)
    accounts = generate_accounts(config, users)
    account_devices = generate_account_devices(config, accounts, devices)

    background = _build_background(
        config, accounts, users, devices, merchants, locations,
        account_devices, n=config.n_transactions,
    )

    fraud_events = build_hard_overlap_events(
        config, accounts, users, devices, merchants, locations,
        account_devices, n_per_scenario=n_per_scenario,
    )
    fraud_df = _fraud_to_df(fraud_events)
    if fraud_df.empty:
        full = background
    else:
        # Deduplicate (account_id, ts) by ±2s jitter
        bg_keys = set(zip(background["account_id"].astype(int).tolist(),
                          background["ts"].astype("int64").tolist()))
        for i, row in fraud_df.iterrows():
            for delta in (0, 1, -1, 2, -2, 3, -3):
                cand = row["ts"] + pd.Timedelta(seconds=delta)
                if (int(row["account_id"]), int(cand.value)) not in bg_keys:
                    fraud_df.at[i, "ts"] = cand
                    bg_keys.add((int(row["account_id"]), int(cand.value)))
                    break
        full = pd.concat([background, fraud_df], ignore_index=True, sort=False)

    full = full.sort_values("ts").reset_index(drop=True)
    full["txn_id"] = np.arange(1, len(full) + 1, dtype=np.int64)
    full = full[["txn_id", "account_id", "device_id", "merchant_id",
                 "location_id", "ts", "amount", "currency", "channel",
                 "status", "scenario_tag", "label_fraud", "scenario_extra"]]

    # Wrap into the same SyntheticEnvironment container as the easy env.
    return SyntheticEnvironment(
        config=config,
        users=users,
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        locations=locations,
        account_devices=account_devices,
        transactions=full,
        metadata={
            "variant": "hard_overlap",
            "n_per_scenario": n_per_scenario,
            "scenario_names": list({e.scenario_tag for e in fraud_events}),
        },
    )
