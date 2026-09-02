"""Transaction generator.

Produces a single ``transactions`` DataFrame whose rows are the union of:

* a background stream of legitimate activity (the bulk)
* ``scenario_events`` produced by each scenario callable in
  :mod:`finsheild.synthetic_env.scenarios`

Background and scenario rows are deduplicated by ``(account_id, ts)``;
collisions are resolved by jittering the timestamp by ±1 second. The
generator never invents a ``fraud_label`` outside of the scenarios
provided.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd

from finsheild.synthetic_env.config import SyntheticEnvConfig
from finsheild.synthetic_env.scenarios import (
    SCENARIO_IMPLEMENTATIONS,
    ScenarioContext,
    ScenarioEvent,
)


def _build_scenario_context(config: SyntheticEnvConfig,
                             accounts: pd.DataFrame,
                             users: pd.DataFrame,
                             devices: pd.DataFrame,
                             merchants: pd.DataFrame,
                             locations: pd.DataFrame,
                             account_devices: pd.DataFrame):
    """Construct a ScenarioContext with table-specific RNGs."""
    from finsheild.synthetic_env.rng import make_random, make_rng

    return ScenarioContext(
        accounts=accounts,
        users=users,
        devices=devices,
        merchants=merchants,
        locations=locations,
        account_devices=account_devices,
        rng=make_rng(config.seed, "scenarios_pooled"),
        rnd=make_random(config.seed, "scenarios_pooled_random"),
        config_obj=config,
        start_ts=pd.Timestamp(config.start_ts),
        end_ts=pd.Timestamp(config.start_ts)
        + pd.Timedelta(days=config.time_span_days),
    )


def _scenario_event_to_dict(ev: ScenarioEvent) -> dict:
    d = asdict(ev)
    d["scenario_extra"] = str(d.pop("extra"))
    return d


def generate_transactions(config: SyntheticEnvConfig,
                           accounts: pd.DataFrame,
                           users: pd.DataFrame,
                           devices: pd.DataFrame,
                           merchants: pd.DataFrame,
                           locations: pd.DataFrame,
                           account_devices: pd.DataFrame) -> pd.DataFrame:
    """Generate the unified ``transactions`` table.

    Output columns
    --------------
    txn_id, account_id, device_id, merchant_id, location_id, ts, amount,
    currency, channel, status, scenario_tag, label_fraud, scenario_extra

    ``label_fraud`` is 1 only for the *flagged* transaction of a scenario
    episode. Surrounding context transactions carry a scenario tag but
    ``label_fraud=0`` — that matches real-world fraud where a scam sits in
    a stream of normal-looking activity.
    """
    if len(accounts) == 0:
        return pd.DataFrame(columns=[
            "txn_id", "account_id", "device_id", "merchant_id", "location_id",
            "ts", "amount", "currency", "channel", "status",
            "scenario_tag", "label_fraud", "scenario_extra"])

    rng = make_rng_factory(config.seed, "transactions_background")()
    rnd = make_random_factory(config.seed, "transactions_background_random")()

    start = pd.Timestamp(config.start_ts)
    end = start + pd.Timedelta(days=config.time_span_days)
    total_seconds = config.total_seconds

    n = config.n_transactions
    # Background legitimate transactions.
    # Account distribution: active accounts get more txns.
    account_status = accounts.set_index("account_id")["status"]
    active_ids = account_status[account_status == "active"].index.to_numpy()
    if len(active_ids) == 0:
        active_ids = accounts["account_id"].to_numpy()
    weights = np.ones(len(active_ids))
    acc_choices = rng.choice(active_ids, size=n, replace=True)

    # Account -> devices (sampled from account_devices)
    acc_dev_map = account_devices.groupby("account_id")["device_id"].apply(
        lambda s: s.to_numpy()).to_dict()
    dev_choices = np.empty(n, dtype=np.int64)
    for i, acc_id in enumerate(acc_choices):
        devs = acc_dev_map.get(int(acc_id))
        if devs is None or len(devs) == 0:
            dev_choices[i] = int(rng.integers(1, len(devices) + 1))
        else:
            dev_choices[i] = int(rng.choice(devs))

    # Merchants weighted toward low/medium risk.
    merchant_weights = np.where(merchants["risk_band"] == "high", 0.05,
                                 1.0)
    merchant_probs = merchant_weights / merchant_weights.sum()
    merchant_choices = rng.choice(merchants["merchant_id"].to_numpy(),
                                   size=n, replace=True, p=merchant_probs)

    # Locations: bias toward user's home country.
    user_country = users.set_index("user_id")["home_country"]
    location_country = locations.set_index("location_id")["country"]
    country_locs = {}
    for loc_id, country in location_country.items():
        country_locs.setdefault(country, []).append(int(loc_id))
    loc_choices = np.empty(n, dtype=np.int64)
    for i, acc_id in enumerate(acc_choices):
        user_id = accounts.set_index("account_id")["user_id"].get(int(acc_id))
        home = user_country.get(user_id, "US")
        pool = country_locs.get(home, country_locs["US"])
        # 90% home country, 10% any
        if rng.random() < 0.9:
            loc_choices[i] = int(rng.choice(pool))
        else:
            loc_choices[i] = int(rng.choice(locations["location_id"].to_numpy()))

    # Timestamps: uniform over window, with mild hourly modulation so it
    # isn't completely uniform (peaks at lunch + evening).
    base_offsets = rng.integers(0, total_seconds, size=n)
    hour_weights = np.array([
        0.4, 0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.6, 1.0, 1.2, 1.4, 1.5,
        1.6, 1.4, 1.3, 1.4, 1.5, 1.6, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hour_choices = rng.choice(24, size=n, p=hour_weights)
    # Replace hour-of-day component with sampled hour, keep date.
    base_ts = start + pd.to_timedelta(base_offsets, unit="s")
    sampled_hour_ts = base_ts.floor("D") + pd.to_timedelta(
        hour_choices * 3600 + rng.integers(0, 3600, size=n), unit="s")
    # ensure still inside window
    sampled_hour_ts = np.where(sampled_hour_ts > end, end, sampled_hour_ts)

    amounts = np.round(rng.lognormal(mean=3.6, sigma=0.95, size=n), 2)

    bg = pd.DataFrame({
        "txn_id": np.arange(1, n + 1, dtype=np.int64),
        "account_id": acc_choices.astype(np.int64),
        "device_id": dev_choices.astype(np.int64),
        "merchant_id": merchant_choices.astype(np.int64),
        "location_id": loc_choices.astype(np.int64),
        "ts": pd.to_datetime(sampled_hour_ts),
        "amount": amounts.astype(float),
        "currency": "USD",
        "channel": rnd.choices(["online", "pos", "atm", "mobile"],
                                k=n),
        "status": "settled",
        "scenario_tag": "background",
        "label_fraud": 0,
        "scenario_extra": "",
    })

    # Scenario events.
    ctx = _build_scenario_context(config, accounts, users, devices,
                                   merchants, locations, account_devices)
    events: List[ScenarioEvent] = []
    for fn in SCENARIO_IMPLEMENTATIONS:
        events.extend(fn(ctx))

    scenario_df = pd.DataFrame([_scenario_event_to_dict(e) for e in events])
    if scenario_df.empty:
        return bg

    # Deduplicate (account_id, ts) by jittering scenario ts ±1s.
    bg_keys = set(zip(bg["account_id"].astype(int).tolist(),
                      bg["ts"].astype("int64").tolist()))
    for i, row in scenario_df.iterrows():
        for delta in (0, 1, -1, 2, -2):
            cand = row["ts"] + pd.Timedelta(seconds=delta)
            if (int(row["account_id"]), int(cand.value)) not in bg_keys:
                scenario_df.at[i, "ts"] = cand
                bg_keys.add((int(row["account_id"]), int(cand.value)))
                break

    # Assign new txn_ids after dedup, sort by ts.
    scenario_df = scenario_df.drop(columns=["txn_id"], errors="ignore")
    full = pd.concat([bg, scenario_df], ignore_index=True, sort=False)
    full["txn_id"] = np.arange(1, len(full) + 1, dtype=np.int64)
    full = full.sort_values("ts").reset_index(drop=True)
    full["txn_id"] = np.arange(1, len(full) + 1, dtype=np.int64)
    return full


def make_rng_factory(seed: int, salt: str):
    """Closure factory so we can build a numpy RNG by name without circular import in ``transactions``."""
    from finsheild.synthetic_env.rng import make_rng

    def _factory():
        return make_rng(seed, salt)

    return _factory


def make_random_factory(seed: int, salt: str):
    """Same, for ``random.Random``."""
    from finsheild.synthetic_env.rng import make_random

    def _factory():
        return make_random(seed, salt)

    return _factory