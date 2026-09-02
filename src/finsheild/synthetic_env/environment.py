"""Top-level orchestrator for the synthetic environment.

Exposes :func:`generate_environment` which returns a :class:`SyntheticEnvironment`
— a small dataclass that bundles all six DataFrames plus generation metadata.
The orchestrator is responsible for the *order* of generation so referential
integrity is satisfied:

1. ``users``        — no dependencies
2. ``locations``    — no dependencies
3. ``devices``      — no dependencies
4. ``merchants``    — no dependencies
5. ``accounts``     — depends on ``users``
6. ``account_devices`` — depends on ``accounts`` and ``devices``
7. ``transactions`` — depends on every other table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from finsheild.synthetic_env.config import SyntheticEnvConfig
from finsheild.synthetic_env.entities import (
    generate_accounts,
    generate_devices,
    generate_locations,
    generate_merchants,
    generate_users,
)
from finsheild.synthetic_env.links import generate_account_devices
from finsheild.synthetic_env.scenarios import SCENARIO_NAMES
from finsheild.synthetic_env.transactions import generate_transactions


@dataclass
class SyntheticEnvironment:
    """Container for all six tables plus generation metadata."""

    config: SyntheticEnvConfig
    users: pd.DataFrame
    accounts: pd.DataFrame
    devices: pd.DataFrame
    merchants: pd.DataFrame
    locations: pd.DataFrame
    account_devices: pd.DataFrame
    transactions: pd.DataFrame

    metadata: Dict = field(default_factory=dict)

    @property
    def tables(self) -> Dict[str, pd.DataFrame]:
        return {
            "users": self.users,
            "accounts": self.accounts,
            "devices": self.devices,
            "merchants": self.merchants,
            "locations": self.locations,
            "account_devices": self.account_devices,
            "transactions": self.transactions,
        }

    def fraud_rate(self) -> float:
        if len(self.transactions) == 0:
            return 0.0
        return float(self.transactions["label_fraud"].mean())

    def scenario_breakdown(self) -> pd.DataFrame:
        """Per-scenario counts (flagged + context) for QA inspection."""
        if len(self.transactions) == 0:
            return pd.DataFrame(columns=["scenario_tag", "n_total", "n_fraud"])
        grp = self.transactions.groupby("scenario_tag", dropna=False)
        out = grp.size().rename("n_total").to_frame()
        out["n_fraud"] = grp["label_fraud"].sum()
        return out.reset_index().sort_values("scenario_tag").reset_index(drop=True)


def generate_environment(config: SyntheticEnvConfig | None = None) -> SyntheticEnvironment:
    """Generate the full synthetic environment deterministically."""
    if config is None:
        config = SyntheticEnvConfig.dev()

    users = generate_users(config)
    locations = generate_locations(config)
    devices = generate_devices(config)
    merchants = generate_merchants(config)
    accounts = generate_accounts(config, users)
    account_devices = generate_account_devices(config, accounts, devices)
    transactions = generate_transactions(
        config=config,
        accounts=accounts,
        users=users,
        devices=devices,
        merchants=merchants,
        locations=locations,
        account_devices=account_devices,
    )

    metadata = {
        "scenario_names": list(SCENARIO_NAMES),
        "n_tables": 7,
        "config": config.to_dict(),
        "fraud_rate": (
            float(transactions["label_fraud"].mean())
            if len(transactions) else 0.0),
    }
    return SyntheticEnvironment(
        config=config,
        users=users,
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        locations=locations,
        account_devices=account_devices,
        transactions=transactions,
        metadata=metadata,
    )