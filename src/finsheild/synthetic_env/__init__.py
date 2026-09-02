"""Finsheild synthetic digital-payment environment (Phase 4).

Generates six interrelated entities — users, accounts, devices, merchants,
locations, transactions — plus an ``account_devices`` link table — and injects
eight inspectable suspicious scenarios:

1. account takeover
2. unusual amount/time
3. transaction velocity
4. new device
5. unusual location
6. device sharing
7. mule behavior
8. unusual merchant

All generation is deterministic given ``SyntheticEnvConfig(seed=...)`` and the
dataset size is configurable via ``SyntheticEnvConfig(..., n_users=...,
n_transactions=...)``.

Design notes:

* Only the ``transactions`` table carries a fraud label. Other tables are pure
  reference data — they never leak the label of a future transaction.
* Scenario tags live alongside the binary label so downstream feature
  engineering can target individual mechanisms (Phase 5+).
* The environment emits ``SyntheticEnvironment`` with six DataFrames plus a
  metadata dict so a caller can persist + reload it cheaply.
"""

from finsheild.synthetic_env.config import SyntheticEnvConfig
from finsheild.synthetic_env.entities import (
    generate_accounts,
    generate_devices,
    generate_locations,
    generate_merchants,
    generate_users,
)
from finsheild.synthetic_env.environment import (
    SCENARIO_NAMES,
    SyntheticEnvironment,
    generate_environment,
)
from finsheild.synthetic_env.links import generate_account_devices
from finsheild.synthetic_env.scenarios import SCENARIO_IMPLEMENTATIONS
from finsheild.synthetic_env.transactions import generate_transactions

__all__ = [
    "SyntheticEnvConfig",
    "SyntheticEnvironment",
    "SCENARIO_NAMES",
    "SCENARIO_IMPLEMENTATIONS",
    "generate_environment",
    "generate_users",
    "generate_accounts",
    "generate_devices",
    "generate_merchants",
    "generate_locations",
    "generate_account_devices",
    "generate_transactions",
]