"""Finsheild feature engineering (Phase 5).

Builds features on top of the synthetic environment (:mod:`finsheild.synthetic_env`)
and the existing ULB pipeline. **No future-data leakage**: every feature for a
transaction at time ``t`` is computed using only rows with ``ts <= t``.

Public entry points:

* :func:`build_features` — main orchestrator. Takes a :class:`SyntheticEnvironment`
  and returns a single feature ``pd.DataFrame`` indexed by ``txn_id``, plus a
  ``feature_columns`` list and a ``metadata`` dict.
* :class:`FeatureConfig` — knobs (velocity windows, history minimum, etc).

The feature families are organised into modules:

* :mod:`finsheild.features.transactional` — amount, hour, day, channel, merchant
  category (no leakage possible)
* :mod:`finsheild.features.behavioral` — user historical averages, deviations
* :mod:`finsheild.features.velocity` — transaction counts and amounts in
  rolling windows
* :mod:`finsheild.features.device` — known/new device indicators, account
  sharing
* :mod:`finsheild.features.location` — distance to previous location, country
  switch indicator
"""

from finsheild.features.config import FeatureConfig
from finsheild.features.engine import build_features

__all__ = ["FeatureConfig", "build_features"]