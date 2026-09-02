"""Risk Fusion — Phase 10.

Combines five fraud signals (XGBoost, Anomaly, Behavioral, Graph, Rules)
into a single risk score + level + decision + evidence list.

Public surface
--------------
* :class:`RiskFusionEngine` — main engine with ``fit`` / ``predict`` / ``predict_batch``.
* :class:`RiskResult` — per-transaction output dataclass.
* :data:`DEFAULT_THRESHOLDS` — default ``{red: 0.7, yellow: 0.3}`` mapping.
* :data:`DEFAULT_WEIGHTS` — default signal weights (sums to 1).

Example
-------
>>> from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment
>>> from finsheild.features import build_features
>>> from finsheild.risk_fusion import RiskFusionEngine
>>> env = generate_environment(SyntheticEnvConfig.ci())
>>> fr = build_features(env)
>>> engine = RiskFusionEngine().fit(env, fr)
>>> result = engine.predict(fr.features.iloc[0])
>>> result.risk_level in {"GREEN", "YELLOW", "RED"}
True
"""

from finsheild.risk_fusion.engine import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    RiskFusionEngine,
    RiskResult,
)

__all__ = ["RiskFusionEngine", "RiskResult", "DEFAULT_THRESHOLDS", "DEFAULT_WEIGHTS"]
