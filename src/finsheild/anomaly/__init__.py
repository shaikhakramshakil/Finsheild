"""FinSheild anomaly detection (Phase 7).

Public surface
--------------
* :class:`AnomalyDetector` — Isolation Forest wrapper with ``[0, 1]`` scores.
* :func:`train_anomaly_detector` — fit on legit/background transactions.
* :func:`score_transactions` — score any feature matrix / build result.

Both helpers accept the Phase-5 :class:`FeatureBuildResult` so callers
can write::

    from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment
    from finsheild.features import build_features
    from finsheild.anomaly import train_anomaly_detector, score_transactions

    env = generate_environment(SyntheticEnvConfig.ci())
    result = build_features(env)
    detector = train_anomaly_detector(result)
    scores = score_transactions(detector, result)  # np.ndarray in [0, 1]
"""

from finsheild.anomaly.detector import AnomalyDetector, score_transactions, train_anomaly_detector

__all__ = ["AnomalyDetector", "train_anomaly_detector", "score_transactions"]
