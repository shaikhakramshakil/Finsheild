"""Risk Fusion Engine — Phase 10.

Combines five signals into a single risk score:

1. XGBoost fraud score (trained on Phase-5 features)
2. Anomaly score (Isolation Forest from Phase 7)
3. Behavioral signals (heuristic + profile-based from Phase 6)
4. Graph signals (device/account sharing proxy; optional real graph)
5. Rule signals (deterministic checks from Phase 8)

Fusion is a weighted sum of normalized signals in [0, 1].
Thresholds map the continuous risk_score to discrete risk_level and
decision.

Design notes
------------
* ``fit(env, feature_result)`` is the primary entry point. ``env`` is a
  ``SyntheticEnvironment`` and ``feature_result`` is a
  ``FeatureBuildResult``. Both are required for training but
  ``predict`` only needs a single feature row (dict/Series/DataFrame).
* Graph signals gracefully degrade when ``finsheild.graph`` is absent
  (Phase 9 not yet landed). A heuristic proxy derived from
  ``device_account_count`` / ``vel_count_300s`` is used instead.
* All ``NaN`` / ``inf`` values are imputed with training medians so
  scoring never crashes on new-user rows.
* Evidence strings mirror the spec examples: ``"high_velocity: 5 txns
  in 5min"``, ``"shared_device: 3 accounts"``, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from finsheild.anomaly.detector import AnomalyDetector
from finsheild.rules.engine import RuleEngine

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "red": 0.7,
    "yellow": 0.3,
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "xgb": 0.35,
    "anomaly": 0.20,
    "behavioral": 0.15,
    "graph": 0.10,
    "rule": 0.20,
}

# severity -> weight for rule_score normalisation
_SEVERITY_WEIGHTS: Dict[str, float] = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(row: Any, key: str, default: Any = None) -> Any:
    """Safe accessor for dict / Series / object."""
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        if isinstance(row, pd.Series):
            # Series.get handles missing keys
            return row.get(key, default)
        if hasattr(row, "__getitem__"):
            try:
                return row[key]  # type: ignore[index]
            except Exception:
                pass
        return getattr(row, key, default)
    except Exception:
        return default


def _is_nan(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd is not None and pd.isna(v):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    try:
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


def _get_amount(row: Any) -> float:
    """Extract transaction amount from a feature row with fallback."""
    for key in ("amount", "amount_x", "amount_y"):
        v = _get(row, key, None)
        if v is not None and not _is_nan(v):
            try:
                return float(v)
            except Exception:
                continue
    # fallback: exp of amount_log
    v = _get(row, "amount_log", None)
    if v is not None and not _is_nan(v):
        try:
            return float(math.expm1(float(v)))
        except Exception:
            pass
    return 0.0


def _normalize_thresholds(user: Dict[str, float] | None) -> Dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS)
    if user:
        for k, v in user.items():
            if k in out:
                out[k] = float(v)
            else:
                # allow alternative names red_threshold etc?
                lk = k.lower()
                if "red" in lk:
                    out["red"] = float(v)
                elif "yellow" in lk:
                    out["yellow"] = float(v)
                else:
                    out[k] = float(v)
    # ensure red > yellow
    if out["red"] <= out["yellow"]:
        # clamp to keep ordering; spec assumes red > yellow
        # nudge red slightly above yellow
        out["red"] = min(1.0, out["yellow"] + 0.1)
    return out


def _normalize_weights(user: Dict[str, float] | None) -> Dict[str, float]:
    out = dict(DEFAULT_WEIGHTS)
    if user:
        for k, v in user.items():
            out[k] = float(v)
    # normalise to sum 1
    total = sum(out.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        for k in out:
            out[k] = out[k] / total
    return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """Output of :meth:`RiskFusionEngine.predict`."""

    risk_score: float
    risk_level: str  # GREEN / YELLOW / RED
    decision: str  # APPROVE / STEP_UP / BLOCK / INVESTIGATE
    evidence: List[str] = field(default_factory=list)
    # detail breakdown for explainability (optional)
    xgb_score: float = 0.0
    anomaly_score: float = 0.0
    behavioral_score: float = 0.0
    graph_score: float = 0.0
    rule_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "evidence": list(self.evidence),
            "xgb_score": self.xgb_score,
            "anomaly_score": self.anomaly_score,
            "behavioral_score": self.behavioral_score,
            "graph_score": self.graph_score,
            "rule_score": self.rule_score,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RiskFusionEngine:
    """Fuse five fraud signals into a single risk decision.

    Parameters
    ----------
    thresholds:
        Mapping with ``red`` and ``yellow`` keys. Defaults to
        ``{red: 0.7, yellow: 0.3}``. ``red`` must be > ``yellow``.
    weights:
        Mapping with ``xgb``, ``anomaly``, ``behavioral``, ``graph``,
        ``rule`` keys. Will be normalised to sum to 1.
    xgboost_params:
        Optional overrides passed to ``XGBClassifier`` constructor.
    anomaly_params:
        Optional overrides for ``AnomalyDetector`` (``contamination``,
        ``random_state``).
    random_state:
        RNG seed for both models.
    xgb_model:
        Optional pre-trained XGBoost model. When supplied, ``fit`` will
        not retrain it.
    anomaly_detector:
        Optional pre-trained ``AnomalyDetector``.
    """

    def __init__(
        self,
        thresholds: Dict[str, float] | None = None,
        weights: Dict[str, float] | None = None,
        xgboost_params: Dict[str, Any] | None = None,
        anomaly_params: Dict[str, Any] | None = None,
        random_state: int = 42,
        xgb_model: Any | None = None,
        anomaly_detector: AnomalyDetector | None = None,
    ) -> None:
        self.thresholds: Dict[str, float] = _normalize_thresholds(thresholds)
        self.weights: Dict[str, float] = _normalize_weights(weights)
        self.random_state = int(random_state)
        self.xgboost_params = dict(xgboost_params) if xgboost_params else {}
        self.anomaly_params = dict(anomaly_params) if anomaly_params else {}
        self.xgb_model = xgb_model
        self.anomaly_detector: AnomalyDetector | None = anomaly_detector
        self.rule_engine = RuleEngine()
        self.feature_columns: List[str] | None = None
        self._medians: np.ndarray | None = None
        self._is_fitted: bool = False
        # try to load graph module lazily (Phase 9)
        self._graph_available: bool = False
        self._graph_scorer: Any | None = None
        try:
            import finsheild.graph as _graph_mod  # type: ignore

            # probe for expected symbols; if missing we stay fallback
            self._graph_available = True
            self._graph_scorer = _graph_mod
        except ImportError:
            self._graph_available = False
            self._graph_scorer = None
        except Exception:
            self._graph_available = False
            self._graph_scorer = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, env, feature_result) -> "RiskFusionEngine":
        """Train XGBoost + IsolationForest on the provided data.

        Parameters
        ----------
        env:
            ``SyntheticEnvironment`` (needed for behavioral context &
            optional graph construction). May be ``None`` when caller
            supplies pre-trained models — in that case behavioral proxy
            is used.
        feature_result:
            ``FeatureBuildResult`` or ``pd.DataFrame`` containing
            feature columns and ``label_fraud``.

        Returns
        -------
        self
        """
        # ------------------------------------------------------------------
        # Extract feature matrix + labels
        # ------------------------------------------------------------------
        if hasattr(feature_result, "features") and hasattr(feature_result, "feature_columns"):
            feature_columns = list(feature_result.feature_columns)  # type: ignore[attr-defined]
            df_features: pd.DataFrame = feature_result.features  # type: ignore[attr-defined]
        elif isinstance(feature_result, pd.DataFrame):
            # assume feature columns are all except known non-feature keys
            non_feature = {"txn_id", "account_id", "ts", "label_fraud", "scenario_tag", "amount", "amount_x", "amount_y", "merchant_id", "tx_country", "prev_location_id"}
            feature_columns = [c for c in feature_result.columns if c not in non_feature]
            df_features = feature_result
        else:
            raise TypeError("feature_result must be FeatureBuildResult or DataFrame")

        self.feature_columns = list(feature_columns)

        # Need label_fraud column
        if "label_fraud" in df_features.columns:
            y = df_features["label_fraud"].to_numpy(dtype=np.int64)
        elif hasattr(feature_result, "y") and callable(getattr(feature_result, "y")):
            y = feature_result.y()  # type: ignore[attr-defined]
            if y is None:
                raise ValueError("No label_fraud found in feature_result")
        else:
            raise ValueError("No label_fraud column found in feature_result")

        X_raw = df_features[self.feature_columns].to_numpy(dtype="float64")
        # Convert inf -> nan for median computation
        X_raw[~np.isfinite(X_raw)] = np.nan
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X_raw, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        self._medians = medians

        # Impute
        X = X_raw.copy()
        mask = np.isnan(X)
        if np.any(mask):
            rows, cols = np.where(mask)
            X[rows, cols] = medians[cols]

        # ------------------------------------------------------------------
        # XGBoost
        # ------------------------------------------------------------------
        if self.xgb_model is None:
            try:
                import xgboost as xgb  # type: ignore
            except ImportError as e:
                raise ImportError("xgboost is required for RiskFusionEngine.fit") from e

            params = dict(
                n_estimators=self.xgboost_params.get("n_estimators", 30),
                max_depth=self.xgboost_params.get("max_depth", 4),
                learning_rate=self.xgboost_params.get("learning_rate", 0.15),
                subsample=self.xgboost_params.get("subsample", 0.8),
                colsample_bytree=self.xgboost_params.get("colsample_bytree", 0.8),
                eval_metric=self.xgboost_params.get("eval_metric", "logloss"),
                random_state=self.random_state,
                n_jobs=self.xgboost_params.get("n_jobs", 1),
                verbosity=self.xgboost_params.get("verbosity", 0),
                tree_method=self.xgboost_params.get("tree_method", "hist"),
            )
            # allow overrides to replace defaults
            for k, v in self.xgboost_params.items():
                if k not in params:
                    params[k] = v
            self.xgb_model = xgb.XGBClassifier(**params)
            self.xgb_model.fit(X, y)
        else:
            # ensure medians are still required for predict path
            pass

        # ------------------------------------------------------------------
        # Anomaly detector
        # ------------------------------------------------------------------
        if self.anomaly_detector is None:
            contamination = float(self.anomaly_params.get("contamination", 0.05))
            rs = int(self.anomaly_params.get("random_state", self.random_state))
            self.anomaly_detector = AnomalyDetector(contamination=contamination, random_state=rs)
            # Train only on legit rows
            legit_mask = y == 0
            if np.sum(legit_mask) == 0:
                # fallback: train on all if no legit (degenerate test env)
                legit_mask = np.ones(len(y), dtype=bool)
            X_legit = X[legit_mask]
            # Use DataFrame for fit so detector captures feature_columns
            # but we already have medians; detector will recompute its own
            # medians internally — that's fine. Pass array.
            self.anomaly_detector.fit(X_legit, feature_columns=self.feature_columns)
        else:
            if self.anomaly_detector.feature_columns is None:
                self.anomaly_detector.feature_columns = list(self.feature_columns)

        # ------------------------------------------------------------------
        # Behavioral profiles (optional, stored for completeness)
        # ------------------------------------------------------------------
        self._env = env
        self._profiles: Dict[int, Any] | None = None
        if env is not None and hasattr(env, "transactions") and hasattr(env, "accounts"):
            try:
                from finsheild.behavioral.profile import build_profiles

                self._profiles = build_profiles(env.transactions, env.accounts)
                # also build account->user map for fast lookup
                try:
                    self._acct_to_user = env.accounts.set_index("account_id")["user_id"].to_dict()
                except Exception:
                    self._acct_to_user = {}
            except Exception:
                self._profiles = None
                self._acct_to_user = {}

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Internal signal computers
    # ------------------------------------------------------------------ #

    def _vector_for_row(self, row: Any) -> np.ndarray:
        """Return 2-D array shape (1, n_features) for a single row."""
        if self.feature_columns is None or self._medians is None:
            raise RuntimeError("Engine not fitted — call fit() first.")
        if isinstance(row, pd.DataFrame):
            if len(row) == 0:
                raise ValueError("Empty DataFrame passed to _vector_for_row")
            row = row.iloc[0]
        # Extract values in feature_columns order
        vals: List[float] = []
        for col in self.feature_columns:
            v = _get(row, col, 0.0)
            if _is_nan(v):
                v = np.nan
            try:
                fv = float(v)
            except Exception:
                fv = np.nan
            if not math.isfinite(fv):
                fv = np.nan
            vals.append(fv)
        arr = np.array(vals, dtype="float64").reshape(1, -1)
        # Impute with medians
        medians = self._medians
        if medians is not None and medians.shape[0] == arr.shape[1]:
            mask = np.isnan(arr)
            if np.any(mask):
                rows, cols = np.where(mask)
                arr[rows, cols] = medians[cols]
            # also handle inf already converted to nan
        else:
            arr = np.where(np.isnan(arr), 0.0, arr)
        return arr

    def _xgb_score(self, row: Any) -> float:
        if self.xgb_model is None or not self._is_fitted:
            return 0.0
        try:
            X = self._vector_for_row(row)
            proba = self.xgb_model.predict_proba(X)
            # XGB binary returns shape (n,2) or (n,) depending on version?
            if proba.ndim == 2 and proba.shape[1] == 2:
                return float(np.clip(proba[0, 1], 0.0, 1.0))
            elif proba.ndim == 1:
                return float(np.clip(proba[0], 0.0, 1.0))
            else:
                return float(np.clip(proba[0], 0.0, 1.0))
        except Exception:
            return 0.0

    def _anomaly_score(self, row: Any) -> float:
        if self.anomaly_detector is None or not self._is_fitted:
            return 0.0
        try:
            X = self._vector_for_row(row)
            s = self.anomaly_detector.score_samples(X)
            return float(np.clip(s[0], 0.0, 1.0))
        except Exception:
            return 0.0

    def _behavioral_score(self, row: Any) -> float:
        """Heuristic behavioral score in [0,1] derived from feature flags."""
        score = 0.0
        # amount_zscore
        z = _get(row, "amount_zscore", 0)
        if _is_nan(z):
            z = 0
        try:
            zf = float(z)
        except Exception:
            zf = 0.0
        if not math.isfinite(zf):
            zf = 0.0
        az = abs(zf)
        if az > 3:
            score += 0.4
        elif az > 2:
            score += 0.25
        elif az > 1:
            score += 0.10

        if int(_get(row, "is_new_device", 0) or 0) == 1:
            score += 0.20
        if int(_get(row, "is_unusual_location", 0) or 0) == 1:
            score += 0.20
        if int(_get(row, "is_new_user", 0) or 0) == 1:
            score += 0.15
        if int(_get(row, "country_switch", 0) or 0) == 1:
            score += 0.15
        if int(_get(row, "device_is_shared", 0) or 0) == 1:
            score += 0.10
        if int(_get(row, "is_offhours", 0) or 0) == 1:
            score += 0.05
        # cap
        return float(min(score, 1.0))

    def _graph_score(self, row: Any) -> float:
        """Graph proxy score in [0,1]."""
        # If real graph module available, try to delegate
        if self._graph_available and self._graph_scorer is not None:
            try:
                # Graph module API is not yet frozen; attempt common names
                # Try scorer.score(row) or graph_score(row) etc.
                for attr in ("score_transaction", "score_row", "graph_score", "compute_graph_score"):
                    fn = getattr(self._graph_scorer, attr, None)
                    if callable(fn):
                        val = fn(row)
                        if isinstance(val, dict):
                            # pick numeric
                            for kk in ("graph_score", "score", "risk"):
                                if kk in val:
                                    return float(np.clip(float(val[kk]), 0.0, 1.0))
                        else:
                            return float(np.clip(float(val), 0.0, 1.0))
            except Exception:
                pass
            # fall through to heuristic if delegation failed

        # Heuristic proxy: device sharing + velocity burst
        dac = _get(row, "device_account_count", 1)
        try:
            dac_i = int(dac)
        except Exception:
            dac_i = 1
        vel = _get(row, "vel_count_300s", 0)
        try:
            vel_i = int(vel)
        except Exception:
            vel_i = 0

        # device sharing is the strongest graph signal
        if dac_i >= 4:
            base = 0.9
        elif dac_i == 3:
            base = 0.7
        elif dac_i == 2:
            base = 0.5
        else:
            base = 0.0

        # velocity amplifies
        if vel_i >= 8:
            base = max(base, 0.85)
        elif vel_i >= 5:
            base = max(base, 0.6)

        # prior_unique_countries as proxy for location graph spread
        puc = _get(row, "prior_unique_countries", 0)
        try:
            puc_i = int(puc)
            if puc_i >= 3:
                base = min(1.0, base + 0.15)
            elif puc_i >= 2:
                base = min(1.0, base + 0.08)
        except Exception:
            pass

        return float(np.clip(base, 0.0, 1.0))

    def _rule_score(self, row: Any) -> tuple[float, List[str]]:
        """Return (rule_score, list_of_triggered_rule_ids)."""
        try:
            results = self.rule_engine.evaluate(row)
        except Exception:
            return 0.0, []
        total = 0.0
        triggered: List[str] = []
        for r in results:
            if r.triggered:
                w = _SEVERITY_WEIGHTS.get(r.severity, 0.5)
                total += w
                triggered.append(r.rule_id)
        # max possible = sum of all severity weights
        max_total = sum(_SEVERITY_WEIGHTS.get(r.severity, 0.5) for r in self.rule_engine.rules)
        if max_total <= 0:
            max_total = len(self.rule_engine.rules) * 0.5
        score = total / max_total if max_total > 0 else 0.0
        return float(np.clip(score, 0.0, 1.0)), triggered

    def _build_evidence(
        self,
        row: Any,
        xgb_s: float,
        ano_s: float,
        beh_s: float,
        graph_s: float,
        rule_s: float,
        triggered_rules: List[str],
    ) -> List[str]:
        ev: List[str] = []

        # Velocity
        vel = _get(row, "vel_count_300s", 0)
        try:
            vel_i = int(vel)
            if vel_i >= 5:
                ev.append(f"high_velocity: {vel_i} txns in 5min")
            if vel_i >= 8:
                ev.append(f"burst_velocity: {vel_i} txns in 5min")
        except Exception:
            pass

        # Shared device
        dac = _get(row, "device_account_count", None)
        is_shared = int(_get(row, "device_is_shared", 0) or 0) == 1
        if is_shared and dac is not None:
            try:
                ev.append(f"shared_device: {int(dac)} accounts")
            except Exception:
                ev.append("shared_device")
        elif is_shared:
            ev.append("shared_device")

        # New device
        if int(_get(row, "is_new_device", 0) or 0) == 1:
            ndhv_amt = _get_amount(row)
            if ndhv_amt > 500:
                ev.append(f"new_device_high_value: amount={ndhv_amt:.2f}")
            else:
                ev.append("new_device")

        # Unusual location
        if int(_get(row, "is_unusual_location", 0) or 0) == 1:
            # include country switch distance if available
            dist = _get(row, "distance_to_prev_km", None)
            if dist is not None and not _is_nan(dist):
                try:
                    ev.append(f"unusual_location: distance={float(dist):.1f}km")
                except Exception:
                    ev.append("unusual_location")
            else:
                ev.append("unusual_location")

        # Country switch
        if int(_get(row, "country_switch", 0) or 0) == 1:
            if "unusual_location" not in " ".join(ev):
                ev.append("country_switch")

        # Unusual amount
        z = _get(row, "amount_zscore", 0)
        try:
            zf = float(z)
            if not _is_nan(zf) and abs(zf) > 2:
                ev.append(f"unusual_amount: z={zf:.1f}")
        except Exception:
            pass

        # Offhours high value
        if int(_get(row, "is_offhours", 0) or 0) == 1:
            amt = _get_amount(row)
            if amt > 1000:
                ev.append(f"offhours_high_value: amount={amt:.2f}")

        # Rule triggers (add any not already covered verbosely)
        for rid in triggered_rules:
            # avoid duplicate if already added with richer text
            if rid == "high_velocity" and any("high_velocity" in e for e in ev):
                continue
            if rid == "shared_device" and any("shared_device" in e for e in ev):
                continue
            if rid == "new_device" and any("new_device" in e for e in ev):
                continue
            if rid == "unusual_location" and any("unusual_location" in e for e in ev):
                continue
            if rid == "unusual_amount" and any("unusual_amount" in e for e in ev):
                continue
            if rid == "offhours_high_value" and any("offhours_high_value" in e for e in ev):
                continue
            if rid == "burst_velocity" and any("burst_velocity" in e for e in ev):
                continue
            # add generic rule evidence
            ev.append(f"rule:{rid}")

        # ML flagged
        if xgb_s > 0.6:
            ev.append(f"ml_flagged: fraud_prob={xgb_s:.2f}")

        # Anomalous pattern
        if ano_s > 0.6:
            ev.append(f"anomalous_pattern: score={ano_s:.2f}")

        # Graph high
        if graph_s > 0.5:
            if not any("shared_device" in e for e in ev):
                ev.append(f"graph_risk: score={graph_s:.2f}")

        # Behavioral new user
        if int(_get(row, "is_new_user", 0) or 0) == 1:
            if "new_device" not in " ".join(ev):
                ev.append("new_user")

        # Deduplicate preserving order
        seen = set()
        deduped: List[str] = []
        for e in ev:
            if e not in seen:
                deduped.append(e)
                seen.add(e)

        # Ensure non-empty for high risk: if risk would be elevated but evidence empty, add fallback
        # (handled by caller after risk_score computed)

        return deduped

    # ------------------------------------------------------------------ #
    # Public predict
    # ------------------------------------------------------------------ #

    def predict(self, feature_row: Any) -> RiskResult:
        """Score a single feature row.

        Parameters
        ----------
        feature_row:
            Dict-like row (``dict``, ``pd.Series``, or single-row
            ``pd.DataFrame``) containing at least the feature columns
            and optionally ``amount`` / ``amount_x``.

        Returns
        -------
        RiskResult
        """
        if not self._is_fitted:
            raise RuntimeError("RiskFusionEngine not fitted — call fit() first.")

        # Normalize DataFrame single row -> Series
        if isinstance(feature_row, pd.DataFrame):
            if len(feature_row) == 0:
                raise ValueError("Empty DataFrame passed to predict")
            feature_row = feature_row.iloc[0]

        xgb_s = self._xgb_score(feature_row)
        ano_s = self._anomaly_score(feature_row)
        beh_s = self._behavioral_score(feature_row)
        graph_s = self._graph_score(feature_row)
        rule_s, triggered = self._rule_score(feature_row)

        # Weighted fusion
        w = self.weights
        risk_score = (
            w["xgb"] * xgb_s
            + w["anomaly"] * ano_s
            + w["behavioral"] * beh_s
            + w["graph"] * graph_s
            + w["rule"] * rule_s
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # Evidence
        evidence = self._build_evidence(feature_row, xgb_s, ano_s, beh_s, graph_s, rule_s, triggered)

        # Ensure evidence non-empty for any suspicious signal:
        # if risk > yellow threshold but evidence empty, add generic
        red = float(self.thresholds["red"])
        yellow = float(self.thresholds["yellow"])
        if not evidence and risk_score > yellow:
            evidence.append(f"elevated_risk: score={risk_score:.2f}")
        # Also if any individual signal high but evidence still empty (e.g. pure ml_flagged missed due to threshold)
        if not evidence and (xgb_s > 0.5 or ano_s > 0.5 or beh_s > 0.3 or graph_s > 0.3 or rule_s > 0.2):
            evidence.append(f"elevated_risk: score={risk_score:.2f}")

        # Risk level
        if risk_score > red:
            level = "RED"
        elif risk_score > yellow:
            level = "YELLOW"
        else:
            level = "GREEN"

        # Decision
        if level == "RED":
            amt = _get_amount(feature_row)
            if amt > 1000:
                decision = "BLOCK"
            else:
                decision = "INVESTIGATE"
        elif level == "YELLOW":
            decision = "STEP_UP"
        else:
            decision = "APPROVE"

        return RiskResult(
            risk_score=risk_score,
            risk_level=level,
            decision=decision,
            evidence=evidence,
            xgb_score=xgb_s,
            anomaly_score=ano_s,
            behavioral_score=beh_s,
            graph_score=graph_s,
            rule_score=rule_s,
        )

    def _prepare_batch_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Return imputed matrix for *df* aligned to ``feature_columns``."""
        if self.feature_columns is None or self._medians is None:
            raise RuntimeError("Engine not fitted — call fit() first.")
        # Ensure all feature columns exist; missing -> fill 0
        cols = self.feature_columns
        # Build matrix column-wise to handle missing cols
        arrays = []
        for col in cols:
            if col in df.columns:
                vals = df[col].to_numpy(dtype="float64")
            else:
                vals = np.zeros(len(df), dtype="float64")
            # Convert non-finite to nan for later imputation
            vals = vals.astype("float64")
            vals[~np.isfinite(vals)] = np.nan
            arrays.append(vals)
        X = np.column_stack(arrays) if arrays else np.zeros((len(df), 0), dtype="float64")
        # Impute with training medians
        medians = self._medians
        if medians is not None and medians.shape[0] == X.shape[1]:
            # vectorised imputation
            # Use masking
            nan_mask = np.isnan(X)
            if np.any(nan_mask):
                rows, ccols = np.where(nan_mask)
                X[rows, ccols] = medians[ccols]
        else:
            X = np.where(np.isnan(X), 0.0, X)
        return X

    def predict_batch(self, feature_df: pd.DataFrame) -> List[RiskResult]:
        """Score a DataFrame of feature rows.

        Parameters
        ----------
        feature_df:
            ``pd.DataFrame`` with one row per transaction. Must contain
            the feature columns used at ``fit`` time. May also be a
            ``FeatureBuildResult`` (extracts ``.features``).

        Returns
        -------
        List[RiskResult]
            One result per row, in input order.
        """
        if not self._is_fitted:
            raise RuntimeError("RiskFusionEngine not fitted — call fit() first.")

        # Allow FeatureBuildResult passthrough
        if hasattr(feature_df, "features") and hasattr(feature_df, "feature_columns"):
            df: pd.DataFrame = feature_df.features  # type: ignore[attr-defined]
        elif isinstance(feature_df, pd.DataFrame):
            df = feature_df
        else:
            raise TypeError("predict_batch expects DataFrame or FeatureBuildResult")

        if len(df) == 0:
            return []

        # Vectorised XGBoost + Anomaly scores
        n = len(df)
        X_batch = self._prepare_batch_matrix(df)
        # XGBoost scores
        if self.xgb_model is not None:
            try:
                proba = self.xgb_model.predict_proba(X_batch)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    xgb_scores = np.clip(proba[:, 1], 0.0, 1.0).astype(float)
                elif proba.ndim == 1:
                    xgb_scores = np.clip(proba, 0.0, 1.0).astype(float)
                else:
                    # fallback: take last column
                    xgb_scores = np.clip(proba[:, -1], 0.0, 1.0).astype(float)
            except Exception:
                xgb_scores = np.zeros(n, dtype=float)
        else:
            xgb_scores = np.zeros(n, dtype=float)

        # Anomaly scores
        if self.anomaly_detector is not None:
            try:
                anomaly_scores = self.anomaly_detector.score_samples(X_batch)
                anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0).astype(float)
            except Exception:
                anomaly_scores = np.zeros(n, dtype=float)
        else:
            anomaly_scores = np.zeros(n, dtype=float)

        results: List[RiskResult] = []
        w = self.weights
        red = float(self.thresholds["red"])
        yellow = float(self.thresholds["yellow"])
        for idx, (_, row) in enumerate(df.iterrows()):
            xgb_s = float(xgb_scores[idx])
            ano_s = float(anomaly_scores[idx])
            beh_s = self._behavioral_score(row)
            graph_s = self._graph_score(row)
            rule_s, triggered = self._rule_score(row)

            risk_score = float(np.clip(
                w["xgb"] * xgb_s
                + w["anomaly"] * ano_s
                + w["behavioral"] * beh_s
                + w["graph"] * graph_s
                + w["rule"] * rule_s,
                0.0, 1.0,
            ))

            evidence = self._build_evidence(row, xgb_s, ano_s, beh_s, graph_s, rule_s, triggered)
            if not evidence and risk_score > yellow:
                evidence.append(f"elevated_risk: score={risk_score:.2f}")
            if not evidence and (xgb_s > 0.5 or ano_s > 0.5 or beh_s > 0.3 or graph_s > 0.3 or rule_s > 0.2):
                evidence.append(f"elevated_risk: score={risk_score:.2f}")

            if risk_score > red:
                level = "RED"
            elif risk_score > yellow:
                level = "YELLOW"
            else:
                level = "GREEN"

            if level == "RED":
                amt = _get_amount(row)
                decision = "BLOCK" if amt > 1000 else "INVESTIGATE"
            elif level == "YELLOW":
                decision = "STEP_UP"
            else:
                decision = "APPROVE"

            results.append(
                RiskResult(
                    risk_score=risk_score,
                    risk_level=level,
                    decision=decision,
                    evidence=evidence,
                    xgb_score=xgb_s,
                    anomaly_score=ano_s,
                    behavioral_score=beh_s,
                    graph_score=graph_s,
                    rule_score=rule_s,
                )
            )
        return results

    # Convenience alias: allow engine(df) style?
    __call__ = predict_batch
