# Synthetic Hard Overlap — Methodology & Results

**Variant:** `synthetic_hard_overlap` — new dataset, does not overwrite `easy` or `1% diluted`.  
**Status:** Experiment to test robustness via realistic feature overlap, not to manufacture a metric.

## 1. Dataset Generation

**Generator:** `src/finsheild/synthetic_env/environment_hard.py` + `scenarios_hard.py`  
**Reuses:** Same entity generators (`generate_users`, `generate_accounts`, `generate_devices`, `generate_merchants`, `generate_locations`, `generate_account_devices`) as easy synthetic, so reference tables are identical in distribution.

**Config (fixed, deterministic):**
- `seed = 1729`
- `n_transactions_bg = 9000` (background legitimate)
- `n_per_scenario = 20` (5 hard scenarios ×20 = 100 fraud)
- Total rows: **9100** (9000 bg + 100 fraud)
- Fraud: **100** (`hard_*` tags), rate **1.0989%**
- `time_span_days = 30`, `n_users=200`, `n_accounts=250`, `n_devices=220`, `n_merchants=80`, `n_locations=60`
- Train/val/test: stratified `70/15/15`, `random_state=42`, scaler fit on train only.

**Fraud prevalence:** ~1.10% (within ±0.3% of target 1% unless architecture requires otherwise — here 9000+100 gives 1.10%, documented).

## 2. Overlap Strategy (per spec, no label leakage)

**Background legitimate now overlaps with fraud:**
- **Amount:** fraud uses `lognormal(3.5-4.0, 0.4-0.6)` vs bg `lognormal(3.6, 0.95)` — substantially overlapping; includes low-value fraud and high-value legit.
- **Velocity:** legit has no injected bursts, but fraud also has **no** velocity bursts in hard variant (all hard fraud are single transactions). Both have same 5-min/1h windows from feature engineering, so velocity is naturally similar — overlap by design.
- **Time:** legit has off-hours via hour weights (0.05 for 2-5 AM); fraud has 30% off-hours in `hard_normal_amount_normal_device_merchant` and business hours (9-19) otherwise — overlap.
- **Location:** legit 30% foreign (travel), fraud 0-20% foreign depending on scenario (e.g., `hard_normal_amount_normal_device_merchant` 20% foreign, `hard_mixed` 25% foreign, others 0% foreign near home). Results: legit foreign rate 0.405, fraud 0.260 — fraud *less* likely foreign than legit (good overlap).
- **Device:** legit 10% new device (previously 0% — fixed to create overlap), fraud 45% new device (2/5 scenarios use new device). Lift now 4.3× (was 450k× before fix) — realistic overlap.
- **Merchant:** legit 20% high-risk (weighted to achieve 19.8% actual), fraud 23% high-risk (mixed: 50% high-risk in one scenario + 25% in mixed). Lift 1.2× (was 13.8×) — excellent overlap.

**Fraud signals preserved (not random noise):**
- No single feature is perfect. Fraud is a *blend*:
  - `hard_moderate_amo_dev`: moderate amount (lognormal 3.9) + new device, normal loc/hour
  - `hard_normal_amount_new_device_merchant`: normal amount + new device + 50% high-risk merchant
  - `hard_high_amount_normal_location_velocity`: high amount only (lognormal 5.5), normal loc/device
  - `hard_normal_amount_normal_device_merchant`: weak signals only (30% off-hours, 20% foreign, normal amount/device)
  - `hard_mixed_signals`: random combo of weak signals (amount 3.5-4.0, foreign/off-hours/new device/high-risk in quarters)

**Leakage audit:**
- No `if fraud: amount > threshold` or `fraud_device=True` or `merchant_type=fraud_merchant`.
- Fraud emerges from correlated behavioral patterns, not explicit labels.
- Amount/device/merchant/location/timestamp/feature names/row order never directly encode label.
- Verified: `label_fraud` not in `feature_cols`, no feature name contains "fraud" or "label".

## 3. Model Configuration (fixed)

- **XGBoost:** `n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric=aucpr, tree_method=hist` (from `src/finsheild/model.py`)
- **LogReg:** `C=1.0, max_iter=1000, solver=lbfgs, random_state=42`
- No hyperparameter tuning, no increase in complexity.
- Same evaluation as previous: `train_test_split` stratified 70/15/15, scaler `StandardScaler` fit on train only, impute NaN/inf with median, threshold 0.5 for F1, 1% FPR tuned on val.

## 4. Results (actual measured, threshold 0.5)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Lift vs Random (1.10% base) |
| ----- | ------: | -----: | -: | --------: | -----: | ---: |
| LogReg | 0.9640 | 0.2200 | 0.1111 | 0.3333 | 0.0667 | 20.0× |
| XGBoost | 0.9505 | 0.3728 | 0.2857 | 0.5000 | 0.2000 | 33.9× |

**Confusion (XGBoost, test n=1365, 15 fraud):**
- TN 1349, FP 1, FN 12, TP 3
- Recall 20%, Precision 50%, Recall@1%FPR 40%

**Confusion (LogReg, test n=1365):**
- TN 1348, FP 2, FN 14, TP 1
- Recall 6.7%, Precision 33%

*Note: PR is primary for imbalanced; F1/Precision/Recall at 0.5 threshold are pessimistic for 1% fraud. Recall@1%FPR is more operationally relevant.*

## 5. Feature Separability (overlap verification)

**Numerical (fraud within legit IQR):**
- `amount`: fraud in legit IQR ~45% (vs 0% for easy extreme)
- `vel_count_300s`: fraud and legit both mostly 0-2 (no burst injection, so overlapping)
- `distance_to_prev_km`: legit has travel (30% foreign) so distance distribution overlaps
- `amount_zscore`: fraud zscore overlaps legit (moderate deviations, not 5σ)

**Categorical (fraud vs legit rates, lift 1.0 = no separation):**
- `is_new_device`: legit **0.106**, fraud **0.450**, lift **4.3×** (was 450k× before — fixed)
- `country_switch`: legit 0.405, fraud 0.260, lift 0.6× (fraud *less* likely to switch — good overlap)
- `is_high_risk_merchant`: legit 0.198, fraud 0.230, lift **1.2×** (was 13.8× — fixed to near-perfect overlap)

Overlap is real, not claimed via metric drop alone. Histograms saved to `evaluation/figures/synthetic_hard_overlap_numerical_overlap.png`.

## 6. Comparison of All Synthetic Versions

| Dataset | Fraud Rate | XGB ROC-AUC | XGB PR-AUC | LogReg PR-AUC |
| ------- | ---------: | ----------: | ---------: | ------------: |
| Easy Synthetic | 11.51% (709/6158) | 0.996 | 0.959 | — |
| 1% Diluted | 1.07% (91/8540) | 0.9927 | 0.5531 | — |
| Hard Overlap | 1.10% (100/9100) | 0.9505 | 0.3728 | 0.2200 |
| Real ULB | 0.17% (492/284807) | 0.9709 | 0.8418 | 0.7005 |

*Easy and 1% diluted metrics from prior runs (easy PR 0.959 from original, diluted PR 0.5531 from 91 fraud experiment). Real ULB from `evaluation/reports/*_metrics.json` after Kaggle pull.*

**Interpretation:**
- PR drops easy (0.959) → diluted (0.553) → hard (0.373) as expected when prevalence and separability decrease. Hard is **harder** than diluted and **harder** than real ULB (0.84), indicating overlap is substantial.
- Hard still has signal: PR 0.37 vs random 0.011 = 33.9× lift, not random (0.01). Model learns *combinations* of weak signals.
- XGB beats LogReg on hard (0.37 vs 0.22) — tree can learn interactions (e.g., moderate amount + new device), linear cannot.
- Real ULB PR 0.84 > hard 0.37 suggests real fraud has stronger signals than our heuristic overlap, or our hard simulation is *too* hard / not calibrated. This is expected — synthetic is still simulated.

**Leakage:** None found. Amount, velocity, device, merchant, location, timestamp, IDs, feature names, row order all audited.

**Limitations:** Heuristic overlap, not calibrated to real banking behavior; still simulated; amount/velocity distributions lognormal, not real; fraud is not ground-truth labeled by investigators.

**Next phase:** Use hard overlap to test risk fusion / graph features (Phases 10-11) — if they help, hard PR should improve without leaking.
