# Dataset Report — Finsheild Phase 1

Generated: 2026-09-02 (actual run)

## Shape & Schema
- Shape: (10000, 31)
- Columns: ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
- Dtypes: all numeric (float64/int64)
- Missing values: 0
- Duplicated rows: 0

## Class Distribution
- Counts: {'0': 9828, '1': 172}
- Percentages: {k: f"{v:.4%}" for k,v in metrics['class_counts_pct'].items()}
- Fraud rate: 1.7200%

## Amount/Time by Class
- Amount mean legit: 49.49, fraud: 43.15
- Time mean legit: 85971.8, fraud: 88374.8
- Figures: evaluation/figures/class_distribution.png, evaluation/figures/amount_time_by_class.png

## Correlation / PCA note
- V1-V28 are PCA-transformed anonymized features. Low pairwise correlation; not individually interpretable but retain fraud signal.

## Splits (stratified 70/15/15, random_state=42)
- Train: 7000 (70.0%) — {'0': 6880, '1': 120}
- Val: 1500 (15.0%) — {'0': 1474, '1': 26}
- Test: 1500 (15.0%) — {'0': 1474, '1': 26}
- Stratification preserved.

## Preprocessing (leakage-safe)
- Scaler: StandardScaler fit ONLY on train for Amount/Time
- Mean_: [50.66625714285715, 85517.34916720264]
- Scale_: [154.0281396159885, 49707.09498178243]
- Train Amount mean after scaling: -0.000000
- Train Time mean after scaling: -0.000000
- Validation/test transformed with train-fitted scaler (no leakage).

## Acquisition
- Source: Kaggle Credit Card Fraud Detection (ULB) — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Local raw: data/raw/creditcard.csv (gitignored)
- Processed: data/processed/train.csv, val.csv, test.csv + scaler.joblib (gitignored)

## Leakage & Time note
- Time is seconds since first transaction; prefer stratified random split for now. Time-based split left as alternative.
