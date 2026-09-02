# Baseline Report — baseline

Model: `logreg`
Phase: 2 (baseline classifier)

## Test metrics
- Precision: 0.0000
- Recall:    0.0000
- F1:        0.0000
- ROC-AUC:   0.5927
- PR-AUC:    0.0267
- Recall @ FPR=1.00%: 0.0000

## Confusion matrix (test, threshold=0.1164)
|        | Predicted legit | Predicted fraud |
|--------|-----------------|-----------------|
| Actual legit | 732 | 5 |
| Actual fraud | 13 | 0 |

Support: 737 legit, 13 fraud
Threshold tuned on val: 0.1164

## Training
- Train seconds: 0.1
- Splits: {'train': 3500, 'val': 750, 'test': 750}

Figures: `evaluation/figures/{out_dir}_pr_curve.png`, `{out_dir}_roc_curve.png`, `{out_dir}_confusion_matrix.png`