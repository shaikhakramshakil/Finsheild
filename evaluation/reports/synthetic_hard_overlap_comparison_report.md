# Synthetic Hard Overlap — Comparison Report
Generated: 2026-09-04 | Seed: 1729

## Experiment Comparison (XGBoost, PR-AUC primary)

| Dataset        | Fraud Rate | XGB ROC-AUC | XGB PR-AUC | LogReg PR-AUC | Lift |
| -------------- | ---------: | ----------: | ---------: | ------------: | ---: |
| Easy Synthetic |   11.51% |      0.9960 |     0.9590 | — | — |
| 1% Diluted     |   1.07% |      0.9927 |     0.5531 | — | — |
| Hard Overlap   |   1.10% |      0.9505 |     0.3728 | 0.2200 | 33.9x |
| Real ULB       |      0.17% |      0.9709 |     0.8418 | 0.7005 | 495x |

## Interpretation
- Feature overlap introduced: fraud amounts, hours, locations, devices and merchants now overlap substantially with legitimate.
- Performance change: PR should drop easy→diluted→hard as signals weaken; hard should approach real difficulty.
- Leakage: audited — no label in amount/velocity/device/merchant/location/timestamp/feature names/row order.

## Dataset
- Rows: 9100 | Fraud: 100 (1.0989%) | Seed: 1729 | Split: train 6370 val 1365 test 1365
