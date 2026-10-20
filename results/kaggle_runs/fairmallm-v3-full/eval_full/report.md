# Evaluation report (cohort = active)

Real training data: n=18166, positive rate 0.406, label SPD +0.016 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 18166 | 0.767±0.005 | 0.753 | 0.704 | 0.831±0.001 | +0.061 | +0.065 |
| TRTR-n (real n=1999) | 1999 | 0.711±0.018 | 0.690 | 0.619 | 0.761±0.019 | +0.016 | +0.045 |
| TSTR full | 1999 | 0.638±0.009 | 0.631 | 0.572 | 0.684±0.001 | -0.092 | -0.061 |
| AUG real-n + full | 3998 | 0.679±0.029 | 0.674 | 0.620 | 0.755±0.011 | -0.005 | +0.037 |

## Synthetic data quality

| data set | n | label SPD | D1 acc (→0.5) | D2 AUC (→0.5) | D2 real same n | GRU disc. (→0) | W1 mean | lag-1 gap | DCR median (real test) |
|---|---|---|---|---|---|---|---|---|---|
| full | 1999 | -0.000 | 0.826 | 0.582 | 0.765 | 0.373 | 0.333 | 0.358 | 5.22 (5.21) |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.