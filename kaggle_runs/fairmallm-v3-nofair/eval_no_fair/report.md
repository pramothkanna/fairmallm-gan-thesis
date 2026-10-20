# Evaluation report (cohort = active)

Real training data: n=18166, positive rate 0.406, label SPD +0.016 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 18166 | 0.767±0.005 | 0.753 | 0.704 | 0.831±0.001 | +0.061 | +0.065 |
| TRTR-n (real n=2001) | 2001 | 0.667±0.014 | 0.669 | 0.626 | 0.742±0.004 | -0.092 | -0.023 |
| TSTR no_fair | 2001 | 0.643±0.009 | 0.634 | 0.572 | 0.692±0.005 | -0.112 | -0.069 |
| AUG real-n + no_fair | 4002 | 0.685±0.019 | 0.668 | 0.597 | 0.746±0.002 | +0.013 | +0.059 |

## Synthetic data quality

| data set | n | label SPD | D1 acc (→0.5) | D2 AUC (→0.5) | D2 real same n | GRU disc. (→0) | W1 mean | lag-1 gap | DCR median (real test) |
|---|---|---|---|---|---|---|---|---|---|
| no_fair | 2001 | +0.017 | 0.831 | 0.627 | 0.774 | 0.353 | 0.359 | 0.342 | 4.90 (5.21) |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.