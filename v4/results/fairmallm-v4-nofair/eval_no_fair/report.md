# Evaluation report (cohort = active)

Real training data: n=18166, positive rate 0.406, label SPD +0.016 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 18166 | 0.767±0.005 | 0.753 | 0.704 | 0.831±0.001 | +0.061 | +0.065 |
| TRTR-n (real n=2001) | 2001 | 0.667±0.014 | 0.669 | 0.626 | 0.742±0.004 | -0.092 | -0.023 |
| TSTR no_fair | 2001 | 0.692±0.011 | 0.661 | 0.562 | 0.736±0.006 | -0.063 | -0.029 |
| AUG real-n + no_fair | 4002 | 0.717±0.008 | 0.704 | 0.646 | 0.776±0.005 | +0.056 | +0.065 |

## Synthetic data quality

| data set | n | label SPD | D1 acc (→0.5) | D2 AUC (→0.5) | D2 real same n | GRU disc. (→0) | W1 mean | lag-1 gap | DCR median (real test) |
|---|---|---|---|---|---|---|---|---|---|
| no_fair | 2001 | +0.017 | 0.711 | 0.716 | 0.774 | 0.390 | 0.203 | 0.170 | 5.42 (5.21) |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.