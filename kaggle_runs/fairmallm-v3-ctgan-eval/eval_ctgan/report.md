# Evaluation report (cohort = active)

Real training data: n=18166, positive rate 0.406, label SPD +0.016 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 18166 | 0.771±0.002 | 0.754 | 0.703 | 0.831±0.001 | +0.065 | +0.075 |
| TRTR-n (real n=2000) | 2000 | 0.697±0.027 | 0.678 | 0.600 | 0.760±0.010 | +0.007 | +0.042 |
| TSTR ctgan | 2000 | 0.674±0.013 | 0.661 | 0.595 | 0.724±0.010 | -0.130 | -0.066 |
| AUG real-n + ctgan | 4000 | 0.684±0.021 | 0.682 | 0.635 | 0.761±0.007 | -0.026 | +0.017 |

## Synthetic data quality

| data set | n | label SPD | D1 acc (→0.5) | D2 AUC (→0.5) | D2 real same n | GRU disc. (→0) | W1 mean | lag-1 gap | DCR median (real test) |
|---|---|---|---|---|---|---|---|---|---|
| ctgan | 2000 | -0.099 | 0.601 | 0.861 | 0.777 | 0.430 | 0.068 | 0.102 | 7.05 (5.21) |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.