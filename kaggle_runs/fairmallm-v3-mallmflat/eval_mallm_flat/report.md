# Evaluation report (cohort = active)

Real training data: n=18166, positive rate 0.406, label SPD +0.016 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 18166 | 0.767±0.005 | 0.753 | 0.704 | 0.831±0.001 | +0.061 | +0.065 |
| TRTR-n (real n=623) | 623 | 0.668±0.016 | 0.662 | 0.606 | 0.732±0.003 | -0.117 | -0.053 |
| TSTR mallm_flat | 623 | 0.675±0.007 | 0.647 | 0.552 | 0.710±0.006 | -0.136 | -0.095 |
| AUG real-n + mallm_flat | 1246 | 0.681±0.013 | 0.657 | 0.566 | 0.731±0.009 | -0.076 | -0.033 |

## Synthetic data quality

| data set | n | label SPD | D1 acc (→0.5) | D2 AUC (→0.5) | D2 real same n | GRU disc. (→0) | W1 mean | lag-1 gap | DCR median (real test) |
|---|---|---|---|---|---|---|---|---|---|
| mallm_flat | 623 | +0.086 | 0.773 | 0.631 | 0.700 | 0.289 | 0.209 | 0.370 | 5.40 (5.21) |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.