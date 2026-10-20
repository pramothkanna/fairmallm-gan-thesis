# Evaluation report (cohort = active)

Real training data: n=18170, positive rate 0.407, label SPD +0.025 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 18170 | 0.768±0.005 | 0.756 | 0.708 | 0.841±0.002 | +0.036 | +0.084 |
| TRTR-n (real n=623) | 623 | 0.685±0.015 | 0.671 | 0.605 | 0.740±0.012 | -0.078 | -0.016 |
| TSTR full | 623 | 0.662±0.012 | 0.648 | 0.578 | 0.715±0.004 | -0.146 | -0.087 |
| AUG real-n + full | 1246 | 0.695±0.017 | 0.675 | 0.603 | 0.747±0.012 | -0.062 | -0.000 |
| TSTR no_fair | 623 | 0.652±0.029 | 0.644 | 0.579 | 0.716±0.007 | -0.169 | -0.110 |
| AUG real-n + no_fair | 1246 | 0.674±0.012 | 0.672 | 0.621 | 0.741±0.013 | -0.109 | -0.036 |
| TSTR mallm_flat | 623 | 0.676±0.020 | 0.656 | 0.575 | 0.723±0.003 | -0.153 | -0.090 |
| AUG real-n + mallm_flat | 1246 | 0.703±0.018 | 0.677 | 0.595 | 0.747±0.019 | -0.052 | -0.005 |
| TSTR ctgan | 623 | 0.679±0.016 | 0.663 | 0.591 | 0.725±0.014 | -0.139 | -0.070 |
| AUG real-n + ctgan | 1246 | 0.682±0.039 | 0.672 | 0.613 | 0.750±0.014 | -0.087 | -0.019 |

## Synthetic data quality

| data set | n | label SPD | D1 acc (→0.5) | D2 AUC (→0.5) | D2 real same n | GRU disc. (→0) | W1 mean | lag-1 gap | DCR median (real test) |
|---|---|---|---|---|---|---|---|---|---|
| full | 623 | -0.021 | 0.827 | 0.588 | 0.701 | 0.321 | 0.342 | 0.371 | 4.94 (5.26) |
| no_fair | 623 | +0.022 | 0.822 | 0.555 | 0.684 | 0.369 | 0.359 | 0.345 | 4.79 (5.26) |
| mallm_flat | 623 | +0.086 | 0.767 | 0.631 | 0.730 | 0.313 | 0.212 | 0.350 | 5.18 (5.26) |
| ctgan | 623 | -0.080 | 0.612 | 0.845 | 0.742 | 0.350 | 0.079 | 0.108 | 7.13 (5.26) |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.