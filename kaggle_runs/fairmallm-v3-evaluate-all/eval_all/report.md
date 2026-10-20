# Evaluation report (cohort = active)

Real training data: n=18166, positive rate 0.406, label SPD +0.016 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 18166 | 0.771±0.002 | 0.754 | 0.703 | 0.831±0.001 | +0.065 | +0.075 |
| TRTR-n (real n=623) | 623 | 0.662±0.025 | 0.655 | 0.597 | 0.728±0.002 | -0.119 | -0.059 |
| TRTR-n (real n=1999) | 1999 | 0.704±0.022 | 0.681 | 0.605 | 0.753±0.019 | -0.029 | +0.015 |
| TRTR-n (real n=2000) | 2000 | 0.697±0.027 | 0.678 | 0.600 | 0.760±0.010 | +0.007 | +0.042 |
| TRTR-n (real n=2001) | 2001 | 0.696±0.033 | 0.681 | 0.616 | 0.754±0.017 | -0.041 | +0.008 |
| TSTR full | 1999 | 0.643±0.009 | 0.631 | 0.563 | 0.686±0.002 | -0.091 | -0.062 |
| AUG real-n + full | 3998 | 0.681±0.029 | 0.677 | 0.626 | 0.752±0.015 | +0.006 | +0.046 |
| TSTR no_fair | 2001 | 0.645±0.006 | 0.631 | 0.559 | 0.692±0.004 | -0.105 | -0.067 |
| AUG real-n + no_fair | 4002 | 0.680±0.021 | 0.667 | 0.603 | 0.744±0.009 | -0.004 | +0.039 |
| TSTR mallm_flat | 623 | 0.669±0.024 | 0.647 | 0.563 | 0.710±0.006 | -0.138 | -0.095 |
| AUG real-n + mallm_flat | 1246 | 0.687±0.011 | 0.663 | 0.578 | 0.732±0.008 | -0.081 | -0.033 |
| TSTR ctgan | 2000 | 0.674±0.013 | 0.661 | 0.595 | 0.724±0.010 | -0.130 | -0.066 |
| AUG real-n + ctgan | 4000 | 0.684±0.021 | 0.682 | 0.635 | 0.761±0.007 | -0.026 | +0.017 |

## Synthetic data quality

| data set | n | label SPD | D1 acc (→0.5) | D2 AUC (→0.5) | D2 real same n | GRU disc. (→0) | W1 mean | lag-1 gap | DCR median (real test) |
|---|---|---|---|---|---|---|---|---|---|
| full | 1999 | -0.000 | 0.826 | 0.582 | 0.765 | 0.370 | 0.333 | 0.358 | 5.22 (5.21) |
| no_fair | 2001 | +0.017 | 0.835 | 0.627 | 0.759 | 0.341 | 0.354 | 0.347 | 4.90 (5.21) |
| mallm_flat | 623 | +0.086 | 0.758 | 0.631 | 0.685 | 0.305 | 0.211 | 0.351 | 5.40 (5.21) |
| ctgan | 2000 | -0.099 | 0.602 | 0.861 | 0.758 | 0.438 | 0.070 | 0.099 | 7.05 (5.21) |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.