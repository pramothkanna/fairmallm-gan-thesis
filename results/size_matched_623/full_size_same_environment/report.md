# Evaluation report (cohort = active)

Real training data: n=18170, positive rate 0.407, label SPD +0.025 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 18170 | 0.768±0.005 | 0.756 | 0.708 | 0.841±0.002 | +0.036 | +0.084 |
| TRTR-n (real n=1999) | 1999 | 0.720±0.016 | 0.698 | 0.625 | 0.777±0.006 | +0.054 | +0.099 |
| TRTR-n (real n=2000) | 2000 | 0.704±0.020 | 0.692 | 0.627 | 0.776±0.007 | +0.034 | +0.087 |
| TRTR-n (real n=2001) | 2001 | 0.707±0.017 | 0.692 | 0.627 | 0.771±0.004 | +0.039 | +0.089 |
| TSTR full | 1999 | 0.658±0.010 | 0.649 | 0.586 | 0.712±0.001 | -0.095 | -0.032 |
| TSTR no_fair | 2001 | 0.660±0.015 | 0.654 | 0.598 | 0.717±0.003 | -0.121 | -0.064 |
| TSTR ctgan | 2000 | 0.678±0.014 | 0.669 | 0.610 | 0.737±0.008 | -0.151 | -0.075 |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.