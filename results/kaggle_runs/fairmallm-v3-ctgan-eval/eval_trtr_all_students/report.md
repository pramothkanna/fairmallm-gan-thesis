# Evaluation report (cohort = all)

Real training data: n=22843, positive rate 0.528, label SPD +0.017 (S=1 female).
SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.
TRTR-n uses the same number of real students as the synthetic set (fair size reference).

## Utility and fairness of the GRU on the REAL test set (mean over seeds)

| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |
|---|---|---|---|---|---|---|---|
| TRTR (all real) | 22843 | 0.812±0.002 | 0.818 | 0.806 | 0.892±0.001 | +0.049 | +0.034 |

Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`.