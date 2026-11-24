# v4 smoke tests on Kaggle (21 Oct 2026, Qwen2.5-7B-Instruct, T4 x2, mode full, 128 students, 2 rounds)

| version | how the plan is given | plan recall | students kept | quiz active weeks | assessment active weeks | final D1 | final D2 | notes |
|---|---|---|---|---|---|---|---|---|
| v3 (Oct full run, reference) | no plan | – | 61 % | 7.2 % | 5.5 % | 0.83 | 0.59 | 2,000 students |
| v4 | text before header | 0.52 | 43 % | 5.0 % | 16.9 % | 0.63 | 0.37 | qz/sc columns swapped, plan echoed |
| v4.1 | fill-in template | 0.49 | 45 % | 12.4 % | 13.6 % | 0.73 | 0.51 | sp/oc collapsed to 0, CUDA OOM |
| v4.2 | constrained decoding | **1.00** | **97 %** | **19.5 %** | **18.6 %** | **0.64** | **0.48** | qz values mostly 1 (real mean 9.6/week) |
| real (train) | | | | 19.0 % | 18.4 % | | 0.76 | |

v4.3 (used for the full runs) adds the donor's number of digits per planned qz/sc value
to fix the "qz = 1" problem. Smoke-test sizes are small (128 students), so D1/D2 here are noisy.
