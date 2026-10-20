# FairMALLM-GAN: Fair and Temporally-Aware LLM-Guided Synthetic Data Generation

Code of my Master thesis (University of Koblenz, Faculty 4 – Computer Science).
Synthetic **weekly student-activity sequences** are generated with a frozen LLM (Qwen2.5-7B-Instruct) that is
steered by two discriminators (D1: realism, D2: gender fairness) through text feedback, and evaluated on the
OULAD early-warning task (withdrawn/fail within the first ~30 % of a module). RQ4 repeats the idea on a
tabular data set (BIBB/BAuA Employment Survey 2018).

> Data are **not** in this repository. OULAD is public (https://analyse.kmi.open.ac.uk/open_dataset, CC-BY 4.0);
> the BIBB/BAuA Campus File is licensed and must be requested from the BIBB/FDZ.

## Folder structure
| Folder | Content |
|---|---|
| `eda/oulad_eda.py`, `notebooks/EDA_OULAD.ipynb` | exploratory data analysis (distributions, missing values, gender, withdrawal timing, activity types) – run **before** any modelling |
| `src/` | preprocessing (`preprocess_oulad.py`), cohort + split (`prepare_cohort.py`, `common.py`), generator (`fair_mallm_gan_v3.py`), CTGAN baseline, evaluation (`evaluate_v3.py`) |
| `src/etb/` | RQ4 pipeline for the ETB 2018 survey |
| `kaggle/` | notebooks used on Kaggle (T4 x2) and the script that builds them |
| `analysis/` | analysis of the finished runs and the size-matched comparison (CPU) |
| `results/eda` | EDA tables; `results/figures` report figures; `results/kaggle_runs` logs, metrics and reports of the Kaggle runs; `results/analysis` analysis tables and figures; `results/size_matched_623` all generators at 623 students |
| `docs/` | file overview (`README.md`), changes v2 → v3 and why, run guide (`RUN_GUIDE.md`), progress log (`PROGRESS.md`) |

## Order of work
1. `python src/preprocess_oulad.py` (raw OULAD → weekly tensors)  → `python src/prepare_cohort.py` (cohort + grouped split)
2. `python eda/oulad_eda.py --raw <OULAD folder> --processed <processed folder> --out results/eda`
3. Generation on Kaggle GPU: `kaggle/fairmallm-v3-{full,nofair,mallmflat}.ipynb`; CTGAN: `src/baseline_ctgan.py`
4. `python src/evaluate_v3.py --data_dir ... --out_dir ... --synth full=... no_fair=... ctgan=...`

5. `python analysis/results_analysis.py ...` (see `docs/RUN_GUIDE.md`)

Details of each file: `docs/README.md`. Progress and plan: `docs/PROGRESS.md`.

## Current results (19 October 2026, active cohort, GRU on real test students, 5 seeds)
| Training data | n | Gender leakage D2 (real 0.77) | Label SPD | Realism D1 (→0.5) | GRU AUC | GRU SPD |
|---|---|---|---|---|---|---|
| Real students (same n) | 2,000 | – | +0.02 | – | 0.753–0.760 | −0.04…+0.01 |
| **LLM + D2 (proposed)** | 1,999 | **0.58** | **0.00** | 0.83 | 0.686 | −0.091 |
| LLM without D2 | 2,001 | 0.63 | +0.02 | 0.84 | 0.692 | −0.105 |
| MALLM-GAN (flat) | 623 | 0.63 | +0.09 | 0.76 | 0.710 (real n=623: 0.728) | −0.138 |
| CTGAN | 2,000 | 0.86 | −0.10 | 0.60 | 0.724 | −0.130 |

The fairness feedback removes most gender information from the data at a small cost (0.006 AUC). Realism
(too few quiz/assessment weeks), diversity and the module-composition proxy in downstream predictions are the
open problems; see `docs/PROGRESS.md`.

AI assistance (Claude) was used for code review and part of the implementation; all design decisions and
results are checked and owned by the author.
