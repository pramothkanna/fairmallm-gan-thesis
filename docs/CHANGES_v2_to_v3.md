# What was checked, what was wrong, what changed (v2 → v3)

Code review of the first version (v2, September 2026) and the reasons for every change in v3.
The reasoning is used in the **Methodology** and **Limitations** chapters. The proposal was not changed.

---

## 1. What was checked and is correct (no change)

| Item | Check | Result |
|---|---|---|
| Composite key `module\|presentation\|id_student` | 32,593 enrolments, keys aligned with all `.npy` files | ✔ |
| 5 weekly features, label, gender coding | same definitions as proposal §3.2 | ✔ |
| Weekly assessment score | re-computed from raw `studentAssessment` + `assessments` | ✔ all 57,556 student-weeks identical |
| Padding | padded weeks are exactly 0, `mask` correct (11 or 12 real weeks) | ✔ |
| Early-warning window | 11/12 *whole* weeks = 31–33 % of module length | ✔ ("approximately 30 %") |
| Scaler | fitted on real training weeks only | ✔ |
| D1/D2 v2 | cross-validated (v1 bug fixed) | ✔ |

## 2. Problems found and how v3 fixes them

### 2.1 Students who had already left were treated as "early-warning" cases
* 6,673 enrolments (20.5 %) unregistered **before** the 30 % cutoff, 3,083 (9.5 %) even **before day 0**.
  Their outcome is already known when the model would make its prediction, they have (almost) no
  activity, and they make prediction look easier than it is (all-zero students: 24 % of positives in
  the full cohort, 5.9 % after the fix).
* **Fix:** main cohort = students still registered at the cutoff (`cohort="active"`,
  25,912 enrolments, positive rate 0.407). The full cohort is still available
  (`cohort="all"`) and is reported as a sensitivity analysis.
* Code: `common.active_at_cutoff()`, `prepare_cohort.py`.

### 2.2 The same student could be in train and test
* `id_student` repeats across presentations; 15 % of test enrolments had the same person in train.
* **Fix:** grouped split by `id_student`, stratified by label × gender, 70/15/15
  (`StratifiedGroupKFold`) → overlap 0 %.

### 2.3 Missing values `?` were not parsed in the notebook
* `studentRegistration.date_unregistration` stayed a text column, so the switch
  `DROP_EARLY_WITHDRAWALS` could never have worked. It was `False`, so the saved tensors are not
  affected. **Fix:** `na_values="?"` everywhere (`preprocess_oulad.py`, `common.load_real`).

### 2.4 The generator created gender bias instead of removing it
* Real data: dropout rate M 0.538, F 0.516 → label SPD **+0.02** (almost no bias).
* v2 synthetic data: M 0.313, F 0.516 → label SPD **−0.20**. The LLM chose S and Y itself.
* **Fix:** *conditioned slots* – the prompt lists the exact headers `<S=..;Y=..;W=..>` to fill.
  In the full model the label rate is the same for both genders (`label_policy="parity"`);
  baselines copy the real rates (`"real"`). Final data are assembled to the target S × Y mix.

### 2.5 D2 ≈ 0.5 was reached for the wrong reason
* On **real** data the same D2 reaches AUC **0.72–0.75** (gender is predictable from engagement,
  mainly through the module mix: BBB 88 % female, EEE 12 %). v2 data reached 0.5 because the LLM
  simply did not model realistic differences, while D1 (realism) never converged (0.56–0.73).
* **Fix:** D2 is now reported next to the real-data reference (`real_D2_auc`), and D1 compares
  against a real sample with the *same* S × Y composition, so a generator cannot "win" D2 by being
  unrealistic. This makes the realism–fairness trade-off (RQ3) visible.

### 2.6 Feedback was too generic
* v2 realism feedback was one fixed sentence → D1 did not improve.
* **Fix (MALLM-GAN style):** numeric feedback – the 3 statistics D1 relies on most and the largest
  real-vs-synthetic gaps per label, e.g. *"Y=1 quiz_interactions % of weeks non-zero: real 9 vs yours 2"*;
  fairness feedback names the leaking feature and its S=0 vs S=1 values. A short aggregate
  **data card** (per-label weekly means, activity rates) is in the system prompt.

### 2.7 Implausible / copied synthetic students
* 40 students with 8 weeks (impossible), 6 identical pairs, 3 gender-swapped twins, quiz/score
  activity far too rare.
* **Fix:** strict parser (exactly W ∈ {11, 12} weeks), **novelty filter** (reject students closer to a
  real training student than 99 % of real hold-out students are; reject duplicates and
  gender-swapped twins of accepted synthetic students), cap on all-zero students (real share + 3 pp).

### 2.8 Too slow for Kaggle (≈ 7 min per 8 students)
* **Fix:** compact serialisation `w0:5,2,10,0,61` (~40 % fewer tokens), **batched generation**
  (8 prompts × 8 students per GPU pass, automatic halving on out-of-memory; ≈ 5× the v2 speed in the Kaggle smoke test), output length sized
  to the request, **time budget** (stops cleanly before Kaggle's 12-h limit), checkpoint/resume.

### 2.9 Evaluation weaknesses
| v2 | v3 |
|---|---|
| 400 synthetic vs 22,814 real | + **TRTR-n** (same number of real students) and AUG (real-n + synthetic) |
| 1 GRU run, 15 fixed epochs | 5 seeds (mean ± sd), early stopping on the *training source's* validation split |
| raw skewed counts | log1p + standardisation (fitted on real train) |
| fixed threshold 0.5 | threshold chosen on that validation split (max. balanced accuracy) |
| acc, F1 | + balanced accuracy, ROC-AUC, per-module SPD |
| no data-level metrics | label SPD, D1, D2 (vs real), GRU discriminative score, Wasserstein distance, weekly curves, lag-1 autocorrelation, correlation gap |
| no privacy metric | distance to closest record (DCR) vs real test → train |

### 2.10 RQ3 baselines and RQ4 were missing
* `mode="no_fair"` – same generator without D2 (ablation).
* `mode="mallm_flat"` – MALLM-GAN baseline: flattened static `column=value` rows, D1 only, plus an
  **optimizer LLM** that rewrites the guidelines (as in the original MALLM-GAN).
* `baseline_ctgan.py` – CTGAN on the flattened weekly panel.
* `etb/` – RQ4 pipeline (preprocessing, tabular generator, LightGBM evaluation, CTGAN).

## 3. Things that remain limitations (write them in the thesis)
* Forum clicks are not one of the 5 features (proposal choice; forum is strongly predictive).
* `sc = 0` mixes "no submission" and "score 0" (only 122 real zero scores).
* Gender in OULAD is mostly predictable through the module mix; equalising features across genders
  necessarily moves the data away from reality (expected trade-off, RQ3).
* The LLM sees aggregate statistics and few-shot examples of real students; the novelty filter and
  DCR check reduce, but do not formally guarantee, privacy (no differential privacy).
* ETB Campus File has 13,342 respondents (13,322 with a valid satisfaction answer), not ~20,000.
  With the chosen target the gender gap in the label is small (label SPD −0.01); the gender
  information is in the features (real D2 AUC ≈ 0.77).
