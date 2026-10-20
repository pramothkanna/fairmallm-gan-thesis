# FairMALLM-GAN v3 — code for the thesis
*Fair and Temporally-Aware LLM-Guided Synthetic Data Generation*

The v3 files replace the first version (v2, `fair_mallm_gan.py`, `evaluate_tstr.py`, not part of this
repository). What was fixed and why: `CHANGES_v2_to_v3.md`. How to run everything on Kaggle: `RUN_GUIDE.md`.
Progress and current status: `PROGRESS.md`.

## Files

| File | What it does | Runs on |
|---|---|---|
| `src/common.py` | shared helpers: loading, cohort, grouped split, D1/D2 features, SPD/EOD | – |
| `src/prepare_cohort.py` | builds the v3 cohort + grouped split and a short report | CPU, seconds |
| `src/preprocess_oulad.py` | script version of `Weekly_binning.ipynb` (with the `?` fix) — only needed to rebuild `oulad_processed` | CPU, ~5 min |
| `src/fair_mallm_gan_v3.py` | **the generator** (modes `full`, `no_fair`, `mallm_flat`) | Kaggle GPU T4 ×2 |
| `src/baseline_ctgan.py` | CTGAN baseline on the flattened weekly panel | GPU or CPU |
| `src/evaluate_v3.py` | GRU TRTR / TRTR-n / TSTR / AUG, fairness, fidelity, privacy, figures, `report.md` | GPU or CPU |
| `src/etb/etb_preprocess.py` | RQ4: ETB 2018 Campus File → table, target, splits | CPU |
| `src/etb/fair_mallm_tabular.py` | RQ4: the same generator for the ETB table | Kaggle GPU |
| `src/etb/evaluate_etb.py` | RQ4: LightGBM TRTR / TSTR + fairness + CTGAN baseline | CPU |
| `kaggle/*.ipynb` | ready-made Kaggle notebooks (smoke test, full, no_fair, mallm_flat, ctgan, evaluate-all) | Kaggle |
| `eda/oulad_eda.py`, `notebooks/EDA_OULAD.ipynb` | exploratory data analysis | CPU, ~5 min |
| `analysis/results_analysis.py` | analysis of the finished runs (activity profile, label signal, module proxy, D2 stability, feature pilot) | CPU, ~1 min |
| `analysis/subsample_to_623.py` | size-matched comparison at 623 students | CPU |
| `kaggle/make_notebooks.py` | rebuilds the notebooks | – |

## The three generator modes (RQ1–RQ3)

| mode | serialisation | discriminators in feedback | label mix | role in thesis |
|---|---|---|---|---|
| `full` | weekly `w0:hp,sp,oc,qz,sc \| w1:...` | D1 realism + D2 fairness | same dropout rate for both genders | proposed model |
| `no_fair` | weekly | D1 only | copied from real data | ablation: effect of the fairness part |
| `mallm_flat` | flattened `hp_w0=..,..,sc_w11=..` | D1 only + optimizer LLM | copied from real data | MALLM-GAN baseline |
| CTGAN | flattened 60 columns | – | learned | non-LLM baseline |

## How one generator run works (plain words)
1. Load the processed OULAD tensors; keep students still enrolled at the 30 % cutoff; split by person.
2. Build a short summary of the real training data (the "data card") for the system prompt.
3. **Rounds (adversarial prompt optimisation):** 8 prompts at once, each with 8 real examples
   (2 per gender × outcome) and 8 empty headers like `<S=1;Y=0;W=12>` that the LLM must fill.
   Parsed students pass the filters (right length, not a copy, not a duplicate).
   D1 (real vs synthetic) and D2 (gender from features + label) are scored with 5-fold CV;
   the result is written into concrete feedback for the next round. Stops when both are within
   ±0.10 of 0.5 for 2 rounds in a row, or after `max_rounds`.
4. **Final generation:** with the best feedback, generate until `n_synthetic` students in the target
   gender × outcome mix are accepted (or the time budget is reached).
5. Outputs in `OUT_DIR`: `synth_X_seq.npy`, `synth_mask.npy`, `synth_S_gender.npy`, `synth_y.npy`,
   `synthetic_weekly_long.csv`, `train_state.json` (all rounds), `final_check.json`,
   `run_log.txt`, `raw_samples.txt` (raw LLM text for debugging), `rejections_final.json`.

## Running locally (CPU) — only for the non-LLM parts
```bash
pip install numpy pandas scikit-learn torch scipy matplotlib ctgan lightgbm
python src/prepare_cohort.py --data_dir "Dataset/oulad_processed" --out_dir results/v3_cohort
python src/evaluate_v3.py --data_dir "Dataset/oulad_processed" --out_dir results/v3_eval \
       --synth full=<folder with synth_*.npy> ctgan=<folder>
python src/etb/etb_preprocess.py --dta "Dataset/BBIB Campus file/BIBBBAuA_2018_CampusFile.dta" --out_dir Dataset/etb_processed
python src/etb/evaluate_etb.py --data_dir Dataset/etb_processed --out_dir results/etb_eval --ctgan
```
A test without GPU/LLM: `python src/fair_mallm_gan_v3.py --DATA_DIR <oulad_processed> --OUT_DIR test --backend mock`
(the mock backend returns perturbed real students; never use it for results).
