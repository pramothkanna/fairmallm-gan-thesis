# Run guide (Kaggle, free T4 x2)

## Inputs
All notebooks are private and use two private Kaggle datasets:
* `oulad-processed`: the processed OULAD tensors (output of `src/preprocess_oulad.py`)
* `fairmallm-v3-code`: the files of `src/` (upload a new version after every code change)

## Generation runs
| Notebook | Produces | Time |
|---|---|---|
| `kaggle/fairmallm-v3-smoke.ipynb` | 64 students, checks GPU, parser and filters | ≈ 25 min |
| `kaggle/fairmallm-v3-full.ipynb` | proposed model, 2,000 students + quick evaluation | ≈ 6.5 h |
| `kaggle/fairmallm-v3-nofair.ipynb` | ablation without D2 (RQ3) | ≈ 6 h |
| `kaggle/fairmallm-v3-mallmflat.ipynb` | MALLM-GAN baseline (RQ3) | stops at the 10.5 h budget (623 students in Oct. 2026) |
| `kaggle/fairmallm-v3-ctgan-eval.ipynb` | CTGAN baseline + evaluation (CPU) | ≈ 1–2 h |
| `kaggle/fairmallm-v3-evaluate-all.ipynb` | joint 5-seed evaluation of all sets (CPU is enough) | ≈ 1 h |

Import a notebook (*File → Import Notebook*), add the inputs (*Add Input*), set *GPU T4 x2* and *Internet on*
for the generation runs, then *Save Version → Save & Run All*. Kaggle allows two GPU sessions at a time and
about 30 GPU hours per week. Every run stops itself before the 12-hour limit and saves what it has.

## After a run
1. Notebook → *Output*: check `run_<mode>/final_check.json` (`n`, `D1_acc`, `D2_auc`, `real_D2_auc`,
   `label_SPD_synth`) and `run_<mode>/run_log.txt` (one line per round).
2. Download the output; copy the small files (json, csv, txt, md, png) to `results/kaggle_runs/<notebook>/`.
   Do not commit `*.npy`, `raw_samples.txt`, `synthetic_*.csv/jsonl`.
3. For the joint evaluation add the finished notebooks as inputs of `fairmallm-v3-evaluate-all`.

## Analysis on a normal PC (CPU)
```bash
python analysis/results_analysis.py --data_dir Dataset/oulad_processed \
  --eval_json results/kaggle_runs/fairmallm-v3-evaluate-all/eval_all/results_v3.json \
  --runs full=<run_full> no_fair=<run_no_fair> mallm_flat=<run_mallm_flat> ctgan=<run_ctgan> \
  --out results/analysis
python analysis/subsample_to_623.py --runs full=<run_full> no_fair=<run_no_fair> ctgan=<run_ctgan> --out sub623
python src/evaluate_v3.py --data_dir Dataset/oulad_processed --out_dir results/size_matched_623 \
  --synth full=sub623/run_full no_fair=sub623/run_no_fair mallm_flat=<run_mallm_flat> ctgan=sub623/run_ctgan
```
The `run_*` folders are the downloaded Kaggle outputs (they contain the `synth_*.npy` files).

## Which result answers which research question
| RQ | Where | What to look at |
|---|---|---|
| RQ1 fairness | `eval_all/data_quality.csv`, `results/analysis/I_D2_stability.csv`, `C_*` | label SPD, D2 vs real same size; SPD/EOD overall and within modules |
| RQ2 temporal fidelity | `eval_all/report.md`, `results/analysis/A_*`, `B_*`, `fig18` | TSTR vs TRTR-n, GRU discriminative score, activity rates, weekly curves |
| RQ3 trade-off | `results/analysis/E_size_matched.csv`, `results/size_matched_623/` | AUC loss vs leakage removed; all generators at equal size |
| RQ4 ETB | `src/etb/` | open (LLM run pending) |
| Privacy | `eval_all/data_quality.csv` | DCR synthetic vs real hold-out, exact copies |

## ETB 2018 (RQ4)
The Campus File is licensed and is never committed or uploaded publicly.
`python src/etb/etb_preprocess.py --dta <CampusFile.dta> --out_dir Dataset/etb_processed`, then
`python src/etb/evaluate_etb.py --data_dir Dataset/etb_processed --out_dir results/etb_eval --ctgan`.
The LLM part (`kaggle/fairmallm-v3-etb-*.ipynb`) needs a GPU and may only run on Kaggle if the BIBB
terms allow a private upload.
