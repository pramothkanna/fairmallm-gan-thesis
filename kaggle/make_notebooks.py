# -*- coding: utf-8 -*-
"""
make_notebooks.py -- builds self-contained Kaggle notebooks from the v3 code.
Inputs every notebook needs (Add Input):  'oulad-processed' (data) and
'fairmallm-v3-code' (the .py files of this folder; the notebook copies them to /kaggle/working).

python make_notebooks.py   ->  writes *.ipynb next to this file
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = ["common.py", "fair_mallm_gan_v3.py", "evaluate_v3.py", "baseline_ctgan.py"]


def md(t):   return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


COPY = r'''import glob, shutil
for f in ["common.py", "fair_mallm_gan_v3.py", "evaluate_v3.py", "baseline_ctgan.py", "prepare_cohort.py"]:
    src = glob.glob(f"/kaggle/input/**/{f}", recursive=True)
    assert src, f"{f} not found - attach the dataset 'fairmallm-v3-code' (Add Input)"
    shutil.copy(src[0], f"/kaggle/working/{f}")
print("code copied")
'''


def writefile_cells():
    return [code(COPY)]


FIND_DATA = r'''import glob, os, sys, json, shutil, torch
sys.path.insert(0, "/kaggle/working"); os.chdir("/kaggle/working")
hits = glob.glob("/kaggle/input/**/oulad_processed/X_seq.npy", recursive=True) or \
       glob.glob("/kaggle/input/**/X_seq.npy", recursive=True)
assert hits, "Attach the dataset 'oulad-processed' (Add Input) first."
DATA_DIR = os.path.dirname(hits[0])
print("DATA_DIR =", DATA_DIR)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
'''


def run_notebook(mode, n_synthetic, max_rounds, budget_h, eval_seeds, title):
    intro = (f"# FairMALLM-GAN v3 — {title}\n\n"
             "Settings: **Accelerator = GPU T4 x2**, **Internet = on**, inputs **oulad-processed** + **fairmallm-v3-code**.\n\n"
             "Run in the background: *Save Version → Save & Run All (Commit)*. Results appear in the "
             "version's **Output** tab (`run_*/` = synthetic data, `eval_*/` = evaluation).\n\n"
             "To run another configuration change `MODE` below: `full` (proposed model), `no_fair` "
             "(ablation without D2), `mallm_flat` (MALLM-GAN baseline).")
    run = f'''MODE = "{mode}"            # full | no_fair | mallm_flat
from fair_mallm_gan_v3 import Config, run
cfg = Config(DATA_DIR=DATA_DIR, OUT_DIR=f"/kaggle/working/run_{{MODE}}", mode=MODE,
             cohort="active", n_synthetic={n_synthetic}, max_rounds={max_rounds},
             prompts_per_batch=8, students_per_prompt=8, time_budget_h={budget_h})
synth, state = run(cfg)
'''
    ev = f'''# quick evaluation of THIS run (full comparison: evaluate_all notebook)
from evaluate_v3 import run as run_eval
res = run_eval(DATA_DIR, f"/kaggle/working/eval_{{MODE}}", {{MODE: cfg.OUT_DIR}},
               cohort="active", seeds={eval_seeds})
print(open(f"/kaggle/working/eval_{{MODE}}/report.md").read())
'''
    cells = [md(intro), code("!pip install -q transformers accelerate\n")] + writefile_cells() + \
            [code(FIND_DATA), code(run), code(ev),
             code('for root, _, fs in os.walk("/kaggle/working"):\n'
                  '    print(root, sorted(fs)[:12])\n')]
    return nb(cells)


def ctgan_notebook():
    intro = ("# FairMALLM-GAN v3 — CTGAN baseline (RQ3)\n\nAccelerator GPU (any), Internet on, input "
             "**oulad-processed**. Trains CTGAN on the flattened weekly panel and evaluates it with the "
             "same GRU protocol.")
    run = '''!python baseline_ctgan.py --data_dir "$DATA_DIR" --out_dir /kaggle/working/run_ctgan --n_synthetic 2000 --epochs 300
from evaluate_v3 import run as run_eval
res = run_eval(DATA_DIR, "/kaggle/working/eval_ctgan", {"ctgan": "/kaggle/working/run_ctgan"},
               cohort="active", seeds=(0, 1, 2, 3, 4))
print(open("/kaggle/working/eval_ctgan/report.md").read())
# sensitivity: TRTR on ALL students (incl. those who already withdrew before the cutoff)
res_all = run_eval(DATA_DIR, "/kaggle/working/eval_trtr_all_students", {}, cohort="all", seeds=(0, 1, 2))
print(open("/kaggle/working/eval_trtr_all_students/report.md").read())
'''
    return nb([md(intro), code("!pip install -q ctgan\n")] + writefile_cells() +
              [code(FIND_DATA + 'os.environ["DATA_DIR"] = DATA_DIR\n'), code(run)])


def evaluate_all_notebook():
    intro = ("# FairMALLM-GAN v3 — final comparison (RQ1–RQ3)\n\n"
             "Run this AFTER the generation notebooks have finished.\n\n"
             "1. *Add Input* → **Your Work / Notebooks** → add the outputs of `fairmallm-v3-full`, "
             "`fairmallm-v3-nofair`, `fairmallm-v3-mallmflat`, `fairmallm-v3-ctgan-eval` "
             "(and `oulad-processed`).\n2. Run all. GPU optional (faster).")
    find = FIND_DATA + r'''
runs = {}
for name in ["full", "no_fair", "mallm_flat", "ctgan"]:
    hit = glob.glob(f"/kaggle/input/**/run_{name}/synth_X_seq.npy", recursive=True)
    if hit: runs[name] = os.path.dirname(hit[0])
print(runs)
'''
    ev = '''from evaluate_v3 import run as run_eval
res = run_eval(DATA_DIR, "/kaggle/working/eval_all", runs, cohort="active", seeds=(0, 1, 2, 3, 4))
print(open("/kaggle/working/eval_all/report.md").read())
'''
    return nb([md(intro)] + writefile_cells() + [code(find), code(ev)])


def etb_notebook(mode):
    intro = (f"# FairMALLM-GAN v3 — RQ4 ETB 2018 ({mode})\n\n"
             "Inputs: **fairmallm-v3-code** and YOUR PRIVATE dataset **etb-processed** (the folder made by "
             "`etb/etb_preprocess.py`: `etb.csv`, `splits.json`, `meta.json`). Only upload it if the BIBB "
             "Campus-File terms allow it, and keep it private. Accelerator GPU T4 x2, Internet on.")
    copy = COPY.replace('"prepare_cohort.py"]', '"prepare_cohort.py", "etb_preprocess.py", '
                                                 '"fair_mallm_tabular.py", "evaluate_etb.py"]')
    run = f'''import glob, os, sys
sys.path.insert(0, "/kaggle/working"); os.chdir("/kaggle/working")
ETB_DIR = os.path.dirname(glob.glob("/kaggle/input/**/etb.csv", recursive=True)[0])
from fair_mallm_tabular import TabConfig, run
run(TabConfig(DATA_DIR=ETB_DIR, OUT_DIR="/kaggle/working/etb_{mode}", mode="{mode}",
              n_synthetic=2000, max_rounds=10, time_budget_h=10.0))
!pip install -q ctgan lightgbm
!python evaluate_etb.py --data_dir "$ETB_DIR" --out_dir /kaggle/working/etb_eval_{mode} --synth {mode}=/kaggle/working/etb_{mode} --ctgan
'''
    return nb([md(intro), code("!pip install -q transformers accelerate\n"), code(copy), code(run)])


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


NOTEBOOKS = {
    "fairmallm-v3-smoke": run_notebook("full", 64, 2, 0.6, (0,), "SMOKE TEST (~30 min)"),
    "fairmallm-v3-full": run_notebook("full", 2000, 10, 10.0, (0, 1, 2), "proposed model (full)"),
    "fairmallm-v3-nofair": run_notebook("no_fair", 2000, 10, 10.0, (0, 1, 2), "ablation without D2 (no_fair)"),
    "fairmallm-v3-mallmflat": run_notebook("mallm_flat", 2000, 10, 10.0, (0, 1, 2), "MALLM-GAN baseline (mallm_flat)"),
    "fairmallm-v3-ctgan-eval": ctgan_notebook(),
    "fairmallm-v3-evaluate-all": evaluate_all_notebook(),
    "fairmallm-v3-etb-full": etb_notebook("full"),
    "fairmallm-v3-etb-nofair": etb_notebook("no_fair"),
}

if __name__ == "__main__":
    for name, n in NOTEBOOKS.items():
        json.dump(n, open(os.path.join(HERE, name + ".ipynb"), "w"), indent=1)
        print("wrote", name + ".ipynb")
