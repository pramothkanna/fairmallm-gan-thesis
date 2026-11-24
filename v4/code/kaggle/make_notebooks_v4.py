# -*- coding: utf-8 -*-
"""Builds SELF-CONTAINED Kaggle notebooks for FairMALLM-GAN v4 (code embedded with
%%writefile, so no code dataset is needed). Generation notebooks need only the input
'oulad-processed'; the joint evaluation additionally needs the OUTPUTS of the generation
notebooks as inputs (Add Input -> Your Work -> Notebooks), CPU is enough.
python make_notebooks_v4.py  -> writes fairmallm-v4-*.ipynb next to this file."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
FILES = ["common.py", "fair_mallm_gan_v4.py", "evaluate_v3.py"]

def md(t):   return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}

FIND = r'''import glob, os, sys, json, torch
os.chdir("/kaggle/working"); sys.path.insert(0, "/kaggle/working")
hits = glob.glob("/kaggle/input/**/oulad_processed/X_seq.npy", recursive=True) or \
       glob.glob("/kaggle/input/**/X_seq.npy", recursive=True)
assert hits, "Attach the dataset 'oulad-processed' (Add Input) first."
DATA_DIR = os.path.dirname(hits[0])
print("DATA_DIR =", DATA_DIR)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.device_count() >= 1, "No GPU: Settings -> Accelerator -> GPU T4 x2"
'''

def notebook(mode, n, rounds, budget, seeds, title, smoke=False):
    intro = (f"# FairMALLM-GAN v4 — {title}\n\n"
             "v4 = v3 + activity plans for quiz/assessment weeks (see the docstring of fair_mallm_gan_v4.py).\n\n"
             "**Settings:** Accelerator **GPU T4 x2**, Internet **on**, input **oulad-processed**.\n\n"
             "Run with *Save Version → Save & Run All (Commit)*. Results: Output tab, `run_*/` and `eval_*/`.")
    cells = [md(intro), code("!pip install -q transformers accelerate\n")]
    for f in FILES:
        cells.append(code(f"%%writefile {f}\n" + open(os.path.join(SRC, f)).read()))
    run = f'''MODE = "{mode}"
from fair_mallm_gan_v4 import Config, run
cfg = Config(DATA_DIR=DATA_DIR, OUT_DIR=f"/kaggle/working/run_{{MODE}}", mode=MODE,
             cohort="active", n_synthetic={n}, max_rounds={rounds},
             prompts_per_batch=8, students_per_prompt=8, time_budget_h={budget})
synth, state = run(cfg)
print(open(f"{{cfg.OUT_DIR}}/final_check.json").read())
'''
    ev = f'''from evaluate_v3 import run as run_eval
res = run_eval(DATA_DIR, f"/kaggle/working/eval_{{MODE}}", {{MODE: cfg.OUT_DIR}},
               cohort="active", seeds={seeds})
print(open(f"/kaggle/working/eval_{{MODE}}/report.md").read())
'''
    cells += [code(FIND), code(run)]
    if not smoke:
        cells.append(code(ev))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

FIND_RUNS = r'''import glob, os, sys, json
os.chdir("/kaggle/working"); sys.path.insert(0, "/kaggle/working")
hits = glob.glob("/kaggle/input/**/oulad_processed/X_seq.npy", recursive=True) or glob.glob("/kaggle/input/**/X_seq.npy", recursive=True)
DATA_DIR = os.path.dirname([h for h in hits if "run_" not in h][0]); print("DATA_DIR", DATA_DIR)
want = {"v4_full": ("fairmallm-v4-full", "run_full"), "v4_no_fair": ("fairmallm-v4-nofair", "run_no_fair"),
        "v3_full": ("fairmallm-v3-full", "run_full"), "v3_no_fair": ("fairmallm-v3-nofair", "run_no_fair"),
        "mallm_flat": ("fairmallm-v3-mallmflat", "run_mallm_flat"), "ctgan": ("fairmallm-v3-ctgan-eval", "run_ctgan")}
runs = {}
for name, (slug, sub) in want.items():
    h = glob.glob(f"/kaggle/input/**/{slug}/**/{sub}/synth_X_seq.npy", recursive=True) or \
        [p for p in glob.glob(f"/kaggle/input/**/{sub}/synth_X_seq.npy", recursive=True) if slug in p]
    if h: runs[name] = os.path.dirname(h[0])
print(json.dumps(runs, indent=1))
assert len(runs) == 6, runs
'''


def joint_eval_notebook():
    intro = ("# FairMALLM-GAN — joint 5-seed evaluation (v4 + v3 + baselines)\n\nCPU only. Inputs: oulad-processed and the outputs of "
             "fairmallm-v4-full, fairmallm-v4-nofair, fairmallm-v3-full, fairmallm-v3-nofair, fairmallm-v3-mallmflat, "
             "fairmallm-v3-ctgan-eval. Same real test students and seeds for every data set. Nothing is regenerated.")
    cells = [md(intro)]
    for f in ["common.py", "evaluate_v3.py"]:
        cells.append(code(f"%%writefile {f}\n" + open(os.path.join(SRC, f)).read()))
    ev = ('from evaluate_v3 import run as run_eval\n'
          'res = run_eval(DATA_DIR, "/kaggle/working/eval_joint", runs, cohort="active", seeds=(0, 1, 2, 3, 4))\n'
          'print(open("/kaggle/working/eval_joint/report.md").read())\n')
    cells += [code(FIND_RUNS), code(ev)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


NB = {
    "fairmallm-v4-smoke": notebook("full", 128, 2, 0.75, (0,), "SMOKE TEST (~40 min): does Qwen follow the activity plans?", smoke=True),
    "fairmallm-v4-full": notebook("full", 2000, 10, 10.0, (0, 1, 2), "proposed model (full)"),
    "fairmallm-v4-nofair": notebook("no_fair", 2000, 10, 10.0, (0, 1, 2), "ablation without D2 (no_fair)"),
    "fairmallm-evaluate-joint": joint_eval_notebook(),
}
if __name__ == "__main__":
    for k, v in NB.items():
        json.dump(v, open(os.path.join(HERE, k + ".ipynb"), "w"), indent=1)
        print("wrote", k + ".ipynb")
