# -*- coding: utf-8 -*-
"""
results_analysis_v4.py  --  analysis of the v4.3 runs (21 Oct 2026) next to the v3 runs

Same questions as results_analysis.py (A activity profile, B label signal, C fairness overall vs
within modules), but for any set of named runs and with the GRU numbers taken from the JOINT
5-seed evaluation (kaggle/fairmallm-evaluate-joint.ipynb -> eval_joint/results_v3.json).

Usage (CPU, about one minute)
  python analysis/results_analysis_v4.py --data_dir <oulad_processed> \
      --eval_json results/kaggle_runs/fairmallm-evaluate-joint/eval_joint/results_v3.json \
      --runs v4_full=<v4 run_full> v4_no_fair=<v4 run_no_fair> v3_full=<v3 run_full> \
             v3_no_fair=<v3 run_no_fair> ctgan=<run_ctgan> \
      --out results/analysis_v4
Run names must match the names used in the joint evaluation (TSTR <name>).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_real, get_split, build_matrix, spd, eod

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--eval_json", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="name=folder with synth_*.npy")
ap.add_argument("--out", default="results/analysis_v4")
args = ap.parse_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
RUNS = dict(r.split("=", 1) for r in args.runs)
NAMES = {"real": "Real (train)", "v4_full": "v4.3 + D2 (proposed)", "v4_no_fair": "v4.3 without D2",
         "v3_full": "v3 + D2", "v3_no_fair": "v3 without D2", "mallm_flat": "MALLM-GAN (flat)", "ctgan": "CTGAN"}
COL = {"real": "#222222", "v4_full": "#1f5fbf", "v4_no_fair": "#7fa7e3", "v3_full": "#b0b0b0",
       "v3_no_fair": "#d6d6d6", "mallm_flat": "#3aa17e", "ctgan": "#e0a030"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
SHORT = ["homepage", "subpage", "content", "quiz", "assessment"]

d = load_real(args.data_dir)
tr, va, te = get_split(d, "active")
X, M, y, S = d["X"], d["mask"].astype(int), d["y"].astype(int), d["S"].astype(int)
module = d["static"]["code_module"].values
sets = {"real": (X[tr], M[tr], y[tr], S[tr])}
for k, p in RUNS.items():
    sets[k] = (np.load(f"{p}/synth_X_seq.npy"), np.load(f"{p}/synth_mask.npy").astype(int),
               np.load(f"{p}/synth_y.npy").astype(int), np.load(f"{p}/synth_S_gender.npy").astype(int))

# ---------------------------------------------------------------- A. activity profile
rows = []
for k, (Xa, Ma, ya, Sa) in sets.items():
    mb = Ma.astype(bool); r = {"data": NAMES.get(k, k), "n": len(ya)}
    for f, nm in enumerate(SHORT):
        r[f"{nm}_active_weeks_%"] = round(100 * float((Xa[:, :, f][mb] > 0).mean()), 1)
    r["quiz_mean_when_active"] = round(float(Xa[:, :, 3][mb][Xa[:, :, 3][mb] > 0].mean()), 1)
    r["score_mean_when_submitted"] = round(float(Xa[:, :, 4][mb][Xa[:, :, 4][mb] > 0].mean()), 1)
    for f, nm in enumerate(SHORT[:3]):
        r[f"{nm}_mean_log"] = round(float(np.log1p(Xa[:, :, f][mb]).mean()), 2)
    r["all_zero_%"] = round(100 * float((Xa.sum((1, 2)) == 0).mean()), 1)
    rows.append(r)
prof = pd.DataFrame(rows); prof.to_csv(f"{OUT}/A_activity_profile_v4.csv", index=False)
print(prof.to_string(index=False))

keys = list(sets.keys()); w = 0.8 / len(keys)
fig, ax = plt.subplots(figsize=(9, 3.6))
for i, k in enumerate(keys):
    vals = [prof.loc[i, f"{nm}_active_weeks_%"] for nm in SHORT]
    ax.bar(np.arange(5) + (i - (len(keys) - 1) / 2) * w, vals, width=w, color=COL.get(k, "#888"), label=NAMES.get(k, k))
ax.set_xticks(range(5)); ax.set_xticklabels(SHORT); ax.set_ylabel("% of weeks with activity")
ax.legend(frameon=False, fontsize=8, ncol=3); ax.set_title("How often each activity occurs: real vs synthetic (v3 and v4.3)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig19_activity_profile_v4.png"); plt.close()

# ---------------------------------------------------------------- B. label signal
Ftest = build_matrix(X[te], M[te]); yte, Ste, mte = y[te], S[te], module[te]
rows = []
for k, (Xa, Ma, ya, Sa) in sets.items():
    F = build_matrix(Xa, Ma)
    tot = np.log1p(Xa[:, :, :4].sum((1, 2)))
    gap = float(tot[ya == 0].mean() - tot[ya == 1].mean())
    quiz_gap = float((Xa[ya == 0][:, :, 3] > 0).mean() - (Xa[ya == 1][:, :, 3] > 0).mean())
    sc = StandardScaler().fit(F); lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(F), ya)
    rows.append({"data": NAMES.get(k, k), "click_gap_pass_minus_risk_log": round(gap, 2),
                 "quiz_week_gap_pp": round(100 * quiz_gap, 1),
                 "LR_AUC_on_real_test": round(float(roc_auc_score(yte, lr.predict_proba(sc.transform(Ftest))[:, 1])), 3)})
sig = pd.DataFrame(rows); sig.to_csv(f"{OUT}/B_label_signal_v4.csv", index=False); print(sig.to_string(index=False))

# ---------------------------------------------------------------- C. GRU fairness overall vs within module (joint evaluation)
ev = json.load(open(args.eval_json))
mods = sorted(ev["TRTR (all real)"]["SPD_by_module"])
cond = [("TRTR (all real)", "All real (18,166)"), ("TRTR-n (real n=2000)", "Real n=2,000")]
cond += [(f"TSTR {k}", NAMES.get(k, k)) for k in ["v4_full", "v4_no_fair", "v3_full", "v3_no_fair", "mallm_flat", "ctgan"]
         if f"TSTR {k}" in ev]
comp_te = pd.crosstab(mte, Ste)                       # students per module x gender in the real test set
rows = []
for c, lab in cond:
    e = ev[c]; bm = e["SPD_by_module"]
    wts = np.array([comp_te.loc[m].sum() for m in mods], float)
    r = {"condition": lab, "AUC": round(e["auc"], 3), "SPD_overall": round(e["SPD"], 3), "SPD_overall_sd": round(e["SPD_sd"], 3),
         "EOD_overall": round(e["EOD"], 3)}
    r.update({m: round(bm[m], 3) for m in mods})
    r["SPD_within_module_weighted"] = round(float(np.average([bm[m] for m in mods], weights=wts)), 3)
    r["mean_abs_within_module"] = round(float(np.mean([abs(bm[m]) for m in mods])), 3)
    rows.append(r)
gm = pd.DataFrame(rows); gm.to_csv(f"{OUT}/C_GRU_SPD_by_module_v4.csv", index=False); print(gm.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.9))
for i in range(len(gm)):
    ax.scatter(gm.loc[i, mods], [i] * len(mods), color="#9a9a9a", s=14, zorder=2)
    ax.scatter(gm.loc[i, "SPD_within_module_weighted"], i, color="#3aa17e", s=46, marker="s", zorder=3)
    ax.scatter(gm.loc[i, "SPD_overall"], i, color="#1f5fbf", s=60, marker="D", zorder=4)
ax.axvline(0, ls="--", color="k", lw=0.8); ax.set_yticks(range(len(gm))); ax.set_yticklabels(gm["condition"])
ax.invert_yaxis(); ax.set_xlabel("SPD (positive = men flagged more often)")
ax.set_title("GRU: overall SPD (blue), size-weighted within-module SPD (green), each module (grey)", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig20_spd_within_modules_v4.png"); plt.close()
print("done ->", OUT)
