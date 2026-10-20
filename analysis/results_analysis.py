# -*- coding: utf-8 -*-
"""
results_analysis.py  --  analysis of the October 2026 runs (after the joint evaluation)

Reads the four synthetic data sets, the processed OULAD tensors and the joint
5-seed evaluation (results_v3.json) and answers questions the evaluation report
does not answer directly:

  A. Activity profile: which weekly behaviours does each generator get right or wrong?
  B. Label signal: does the synthetic data keep the difference between students who
     pass and students who drop out? (explains the utility gap)
  C. Downstream fairness: why are GRUs trained on synthetic data *less* fair overall
     although the synthetic data is fairer? (overall vs within-module SPD)
  D. Convergence and cost of the three LLM runs (rounds, tokens/s, acceptance)
  E. Results against the targets stated in the proposal (scorecard)

Usage
  python analysis/results_analysis.py --data_dir <oulad_processed> \
      --eval_json results/kaggle_runs/fairmallm-v3-evaluate-all/eval_all/results_v3.json \
      --runs full=<run_full> no_fair=<run_no_fair> mallm_flat=<run_mallm_flat> ctgan=<run_ctgan> \
      --out results/analysis
Runs on CPU in about one minute.
"""
import argparse, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_real, get_split, build_matrix, spd, eod, FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--eval_json", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="name=folder with synth_*.npy (and train_state.json/run_log.txt)")
ap.add_argument("--out", default="results/analysis")
args = ap.parse_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
RUNS = dict(r.split("=", 1) for r in args.runs)
NAMES = {"real": "Real (train)", "full": "LLM + D2 (proposed)", "no_fair": "LLM without D2",
         "mallm_flat": "MALLM-GAN (flat)", "ctgan": "CTGAN"}
COL = {"real": "#222222", "full": "#2f6fd6", "no_fair": "#8fb0e8", "mallm_flat": "#3aa17e", "ctgan": "#e0a030"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
SHORT = ["homepage", "subpage", "content", "quiz", "assessment"]
summary = {}

# ---------------------------------------------------------------- data
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
    mb = Ma.astype(bool); r = {"data": NAMES[k], "n": len(ya)}
    for f, nm in enumerate(SHORT):
        r[f"{nm}_active_weeks_%"] = round(100 * float((Xa[:, :, f][mb] > 0).mean()), 1)
    for f, nm in enumerate(SHORT[:3]):
        r[f"{nm}_mean_log"] = round(float(np.log1p(Xa[:, :, f][mb]).mean()), 2)
    r["all_zero_%"] = round(100 * float((Xa.sum((1, 2)) == 0).mean()), 1)
    r["12_weeks_%"] = round(100 * float((Ma.sum(1) == 12).mean()), 1)
    rows.append(r)
prof = pd.DataFrame(rows); prof.to_csv(f"{OUT}/A_activity_profile.csv", index=False)
print(prof.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.16
for i, (k, _) in enumerate(sets.items()):
    vals = [prof.loc[i, f"{nm}_active_weeks_%"] for nm in SHORT]
    ax.bar(np.arange(5) + (i - 2) * w, vals, width=w, color=COL[k], label=NAMES[k])
ax.set_xticks(range(5)); ax.set_xticklabels(SHORT); ax.set_ylabel("% of weeks with activity")
ax.legend(frameon=False, fontsize=8, ncol=3); ax.set_title("How often each activity occurs: real vs synthetic", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig14_activity_profile.png"); plt.close()

# ---------------------------------------------------------------- B. label signal
def zfit(A):
    return StandardScaler().fit(A)
Ftest = build_matrix(X[te], M[te]); yte, Ste, mte = y[te], S[te], module[te]
rows = []
for k, (Xa, Ma, ya, Sa) in sets.items():
    F = build_matrix(Xa, Ma)
    tot = np.log1p(Xa[:, :, :4].sum((1, 2)))
    gap = float(tot[ya == 0].mean() - tot[ya == 1].mean())          # pass minus at-risk, log clicks
    quiz_gap = float((Xa[ya == 0][:, :, 3] > 0).mean() - (Xa[ya == 1][:, :, 3] > 0).mean())
    # in-sample separability (how strongly the label is written into the data)
    sc = zfit(F); lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(F), ya)
    rows.append({"data": NAMES[k], "click_gap_pass_minus_risk_log": round(gap, 2),
                 "quiz_week_gap_pp": round(100 * quiz_gap, 1),
                 "LR_AUC_on_real_test": round(float(roc_auc_score(yte, lr.predict_proba(sc.transform(Ftest))[:, 1])), 3)})
sig = pd.DataFrame(rows); sig.to_csv(f"{OUT}/B_label_signal.csv", index=False); print(sig.to_string(index=False))

# ---------------------------------------------------------------- C. downstream fairness: overall vs within module
def within_module(pred):
    out = {}
    for m in np.unique(mte):
        k = mte == m
        if (Ste[k] == 0).sum() >= 20 and (Ste[k] == 1).sum() >= 20:
            out[m] = spd(pred[k], Ste[k])
    return out

rng = np.random.default_rng(0)
rows = []
for k, (Xa, Ma, ya, Sa) in sets.items():
    reps = 5 if k == "real" else 1
    for r_ in range(reps):
        if k == "real":                                   # real students, same size as the synthetic sets
            idx = rng.choice(len(ya), 2000, replace=False); F = build_matrix(Xa[idx], Ma[idx]); yy = ya[idx]
        else:
            F = build_matrix(Xa, Ma); yy = ya
        sc = zfit(F); lr = LogisticRegression(max_iter=2000).fit(sc.transform(F), yy)
        p = lr.predict_proba(sc.transform(Ftest))[:, 1]
        thr = np.quantile(p, 1 - yte.mean())             # same positive-prediction rate for every model
        pred = (p >= thr).astype(int); wm = within_module(pred)
        sizes = np.array([(mte == m).sum() for m in wm])
        rows.append({"data": NAMES[k], "rep": r_, "SPD_overall": spd(pred, Ste), "EOD_overall": eod(yte, pred, Ste),
                     "SPD_within_module_weighted": float(np.average(list(wm.values()), weights=sizes)),
                     "mean_risk_F": float(p[Ste == 1].mean()), "mean_risk_M": float(p[Ste == 0].mean()), **{f"SPD_{m}": v for m, v in wm.items()}})
fair = pd.DataFrame(rows).groupby("data", sort=False).mean(numeric_only=True).drop(columns="rep").round(3)
fair.to_csv(f"{OUT}/C_overall_vs_within_module_SPD_logreg.csv"); print(fair.T.to_string())
# how differently women and men are distributed over modules in the real test set
comp = pd.crosstab(mte, Ste, normalize="columns").rename(columns={0: "share_of_men", 1: "share_of_women"})
comp["dropout_rate"] = pd.Series(yte).groupby(mte).mean().values
comp["median_window_clicks"] = pd.Series(X[te][:, :, :4].sum((1, 2))).groupby(mte).median().values
comp.round(3).to_csv(f"{OUT}/C_module_composition_test.csv"); print(comp.round(3))

# GRU numbers from the joint evaluation: per-module SPD
ev = json.load(open(args.eval_json))
cond = [("TRTR (all real)", "All real (18,166)"), ("TRTR-n (real n=2000)", "Real n=2,000"),
        ("TSTR full", NAMES["full"]), ("TSTR no_fair", NAMES["no_fair"]),
        ("TSTR mallm_flat", NAMES["mallm_flat"] + " n=623"), ("TSTR ctgan", NAMES["ctgan"])]
mods = sorted(ev["TRTR (all real)"]["SPD_by_module"])
rows = []
for c, lab in cond:
    r = {"condition": lab, "SPD_overall": round(ev[c]["SPD"], 3), "SPD_overall_sd": round(ev[c]["SPD_sd"], 3)}
    r.update({m: round(ev[c]["SPD_by_module"][m], 3) for m in mods})
    r["mean_abs_within_module"] = round(float(np.mean([abs(ev[c]["SPD_by_module"][m]) for m in mods])), 3)
    rows.append(r)
gm = pd.DataFrame(rows); gm.to_csv(f"{OUT}/C_GRU_SPD_by_module.csv", index=False); print(gm.to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axs[0]
for i, (c, lab) in enumerate(cond):
    ax.scatter(gm.loc[i, mods], [i] * len(mods), color="#9a9a9a", s=14, zorder=2)
    ax.scatter(gm.loc[i, "SPD_overall"], i, color="#2f6fd6", s=60, marker="D", zorder=3)
ax.axvline(0, ls="--", color="k", lw=0.8); ax.set_yticks(range(len(cond))); ax.set_yticklabels([l for _, l in cond])
ax.invert_yaxis(); ax.set_xlabel("SPD (positive = men flagged more often)")
ax.set_title("GRU: overall SPD (blue) vs SPD inside each module (grey)", loc="left", fontsize=10)
ax = axs[1]
ax.scatter(comp["share_of_women"] - comp["share_of_men"], comp["median_window_clicks"], color="#2f6fd6")
for m, r in comp.iterrows():
    ax.annotate(m, (r["share_of_women"] - r["share_of_men"], r["median_window_clicks"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
ax.axvline(0, ls=":", color="k", lw=0.8); ax.set_xlabel("share of women minus share of men in module")
ax.set_ylabel("median early clicks"); ax.set_title("Women study in modules with fewer clicks", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig15_spd_within_modules.png"); plt.close()

# ---------------------------------------------------------------- D. convergence and cost of the LLM runs
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True); rows = []
for ax, k in zip(axs, [k for k in ["full", "no_fair", "mallm_flat"] if k in RUNS]):
    st = json.load(open(f"{RUNS[k]}/train_state.json")); h = [x for x in st["history"] if "D1_acc" in x]
    r = [x["round"] for x in h]
    ax.axhspan(0.4, 0.6, color="#eeeeee"); ax.axhline(0.5, ls="--", color="k", lw=0.8)
    ax.plot(r, [x["D1_acc"] for x in h], marker="o", ms=3, color="#2f6fd6", label="D1 realism (acc.)")
    ax.plot(r, [x["D2_auc"] for x in h], marker="o", ms=3, color="#e0a030", label="D2 gender (AUC)")
    ax.set_title(NAMES[k], fontsize=10); ax.set_xlabel("round")
    log = open(f"{RUNS[k]}/run_log.txt").read()
    hrs = re.search(r"done in ([\d.]+) h", log); fc = json.load(open(f"{RUNS[k]}/final_check.json"))
    rej = json.load(open(f"{RUNS[k]}/rejections_final.json"))
    rows.append({"run": NAMES[k], "scored_rounds": len(h), "converged": st.get("converged"), "best_round": st.get("best_round"),
                 "min_D1": round(min(x["D1_acc"] for x in h), 3), "min_D2": round(min(x["D2_auc"] for x in h), 3),
                 "tokens_per_s": round(float(np.mean([x.get("tok_per_s", np.nan) for x in h])), 1),
                 "min_per_round": round(float(np.mean([x.get("minutes", np.nan) for x in h])), 1),
                 "final_acceptance_%": round(100 * rej["accepted"] / rej["requested"], 1),
                 "students": fc["n"], "hours": float(hrs.group(1)) if hrs else 10.0})
axs[0].set_ylabel("score"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("Discriminator scores per round (grey band = target 0.4–0.6)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig16_rounds_all_llm_runs.png"); plt.close()
cost = pd.DataFrame(rows); cost.to_csv(f"{OUT}/D_convergence_and_cost.csv", index=False); print(cost.to_string(index=False))

# ---------------------------------------------------------------- E. size-matched utility and fairness
dq = ev["data_quality"]
ref = {"full": "TRTR-n (real n=1999)", "no_fair": "TRTR-n (real n=2001)", "mallm_flat": "TRTR-n (real n=623)", "ctgan": "TRTR-n (real n=2000)"}
rows = []
for k in [k for k in ["full", "no_fair", "mallm_flat", "ctgan"] if k in dq]:
    rows.append({"data": NAMES[k], "n": dq[k]["n"], "AUC_TSTR": ev[f"TSTR {k}"]["auc"], "AUC_real_same_n": ev[ref[k]]["auc"],
                 "AUC_gap": ev[f"TSTR {k}"]["auc"] - ev[ref[k]]["auc"], "D2": dq[k]["D2_auc"], "D2_real_same_n": dq[k]["D2_auc_real_same_size"],
                 "D2_reduction": dq[k]["D2_auc_real_same_size"] - dq[k]["D2_auc"], "AUC_AUG": ev[f"AUG real-n + {k}"]["auc"],
                 "SPD_TSTR": ev[f"TSTR {k}"]["SPD"], "SPD_real_same_n": ev[ref[k]]["SPD"], "SPD_AUG": ev[f"AUG real-n + {k}"]["SPD"]})
sm = pd.DataFrame(rows).round(3); sm.to_csv(f"{OUT}/E_size_matched.csv", index=False); print(sm.to_string(index=False))
fig, ax = plt.subplots(figsize=(6.6, 4))
for _, r in sm.iterrows():
    k = [kk for kk, v in NAMES.items() if v == r["data"]][0]
    ax.scatter(r["D2_reduction"], r["AUC_gap"], s=40 + r["n"] / 25, color=COL[k], zorder=3)
    ax.annotate(f'{r["data"]} (n={int(r["n"])})', (r["D2_reduction"], r["AUC_gap"]), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("gender leakage removed (D2 real − D2 synthetic)"); ax.set_ylabel("AUC vs same number of real students")
ax.set_title("Size-matched trade-off: right = fairer data, up = more useful", loc="left", fontsize=10)
ax.set_xlim(-0.15, 0.25); fig.tight_layout(); fig.savefig(f"{OUT}/fig17_size_matched_tradeoff.png"); plt.close()

# ---------------------------------------------------------------- F. scorecard against the proposal
f = dq["full"]; tf = ev["TSTR full"]
score = [
    ("RQ1", "D2 AUC on synthetic data", "≈ 0.5", f"{f['D2_auc']:.2f} (real {f['D2_auc_real_same_size']:.2f})", "partly"),
    ("RQ1", "Label SPD of synthetic data", "≈ 0", f"{f['label_SPD']:+.3f} (real {ev['real_reference']['label_SPD_train']:+.3f})", "met"),
    ("RQ1", "D1 accuracy (realism)", "≈ 0.5", f"{f['D1_acc']:.2f}", "not met"),
    ("RQ1", "Feedback loop reaches chance level", "both ≈ 0.5", "no run converged in 10 rounds", "not met"),
    ("RQ1", "SPD / EOD of GRU trained on synthetic data", "≈ 0", f"{tf['SPD']:+.3f} / {tf['EOD']:+.3f}", "not met"),
    ("RQ2", "TSTR AUC close to TRTR at same size", "small gap", f"{tf['auc']:.3f} vs {ev['TRTR-n (real n=1999)']['auc']:.3f}", "not met"),
    ("RQ3", "Accuracy cost of the fairness feedback", "proportionate", f"−{ev['TSTR no_fair']['auc'] - tf['auc']:.3f} AUC for −{dq['no_fair']['D2_auc'] - f['D2_auc']:.3f} D2", "met"),
    ("RQ3", "Temporal format better than flattened baselines", "better", f"CTGAN {ev['TSTR ctgan']['auc']:.3f}, MALLM-GAN {ev['TSTR mallm_flat']['auc']:.3f} (n=623)", "not met"),
    ("RQ4", "Same framework on ETB 2018", "comparable", "LLM run pending (data licence)", "open")]
pd.DataFrame(score, columns=["RQ", "criterion", "target (proposal)", "result", "status"]).to_csv(f"{OUT}/F_scorecard.csv", index=False)
print(pd.DataFrame(score).to_string(index=False))

# ---------------------------------------------------------------- G. weekly patterns of all four sets, split by outcome
fig, axs = plt.subplots(2, 3, figsize=(12, 5.6), sharex=True)
for row, lab, labname in [(0, 0, "students who pass"), (1, 1, "students who withdraw/fail")]:
    for ax, (f, kind, nm) in zip(axs[row], [(0, "mean", "homepage clicks, mean log(1+x)"), (3, "share", "% with quiz activity"), (4, "share", "% submitting an assessment")]):
        for k, (Xa, Ma, ya, Sa) in sets.items():
            def st(w):
                kk = (Ma[:, w] == 1) & (ya == lab); v = Xa[kk, w, f]
                return 100 * (v > 0).mean() if kind == "share" else np.log1p(v).mean()
            ax.plot(range(12), [st(w) for w in range(12)], color=COL[k], lw=2.2 if k == "real" else 1.3,
                    marker=None if k == "real" else "o", ms=2.5, label=NAMES[k])
        ax.set_title(f"{nm}\n{labname}", fontsize=9)
        if row == 1: ax.set_xlabel("week")
axs[0][0].legend(frameon=False, fontsize=7.5)
fig.suptitle("Weekly patterns by outcome: real vs the four synthetic sets", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig18_weekly_patterns_by_outcome.png"); plt.close()

# ---------------------------------------------------------------- H. pilot: clicks relative to the module average
# Idea from C: click volume differs mainly between modules, and women are concentrated in low-click
# modules. Dividing click counts by the module's average weekly clicks (computed on real train only)
# removes this between-module proxy. Quick check with logistic regression on REAL data only.
def module_relative(Xa, ref_idx):
    Xn = Xa.astype(float).copy()
    for m in np.unique(module):
        r = ref_idx[module[ref_idx] == m]
        for f in range(4):
            Xn[module == m, :, f] = Xa[module == m, :, f] / (X[r][:, :, f][M[r] == 1].mean() + 1e-6) * 10
    return Xn
rows = []
for nm, Xa in [("raw weekly counts", X), ("counts relative to module average", module_relative(X, tr))]:
    F = build_matrix(Xa[tr], M[tr]); Ft = build_matrix(Xa[te], M[te]); sc = zfit(F)
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(F), y[tr]); p = lr.predict_proba(sc.transform(Ft))[:, 1]
    pred = (p >= np.quantile(p, 1 - yte.mean())).astype(int)
    rows.append({"features": nm, "AUC": round(float(roc_auc_score(yte, p)), 3), "SPD": round(spd(pred, Ste), 3),
                 "EOD": round(eod(yte, pred, Ste), 3), "mean_within_module_SPD": round(float(np.mean(list(within_module(pred).values()))), 3)})
pilot = pd.DataFrame(rows); pilot.to_csv(f"{OUT}/H_pilot_module_relative_clicks.csv", index=False); print(pilot.to_string(index=False))

# ---------------------------------------------------------------- I. how stable is D2? (20 random subsets per size)
from evaluate_v3 import d2_auc
rows = []
for k in [k for k in ["real", "full", "no_fair", "mallm_flat", "ctgan"] if k in sets]:
    Xa, Ma, ya, Sa = sets[k]
    for n in (623, 1500):
        if n > len(ya):
            continue
        vals = []
        for r_ in range(20):
            ix = np.random.default_rng(100 + r_).choice(len(ya), n, replace=False)
            vals.append(d2_auc(build_matrix(Xa[ix], Ma[ix]), Sa[ix], ya[ix], seed=r_))
        rows.append({"data": NAMES[k], "n": n, "D2_mean": round(float(np.mean(vals)), 3), "D2_sd": round(float(np.std(vals)), 3),
                     "D2_min": round(float(np.min(vals)), 3), "D2_max": round(float(np.max(vals)), 3)})
stab = pd.DataFrame(rows); stab.to_csv(f"{OUT}/I_D2_stability.csv", index=False); print(stab.to_string(index=False))
print("written to", OUT)
