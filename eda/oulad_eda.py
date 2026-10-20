# %% [markdown]
# # Exploratory data analysis of OULAD (before any modelling)
# Pramoth Ponnusamy, Master thesis "Fair and Temporally-Aware LLM-Guided Synthetic Data Generation".
#
# Questions answered here, in this order (this is the order that justifies the preprocessing):
# 1. What does the data look like (outcomes, gender, modules, missing values)?
# 2. Is there gender bias in the *label* (dropout)?
# 3. When do students withdraw -> who should be in an early-warning cohort?
# 4. How are the weekly click features distributed -> why log transform?
# 5. Which activity types carry signal in the first 30 % -> why these 5 features?
# 6. How do the processed features differ between students who pass and who drop out?
#
# Run:  python eda/oulad_eda.py --raw <OULAD csv folder> --processed <oulad_processed folder> --out results/eda
# (works as a notebook too: every "# %%" block is one cell; notebooks/EDA_OULAD.ipynb is generated from this file)

# %%
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, skew, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from common import load_real, FEATURES, active_at_cutoff   # the same helpers the generator uses

try:                                    # notebook: edit these two paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="Dataset/OULAD")
    ap.add_argument("--processed", default="Dataset/oulad_processed")
    ap.add_argument("--out", default="results/eda")
    ap.add_argument("--skip_vle", action="store_true", help="skip the big studentVle.csv part (450 MB)")
    args, _ = ap.parse_known_args()
except SystemExit:
    raise
RAW, PROC, OUT = args.raw, args.processed, args.out
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BLUE, GREY = "#2f6fd6", "#9a9a9a"
summary = {}

# %% [markdown]
# ## 1. Overview: outcomes, missing values
# Label used in the thesis: y = 1 if final_result is Withdrawn or Fail (positive class = at risk).

# %%
info = pd.read_csv(f"{RAW}/studentInfo.csv", na_values="?")          # '?' = missing in OULAD
reg = pd.read_csv(f"{RAW}/studentRegistration.csv", na_values="?")
courses = pd.read_csv(f"{RAW}/courses.csv")
df = (info.merge(courses, on=["code_module", "code_presentation"])
          .merge(reg, on=["code_module", "code_presentation", "id_student"], how="left"))
df["y"] = df.final_result.isin(["Withdrawn", "Fail"]).astype(int)
print("enrolments:", len(df), " distinct students:", df.id_student.nunique())
print(df.final_result.value_counts(normalize=True).round(3))
summary["n_enrolments"] = int(len(df)); summary["n_students"] = int(df.id_student.nunique())
summary["outcomes"] = df.final_result.value_counts(normalize=True).round(4).to_dict()
summary["missing_share"] = {c: round(float(df[c].isna().mean()), 4)
                            for c in ["imd_band", "date_registration", "date_unregistration"]}
print(summary["missing_share"])

# %% [markdown]
# ## 2. Gender and the label
# The label itself is almost unbiased overall, but the module mix differs strongly by gender
# (this is why gender is predictable from engagement and why D2 on real data is not 0.5).

# %%
rows = []
for m, g in df.groupby("code_module"):
    rows.append(dict(module=m, n=len(g), female_share=round((g.gender == "F").mean(), 3),
                     pos_F=round(g.y[g.gender == "F"].mean(), 3), pos_M=round(g.y[g.gender == "M"].mean(), 3)))
mg = pd.DataFrame(rows).sort_values("female_share", ascending=False)
mg.to_csv(f"{OUT}/module_gender.csv", index=False); print(mg)
summary["dropout_by_gender"] = {"F": round(df.y[df.gender == "F"].mean(), 3), "M": round(df.y[df.gender == "M"].mean(), 3)}
summary["label_SPD_real"] = round(summary["dropout_by_gender"]["F"] - summary["dropout_by_gender"]["M"], 3)

fig, ax = plt.subplots(figsize=(7, 3.6)); yy = np.arange(len(mg))[::-1]
for yi, (_, r) in zip(yy, mg.iterrows()):
    ax.plot([r.pos_F, r.pos_M], [yi, yi], color=GREY, lw=1.5, zorder=1)
ax.scatter(mg.pos_M, yy, color=GREY, label="men", zorder=2); ax.scatter(mg.pos_F, yy, color=BLUE, label="women", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([f"{m} ({round(100*s)} % women)" for m, s in zip(mg.module, mg.female_share)])
ax.set_xlabel("share withdrawn or failed"); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.legend(frameon=False, loc="lower right"); ax.set_title("Dropout by module and gender", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig01_dropout_by_module_gender.png"); plt.close()

# association of every demographic with the outcome (Cramer's V)
def cramers_v(a, b):
    t = pd.crosstab(a, b); c = chi2_contingency(t)[0]; n = t.values.sum()
    return round(float(np.sqrt(c / (n * (min(t.shape) - 1)))), 3)
summary["cramers_v_with_result"] = {c: cramers_v(df[c].fillna("missing"), df.final_result)
    for c in ["highest_education", "imd_band", "age_band", "region", "disability", "gender", "code_module"]}
print(summary["cramers_v_with_result"])

# %% [markdown]
# ## 3. When do students withdraw?  -> cohort decision
# 30 % of the module length is the prediction point. A student who unregistered *before* it already has a
# known outcome, so including them makes "early warning" look easier than it is.

# %%
w = df[(df.final_result == "Withdrawn") & df.date_unregistration.notna()].copy()
w["pct"] = 100 * w.date_unregistration / w.module_presentation_length
bins = list(range(-40, 105, 10)); h = np.histogram(w.pct.clip(-40, 99.9), bins=bins)[0]
ut = pd.DataFrame(dict(from_pct=bins[:-1], to_pct=bins[1:], share=h / len(w)))
ut.to_csv(f"{OUT}/unreg_timing.csv", index=False)
summary["withdrawn_before_cutoff_share"] = round(float((w.pct < 30).mean()), 3)
summary["withdrawn_before_day0_share"] = round(float((w.pct <= 0).mean()), 3)
print("withdrawals before the 30 % cutoff:", summary["withdrawn_before_cutoff_share"])
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(ut.from_pct + 5, ut.share * 100, width=9, color=[BLUE if t <= 30 else GREY for t in ut.to_pct])
ax.axvline(0, ls=":", color="k", lw=1); ax.axvline(30, ls="--", color="k", lw=1.2)
ax.set_xlabel("unregistration date, % of module length (first bar: < -30 %)"); ax.set_ylabel("% of withdrawals")
ax.set_title("Timing of withdrawals", loc="left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig02_withdrawal_timing.png"); plt.close()

# %% [markdown]
# ## 4. The processed weekly tensors (output of preprocess_oulad.py)
# X_seq: (enrolments, 12 weeks, 5 features); mask marks the real weeks (11 or 12), padded weeks are 0.

# %%
d = load_real(PROC); X, M, y, S = d["X"], d["mask"], d["y"], d["S"]
act = active_at_cutoff(d["static"]); mb = M.astype(bool)
print("X", X.shape, "| positive rate all:", round(y.mean(), 3), "| still enrolled at cutoff:", int(act.sum()),
      "positive rate:", round(y[act].mean(), 3))
summary["cohort"] = dict(all=int(len(y)), active=int(act.sum()), pos_all=round(float(y.mean()), 3), pos_active=round(float(y[act].mean()), 3))

fs = []
for f, nm in enumerate(FEATURES):
    v = X[:, :, f][mb]
    fs.append(dict(feature=nm, mean=round(float(v.mean()), 2), median=float(np.median(v)),
                   zero_weeks=round(float((v == 0).mean()), 3), p99=float(np.percentile(v, 99)), max=float(v.max()),
                   skew_raw=round(float(skew(v)), 1), skew_log=round(float(skew(np.log1p(v))), 2)))
fs = pd.DataFrame(fs); fs.to_csv(f"{OUT}/feature_stats.csv", index=False); print(fs.to_string(index=False))
# -> heavy right skew + many zero weeks: this is why every model uses log(1+x) and standardisation

fig, axs = plt.subplots(1, 5, figsize=(13, 2.8))
for f, ax in enumerate(axs):
    v = X[:, :, f][mb]; ax.hist(np.log1p(v), bins=40, color=BLUE)
    ax.set_title(FEATURES[f].replace("_", " "), fontsize=9); ax.set_xlabel("log(1+x)")
    ax.text(0.5, 0.85, f"{100*(v==0).mean():.0f} % zeros", transform=ax.transAxes, fontsize=8)
fig.suptitle("Weekly feature distributions (log scale), early-warning window", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig03_feature_distributions.png"); plt.close()

rho = spearmanr(X[mb]).correlation
fig, ax = plt.subplots(figsize=(4.6, 4)); ax.imshow(rho, cmap="Blues", vmin=0, vmax=1)
nm5 = ["homepage", "subpage", "content", "quiz", "score"]
ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(nm5, rotation=45); ax.set_yticklabels(nm5)
for i in range(5):
    for j in range(5): ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if rho[i, j] > .6 else "black")
ax.set_title("Spearman correlation of weekly features", loc="left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig04_feature_correlation.png"); plt.close()

# %% [markdown]
# ## 5. Weekly engagement by outcome (students still enrolled at the cutoff)

# %%
fig, axs = plt.subplots(1, 5, figsize=(13, 2.8)); wm = []
for f, ax in enumerate(axs):
    for lab, c, nm in [(0, GREY, "pass"), (1, BLUE, "withdraw/fail")]:
        k = act & (y == lab)
        ax.plot([np.log1p(X[k & (M[:, w] == 1), w, f]).mean() for w in range(12)], color=c, label=nm, marker="o", ms=3)
    ax.set_title(FEATURES[f].replace("_", " "), fontsize=9); ax.set_xlabel("week")
axs[0].set_ylabel("mean log(1+x)"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("Weekly engagement by outcome, students still enrolled at the cutoff", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig05_weekly_engagement_by_outcome.png"); plt.close()
for lab in (0, 1):
    for wk in range(12):
        k = act & (y == lab) & (M[:, wk] == 1)
        r = dict(label=lab, week=wk); r.update({nm: round(float(X[k, wk, f].mean()), 2) for f, nm in enumerate(FEATURES)})
        r["assessment_share"] = round(float((X[k, wk, 4] > 0).mean()), 3); wm.append(r)
pd.DataFrame(wm).to_csv(f"{OUT}/window_weekly_by_label_active.csv", index=False)
tot = X.sum((1, 2))
summary["all_zero_students_share_of_positives"] = {"all": round(float((tot[y == 1] == 0).mean()), 3),
                                                   "active_cohort": round(float((tot[act & (y == 1)] == 0).mean()), 3)}
clicks = X[:, :, :4].sum((1, 2))
summary["median_window_clicks_by_gender"] = {"F": float(np.median(clicks[S == 1])), "M": float(np.median(clicks[S == 0]))}

# %% [markdown]
# ## 6. VLE activity types and cumulative signal (needs the 450 MB studentVle.csv, read in chunks)
# Why these features? Which click types carry signal in the first 30 %, and when does the signal appear?

# %%
if not args.skip_vle and os.path.exists(f"{RAW}/studentVle.csv"):
    vle = pd.read_csv(f"{RAW}/vle.csv", na_values="?")[["id_site", "code_module", "code_presentation", "activity_type"]]
    st = d["static"][["student_key", "code_module", "code_presentation", "id_student", "cutoff_days"]].copy()
    st["y"], st["act"] = y, act
    tmp = []
    for ch in pd.read_csv(f"{RAW}/studentVle.csv", usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
                          chunksize=2_000_000):
        ch = ch.merge(vle, on=["id_site", "code_module", "code_presentation"], how="left")
        ch["week"] = (ch.date.clip(lower=0) // 7).astype(int)
        tmp.append(ch.groupby(["code_module", "code_presentation", "id_student", "activity_type", "week"], as_index=False).sum_click.sum())
    g = pd.concat(tmp).groupby(["code_module", "code_presentation", "id_student", "activity_type", "week"], as_index=False).sum_click.sum()
    g = g.merge(st, on=["code_module", "code_presentation", "id_student"])
    g["in_window"] = g.week * 7 < g.cutoff_days
    # (a) activity types inside the early window
    e = g[g.in_window].groupby(["student_key", "activity_type"], as_index=False).sum_click.sum()
    tot_by = e.groupby("activity_type").sum_click.sum(); share = tot_by / tot_by.sum()
    base = st.set_index("student_key"); res = []
    for a, sub in e.groupby("activity_type"):
        per = sub.set_index("student_key").sum_click.reindex(base.index).fillna(0)
        res.append(dict(activity_type=a, early_click_share=share[a],
                        auc_all=roc_auc_score(base.y, per), auc_active=roc_auc_score(base.y[base.act], per[base.act]),
                        share_students_using=float((per > 0).mean())))
    at = pd.DataFrame(res).sort_values("early_click_share", ascending=False)
    at.to_csv(f"{OUT}/early_activity_types.csv", index=False); print(at.head(8))
    used = {"oucontent", "homepage", "quiz", "subpage"}; top = at.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.4)); ax.barh(top.activity_type, top.early_click_share * 100, color=[BLUE if t in used else GREY for t in top.activity_type])
    for i, (s_, a_) in enumerate(zip(top.early_click_share, top.auc_active)): ax.text(s_ * 100 + .4, i, f"{s_*100:.1f} %  (AUC {a_:.2f})", va="center", fontsize=8)
    ax.set_xlabel("% of clicks in the first 30 %"); ax.set_title("Activity types in the early window (blue = used as feature)", loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig10_activity_types.png"); plt.close()
    # (b) cumulative AUC over the whole course
    wk = g.groupby(["student_key", "week"], as_index=False).sum_click.sum()
    mat = wk.pivot(index="student_key", columns="week", values="sum_click").reindex(base.index).fillna(0)
    cum = mat.cumsum(axis=1); rows = []
    for w_ in range(min(40, cum.shape[1])):
        c = cum.iloc[:, w_]
        rows.append(dict(week=w_, auc_all=roc_auc_score(base.y, c), auc_active=roc_auc_score(base.y[base.act], c[base.act])))
    ca = pd.DataFrame(rows); ca.to_csv(f"{OUT}/cumulative_auc_by_week.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.2)); ax.plot(ca.week, ca.auc_all, color=GREY, label="all students")
    ax.plot(ca.week, ca.auc_active, color=BLUE, label="still enrolled at cutoff"); ax.axvline(11, ls="--", color="k", lw=1)
    ax.set_xlabel("clicks counted up to week"); ax.set_ylabel("ROC-AUC"); ax.legend(frameon=False)
    ax.set_title("Predictive signal of cumulative clicks", loc="left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig09_cumulative_auc.png"); plt.close()
else:
    print("studentVle.csv not found / skipped: section 6 not run")

# %%
json.dump(summary, open(f"{OUT}/eda_summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
