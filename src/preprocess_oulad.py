# -*- coding: utf-8 -*-
"""
preprocess_oulad.py  --  script version of Weekly_binning.ipynb (v3)
=====================================================================
Identical logic to the notebook that produced oulad_processed/, with ONE fix:
all OULAD csv files are read with na_values="?" (OULAD encodes missing as "?").
In the notebook, studentRegistration.date_unregistration stayed a string column,
which silently broke the DROP_EARLY_WITHDRAWALS switch (it was False, so the
saved outputs are NOT affected). Early withdrawals are now handled at analysis
time by prepare_cohort.py instead.

Usage (e.g. on Kaggle, where the raw OULAD csvs are available):
    python preprocess_oulad.py --data_dir <raw OULAD folder> --out_dir <output folder>
"""
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--data_dir", required=True)
_ap.add_argument("--out_dir", required=True)
_args = _ap.parse_args()



# ======================================================================================
#  STEP 0 — CONFIGURATION
# ======================================================================================


import os, json, math, numpy as np, pandas as pd
DATA_DIR = _args.data_dir
OUT_DIR  = _args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)


# --- Early-warning cutoff (proposal: "fixed at 30% of module length") ----------------
CUTOFF_FRACTION = 0.30

# --- The FIVE features (proposal Section 3.2). First four come from studentVle
#     activity_type; the fifth ("weekly assessment score") comes from studentAssessment.
ACTIVITY_MAP = {
    "homepage_clicks":  ["homepage"],
    "subpage_clicks":   ["subpage"],
    "oucontent_clicks": ["oucontent"],
    "quiz_interactions":["quiz"],          # add "externalquiz" here if you want a broader quiz signal
}
CLICK_FEATURES = list(ACTIVITY_MAP.keys())               # 4 click-based features
ASSESS_FEATURE = "weekly_assessment_score"               # 5th feature
FEATURES = CLICK_FEATURES + [ASSESS_FEATURE]             # final 5-feature order

# --- Label definition (binary at-risk target for early dropout prediction) -----------
#   "withdrawn_or_fail" : y=1 if final_result in {Withdrawn, Fail}   (common OULAD at-risk def.)
#   "withdrawn_only"    : y=1 if final_result == Withdrawn           (pure dropout def.)
LABEL_MODE = "withdrawn_or_fail"

# --- Protected attribute encoding ----------------------------------------------------
#   S = 1 for Female, 0 for Male  (choice is arbitrary but must be documented for SPD/EOD)
GENDER_POSITIVE = "F"

# --- Preprocessing switches (all documented) -----------------------------------------
DROP_NEGATIVE_DAYS      = True    # drop pre-start interactions (day < 0); week 0 = start of presentation
EXCLUDE_BANKED_ASSESS   = True    # drop is_banked==1 (score transferred from a previous presentation)
WEIGHTED_ASSESS_SCORE   = False   # False = simple mean of scores in a week; True = weight-weighted mean
DROP_EARLY_WITHDRAWALS  = False   # if True, drop students who unregistered BEFORE the cutoff (incomplete window)
RANDOM_SEED             = 42

np.random.seed(RANDOM_SEED)


# ======================================================================================
#  STEP 1 — LOAD THE OULAD TABLES (memory-efficient dtypes for the 443 MB studentVle)
# ======================================================================================
def p(name): return os.path.join(DATA_DIR, name)

print("Loading tables ...")
courses = pd.read_csv(p("courses.csv"), na_values="?")                       # code_module, code_presentation, module_presentation_length
studentInfo = pd.read_csv(p("studentInfo.csv"), na_values="?")               # triple + demographics + final_result
studentReg  = pd.read_csv(p("studentRegistration.csv"), na_values="?")       # triple + date_registration + date_unregistration
assessments = pd.read_csv(p("assessments.csv"), na_values="?")               # id_assessment -> module/presentation, date, weight
studentAss  = pd.read_csv(p("studentAssessment.csv"), na_values="?")         # id_assessment, id_student, date_submitted, is_banked, score

for col in ["score", "date_submitted"]:
  if col in studentAss.columns:
    studentAss[col] = pd.to_numeric(studentAss[col], errors="coerce")

for col in ["date", "weight"]:
  if col in assessments.columns:
    assessments[col] = pd.to_numeric(assessments[col], errors="coerce")

# The length column is named "module_presentation_length" in OULAD; normalise to "length".
if "module_presentation_length" in courses.columns:
    courses = courses.rename(columns={"module_presentation_length": "length"})

studentVle = pd.read_csv(
    p("studentVle.csv"),
    usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
    dtype={"code_module": "category", "code_presentation": "category",
           "id_student": "int32", "id_site": "int32", "date": "int32", "sum_click": "int32"},
)
vle = pd.read_csv(
    p("vle.csv"),
    usecols=["id_site", "code_module", "code_presentation", "activity_type"],
    dtype={"id_site": "int32", "code_module": "category",
           "code_presentation": "category", "activity_type": "category"},
)
print(f"  studentVle rows: {len(studentVle):,} | studentInfo rows: {len(studentInfo):,}")

# ======================================================================================
#  STEP 2 — ENTITY KEY, STATIC ATTRIBUTES, LABEL, GENDER
#  NOTE: id_student is NOT globally unique. The true key is (module, presentation, student).
# ======================================================================================
def make_key(df):
    return (df["code_module"].astype(str) + "|" +
            df["code_presentation"].astype(str) + "|" +
            df["id_student"].astype(str))

studentInfo["student_key"] = make_key(studentInfo)
studentReg["student_key"]  = make_key(studentReg)

# --- Binary label -------------------------------------------------------------------
def to_label(fr):
    if LABEL_MODE == "withdrawn_only":
        return int(fr == "Withdrawn")
    return int(fr in ("Withdrawn", "Fail"))          # default: withdrawn_or_fail
studentInfo["label"] = studentInfo["final_result"].map(to_label).astype("int8")

# --- Protected attribute S ----------------------------------------------------------
studentInfo["S_gender"] = (studentInfo["gender"] == GENDER_POSITIVE).astype("int8")

# --- Merge module length + 30% cutoff (in weeks) per presentation -------------------
courses = courses.copy()
courses["n_weeks_total"] = np.ceil(courses["length"] / 7).astype(int)
courses["cutoff_days"]   = (CUTOFF_FRACTION * courses["length"]).round().astype(int)
courses["n_weeks_keep"]  = np.ceil(CUTOFF_FRACTION * courses["length"] / 7).astype(int)  # weeks 0..n_weeks_keep-1
GLOBAL_T = int(courses["n_weeks_keep"].max())          # rectangular sequence length for the GRU
print(f"  Per-presentation weeks kept (30% cutoff): "
      f"{courses['n_weeks_keep'].min()}–{courses['n_weeks_keep'].max()}  ->  GLOBAL_T = {GLOBAL_T}")

cutoff_cols = ["code_module", "code_presentation", "length", "cutoff_days", "n_weeks_keep"]
studentInfo = studentInfo.merge(courses[cutoff_cols], on=["code_module", "code_presentation"], how="left")

# --- Attach unregistration date (for optional early-withdrawal filter) --------------
studentInfo = studentInfo.merge(
    studentReg[["student_key", "date_registration", "date_unregistration"]],
    on="student_key", how="left")

if DROP_EARLY_WITHDRAWALS:
    before = len(studentInfo)
    keep = ~(studentInfo["date_unregistration"].notna() &
             (studentInfo["date_unregistration"] < studentInfo["cutoff_days"]))
    studentInfo = studentInfo[keep].copy()
    print(f"  Dropped {before - len(studentInfo):,} students who unregistered before the cutoff.")

# Static table (one row per student) -------------------------------------------------
STATIC_COLS = ["student_key", "code_module", "code_presentation", "id_student",
               "gender", "S_gender", "region", "highest_education", "imd_band",
               "age_band", "num_of_prev_attempts", "studied_credits", "disability",
               "final_result", "label", "length", "cutoff_days", "n_weeks_keep",
               "date_registration", "date_unregistration"]
static_df = studentInfo[STATIC_COLS].drop_duplicates("student_key").reset_index(drop=True)
valid_keys = set(static_df["student_key"])
print(f"  Cohort size (students): {len(static_df):,} | positive-label rate: {static_df['label'].mean():.3f}")



# ======================================================================================
#  STEP 3 — WEEKLY CLICK FEATURES (join studentVle -> vle to recover activity_type)
# ======================================================================================
print("Building weekly click features ...")
# activity_type lives in the vle catalogue, keyed by (module, presentation, id_site).
sv = studentVle.merge(vle, on=["code_module", "code_presentation", "id_site"], how="left")

# Keep only the activity types we need (drastically shrinks 10.6M rows before aggregation).
wanted_types = [t for lst in ACTIVITY_MAP.values() for t in lst]
sv = sv[sv["activity_type"].isin(wanted_types)].copy()

# Map raw activity_type -> our feature name.
type_to_feat = {t: feat for feat, lst in ACTIVITY_MAP.items() for t in lst}
sv["feature"] = sv["activity_type"].astype(str).map(type_to_feat)

# Entity key + week index (week 0 = first 7 days from presentation start).
sv["student_key"] = make_key(sv)
if DROP_NEGATIVE_DAYS:
    sv = sv[sv["date"] >= 0]
sv["week"] = (sv["date"] // 7).astype(int)

# Restrict to this cohort and to each presentation's own early-warning window.
sv = sv[sv["student_key"].isin(valid_keys)]
sv = sv.merge(static_df[["student_key", "n_weeks_keep"]], on="student_key", how="left")
sv = sv[sv["week"] < sv["n_weeks_keep"]]

# Aggregate clicks -> (student, week, feature) then pivot features to columns.
clicks = (sv.groupby(["student_key", "week", "feature"], observed=True)["sum_click"]
            .sum().reset_index())
clicks_wide = clicks.pivot_table(index=["student_key", "week"], columns="feature",
                                 values="sum_click", fill_value=0).reset_index()
for f in CLICK_FEATURES:                      # ensure all 4 click columns exist
    if f not in clicks_wide.columns:
        clicks_wide[f] = 0
print(f"  Click rows (student-week): {len(clicks_wide):,}")


# ======================================================================================
#  STEP 4 — WEEKLY ASSESSMENT SCORE (5th feature) from studentAssessment + assessments
# ======================================================================================
print("Building weekly assessment score ...")
ass = studentAss.merge(
    assessments[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
    on="id_assessment", how="left")

if EXCLUDE_BANKED_ASSESS:
    ass = ass[ass["is_banked"] != 1]

ass["student_key"] = make_key(ass)
ass = ass[ass["student_key"].isin(valid_keys)].copy()

# Bin by the day the student actually submitted (aligns the score with observed engagement).
ass = ass[ass["date_submitted"].notna()]
if DROP_NEGATIVE_DAYS:
    ass = ass[ass["date_submitted"] >= 0]
ass["week"] = (ass["date_submitted"] // 7).astype(int)
ass = ass.merge(static_df[["student_key", "n_weeks_keep"]], on="student_key", how="left")
ass = ass[ass["week"] < ass["n_weeks_keep"]]
ass = ass[ass["score"].notna()]

if WEIGHTED_ASSESS_SCORE:
    ass["w"] = ass["weight"].fillna(1.0).replace(0, 1.0)
    ass["ws"] = ass["score"] * ass["w"]
    grp = ass.groupby(["student_key", "week"], observed=True)
    assess_week = (grp["ws"].sum() / grp["w"].sum()).reset_index()
    assess_week.columns = ["student_key", "week", ASSESS_FEATURE]
else:
    assess_week = (ass.groupby(["student_key", "week"], observed=True)["score"]
                     .mean().reset_index().rename(columns={"score": ASSESS_FEATURE}))
print(f"  Assessment rows (student-week): {len(assess_week):,}")


# ======================================================================================
#  STEP 5 — ASSEMBLE THE COMPLETE RECTANGULAR WEEKLY PANEL (student × week × 5 features)
#  Missing weeks/features are real zero-activity -> filled with 0 (documented).
# ======================================================================================
print("Assembling full weekly panel ...")
panel = clicks_wide.merge(assess_week, on=["student_key", "week"], how="outer")

# Build the complete (student, week) grid up to each student's own cutoff, then pad to GLOBAL_T.
grid = (static_df[["student_key", "n_weeks_keep"]]
        .assign(week=lambda d: d["n_weeks_keep"].apply(lambda k: list(range(GLOBAL_T))))
        .explode("week"))
grid["week"] = grid["week"].astype(int)
grid["in_window"] = (grid["week"] < grid["n_weeks_keep"]).astype("int8")   # mask: 1 real week, 0 padded

panel = grid.merge(panel, on=["student_key", "week"], how="left")
for f in FEATURES:
    if f not in panel.columns:
        panel[f] = 0.0
    panel[f] = panel[f].fillna(0.0)
# Zero-out any values that fall in padded (out-of-window) weeks.
panel.loc[panel["in_window"] == 0, FEATURES] = 0.0

# Attach static context (gender, label, demographics) to the long panel for generation use.
panel = panel.merge(
    static_df[["student_key", "code_module", "code_presentation", "id_student",
               "gender", "S_gender", "label", "region", "highest_education",
               "imd_band", "age_band", "disability"]],
    on="student_key", how="left")
panel = panel.sort_values(["student_key", "week"]).reset_index(drop=True)

# ======================================================================================
#  STEP 6 — SAVE ARTEFACTS
#  (A) generation-ready (before encoding): long panel, wide panel, static, serialized text
#  (B) GRU-ready: X_seq (N,T,5), mask, y, S, ids, splits, optional scaled tensor
# ======================================================================================
print("Saving artefacts ...")

# ---- (A) LONG / TIDY PANEL — raw counts, before any encoding/scaling ----------------
long_cols = (["student_key", "code_module", "code_presentation", "id_student", "week"]
             + FEATURES + ["S_gender", "gender", "label", "in_window",
                           "region", "highest_education", "imd_band", "age_band", "disability"])
panel[long_cols].to_csv(os.path.join(OUT_DIR, "weekly_panel_long.csv"), index=False)

# ---- WIDE PANEL — one row per student, features flattened (feeds CTGAN / MALLM-GAN flattened baselines)
wide = panel.pivot_table(index="student_key", columns="week", values=FEATURES)
wide.columns = [f"{feat}_w{int(w)}" for feat, w in wide.columns]      # e.g. homepage_clicks_w0
wide = wide.reset_index().merge(
    static_df[["student_key", "S_gender", "label"]], on="student_key", how="left")
wide.to_csv(os.path.join(OUT_DIR, "weekly_panel_wide.csv"), index=False)

# ---- STATIC table -------------------------------------------------------------------
static_df.to_csv(os.path.join(OUT_DIR, "student_static.csv"), index=False)

# ---- (B) 3D TENSOR for the GRU: X_seq (N, T, 5), mask (N, T), y (N,), S (N,) ---------
keys = static_df["student_key"].tolist()
key_to_idx = {k: i for i, k in enumerate(keys)}
N, T, F = len(keys), GLOBAL_T, len(FEATURES)

X_seq = np.zeros((N, T, F), dtype=np.float32)
mask  = np.zeros((N, T), dtype=np.int8)

# Fill from the long panel (vectorised).
pi = panel.copy()
pi["si"] = pi["student_key"].map(key_to_idx)
for fj, f in enumerate(FEATURES):
    X_seq[pi["si"].values, pi["week"].values, fj] = pi[f].values.astype(np.float32)
mask[pi["si"].values, pi["week"].values] = pi["in_window"].values.astype(np.int8)

y = static_df.set_index("student_key").loc[keys, "label"].values.astype(np.int8)
S = static_df.set_index("student_key").loc[keys, "S_gender"].values.astype(np.int8)

np.save(os.path.join(OUT_DIR, "X_seq.npy"), X_seq)     # RAW counts (before encoding) — use for generation & GRU
np.save(os.path.join(OUT_DIR, "mask.npy"),  mask)
np.save(os.path.join(OUT_DIR, "y.npy"),     y)
np.save(os.path.join(OUT_DIR, "S_gender.npy"), S)
np.save(os.path.join(OUT_DIR, "student_keys.npy"), np.array(keys))

# ---- STRATIFIED train/val/test split (stratify jointly by label × gender) -----------
strata = (y.astype(int) * 2 + S.astype(int))
rng = np.random.default_rng(RANDOM_SEED)
train_idx, val_idx, test_idx = [], [], []
for s in np.unique(strata):
    idx = np.where(strata == s)[0]
    rng.shuffle(idx)
    n = len(idx); n_tr = int(0.70 * n); n_va = int(0.15 * n)
    train_idx += idx[:n_tr].tolist()
    val_idx   += idx[n_tr:n_tr + n_va].tolist()
    test_idx  += idx[n_tr + n_va:].tolist()
splits = {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}
with open(os.path.join(OUT_DIR, "splits.json"), "w") as fjs:
    json.dump(splits, fjs)

# ---- OPTIONAL normalized tensor for the GRU (fit on TRAIN real weeks only; no leakage)
tr = np.array(splits["train"])
tr_mask = mask[tr].astype(bool)
means = np.zeros(F, dtype=np.float32); stds = np.ones(F, dtype=np.float32)
for fj in range(F):
    vals = X_seq[tr][..., fj][tr_mask]
    means[fj] = vals.mean() if vals.size else 0.0
    stds[fj]  = vals.std()  if vals.size and vals.std() > 1e-6 else 1.0
X_scaled = ((X_seq - means) / stds).astype(np.float32)
X_scaled[mask == 0] = 0.0                       # keep padded weeks at 0
np.save(os.path.join(OUT_DIR, "X_seq_scaled.npy"), X_scaled)

with open(os.path.join(OUT_DIR, "scaler_stats.json"), "w") as fjs:
    json.dump({"feature_order": FEATURES,
               "mean": means.tolist(), "std": stds.tolist()}, fjs, indent=2)

# ---- (A) OPTIONAL SDForger-style serialization (short structured text per student) --
def serialize_student(g):
    g = g.sort_values("week")
    win = g[g["in_window"] == 1]
    weeks_txt = "; ".join(
        f"w{int(r.week)}[hp={int(r.homepage_clicks)},sp={int(r.subpage_clicks)},"
        f"oc={int(r.oucontent_clicks)},qz={int(r.quiz_interactions)},"
        f"score={r.weekly_assessment_score:.0f}]"
        for r in win.itertuples(index=False))
    head = g.iloc[0]
    return (f"gender={head.gender}; module={head.code_module}; "
            f"label={'at_risk' if head.label == 1 else 'not_at_risk'}; {weeks_txt}")

serialized = (panel.groupby("student_key", group_keys=False)
              .apply(serialize_student, include_groups=False)
              .reset_index().rename(columns={0: "text"}))
serialized.to_csv(os.path.join(OUT_DIR, "serialized_panels.csv"), index=False)

# ---- Metadata (feature order, config, shapes) — reproducibility ---------------------
meta = {
    "feature_order": FEATURES,
    "protected_attribute": "S_gender (1=%s)" % GENDER_POSITIVE,
    "label_mode": LABEL_MODE,
    "cutoff_fraction": CUTOFF_FRACTION,
    "GLOBAL_T_weeks": GLOBAL_T,
    "n_students": int(N),
    "positive_label_rate": float(y.mean()),
    "female_rate": float(S.mean()),
    "switches": {
        "DROP_NEGATIVE_DAYS": DROP_NEGATIVE_DAYS,
        "EXCLUDE_BANKED_ASSESS": EXCLUDE_BANKED_ASSESS,
        "WEIGHTED_ASSESS_SCORE": WEIGHTED_ASSESS_SCORE,
        "DROP_EARLY_WITHDRAWALS": DROP_EARLY_WITHDRAWALS,
    },
    "shapes": {"X_seq": list(X_seq.shape), "mask": list(mask.shape),
               "y": list(y.shape), "S_gender": list(S.shape)},
    "files": {
        "generation_before_encoding": ["weekly_panel_long.csv", "serialized_panels.csv",
                                        "weekly_panel_wide.csv", "student_static.csv"],
        "gru_ready": ["X_seq.npy", "X_seq_scaled.npy", "mask.npy", "y.npy",
                      "S_gender.npy", "student_keys.npy", "splits.json", "scaler_stats.json"],
    },
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as fjs:
    json.dump(meta, fjs, indent=2)




# ======================================================================================
#  STEP 7 — SANITY CHECKS
# ======================================================================================
print("\n================= SUMMARY =================")
print(f"Students (N)           : {N:,}")
print(f"Sequence length (T)    : {T} weeks   (5 features)")
print(f"X_seq shape            : {X_seq.shape}")
print(f"Positive label rate    : {y.mean():.3f}   ({LABEL_MODE})")
print(f"Female rate (S=1)      : {S.mean():.3f}")
print(f"Split sizes            : train={len(splits['train'])}, "
      f"val={len(splits['val'])}, test={len(splits['test'])}")
print(f"Mean clicks/week (real): "
      + ", ".join(f"{f}={X_seq[...,i][mask.astype(bool)].mean():.2f}"
                  for i, f in enumerate(FEATURES)))
print(f"\nAll artefacts written to: {OUT_DIR}")
print("===========================================")
