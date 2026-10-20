# -*- coding: utf-8 -*-
"""
common.py  --  shared helpers for FairMALLM-GAN v3
===================================================
Used by: prepare_cohort.py, fair_mallm_gan_v3.py, baseline_ctgan.py, evaluate_v3.py

Contents
  * constants (feature order, short tags)
  * load_real()          : loads the processed OULAD tensors (unchanged files)
  * build_cohort_splits(): cohort filter + grouped, stratified 70/15/15 split
  * summary features     : 31-dim temporal summary vector per student (D1/D2)
  * fairness metrics     : SPD, EOD
"""
import os, json
import numpy as np
import pandas as pd

FEATURES = ["homepage_clicks", "subpage_clicks", "oucontent_clicks",
            "quiz_interactions", "weekly_assessment_score"]
TAGS = ["hp", "sp", "oc", "qz", "sc"]
SCORE_IDX = 4                     # weekly_assessment_score is bounded 0..100
T_WEEKS = 12


# =============================================================================
# 1. LOAD THE PROCESSED OULAD FILES (produced by Weekly_binning.ipynb)
# =============================================================================
def load_real(data_dir):
    p = lambda f: os.path.join(data_dir, f)
    d = dict(X=np.load(p("X_seq.npy")).astype(np.float32),
             mask=np.load(p("mask.npy")).astype(np.int8),
             y=np.load(p("y.npy")).astype(np.int8),
             S=np.load(p("S_gender.npy")).astype(np.int8),
             keys=np.load(p("student_keys.npy"), allow_pickle=True))
    # '?' = missing in OULAD -> must be parsed as NaN (v2 notebook did not do this)
    st = pd.read_csv(p("student_static.csv"), na_values="?", low_memory=False)
    assert (st["student_key"].values == d["keys"]).all(), "static table not aligned"
    d["static"] = st
    return d


# =============================================================================
# 2. COHORT + GROUPED STRATIFIED SPLIT
# =============================================================================
def active_at_cutoff(static):
    """True if the student was still registered at the early-warning cutoff.
    Students who unregistered before the cutoff (incl. before day 0) already
    have a known outcome at prediction time -> not an early-warning case."""
    unreg = pd.to_numeric(static["date_unregistration"], errors="coerce")
    return ~(unreg.notna() & (unreg < static["cutoff_days"])).values


def build_cohort_splits(d, seed=42):
    """Grouped (by id_student) split, stratified by label x gender, 70/15/15.
    The same person never appears in two splits (v2 split leaked 15%).
    Returns dict with index arrays for cohort 'all' and 'active'."""
    from sklearn.model_selection import StratifiedGroupKFold
    st = d["static"]
    strata = d["y"].astype(int) * 2 + d["S"].astype(int)
    groups = st["id_student"].values
    sgk = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.empty(len(strata), int)
    for k, (_, te) in enumerate(sgk.split(np.zeros(len(strata)), strata, groups)):
        fold[te] = k
    split = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
    act = active_at_cutoff(st)
    out = {"split": split, "active": act}
    for coh, keep in [("all", np.ones(len(act), bool)), ("active", act)]:
        for s in ["train", "val", "test"]:
            out[f"{coh}_{s}"] = np.where((split == s) & keep)[0]
    return out


def get_split(d, cohort="active", seed=42):
    sp = build_cohort_splits(d, seed)
    return sp[f"{cohort}_train"], sp[f"{cohort}_val"], sp[f"{cohort}_test"]


# =============================================================================
# 3. TEMPORAL SUMMARY FEATURES FOR D1 / D2  (log scale, 31 dims)
# =============================================================================
STATS = ["total", "weekly mean", "last week", "trend", "share of active weeks"]


def build_matrix(X, M):
    """Vectorised: (N,T,F) counts + (N,T) mask -> (N, 5*F+1) summary matrix.
    Per feature (log1p scale): total, weekly mean, last real week,
    least-squares trend over the real weeks, share of weeks with activity.
    Assumes real weeks are contiguous from week 0 (true for OULAD + v3 output)."""
    X = np.clip(np.asarray(X, np.float64), 0, None)
    M = np.asarray(M, np.float64)
    N, T, F = X.shape
    L = np.maximum(M.sum(1), 1)                               # (N,)
    V = np.log1p(X) * M[:, :, None]                           # (N,T,F)
    tot = V.sum(1)
    mean = tot / L[:, None]
    last = V[np.arange(N), (L - 1).astype(int)]               # (N,F)
    t = np.arange(T)[None, :] * M                             # (N,T)
    tbar = t.sum(1) / L
    dt = (np.arange(T)[None, :] - tbar[:, None]) * M          # (N,T)
    den = np.maximum((dt ** 2).sum(1), 1e-9)
    slope = (dt[:, :, None] * (V - mean[:, None, :] * M[:, :, None])).sum(1) / den[:, None]
    active = ((X > 0) * M[:, :, None]).sum(1) / L[:, None]
    out = np.stack([tot, mean, last, slope, active], axis=2).reshape(N, F * 5)
    return np.column_stack([out, M.sum(1)]).astype(np.float32)


def to_vec(x, m):
    return build_matrix(x[None], m[None])[0]


def raw_stat(X, M, f, stat):
    """The same statistics on the ORIGINAL scale (for human-readable feedback)."""
    X = np.clip(np.asarray(X, np.float64), 0, None); M = np.asarray(M, np.float64)
    L = np.maximum(M.sum(1), 1); v = X[:, :, f] * M
    if stat == "total":        return v.sum(1)
    if stat == "weekly mean":  return v.sum(1) / L
    if stat == "last week":    return v[np.arange(len(v)), (L - 1).astype(int)]
    if stat == "share of active weeks": return ((v > 0) * M).sum(1) / L
    if stat == "trend":
        return (v[np.arange(len(v)), (L - 1).astype(int)] - v[:, 0]) / np.maximum(L - 1, 1)
    raise ValueError(stat)


def vec_name(i):
    if i >= 5 * len(STATS):
        return "sequence length"
    return f"{FEATURES[i // len(STATS)]} ({STATS[i % len(STATS)]})"


# =============================================================================
# 4. FAIRNESS METRICS   (S=1 female, S=0 male)  -- same definitions as proposal
# =============================================================================
def spd(y_pred, S):
    """SPD = P(yhat=1 | S=0) - P(yhat=1 | S=1)"""
    y_pred, S = np.asarray(y_pred), np.asarray(S)
    return float(y_pred[S == 0].mean() - y_pred[S == 1].mean())


def eod(y_true, y_pred, S):
    """EOD = P(yhat=1 | y=1, S=0) - P(yhat=1 | y=1, S=1)"""
    y_true, y_pred, S = map(np.asarray, (y_true, y_pred, S))
    a = y_pred[(y_true == 1) & (S == 0)].mean()
    b = y_pred[(y_true == 1) & (S == 1)].mean()
    return float(a - b)


# =============================================================================
# 5. SMALL UTILITIES
# =============================================================================
def flat_log(X, M):
    """(N,T,F) -> (N,T*F) log1p with padded weeks = 0 (used for privacy DCR)."""
    Z = np.log1p(np.clip(X, 0, None)) * M[:, :, None]
    return Z.reshape(len(Z), -1)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=conv)
