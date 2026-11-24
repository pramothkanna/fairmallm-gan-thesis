# -*- coding: utf-8 -*-
"""
evaluate_v3.py  --  evaluation for RQ1, RQ2, RQ3 (FairMALLM-GAN v3)
===================================================================
For every synthetic data set (FairMALLM full, no_fair ablation, MALLM-GAN flat,
CTGAN, old v2 ...) this script computes, on the SAME real test students:

 Utility (RQ2/RQ3) - GRU early-warning model, 5 seeds, mean +/- sd
   TRTR    : train on all real training students
   TRTR-n  : train on n real students (n = synthetic size)  <- fair size reference
   TSTR    : train on the synthetic students
   AUG     : n real + synthetic (data-augmentation view)
   metrics : accuracy, balanced accuracy, F1, ROC-AUC (threshold tuned on the
             validation split of the TRAINING source, never on real test data)
 Fairness of the predictions (RQ1/RQ3) on the real test set:
   SPD = P(yhat=1|S=0) - P(yhat=1|S=1),  EOD = TPR(S=0) - TPR(S=1); per-module SPD
 Data-level fairness (RQ1): label SPD of the data set, D2 AUC (gender from
   features + label) vs. the same quantity on a same-size REAL sample
 Temporal fidelity (RQ2): D1 accuracy (LR, CV), post-hoc GRU discriminative
   score |acc-0.5|, Wasserstein distance per feature (log scale), weekly-mean
   curve error, lag-1 autocorrelation gap, cross-feature correlation gap
 Privacy: distance to closest real training record (DCR) vs. real test -> train

Usage:
  python evaluate_v3.py --data_dir <oulad_processed> --out_dir <folder> \
     --synth full=<dir> no_fair=<dir> mallm_flat=<dir> ctgan=<dir> [--cohort active]
Outputs: results_v3.json, utility_fairness.csv, data_quality.csv, figures/*.png, report.md
"""
import argparse, os, json, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance

from common import (FEATURES, load_real, get_split, build_matrix, spd, eod, flat_log,
                    save_json, vec_name)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# GRU early-warning model
# =============================================================================
class GRUClassifier(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1]).squeeze(-1)


class Scaler:
    """log1p, then standardise with REAL training weeks only; padded weeks -> 0."""
    def fit(self, X, M):
        v = np.log1p(np.clip(X, 0, None))[M.astype(bool)]
        self.mu, self.sd = v.mean(0), v.std(0) + 1e-6
        return self

    def __call__(self, X, M):
        Z = (np.log1p(np.clip(X, 0, None)) - self.mu) / self.sd
        Z[M == 0] = 0
        return Z.astype(np.float32)


def _tensors(Z, M):
    L = np.clip(M.sum(1), 1, Z.shape[1]).astype(np.int64)
    return torch.tensor(Z), torch.tensor(L)


@torch.no_grad()
def predict_proba(model, Z, M, bs=4096):
    model.eval(); out = []
    x, l = _tensors(Z, M)
    for i in range(0, len(x), bs):
        out.append(torch.sigmoid(model(x[i:i + bs].to(DEV), l[i:i + bs])).cpu().numpy())
    return np.concatenate(out)


def train_gru(Ztr, Mtr, ytr, Zva, Mva, yva, seed, epochs=40, patience=5, lr=1e-3, bs=128):
    """Adam + BCE, early stopping on validation ROC-AUC of the TRAINING SOURCE
    (real val for TRTR, held-out synthetic for TSTR -> no real-data leakage)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GRUClassifier(Ztr.shape[2]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    x, l = _tensors(Ztr, Mtr); yt = torch.tensor(ytr.astype(np.float32))
    best, best_state, bad = -1, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(x[b].to(DEV), l[b]), yt[b].to(DEV))
            loss.backward(); opt.step()
        p = predict_proba(model, Zva, Mva)
        auc = roc_auc_score(yva, p) if len(np.unique(yva)) > 1 else 0.5
        if auc > best + 1e-4:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    # decision threshold chosen on the SAME validation data (max. balanced accuracy);
    # a fixed 0.5 collapses to "all negative" for weak small-data models
    p = predict_proba(model, Zva, Mva)
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49)))
    model.threshold = float(max(grid, key=lambda t: balanced_accuracy_score(yva, p >= t))) \
        if len(np.unique(yva)) > 1 else 0.5
    return model


def metrics(y, p, S, modules=None, thr=0.5):
    yh = (p >= thr).astype(int)
    r = dict(acc=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh),
             f1=f1_score(y, yh, zero_division=0), auc=roc_auc_score(y, p),
             SPD=spd(yh, S), EOD=eod(y, yh, S), pos_pred_rate=float(yh.mean()))
    if modules is not None:
        r["SPD_by_module"] = {m: spd(yh[modules == m], S[modules == m])
                              for m in np.unique(modules)
                              if len(np.unique(S[modules == m])) == 2}
    return r


def summarise(runs):
    keys = [k for k in runs[0] if k != "SPD_by_module"]
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out.update({k + "_sd": float(np.std([r[k] for r in runs])) for k in keys})
    if "SPD_by_module" in runs[0]:
        mods = runs[0]["SPD_by_module"].keys()
        out["SPD_by_module"] = {m: float(np.mean([r["SPD_by_module"][m] for r in runs])) for m in mods}
    out["n_runs"] = len(runs)
    return out


def split_val(n, y, seed, frac=0.15):
    idx = np.arange(n)
    tr, va = train_test_split(idx, test_size=frac, random_state=seed, stratify=y)
    return tr, va


def run_condition(name, X, M, y, sc, test, seeds, log, sample_fn=None):
    """sample_fn(seed) -> (X, M, y) lets TRTR-n draw a new real subsample per seed."""
    Zte, Mte, yte, Ste, mod = test
    runs = []
    for sd in seeds:
        Xs, Ms, ys = sample_fn(sd) if sample_fn else (X, M, y)
        a, b = split_val(len(ys), ys, sd)
        mdl = train_gru(sc(Xs[a], Ms[a]), Ms[a], ys[a], sc(Xs[b], Ms[b]), Ms[b], ys[b], sd)
        runs.append(metrics(yte, predict_proba(mdl, Zte, Mte), Ste, mod, mdl.threshold))
    s = summarise(runs)
    log(f"  {name:28s} acc={s['acc']:.3f}±{s['acc_sd']:.3f} bal={s['bal_acc']:.3f} "
        f"f1={s['f1']:.3f} auc={s['auc']:.3f}±{s['auc_sd']:.3f} SPD={s['SPD']:+.3f} EOD={s['EOD']:+.3f}")
    return s


# =============================================================================
# Data-level quality: fidelity, fairness, privacy
# =============================================================================
def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def d1_acc(rv, sv, seed=0):
    X = np.vstack([rv, sv]); lab = np.r_[np.ones(len(rv)), np.zeros(len(sv))]
    p = cross_val_predict(_clf(), X, lab, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return float(accuracy_score(lab, p))


def d2_auc(v, S, y, seed=0):
    X = np.column_stack([v, y])
    p = cross_val_predict(_clf(), X, S, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(S, p))


def matched_real(d, tr, S, y, rng):
    idx = []
    for s in (0, 1):
        for yy in (0, 1):
            k = int(((S == s) & (y == yy)).sum())
            pool = tr[(d["S"][tr] == s) & (d["y"][tr] == yy)]
            if k:
                idx += list(rng.choice(pool, k, replace=len(pool) < k))
    return np.array(idx)


def discriminative_score(Xr, Mr, Xs, Ms, sc, seed=0):
    """Post-hoc GRU discriminator (TimeGAN-style). Returns |acc - 0.5| on held-out data."""
    n = min(len(Xr), len(Xs))
    X = np.concatenate([Xr[:n], Xs[:n]]); M = np.concatenate([Mr[:n], Ms[:n]])
    lab = np.r_[np.ones(n), np.zeros(n)]
    a, b = train_test_split(np.arange(2 * n), test_size=0.3, random_state=seed, stratify=lab)
    a2, v = train_test_split(a, test_size=0.2, random_state=seed, stratify=lab[a])
    mdl = train_gru(sc(X[a2], M[a2]), M[a2], lab[a2], sc(X[v], M[v]), M[v], lab[v], seed, epochs=20)
    p = predict_proba(mdl, sc(X[b], M[b]), M[b])
    return float(abs(accuracy_score(lab[b], p >= mdl.threshold) - 0.5))


def fidelity(Xr, Mr, Xs, Ms):
    out = {}
    br, bs = Mr.astype(bool), Ms.astype(bool)
    for f, nm in enumerate(FEATURES):
        vr, vs = np.log1p(Xr[:, :, f][br]), np.log1p(Xs[:, :, f][bs])
        out[f"W1_{nm}"] = float(wasserstein_distance(vr, vs))
        cr = [np.log1p(Xr[:, w, f][br[:, w]]).mean() for w in range(Xr.shape[1]) if br[:, w].any() and bs[:, w].any()]
        cs = [np.log1p(Xs[:, w, f][bs[:, w]]).mean() for w in range(Xs.shape[1]) if br[:, w].any() and bs[:, w].any()]
        out[f"curveMAE_{nm}"] = float(np.mean(np.abs(np.array(cr) - np.array(cs))))
        out[f"lag1_gap_{nm}"] = float(abs(_lag1(Xr[:, :, f], Mr) - _lag1(Xs[:, :, f], Ms)))
    out["W1_mean"] = float(np.mean([out[f"W1_{n}"] for n in FEATURES]))
    out["curveMAE_mean"] = float(np.mean([out[f"curveMAE_{n}"] for n in FEATURES]))
    out["lag1_gap_mean"] = float(np.mean([out[f"lag1_gap_{n}"] for n in FEATURES]))
    cr = np.corrcoef(np.log1p(Xr[br]).T); cs = np.corrcoef(np.log1p(Xs[bs]).T)
    out["corr_gap_fro"] = float(np.nan_to_num(np.linalg.norm(cr - cs)))
    return out


def _lag1(V, M):
    V = np.log1p(V); a, b = [], []
    for w in range(V.shape[1] - 1):
        k = (M[:, w] == 1) & (M[:, w + 1] == 1)
        a.append(V[k, w]); b.append(V[k, w + 1])
    a, b = np.concatenate(a), np.concatenate(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def privacy(d, tr, te, Xs, Ms):
    rf = flat_log(d["X"][tr], d["mask"][tr])
    nn_ = NearestNeighbors(n_neighbors=1).fit(rf)
    sf = flat_log(Xs, Ms); tf = flat_log(d["X"][te], d["mask"][te])
    rich_s, rich_t = (sf > 0).sum(1) >= 5, (tf > 0).sum(1) >= 5
    ds = nn_.kneighbors(sf[rich_s])[0][:, 0] if rich_s.any() else np.array([np.nan])
    dt = nn_.kneighbors(tf[rich_t])[0][:, 0]
    return dict(DCR_synth_median=float(np.nanmedian(ds)), DCR_synth_p05=float(np.nanpercentile(ds, 5)),
                DCR_realtest_median=float(np.median(dt)), DCR_realtest_p05=float(np.percentile(dt, 5)),
                exact_copies=int((ds == 0).sum()),
                share_closer_than_real_p05=float((ds < np.percentile(dt, 5)).mean()))


def data_quality(name, d, tr, te, Xs, Ms, Ss, ys, sc, rng, log):
    q = dict(n=len(Xs), female_rate=float(Ss.mean()), pos_rate=float(ys.mean()),
             label_SPD=spd(ys, Ss), all_zero_share=float((Xs.sum((1, 2)) == 0).mean()))
    m = matched_real(d, tr, Ss, ys, rng)
    rv, sv = build_matrix(d["X"][m], d["mask"][m]), build_matrix(Xs, Ms)
    q["D1_acc"] = d1_acc(rv, sv)
    q["D2_auc"] = d2_auc(sv, Ss, ys) if len(np.unique(Ss)) == 2 else float("nan")
    q["D2_auc_real_same_size"] = d2_auc(rv, d["S"][m], d["y"][m])
    q["disc_score_GRU"] = discriminative_score(d["X"][m], d["mask"][m], Xs, Ms, sc)
    q.update(fidelity(d["X"][m], d["mask"][m], Xs, Ms))
    q.update(privacy(d, tr, te, Xs, Ms))
    log(f"  [{name}] n={q['n']} D1={q['D1_acc']:.3f} D2={q['D2_auc']:.3f} (real {q['D2_auc_real_same_size']:.3f}) "
        f"labelSPD={q['label_SPD']:+.3f} disc={q['disc_score_GRU']:.3f} W1={q['W1_mean']:.3f} "
        f"DCR med={q['DCR_synth_median']:.2f} (real {q['DCR_realtest_median']:.2f})")
    return q


# =============================================================================
# Figures + report
# =============================================================================
def figures(out_dir, d, tr, synths, uf):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    fig, ax = plt.subplots(1, 5, figsize=(22, 4))
    Xr, Mr = d["X"][tr], d["mask"][tr].astype(bool)
    for f, nm in enumerate(FEATURES):
        ax[f].plot([np.log1p(Xr[:, w, f][Mr[:, w]]).mean() for w in range(12)], "k-o", lw=2, label="real")
        for name, (Xs, Ms, _, _) in synths.items():
            Mb = Ms.astype(bool)
            ax[f].plot([np.log1p(Xs[:, w, f][Mb[:, w]]).mean() if Mb[:, w].any() else np.nan
                        for w in range(12)], "-o", ms=3, label=name)
        ax[f].set_title(nm); ax[f].set_xlabel("week"); ax[f].set_ylabel("mean log1p(value)")
    ax[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "weekly_curves.png"), dpi=130); plt.close(fig)
    # fairness-utility trade-off (RQ3)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for _, r in uf.iterrows():
        ax.errorbar(abs(r["SPD"]), r["auc"], yerr=r["auc_sd"], fmt="o")
        ax.annotate(r["condition"], (abs(r["SPD"]), r["auc"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel("|SPD| of predictions on real test set (lower = fairer)")
    ax.set_ylabel("ROC-AUC on real test set (higher = more useful)")
    ax.set_title("Fairness-utility trade-off"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fairness_utility.png"), dpi=130); plt.close(fig)


def write_report(out_dir, uf, dq, cohort, notes):
    L = [f"# Evaluation report (cohort = {cohort})", "", *notes, "",
         "## Utility and fairness of the GRU on the REAL test set (mean over seeds)", "",
         "| condition | n train | acc | bal. acc | F1 | AUC | SPD | EOD |", "|---|---|---|---|---|---|---|---|"]
    for _, r in uf.iterrows():
        L.append(f"| {r['condition']} | {int(r['n_train'])} | {r['acc']:.3f}±{r['acc_sd']:.3f} | "
                 f"{r['bal_acc']:.3f} | {r['f1']:.3f} | {r['auc']:.3f}±{r['auc_sd']:.3f} | "
                 f"{r['SPD']:+.3f} | {r['EOD']:+.3f} |")
    if len(dq):
        L += ["", "## Synthetic data quality", "",
              "| data set | n | label SPD | D1 acc (→0.5) | D2 AUC (→0.5) | D2 real same n | GRU disc. (→0) "
              "| W1 mean | lag-1 gap | DCR median (real test) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in dq.iterrows():
            L.append(f"| {r['dataset']} | {int(r['n'])} | {r['label_SPD']:+.3f} | {r['D1_acc']:.3f} | "
                     f"{r['D2_auc']:.3f} | {r['D2_auc_real_same_size']:.3f} | {r['disc_score_GRU']:.3f} | "
                     f"{r['W1_mean']:.3f} | {r['lag1_gap_mean']:.3f} | {r['DCR_synth_median']:.2f} "
                     f"({r['DCR_realtest_median']:.2f}) |")
    L += ["", "Figures: `figures/weekly_curves.png`, `figures/fairness_utility.png`."]
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(L))


# =============================================================================
# MAIN
# =============================================================================
def run(data_dir, out_dir, synth_dirs, cohort="active", seeds=(0, 1, 2, 3, 4), augment=True,
        quality=True, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "eval_log.txt"), "a")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    t0 = time.time()
    d = load_real(data_dir)
    tr, va, te = get_split(d, cohort, seed)
    X, M, y, S = d["X"], d["mask"], d["y"], d["S"]
    mod = d["static"]["code_module"].values
    sc = Scaler().fit(X[tr], M[tr])
    test = (sc(X[te], M[te]), M[te], y[te], S[te], mod[te])
    log(f"[eval] cohort={cohort} train={len(tr)} val={len(va)} test={len(te)} device={DEV}")
    synths = {}
    for name, p in synth_dirs.items():
        try:
            synths[name] = (np.load(f"{p}/synth_X_seq.npy").astype(np.float32),
                            np.load(f"{p}/synth_mask.npy").astype(np.int8),
                            np.load(f"{p}/synth_S_gender.npy").astype(np.int8),
                            np.load(f"{p}/synth_y.npy").astype(np.int8))
            log(f"[eval] loaded {name}: {len(synths[name][0])} students from {p}")
        except FileNotFoundError:
            log(f"[eval] WARNING: no synthetic files in {p} -> skipped")
    res, rows = {}, []

    def add(cond, n, s):
        res[cond] = s; rows.append(dict(condition=cond, n_train=n, **{k: v for k, v in s.items()
                                                                       if k != "SPD_by_module"}))
    log("[eval] utility + fairness (GRU, real test set)")
    # TRTR: real train, early stopping on real val
    runs = []
    for sd in seeds:
        mdl = train_gru(sc(X[tr], M[tr]), M[tr], y[tr], sc(X[va], M[va]), M[va], y[va], sd)
        runs.append(metrics(y[te], predict_proba(mdl, *test[:2]), S[te], mod[te], mdl.threshold))
    s = summarise(runs); add("TRTR (all real)", len(tr), s)
    log(f"  {'TRTR (all real)':28s} acc={s['acc']:.3f}±{s['acc_sd']:.3f} auc={s['auc']:.3f} "
        f"SPD={s['SPD']:+.3f} EOD={s['EOD']:+.3f}")
    sizes = sorted({len(v[0]) for v in synths.values()})
    for n in sizes:
        def real_n(sd, n=n):
            ix = train_test_split(tr, train_size=min(n, len(tr) - 2), random_state=1000 + sd,
                                  stratify=y[tr] * 2 + S[tr])[0]
            return X[ix], M[ix], y[ix]
        add(f"TRTR-n (real n={n})", n, run_condition(f"TRTR-n (real n={n})", None, None, None, sc,
                                                       test, seeds, log, real_n))
    for name, (Xs, Ms, Ss, ys) in synths.items():
        add(f"TSTR {name}", len(Xs), run_condition(f"TSTR {name}", Xs, Ms, ys, sc, test, seeds, log))
        if augment:
            def aug(sd, Xs=Xs, Ms=Ms, ys=ys):
                n = len(Xs)
                ix = train_test_split(tr, train_size=min(n, len(tr) - 2), random_state=1000 + sd,
                                      stratify=y[tr] * 2 + S[tr])[0]
                return (np.concatenate([X[ix], Xs]), np.concatenate([M[ix], Ms]),
                        np.concatenate([y[ix], ys]))
            add(f"AUG real-n + {name}", 2 * len(Xs), run_condition(f"AUG real-n + {name}", None, None,
                                                                    None, sc, test, seeds, log, aug))
    uf = pd.DataFrame(rows)
    uf.to_csv(os.path.join(out_dir, "utility_fairness.csv"), index=False)

    dq = pd.DataFrame()
    if quality and synths:
        log("[eval] synthetic data quality (fidelity / fairness / privacy)")
        rng = np.random.default_rng(seed)
        q = {name: data_quality(name, d, tr, te, *v, sc, rng, log) for name, v in synths.items()}
        res["data_quality"] = q
        dq = pd.DataFrame([dict(dataset=k, **v) for k, v in q.items()])
        dq.to_csv(os.path.join(out_dir, "data_quality.csv"), index=False)
    res["real_reference"] = dict(label_SPD_train=spd(y[tr], S[tr]), pos_rate_train=float(y[tr].mean()),
                                 female_rate_train=float(S[tr].mean()))
    save_json(res, os.path.join(out_dir, "results_v3.json"))
    figures(out_dir, d, tr, synths, uf)
    notes = [f"Real training data: n={len(tr)}, positive rate {y[tr].mean():.3f}, "
             f"label SPD {spd(y[tr], S[tr]):+.3f} (S=1 female).",
             "SPD = P(ŷ=1|S=0) − P(ŷ=1|S=1); EOD = TPR(S=0) − TPR(S=1). Target for both: 0.",
             "TRTR-n uses the same number of real students as the synthetic set (fair size reference)."]
    write_report(out_dir, uf, dq, cohort, notes)
    log(f"[eval] done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--synth", nargs="*", default=[], help="name=dir pairs")
    ap.add_argument("--cohort", default="active", choices=["active", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--no_quality", action="store_true")
    a = ap.parse_args()
    run(a.data_dir, a.out_dir, dict(s.split("=", 1) for s in a.synth), a.cohort,
        tuple(range(a.seeds)), not a.no_augment, not a.no_quality)
