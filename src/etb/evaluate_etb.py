# -*- coding: utf-8 -*-
"""
evaluate_etb.py  --  RQ4 evaluation on ETB 2018 with LightGBM (TRTR / TRTR-n / TSTR)
===================================================================================
Same metrics as OULAD: accuracy, balanced accuracy, F1, ROC-AUC, SPD, EOD on the
REAL test respondents (5 seeds), plus data-level D1 / D2 / label SPD.
Optional CTGAN baseline (--ctgan) trained on the ETB training split.

Usage:
  python evaluate_etb.py --data_dir <etb_processed> --out_dir <folder> \
        --synth full=<dir with synthetic_etb.csv> no_fair=<dir> [--ctgan]
"""
import argparse, os, json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etb_preprocess import FEATURES, NOMINAL                                    # noqa: E402
from fair_mallm_tabular import Encoder, d1_score, d2_score                       # noqa: E402
from common import spd, eod, save_json                                           # noqa: E402


def fit_predict(train, test, seed):
    Xtr, Xte = train[FEATURES].copy(), test[FEATURES].copy()
    for c in NOMINAL:
        cats = sorted(pd.concat([Xtr[c], Xte[c]]).dropna().unique())
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats); Xte[c] = pd.Categorical(Xte[c], categories=cats)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(Xtr, train["y"].astype(int))
    return m.predict_proba(Xte)[:, 1]


def metr(test, p):
    y, S = test["y"].values, test["S"].values; yh = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh),
                f1=f1_score(y, yh, zero_division=0), auc=roc_auc_score(y, p),
                SPD=spd(yh, S), EOD=eod(y, yh, S))


def summ(runs):
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out.update({k + "_sd": float(np.std([r[k] for r in runs])) for k in runs[0]})
    return out


def ctgan_baseline(train, n, out_dir, epochs=300, seed=42):
    from ctgan import CTGAN
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    df = train[FEATURES + ["S", "y"]].copy()
    disc = NOMINAL + ["S", "y"]
    for c in disc:
        df[c] = df[c].fillna(-1).astype(int).astype(str)
    num = [c for c in FEATURES if c not in NOMINAL]
    miss = {c: df[c].isna() for c in num}
    for c in num:
        df[c] = df[c].fillna(df[c].median())
    m = CTGAN(epochs=epochs, batch_size=500, verbose=False)
    m.fit(df, discrete_columns=disc)
    s = m.sample(n)
    for c in disc:
        s[c] = s[c].astype(int).replace(-1, np.nan)
    for c in num:
        lo, hi = train[c].min(), train[c].max()
        s[c] = s[c].clip(lo, hi).round()
    s["S"] = s["S"].astype(int); s["y"] = s["y"].astype(int)
    os.makedirs(out_dir, exist_ok=True)
    s.to_csv(os.path.join(out_dir, "synthetic_etb.csv"), index=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--ctgan", action="store_true")
    ap.add_argument("--ctgan_n", type=int, default=2000)
    ap.add_argument("--ctgan_epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(a.data_dir, "etb.csv"))
    sp = json.load(open(os.path.join(a.data_dir, "splits.json")))
    tr, te = df.iloc[sp["train"]].reset_index(drop=True), df.iloc[sp["test"]].reset_index(drop=True)
    synth = {k: pd.read_csv(os.path.join(v, "synthetic_etb.csv")) for k, v in
             (s.split("=", 1) for s in a.synth)}
    if a.ctgan:
        synth["ctgan"] = ctgan_baseline(tr, a.ctgan_n, os.path.join(a.out_dir, "ctgan_etb"), a.ctgan_epochs)
    seeds = range(a.seeds); res = {}; rows = []
    res["TRTR (all real)"] = summ([metr(te, fit_predict(tr, te, s)) for s in seeds])
    for name, syn in synth.items():
        n = len(syn)
        res[f"TRTR-n (real n={n})"] = summ([metr(te, fit_predict(
            tr.groupby(["y", "S"], group_keys=False).apply(lambda g: g.sample(frac=min(1, n / len(tr)), random_state=s)),
            te, s)) for s in seeds])
        res[f"TSTR {name}"] = summ([metr(te, fit_predict(syn, te, s)) for s in seeds])
    enc = Encoder(tr); dq = {}
    for name, syn in synth.items():
        d1, _ = d1_score(enc, tr, syn, 0); d2, _ = d2_score(enc, syn, 0)
        samp = tr.sample(len(syn), random_state=0, replace=len(syn) > len(tr))
        dq[name] = dict(n=len(syn), D1_acc=d1, D2_auc=d2, D2_auc_real_same_size=d2_score(enc, samp, 0)[0],
                        label_SPD=spd(syn.y.values, syn.S.values), female_rate=float(syn.S.mean()),
                        pos_rate=float(syn.y.mean()))
    res["data_quality"] = dq
    res["real_reference"] = dict(label_SPD_train=spd(tr.y.values, tr.S.values), n_train=len(tr), n_test=len(te))
    save_json(res, os.path.join(a.out_dir, "results_etb.json"))
    for k, v in res.items():
        if "acc" in v:
            print(f"{k:28s} acc={v['acc']:.3f} bal={v['bal_acc']:.3f} f1={v['f1']:.3f} auc={v['auc']:.3f} "
                  f"SPD={v['SPD']:+.3f} EOD={v['EOD']:+.3f}")
    for k, v in dq.items():
        print(k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()})


if __name__ == "__main__":
    main()
