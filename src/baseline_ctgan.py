# -*- coding: utf-8 -*-
"""
baseline_ctgan.py  --  CTGAN baseline for RQ3 (non-LLM, non-sequential)
=======================================================================
As in the proposal (Section 3.3): the weekly data are FLATTENED to one static
row per student (60 columns = 5 features x 12 weeks) plus S (gender), Y (label)
and W (number of real weeks). CTGAN (Xu et al., 2019) is trained on the
training cohort; the sampled rows are reshaped back to (N, 12, 5) so they can
be evaluated with exactly the same GRU TSTR protocol as the LLM generators.

Counts are log1p-transformed before CTGAN (very skewed) and transformed back.

Usage:
  python baseline_ctgan.py --data_dir <oulad_processed> --out_dir <folder> \
         [--n_synthetic 2000] [--epochs 300] [--n_train 0 (=all)] [--cohort active]
Writes synth_X_seq.npy, synth_mask.npy, synth_S_gender.npy, synth_y.npy,
       synthetic_weekly_long.csv, final_check.json
"""
import argparse, os, time
import numpy as np
import pandas as pd
from common import FEATURES, TAGS, SCORE_IDX, T_WEEKS, load_real, get_split, spd, save_json


def to_wide(X, M, S, y):
    cols = {f"{TAGS[f]}_w{w}": np.log1p(X[:, w, f]) for f in range(len(TAGS)) for w in range(T_WEEKS)}
    df = pd.DataFrame(cols)
    df["S"] = S.astype(str); df["Y"] = y.astype(str); df["W"] = M.sum(1).astype(int).astype(str)
    return df


def from_wide(df, clip_max):
    n = len(df)
    W = df["W"].astype(int).clip(1, T_WEEKS).values
    M = (np.arange(T_WEEKS)[None, :] < W[:, None]).astype(np.int8)
    X = np.zeros((n, T_WEEKS, len(TAGS)), np.float32)
    for f in range(len(TAGS)):
        for w in range(T_WEEKS):
            X[:, w, f] = np.round(np.expm1(np.clip(df[f"{TAGS[f]}_w{w}"].values, 0, None)))
    X = np.minimum(X, clip_max[None, None, :])
    X[:, :, SCORE_IDX] = np.minimum(X[:, :, SCORE_IDX], 100)
    X *= M[:, :, None]
    return X, M, df["S"].astype(int).values.astype(np.int8), df["Y"].astype(int).values.astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cohort", default="active")
    ap.add_argument("--n_synthetic", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n_train", type=int, default=0, help="0 = all training students; "
                    "e.g. 1000 for a data-scarce variant")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from ctgan import CTGAN
    import torch
    np.random.seed(a.seed); torch.manual_seed(a.seed)

    d = load_real(a.data_dir)
    tr, _, _ = get_split(d, a.cohort, a.seed)
    if a.n_train and a.n_train < len(tr):
        tr = np.random.default_rng(a.seed).choice(tr, a.n_train, replace=False)
    df = to_wide(d["X"][tr], d["mask"][tr], d["S"][tr], d["y"][tr])
    t0 = time.time()
    model = CTGAN(epochs=a.epochs, batch_size=500, verbose=True, cuda=torch.cuda.is_available())
    model.fit(df, discrete_columns=["S", "Y", "W"])
    print(f"[ctgan] trained on {len(df)} rows in {(time.time()-t0)/60:.1f} min")
    syn = model.sample(a.n_synthetic)
    clip_max = d["X"][tr].reshape(-1, len(TAGS)).max(0)
    X, M, S, y = from_wide(syn, clip_max)
    for n, arr in [("synth_X_seq", X), ("synth_mask", M), ("synth_S_gender", S), ("synth_y", y)]:
        np.save(os.path.join(a.out_dir, f"{n}.npy"), arr)
    rows = [dict(synth_id=i, week=w, S_gender=int(S[i]), label=int(y[i]),
                 **{f: float(X[i, w, j]) for j, f in enumerate(FEATURES)})
            for i in range(len(X)) for w in range(int(M[i].sum()))]
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "synthetic_weekly_long.csv"), index=False)
    save_json(dict(mode="ctgan", cohort=a.cohort, n=len(X), n_train=len(tr), epochs=a.epochs,
                   label_SPD_synth=spd(y, S), female_rate=float(S.mean()), pos_rate=float(y.mean()),
                   minutes=(time.time() - t0) / 60),
              os.path.join(a.out_dir, "final_check.json"))
    print(f"[ctgan] saved {len(X)} synthetic students -> {a.out_dir}")


if __name__ == "__main__":
    main()
