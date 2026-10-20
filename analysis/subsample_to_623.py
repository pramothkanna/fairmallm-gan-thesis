"""Draw a random subset of 623 students from each synthetic set (seed 0), so that all four
generators can be compared at the size the MALLM-GAN baseline reached.
Usage: python analysis/subsample_to_623.py --runs full=<dir> no_fair=<dir> ctgan=<dir> --out <dir> [--n 623]"""
import argparse, os, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=623); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
for r in a.runs:
    name, p = r.split("=", 1)
    files = ["synth_X_seq.npy", "synth_mask.npy", "synth_S_gender.npy", "synth_y.npy"]
    arr = {f: np.load(os.path.join(p, f)) for f in files}
    ix = np.random.default_rng(a.seed).choice(len(arr["synth_y.npy"]), a.n, replace=False)
    o = os.path.join(a.out, f"run_{name}"); os.makedirs(o, exist_ok=True)
    for f in files: np.save(os.path.join(o, f), arr[f][ix])
    print(name, "->", o, a.n)
