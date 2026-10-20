# -*- coding: utf-8 -*-
"""
prepare_cohort.py  --  cohort definition + grouped split for FairMALLM-GAN v3
=============================================================================
Runs on the EXISTING oulad_processed folder (no need to re-run preprocessing).

Why:
  * 20.5% of enrolments had already unregistered before the 30% cutoff
    (9.5% even before day 0). At prediction time their outcome is known, so
    they are not early-warning cases and they inflate accuracy.
    -> main cohort = students still registered at the cutoff ("active").
  * v2 split was by enrolment; 15% of test students also had another
    enrolment in train. v3 splits by id_student (grouped), stratified by
    label x gender, 70/15/15.

Usage:
    python prepare_cohort.py --data_dir <oulad_processed> --out_dir <folder>
Writes: splits_v3.json, cohort_report.json
"""
import argparse, os
import numpy as np
from common import load_real, build_cohort_splits, save_json, spd


def report(d, sp):
    y, S = d["y"], d["S"]
    rep = {}
    for coh in ["all", "active"]:
        r = {}
        for s in ["train", "val", "test"]:
            ix = sp[f"{coh}_{s}"]
            r[s] = dict(n=len(ix), pos_rate=float(y[ix].mean()),
                        female_rate=float(S[ix].mean()),
                        label_SPD=spd(y[ix], S[ix]))
        allix = np.concatenate([sp[f"{coh}_{s}"] for s in ["train", "val", "test"]])
        tot = d["X"][allix].sum((1, 2))
        r["all_zero_share_y1"] = float((tot[y[allix] == 1] == 0).mean())
        r["all_zero_share_y0"] = float((tot[y[allix] == 0] == 0).mean())
        rep[coh] = r
    ids = d["static"]["id_student"].values
    tr = set(ids[sp["all_train"]])
    rep["test_ids_also_in_train"] = float(np.mean([i in tr for i in ids[sp["all_test"]]]))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = load_real(a.data_dir)
    sp = build_cohort_splits(d, a.seed)
    save_json({k: v for k, v in sp.items() if k not in ("split", "active")},
              os.path.join(a.out_dir, "splits_v3.json"))
    rep = report(d, sp)
    save_json(rep, os.path.join(a.out_dir, "cohort_report.json"))
    for coh in ["all", "active"]:
        print(coh, {s: rep[coh][s]["n"] for s in ["train", "val", "test"]},
              "pos_rate(train)=%.3f" % rep[coh]["train"]["pos_rate"],
              "zero-activity among y=1: %.3f" % rep[coh]["all_zero_share_y1"])
    print("test ids also in train:", rep["test_ids_also_in_train"])


if __name__ == "__main__":
    main()
