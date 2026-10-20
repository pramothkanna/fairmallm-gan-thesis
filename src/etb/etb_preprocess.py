# -*- coding: utf-8 -*-
"""
etb_preprocess.py  --  RQ4: BIBB/BAuA Employment Survey 2018 (Campus File)
=========================================================================
Builds a cross-sectional table for the fairness-aware generator (no time axis).

Target  y = 1 ("at risk") if
          F1451 (overall job satisfaction) = 3 "weniger zufrieden" or 4 "nicht zufrieden"
       OR F1230 (job demands vs. own skills) = 3 "Anforderungen sind geringer"
          (= over-qualified / skills not used -> qualification mismatch)
        y = 0 otherwise.  Rows with F1451 = 9 (no answer) are dropped.
Protected attribute S = 1 female (S1 = 2), 0 male.
Features (no satisfaction sub-items -> no target leakage):
  numeric : age (zpalter), weekly_hours (az), monthly_income (F518_SUF, EUR)
  ordinal : time_pressure (F411_01), must_work_fast (F411_13), plans_own_work (F700_02),
            work_feels_important (F700_07), part_of_community (F700_10)  [1 often .. 4 never]
            health (F1502) [1 excellent .. 5 poor], job_level (F100_kldb2010_AN) [1 helper .. 4 expert]
  nominal : job_status (Stib), education (max1202), sector (F512_neu), has_children (F1604),
            town_size (BIK10, 1 small .. 10 metropolitan core)
Missing codes (9 / 99 / 9999 / 99998 / 99999 / -1 / -4) -> NaN.

NOTE: the Campus File is licensed by BIBB/FDZ for teaching. Do NOT upload it
or the processed table to GitHub / public Kaggle.

Usage: python etb_preprocess.py --dta <BIBBBAuA_2018_CampusFile.dta> --out_dir <etb_processed>
"""
import argparse, os, json
import numpy as np
import pandas as pd

NUMERIC = ["age", "weekly_hours", "monthly_income"]
ORDINAL = ["time_pressure", "must_work_fast", "plans_own_work", "work_feels_important",
           "part_of_community", "health", "job_level"]
NOMINAL = ["job_status", "education", "sector", "has_children", "town_size"]
FEATURES = NUMERIC + ORDINAL + NOMINAL

LABELS = {   # English labels used in the LLM serialisation
    "job_status": {1: "worker", 2: "employee", 3: "civil_servant", 4: "self_employed",
                   5: "freelancer", 6: "other", 7: "other"},
    "education": {1: "no_vocational_degree", 2: "vocational_training", 3: "master_technician",
                  4: "university"},
    "sector": {1: "public_service", 2: "industry", 3: "crafts", 4: "trade", 5: "other_services",
               6: "other_sector", 100: "associations"},
    "has_children": {1: "yes", 2: "no"},
}


def build(dta):
    d = pd.read_stata(dta, convert_categoricals=False)
    d = d[d["F1451"].isin([1, 2, 3, 4])].copy()
    out = pd.DataFrame(index=d.index)
    out["y"] = (d["F1451"].isin([3, 4]) | (d["F1230"] == 3)).astype(int)
    out["S"] = (d["S1"] == 2).astype(int)
    na = lambda s, bad: s.where(~s.isin(bad))
    out["age"] = na(d["zpalter"], [9999]).astype(float)
    out["weekly_hours"] = d["az"].astype(float)
    inc = na(d["F518_SUF"].astype(float), [99998, 99999])
    out["monthly_income"] = inc.clip(upper=15000)
    for new, old in [("time_pressure", "F411_01"), ("must_work_fast", "F411_13"),
                     ("plans_own_work", "F700_02"), ("work_feels_important", "F700_07"),
                     ("part_of_community", "F700_10")]:
        out[new] = na(d[old].astype(float), [9])
    out["health"] = na(d["F1502"].astype(float), [9])
    out["job_level"] = na(d["F100_kldb2010_AN"].astype(float), [-1, -4])
    out["job_status"] = na(d["Stib"].astype(float), [9, 99])
    out["education"] = na(d["max1202"].astype(float), [9, 99])
    out["sector"] = na(d["F512_neu"].astype(float), [9])
    out["has_children"] = na(d["F1604"].astype(float), [9])
    out["town_size"] = d["BIK10"].astype(float)
    return out.reset_index(drop=True)


def split(df, seed=42):
    rng = np.random.default_rng(seed)
    strata = df["y"] * 2 + df["S"]
    sp = {"train": [], "val": [], "test": []}
    for s in sorted(strata.unique()):
        idx = np.where(strata == s)[0]; rng.shuffle(idx)
        n = len(idx); a, b = int(.70 * n), int(.85 * n)
        sp["train"] += idx[:a].tolist(); sp["val"] += idx[a:b].tolist(); sp["test"] += idx[b:].tolist()
    return {k: sorted(v) for k, v in sp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dta", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = build(a.dta)
    sp = split(df)
    df.to_csv(os.path.join(a.out_dir, "etb.csv"), index=False)
    json.dump(sp, open(os.path.join(a.out_dir, "splits.json"), "w"))
    meta = dict(n=len(df), pos_rate=float(df.y.mean()), female_rate=float(df.S.mean()),
                pos_rate_F=float(df.y[df.S == 1].mean()), pos_rate_M=float(df.y[df.S == 0].mean()),
                label_SPD=float(df.y[df.S == 0].mean() - df.y[df.S == 1].mean()),
                numeric=NUMERIC, ordinal=ORDINAL, nominal=NOMINAL,
                missing_share={c: float(df[c].isna().mean()) for c in FEATURES},
                sizes={k: len(v) for k, v in sp.items()})
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
