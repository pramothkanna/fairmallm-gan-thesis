# -*- coding: utf-8 -*-
"""
fair_mallm_tabular.py  --  FairMALLM-GAN for a cross-sectional table (RQ4, ETB 2018)
==================================================================================
Same framework as fair_mallm_gan_v3.py (single frozen LLM, D1 realism + D2 fairness
logistic-regression discriminators, natural-language feedback, conditioned
<S;Y> slots, novelty filter, time budget) applied to the ETB table built by
etb_preprocess.py. No temporal encoding (ETB is a single survey wave).

Modes:  full     = D1 + D2 feedback, parity label mix (proposed framework)
        no_fair  = D1 only, real label mix (baseline without fairness)

Run on Kaggle (GPU) the same way as the OULAD generator, e.g.
    from fair_mallm_tabular import TabConfig, run
    run(TabConfig(DATA_DIR=".../etb_processed", OUT_DIR="/kaggle/working/etb_full", mode="full"))
"""
import os, sys, re, json, time, math
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_mallm_gan_v3 import HFGenerator, VLLMGenerator, OpenAIGenerator, _log   # noqa: E402
from etb_preprocess import NUMERIC, ORDINAL, NOMINAL, FEATURES, LABELS            # noqa: E402

ORD_RANGE = {"time_pressure": (1, 4), "must_work_fast": (1, 4), "plans_own_work": (1, 4),
             "work_feels_important": (1, 4), "part_of_community": (1, 4), "health": (1, 5),
             "job_level": (1, 4)}


@dataclass
class TabConfig:
    DATA_DIR: str = "/kaggle/input/etb-processed"
    OUT_DIR: str = "/kaggle/working/etb_full"
    mode: str = "full"
    backend: str = "hf"
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = False
    allow_cpu: bool = False
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    prompts_per_batch: int = 8
    students_per_prompt: int = 10          # rows per prompt
    shots_per_cell: int = 3
    temperature: float = 0.9
    top_p: float = 0.95
    max_rounds: int = 10
    min_rounds: int = 3
    min_eval_rows: int = 80
    tol_d1: float = 0.10
    tol_d2: float = 0.10
    patience: int = 2
    label_policy: str = "auto"
    n_synthetic: int = 2000
    time_budget_h: float = 10.5
    train_budget_frac: float = 0.35
    seed: int = 42
    fmt: str = "tabular"
    optimizer: bool = False

    def resolve(self):
        if self.label_policy == "auto":
            self.label_policy = "parity" if self.mode == "full" else "real"
        return self


# ---------------------------------------------------------------- serialisation
def _val_txt(c, v):
    if pd.isna(v):
        return "unknown"
    if c in LABELS:
        return LABELS[c].get(int(v), "other")
    return str(int(round(v)))


def serialize(row, s, y):
    return f"<S={int(s)};Y={int(y)}> " + ", ".join(f"{c}={_val_txt(c, row[c])}" for c in FEATURES) + " </end>"


_HEAD = re.compile(r"<\s*S\s*=\s*([01])\s*;\s*Y\s*=\s*([01])\s*>")
_KV = re.compile(r"([a-z_]+)\s*=\s*([A-Za-z_0-9.\-]+)")


def parse(text, bounds):
    rows, bad = [], {}
    heads = list(_HEAD.finditer(text))
    for k, h in enumerate(heads):
        seg = text[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)].split("</end>")[0]
        kv = dict(_KV.findall(seg)); r = {}
        ok = True
        for c in FEATURES:
            v = kv.get(c)
            if v is None:
                ok = False; break
            if v == "unknown":
                r[c] = np.nan; continue
            if c in LABELS:
                inv = {lab: code for code, lab in LABELS[c].items()}
                if v not in inv:
                    ok = False; break
                r[c] = float(inv[v])
            else:
                try:
                    x = float(v)
                except ValueError:
                    ok = False; break
                lo, hi = bounds[c]
                if not lo <= x <= hi:
                    ok = False; break
                r[c] = x
        if not ok:
            bad["invalid_row"] = bad.get("invalid_row", 0) + 1; continue
        rows.append((r, int(h.group(1)), int(h.group(2))))
    return rows, bad


# ---------------------------------------------------------------- encoding for D1 / D2
class Encoder:
    def __init__(self, df):
        self.med = {c: df[c].median() for c in NUMERIC + ORDINAL}
        self.cats = {c: sorted(df[c].dropna().unique().tolist()) for c in NOMINAL}

    def __call__(self, df):
        cols = []
        for c in NUMERIC + ORDINAL:
            v = df[c].astype(float)
            cols += [np.log1p(v.fillna(self.med[c])) if c == "monthly_income" else v.fillna(self.med[c]),
                     v.isna().astype(float)]
        for c in NOMINAL:
            for k in self.cats[c]:
                cols.append((df[c] == k).astype(float))
            cols.append(df[c].isna().astype(float))
        return np.column_stack(cols).astype(np.float32)

    def names(self):
        out = []
        for c in NUMERIC + ORDINAL:
            out += [c, f"{c} missing"]
        for c in NOMINAL:
            out += [f"{c}={_val_txt(c, k)}" for k in self.cats[c]] + [f"{c} missing"]
        return out


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def d1_score(enc, real, syn, seed):
    rng = np.random.default_rng(seed); parts = []
    for (s, y), g in syn.groupby(["S", "y"]):
        pool = real[(real.S == s) & (real.y == y)]
        parts.append(pool.iloc[rng.choice(len(pool), len(g), replace=len(pool) < len(g))])
    r = pd.concat(parts)
    X = np.vstack([enc(r), enc(syn)]); lab = np.r_[np.ones(len(r)), np.zeros(len(syn))]
    acc = accuracy_score(lab, cross_val_predict(_clf(), X, lab, cv=StratifiedKFold(5, shuffle=True, random_state=seed)))
    coef = _clf().fit(X, lab)[-1].coef_[0]
    return float(acc), [int(i) for i in np.argsort(-np.abs(coef))[:3]]


def d2_score(enc, syn, seed):
    S = syn.S.values
    if len(np.unique(S)) < 2 or np.bincount(S).min() < 5:
        return 1.0, 0
    X = np.column_stack([enc(syn), syn.y.values])
    p = cross_val_predict(_clf(), X, S, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    coef = _clf().fit(X, S)[-1].coef_[0][:-1]
    return float(roc_auc_score(S, p)), int(np.argmax(np.abs(coef)))


def _describe(df, c):
    if c in NUMERIC + ORDINAL:
        return f"mean {df[c].mean():.1f}"
    vc = df[c].map(lambda v: _val_txt(c, v)).value_counts(normalize=True)
    return ", ".join(f"{k} {100*v:.0f}%" for k, v in vc.head(3).items())


def feedback(cfg, enc, real, syn, d1, top, d2, li):
    names = enc.names()
    feat_of = lambda n: n.split("=")[0].replace(" missing", "")
    L = [f"- REALISM: D1 accuracy {d1:.2f} (target 0.50)." + (" Good, keep this style." if abs(d1 - .5) <= cfg.tol_d1 else " Fix:")]
    for i in top:
        c = feat_of(names[i])
        L.append(f"  * {c}: real {_describe(real, c)} | yours {_describe(syn, c)}")
    if cfg.mode == "full":
        if abs(d2 - .5) <= cfg.tol_d2:
            L.append(f"- FAIRNESS: good (D2 AUC {d2:.2f}). S must keep not changing any value.")
        else:
            c = feat_of(names[li])
            L.append(f"- FAIRNESS: gender predictable (D2 AUC {d2:.2f}, target 0.50). {c}: "
                     f"S=0 {_describe(syn[syn.S == 0], c)} | S=1 {_describe(syn[syn.S == 1], c)}. "
                     f"Make {c} follow the same distribution for S=0 and S=1 with the same Y.")
    return "\n".join(L)


def data_card(real):
    L = ["REAL DATA SUMMARY (training respondents):"]
    for y, lab in [(0, "Y=0 (satisfied, skills used)"), (1, "Y=1 (dissatisfied or over-qualified)")]:
        g = real[real.y == y]
        L.append(f"{lab}: " + "; ".join(f"{c}: {_describe(g, c)}" for c in FEATURES))
    return "\n".join(L)


def system_prompt(cfg, real):
    t = ("You generate synthetic respondents of the German BIBB/BAuA Employment Survey 2018.\n"
         "FORMAT - one respondent per line:\n<S=s;Y=y> " + ", ".join(f"{c}=.." for c in FEATURES) + " </end>\n"
         "- S = gender code (0/1). Y = 1 if dissatisfied with the job or over-qualified for it, else 0.\n"
         "- age in years, weekly_hours, monthly_income in EUR gross (integers or 'unknown').\n"
         "- time_pressure, must_work_fast, plans_own_work, work_feels_important, part_of_community: "
         "1=often 2=sometimes 3=rarely 4=never. health: 1=excellent .. 5=poor. job_level: 1=helper .. 4=highly complex.\n"
         "- job_status: " + "|".join(sorted(set(LABELS['job_status'].values()))) +
         "; education: " + "|".join(LABELS['education'].values()) +
         "; sector: " + "|".join(LABELS['sector'].values()) +
         "; has_children: yes|no; town_size: 1 (village) .. 10 (big-city core).\n"
         "- Output ONLY respondent lines. Every respondent must be new; never copy an example.\n")
    if cfg.mode == "full":
        t += "- FAIRNESS: S must not influence any value; same Y -> same distribution for S=0 and S=1.\n"
    return t + "\n" + data_card(real)


# ---------------------------------------------------------------- main loop
def run(cfg: TabConfig):
    cfg.resolve(); os.makedirs(cfg.OUT_DIR, exist_ok=True)
    json.dump(asdict(cfg), open(os.path.join(cfg.OUT_DIR, "config.json"), "w"), indent=2)
    t0 = time.time()
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, "etb.csv"))
    sp = json.load(open(os.path.join(cfg.DATA_DIR, "splits.json")))
    real = df.iloc[sp["train"]].reset_index(drop=True)
    enc = Encoder(real)
    bounds = {c: (real[c].min(), real[c].max()) for c in NUMERIC + ORDINAL}
    bounds.update({c: (1, 10) for c in ["town_size"]})
    f = real.S.mean(); p = real.y.mean()
    py = {0: p, 1: p} if cfg.label_policy == "parity" else {s: real.y[real.S == s].mean() for s in (0, 1)}
    cell_p = {(s, y): (f if s else 1 - f) * (py[s] if y else 1 - py[s]) for s in (0, 1) for y in (0, 1)}
    real_keys = set(real[FEATURES].round(0).astype(str).agg("|".join, axis=1))
    gen = {"hf": HFGenerator, "vllm": VLLMGenerator, "openai": OpenAIGenerator}[cfg.backend](cfg)
    system = system_prompt(cfg, real)
    per = max(gen.n_tokens(serialize(real.iloc[i], 0, 0)) for i in range(20))
    max_new = int(per * cfg.students_per_prompt * 1.25) + 64
    rng = np.random.default_rng(cfg.seed)
    seen = set()

    def step(guidance, need=None):
        users = []
        cells = list(cell_p)
        for _ in range(cfg.prompts_per_batch):
            shots = []
            for (s, y) in cells:
                g = real[(real.S == s) & (real.y == y)]
                for i in rng.choice(len(g), cfg.shots_per_cell, replace=False):
                    shots.append(serialize(g.iloc[i], s, y))
            rng.shuffle(shots)
            w = np.array([max(need[c], 0) if need else cell_p[c] for c in cells], float)
            w = w / w.sum() if w.sum() else np.array([cell_p[c] for c in cells])
            slots = [cells[k] for k in rng.choice(4, cfg.students_per_prompt, p=w)]
            g_txt = f"\nFEEDBACK FROM THE DISCRIMINATORS (apply it):\n{guidance}\n" if guidance else ""
            users.append("REAL EXAMPLE RESPONDENTS:\n" + "\n".join(shots) + "\n" + g_txt +
                         f"\nWrite {len(slots)} NEW respondents, one per line, using exactly these headers in this order:\n" +
                         "\n".join(f"<S={s};Y={y}>" for s, y in slots))
        rows, rej = [], {}
        for text in gen.chat_batch(system, users, max_new):
            got, bad = parse(text, bounds)
            for k, v in bad.items():
                rej[k] = rej.get(k, 0) + v
            for r, s, y in got:
                key = "|".join(str(round(r[c])) if not pd.isna(r[c]) else "nan" for c in FEATURES)
                if key in real_keys:
                    rej["copy_of_real"] = rej.get("copy_of_real", 0) + 1; continue
                if key in seen:
                    rej["duplicate_synthetic"] = rej.get("duplicate_synthetic", 0) + 1; continue
                seen.add(key); rows.append(dict(r, S=s, y=y))
        return pd.DataFrame(rows), rej

    hist, guidance, best = [], None, (9, None)
    streak = 0
    for r in range(cfg.max_rounds):
        if time.time() - t0 > cfg.time_budget_h * 3600 * cfg.train_budget_frac:
            break
        syn, rej = step(guidance)
        if len(syn) < cfg.min_eval_rows:
            _log(cfg, f"[etb round {r}] only {len(syn)} rows {rej}"); continue
        d1, top = d1_score(enc, real, syn, cfg.seed + r)
        d2, li = d2_score(enc, syn, cfg.seed + r)
        w = 1.0 if cfg.mode == "full" else 0.0
        score = abs(d1 - .5) + w * abs(d2 - .5)
        ok = abs(d1 - .5) <= cfg.tol_d1 and (cfg.mode != "full" or abs(d2 - .5) <= cfg.tol_d2)
        streak = streak + 1 if ok else 0
        hist.append(dict(round=r, n=len(syn), D1_acc=d1, D2_auc=d2, score=score, rej=rej,
                         label_SPD=float(syn.y[syn.S == 0].mean() - syn.y[syn.S == 1].mean())))
        _log(cfg, f"[etb round {r}] n={len(syn)} D1={d1:.3f} D2={d2:.3f} score={score:.3f} rej={rej}")
        if score < best[0]:
            best = (score, guidance)
        guidance = feedback(cfg, enc, real, syn, d1, top, d2, li)
        json.dump(dict(history=hist, guidance=guidance, best_guidance=best[1]),
                  open(os.path.join(cfg.OUT_DIR, "train_state.json"), "w"), indent=2)
        if streak >= cfg.patience and r + 1 >= cfg.min_rounds:
            break
    target = {c: int(round(cfg.n_synthetic * q)) for c, q in cell_p.items()}
    out = []
    have = {c: 0 for c in target}
    while sum(have.values()) < sum(target.values()) and time.time() - t0 < cfg.time_budget_h * 3600 - 900:
        syn, _ = step(best[1], need={c: target[c] - have[c] for c in target})
        for _, row in syn.iterrows():
            c = (int(row.S), int(row.y))
            if have[c] < target[c]:
                have[c] += 1; out.append(row)
        _log(cfg, f"[etb final] {sum(have.values())}/{sum(target.values())}")
    res = pd.DataFrame(out)[FEATURES + ["S", "y"]]
    res.to_csv(os.path.join(cfg.OUT_DIR, "synthetic_etb.csv"), index=False)
    _log(cfg, f"[etb] saved {len(res)} rows in {(time.time() - t0) / 3600:.2f} h")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    for k, v in asdict(TabConfig()).items():
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else (lambda s: s.lower() in ("1", "true")), default=v)
    run(TabConfig(**vars(ap.parse_args())))
