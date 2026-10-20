# -*- coding: utf-8 -*-
"""
FairMALLM-GAN v3 : Fair & Temporally-Aware LLM-Guided Synthetic Data Generation
================================================================================
One frozen generator LLM + two logistic-regression discriminators
  D1 = realism  (real vs synthetic weekly sequences)   -> target accuracy 0.5
  D2 = fairness (predict gender S from features + Y)   -> target AUC      0.5
Discriminator results are turned into natural-language feedback and put into
the next generation prompt (prompt optimisation, no fine-tuning).

WHAT CHANGED vs v2 (details in CHANGES_v2_to_v3.md)
  1. Conditioned slots: the prompt lists the exact headers <S=..;Y=..;W=..> to
     fill, so the gender x label mix is controlled (v2 let the LLM choose and
     the synthetic data ended up with SPD -0.20 although real SPD is +0.02).
  2. Compact serialisation  w0:5,2,10,0,61 | w1:...  (~40% fewer tokens) and
     a strict parser (wrong number of weeks -> rejected; v2 accepted 8 weeks).
  3. Batched generation: many prompts in one GPU pass (v2: one prompt at a
     time, ~7 min per 8 students).
  4. Few-shot examples stratified by S x Y; aggregate "data card" of the real
     training data in the system prompt.
  5. D1/D2 on log-scaled temporal summaries; D1 compares against a real
     sample with the same S x Y mix; specific, numeric feedback (which
     statistic differs, real vs synthetic) instead of one generic sentence.
  6. Rejection filters: near-copies of real training students, duplicates and
     gender-swapped twins of accepted synthetic students, impossible lengths,
     too many all-zero students.
  7. Three modes for RQ3:  full | no_fair | mallm_flat (see Config.mode)
  8. Time budget -> clean stop before Kaggle's 12 h limit; checkpoint/resume.
"""
import os, re, json, time, math, warnings
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import NearestNeighbors

from common import (FEATURES, TAGS, SCORE_IDX, T_WEEKS, STATS, load_real,
                    get_split, build_matrix, vec_name, raw_stat, spd,
                    flat_log, save_json)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    DATA_DIR: str = "/kaggle/input/oulad-processed/oulad_processed"
    OUT_DIR: str = "/kaggle/working/run_full"

    # full       = temporal serialisation + D1 + D2 + parity labels  (proposed model)
    # no_fair    = temporal serialisation + D1 only, real label mix   (ablation, RQ3)
    # mallm_flat = flattened (static) serialisation + D1 only + optimizer LLM
    #              that rewrites the guidelines = MALLM-GAN baseline (RQ3)
    mode: str = "full"
    cohort: str = "active"            # "active" (still enrolled at cutoff) | "all"

    backend: str = "hf"               # hf | vllm | openai | mock
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = False
    allow_cpu: bool = False           # only for testing with a tiny model
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # ---- generation ----
    prompts_per_batch: int = 8        # prompts per GPU pass (16 ran out of memory on T4 x2)
    students_per_prompt: int = 8
    shots_per_cell: int = 2           # few-shot examples per S x Y cell (4 cells)
    temperature: float = 0.9
    top_p: float = 0.95

    # ---- adversarial loop ----
    max_rounds: int = 10
    min_rounds: int = 3
    pool_rounds: int = 3              # max rounds pooled for scoring
    min_eval_rows: int = 60
    tol_d1: float = 0.10
    tol_d2: float = 0.10
    patience: int = 2
    fairness_weight: float = 1.0      # forced to 0 outside mode="full"
    label_policy: str = "auto"        # auto | parity | real
    use_data_card: bool = True
    use_optimizer_llm: str = "auto"   # auto -> True only for mallm_flat

    # ---- final data ----
    n_synthetic: int = 2000
    zero_cap_margin: float = 0.03     # all-zero share may exceed real share by this
    copy_quantile: float = 0.01       # novelty threshold = this quantile of real-real NN distance
    time_budget_h: float = 10.5       # whole run (Kaggle kills at 12 h)
    train_budget_frac: float = 0.35   # share of the budget the adversarial loop may use
    seed: int = 42

    # derived (filled by resolve())
    fmt: str = ""
    optimizer: bool = False

    def resolve(self):
        assert self.mode in ("full", "no_fair", "mallm_flat"), self.mode
        self.fmt = "flat" if self.mode == "mallm_flat" else "temporal"
        if self.label_policy == "auto":
            self.label_policy = "parity" if self.mode == "full" else "real"
        if self.mode != "full":
            self.fairness_weight = 0.0
        self.optimizer = (self.mode == "mallm_flat") if self.use_optimizer_llm == "auto" \
            else bool(self.use_optimizer_llm)
        return self


def _log(cfg, msg):
    print(msg, flush=True)
    try:
        with open(os.path.join(cfg.OUT_DIR, "run_log.txt"), "a") as fh:
            fh.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except OSError:
        pass


# =============================================================================
# 1. SERIALISATION + STRICT PARSER
# =============================================================================
def serialize(x, m, s, y, fmt):
    W = int(m.sum())
    head = f"<S={int(s)};Y={int(y)};W={W}>"
    if fmt == "temporal":
        body = " | ".join(f"w{w}:" + ",".join(str(int(round(v))) for v in x[w])
                          for w in range(W))
    else:   # flat, feature-major "column=value" like GReaT / MALLM-GAN
        body = ", ".join(f"{TAGS[f]}_w{w}={int(round(x[w, f]))}"
                         for f in range(len(TAGS)) for w in range(W))
    return f"{head} {body} </end>"


_HEAD = re.compile(r"<\s*S\s*=\s*([01])\s*;\s*Y\s*=\s*([01])\s*;\s*W\s*=\s*(\d+)\s*>")
_WEEK = re.compile(r"w(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
_FLAT = re.compile(r"(hp|sp|oc|qz|sc)_w(\d+)\s*=\s*(\d+)")


def parse_generation(text, fmt, allowed_W, clip_max):
    """Returns (rows, reasons). A row is (x, m, s, y). Students are segmented by
    their header, so missing newlines do not break parsing."""
    rows, reasons = [], []
    heads = list(_HEAD.finditer(text))
    for k, h in enumerate(heads):
        seg = text[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        seg = seg.split("</end>")[0]
        s, y, W = int(h.group(1)), int(h.group(2)), int(h.group(3))
        if W not in allowed_W:
            reasons.append("bad_W"); continue
        x = np.zeros((T_WEEKS, len(TAGS)), np.float32)
        m = np.zeros(T_WEEKS, np.int8)
        if fmt == "temporal":
            wk = _WEEK.findall(seg)
            idx = [int(a[0]) for a in wk]
            if idx[:W] != list(range(W)):          # extra weeks beyond W are ignored
                reasons.append("week_count"); continue
            for a in wk[:W]:
                x[int(a[0])] = [float(v) for v in a[1:]]
        else:
            cells = _FLAT.findall(seg)
            seen = set()
            for tag, w, v in cells:
                w = int(w)
                if w < W:
                    x[w, TAGS.index(tag)] = float(v); seen.add((tag, w))
            if len(seen) < len(TAGS) * W or any(int(w) >= W for _, w, _ in cells):
                reasons.append("week_count"); continue
        m[:W] = 1
        x = np.minimum(x, clip_max[None, :])
        x[:, SCORE_IDX] = np.minimum(x[:, SCORE_IDX], 100.0)
        rows.append((x, m, s, y))
    if not heads:
        reasons.append("no_header")
    return rows, reasons


# =============================================================================
# 2. LLM BACKENDS  (batched)
# =============================================================================
class HFGenerator:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.cfg = torch, cfg
        cpu = not torch.cuda.is_available()
        if cpu and not cfg.allow_cpu:
            raise RuntimeError("No GPU. On Kaggle: Settings > Accelerator > GPU T4 x2.")
        _log(cfg, f"[LLM] HF {cfg.hf_model} on " + ("CPU (test only)" if cpu else
                  f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"))
        self.tok = AutoTokenizer.from_pretrained(cfg.hf_model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map=None if cpu else "auto")
        dtype = torch.float32 if cpu else torch.float16
        if cfg.load_in_4bit and not cpu:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        import transformers
        from packaging.version import Version
        new_api = Version(transformers.__version__) >= Version("4.56")
        kw["dtype" if new_api else "torch_dtype"] = dtype     # avoid silent float32 load
        self.model = AutoModelForCausalLM.from_pretrained(cfg.hf_model, **kw)
        self.model.eval()
        devs = sorted(set(str(v) for v in getattr(self.model, "hf_device_map", {}).values()))
        _log(cfg, f"[LLM] transformers {transformers.__version__}, dtype={self.model.dtype}, "
                  f"devices={devs or [str(self.model.device)]}")
        if any("cpu" in d_ or "disk" in d_ for d_ in devs):
            _log(cfg, "[LLM] WARNING: part of the model is offloaded to CPU/disk -> very slow. "
                      "Check that the accelerator is 'GPU T4 x2'.")
        self.bs = cfg.prompts_per_batch
        self.tokens_out = 0; self.seconds = 0.0

    def n_tokens(self, text):
        return len(self.tok(text)["input_ids"])

    def _prompt(self, system, user):
        return self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    def chat_batch(self, system, users, max_new_tokens):
        torch = self.torch
        prompts = [self._prompt(system, u) for u in users]
        outs, i = [], 0
        while i < len(prompts):
            chunk = prompts[i:i + self.bs]
            try:
                t0 = time.time()
                enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.model.device)
                with torch.inference_mode():
                    gen = self.model.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                        pad_token_id=self.tok.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                self.tokens_out += int((new != self.tok.pad_token_id).sum())
                self.seconds += time.time() - t0
                outs += self.tok.batch_decode(new, skip_special_tokens=True)
                i += len(chunk)
            except torch.OutOfMemoryError if hasattr(torch, "OutOfMemoryError") else torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.bs == 1:
                    raise
                self.bs = max(1, self.bs // 2)
                _log(self.cfg, f"[LLM] CUDA OOM -> prompts_per_batch reduced to {self.bs}")
        return outs


class VLLMGenerator:
    def __init__(self, cfg):
        import torch
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self.cfg, self.SP = cfg, SamplingParams
        self.tok = AutoTokenizer.from_pretrained(cfg.hf_model)
        self.llm = LLM(model=cfg.hf_model, dtype="half",
                       tensor_parallel_size=max(1, torch.cuda.device_count()),
                       gpu_memory_utilization=0.90, max_model_len=8192, seed=cfg.seed)
        self.tokens_out = 0; self.seconds = 0.0
        _log(cfg, "[LLM] vLLM backend ready")

    def n_tokens(self, text):
        return len(self.tok(text)["input_ids"])

    def chat_batch(self, system, users, max_new_tokens):
        prompts = [self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True) for u in users]
        t0 = time.time()
        res = self.llm.generate(prompts, self.SP(
            temperature=self.cfg.temperature, top_p=self.cfg.top_p,
            max_tokens=max_new_tokens))
        self.seconds += time.time() - t0
        outs = [r.outputs[0].text for r in res]
        self.tokens_out += sum(len(r.outputs[0].token_ids) for r in res)
        return outs


class OpenAIGenerator:
    def __init__(self, cfg):
        from openai import OpenAI
        self.cfg = cfg
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),
                             base_url=cfg.openai_base_url)
        self.tokens_out = 0; self.seconds = 0.0

    def n_tokens(self, text):
        return len(text) // 3

    def chat_batch(self, system, users, max_new_tokens):
        outs = []
        for u in users:
            t0 = time.time()
            r = self.client.chat.completions.create(
                model=self.cfg.openai_model, temperature=self.cfg.temperature,
                max_tokens=max_new_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": u}])
            self.seconds += time.time() - t0
            outs.append(r.choices[0].message.content)
        return outs


class MockGenerator:
    """CPU test double: answers each requested header with a perturbed real
    student of the same S, Y, W (and some deliberately broken lines), so the
    whole loop can be tested without a GPU. NOT for thesis results."""
    def __init__(self, cfg, ctx):
        self.cfg, self.ctx = cfg, ctx
        self.rng = np.random.default_rng(cfg.seed + 7)
        self.tokens_out = 0; self.seconds = 0.0

    def n_tokens(self, text):
        return len(text) // 3

    def chat_batch(self, system, users, max_new_tokens):
        if "prompt optimizer" in system:
            return ["- Match the weekly click levels of the real rows.\n- Keep zeros realistic."] * len(users)
        outs = []
        d, tr = self.ctx["data"], self.ctx["tr"]
        for u in users:
            lines = []
            for h in _HEAD.finditer(u.split("using exactly these headers")[-1]):
                s, y, W = int(h.group(1)), int(h.group(2)), int(h.group(3))
                pool = tr[(d["S"][tr] == s) & (d["y"][tr] == y) & (d["mask"][tr].sum(1) == W)]
                if len(pool) == 0:
                    continue
                i = self.rng.choice(pool)
                x = d["X"][i] * self.rng.lognormal(0, 0.35, d["X"][i].shape)
                if self.rng.random() < 0.05:          # broken line -> must be rejected
                    lines.append(f"<S={s};Y={y};W={W}> w0:1,2,3 | w1:4 </end>")
                    continue
                lines.append(serialize(np.round(x), d["mask"][i], s, y, self.cfg.fmt))
            outs.append("\n".join(lines))
        return outs


def make_generator(cfg, ctx):
    if cfg.backend == "mock":
        return MockGenerator(cfg, ctx)
    if cfg.backend == "openai":
        return OpenAIGenerator(cfg)
    if cfg.backend == "vllm":
        try:
            return VLLMGenerator(cfg)
        except Exception as e:                 # fall back, never crash the run
            _log(cfg, f"[LLM] vLLM unavailable ({type(e).__name__}: {e}) -> using HF")
    return HFGenerator(cfg)


# =============================================================================
# 3. REAL-DATA CONTEXT (reference sample, data card, novelty filter)
# =============================================================================
def build_context(cfg):
    d = load_real(cfg.DATA_DIR)
    tr, va, te = get_split(d, cfg.cohort, cfg.seed)
    X, M, y, S = d["X"], d["mask"], d["y"], d["S"]
    ctx = dict(data=d, tr=tr, va=va)
    ctx["real_vecs"] = build_matrix(X[tr], M[tr])
    ctx["cell_idx"] = {(s, yy): np.where((S[tr] == s) & (y[tr] == yy))[0]
                       for s in (0, 1) for yy in (0, 1)}
    W = M[tr].sum(1).astype(int)
    ctx["allowed_W"] = sorted(set(W.tolist()))
    ctx["W_marg"] = {w: float((W == w).mean()) for w in ctx["allowed_W"]}
    ctx["W_cell"] = {c: {w: float((W[ix] == w).mean()) for w in ctx["allowed_W"]}
                     for c, ix in ctx["cell_idx"].items()}
    ctx["clip_max"] = X[tr].reshape(-1, X.shape[2]).max(0)
    f_rate = float(S[tr].mean()); p = float(y[tr].mean())
    if cfg.label_policy == "parity":
        py = {0: p, 1: p}
    else:
        py = {s: float(y[tr][S[tr] == s].mean()) for s in (0, 1)}
    ctx["cell_prob"] = {(s, yy): (f_rate if s else 1 - f_rate) * (py[s] if yy else 1 - py[s])
                        for s in (0, 1) for yy in (0, 1)}
    tot = X[tr].sum((1, 2))
    ctx["zero_share"] = {yy: float((tot[y[tr] == yy] == 0).mean()) for yy in (0, 1)}
    # real D2 reference (how predictable gender is in REAL data, same feature set)
    rng = np.random.default_rng(cfg.seed)
    sub = rng.choice(len(tr), min(2000, len(tr)), replace=False)
    ctx["real_D2_auc"] = score_D2(ctx["real_vecs"][sub], S[tr][sub], y[tr][sub], cfg.seed)[0]
    # novelty filter threshold: low quantile of real(val)->real(train) NN distance
    ctx["real_flat"] = flat_log(X[tr], M[tr])
    ctx["nn"] = NearestNeighbors(n_neighbors=1).fit(ctx["real_flat"])
    vf = flat_log(X[va], M[va])
    rich = (vf > 0).sum(1) >= 5
    dist = ctx["nn"].kneighbors(vf[rich])[0][:, 0]
    ctx["copy_thr"] = float(np.quantile(dist, cfg.copy_quantile))
    ctx["data_card"] = data_card(X[tr], M[tr], y[tr])
    _log(cfg, f"[data] cohort={cfg.cohort} train={len(tr)} pos={p:.3f} female={f_rate:.3f} "
              f"label_policy={cfg.label_policy} real_D2_auc={ctx['real_D2_auc']:.3f} "
              f"copy_thr={ctx['copy_thr']:.3f}")
    return ctx


def data_card(X, M, y):
    """Aggregate (non-identifying) description of the real training data."""
    names = ["homepage clicks (hp)", "subpage clicks (sp)", "course-content clicks (oc)",
             "quiz interactions (qz)"]
    out = ["REAL DATA SUMMARY (training students; statistics per real week):"]
    for yy, lab in [(0, "Y=0 (passed)"), (1, "Y=1 (withdrew or failed)")]:
        Xi, Mi = X[y == yy], M[y == yy].astype(bool)
        out.append(f"{lab}:")
        for f, nm in enumerate(names):
            v = Xi[:, :, f][Mi]; nz = v[v > 0]
            out.append(f"  - {nm}: mean {v.mean():.1f}/week, non-zero in {100*(v>0).mean():.0f}% "
                       f"of weeks, typical non-zero value {np.median(nz):.0f} "
                       f"(90th percentile {np.percentile(nz, 90):.0f})")
        v = Xi[:, :, SCORE_IDX][Mi]; nz = v[v > 0]
        out.append(f"  - assessment score (sc): submitted in {100*(v>0).mean():.0f}% of weeks, "
                   f"mean score when submitted {nz.mean():.0f}")
        wk = [Xi[:, w, 0][Mi[:, w]].mean() for w in range(X.shape[1])]
        out.append("  - mean hp by week: " + ", ".join(f"w{w}={v:.0f}" for w, v in enumerate(wk)))
        sw = [100 * (Xi[:, w, SCORE_IDX][Mi[:, w]] > 0).mean() for w in range(X.shape[1])]
        out.append("  - % with an assessment by week: " +
                   ", ".join(f"w{w}={v:.0f}" for w, v in enumerate(sw)))
        tot = Xi.sum((1, 2))
        out.append(f"  - students with no activity at all: {100*(tot==0).mean():.0f}%")
    out.append("Counts are skewed: many small values, a few large ones. "
               "Activity of a student is correlated across hp, sp, oc in the same week.")
    return "\n".join(out)


class NoveltyFilter:
    """Rejects (a) near-copies of real training students and (b) duplicates of
    already accepted synthetic students - gender is ignored, so gender-swapped
    twins are caught. Students with < min_nonzero non-zero values (e.g. all
    zero) are not identifying and are exempt."""
    def __init__(self, ctx, min_nonzero=5):
        self.nn, self.thr, self.mnz = ctx["nn"], ctx["copy_thr"], min_nonzero
        self.syn = np.zeros((0, ctx["real_flat"].shape[1]), np.float32)

    def check(self, x, m):
        v = flat_log(x[None], m[None])[0]
        if (v > 0).sum() < self.mnz:
            return None, v
        if self.nn.kneighbors(v[None])[0][0, 0] < self.thr:
            return "copy_of_real", v
        if len(self.syn) and np.sqrt(((self.syn - v) ** 2).sum(1)).min() < self.thr:
            return "duplicate_synthetic", v
        return None, v

    def add(self, v):
        self.syn = np.vstack([self.syn, v[None]])


# =============================================================================
# 4. DISCRIMINATORS  (held-out, cross-validated)
# =============================================================================
def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def score_D1(ctx, sv, sS, sy, seed):
    """Real vs synthetic. Real sample has the SAME S x Y composition as the
    synthetic sample, so D1 judges realism, not the label mix.
    Returns CV accuracy and the 3 most discriminating features (+ direction)."""
    rng = np.random.default_rng(seed)
    r = []
    for (s, yy), ix in ctx["cell_idx"].items():
        k = int(((sS == s) & (sy == yy)).sum())
        if k:
            r.append(ctx["real_vecs"][rng.choice(ix, k, replace=len(ix) < k)])
    rv = np.vstack(r)
    X = np.vstack([rv, sv]); lab = np.r_[np.ones(len(rv)), np.zeros(len(sv))]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = float(accuracy_score(lab, cross_val_predict(_clf(), X, lab, cv=cv)))
    coef = _clf().fit(X, lab)[-1].coef_[0]
    top = [(int(i), "higher" if coef[i] > 0 else "lower") for i in np.argsort(-np.abs(coef))[:3]]
    return acc, top


def score_D2(sv, S, y, seed):
    """Predict gender from synthetic features + label; CV AUC (FairGAN idea)."""
    S = np.asarray(S).astype(int)
    if len(np.unique(S)) < 2 or np.bincount(S).min() < 5:
        return 1.0, 0
    X = np.column_stack([sv, y])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(_clf(), X, S, cv=cv, method="predict_proba")[:, 1]
    coef = _clf().fit(X, S)[-1].coef_[0][:sv.shape[1]]
    return float(roc_auc_score(S, p)), int(np.argmax(np.abs(coef)))


# =============================================================================
# 5. FEEDBACK (specific, numeric)
# =============================================================================
def _stat_of_vec(i):
    if i >= 5 * len(STATS):
        return None, None
    return i // len(STATS), STATS[i % len(STATS)]


def _fmt(v):
    return f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}"


def realism_feedback(ctx, d1, top, sx, sm, sy, tol):
    d = ctx["data"]; tr = ctx["tr"]
    X, M, y = d["X"][tr], d["mask"][tr], d["y"][tr]
    if abs(d1 - 0.5) <= tol:
        head = f"- REALISM: good (D1 accuracy {d1:.2f}, target 0.50). Keep this style."
    else:
        head = (f"- REALISM: a classifier still separates real from your students "
                f"(D1 accuracy {d1:.2f}, target 0.50). Fix these differences:")
    lines = [head]
    for i, direction in top:                       # what D1 relies on most
        f, st = _stat_of_vec(i)
        if f is None:
            continue
        r = raw_stat(X, M, f, st).mean(); s = raw_stat(sx, sm, f, st).mean()
        lines.append(f"  * {FEATURES[f]} ({st}): real {_fmt(r)} vs yours {_fmt(s)}")
    # largest relative gaps of simple per-label statistics
    gaps = []
    for yy in (0, 1):
        rm, smm = M[y == yy].astype(bool), sm[sy == yy].astype(bool)
        if smm.sum() == 0:
            continue
        for f in range(len(FEATURES)):
            rv, svv = X[y == yy][:, :, f][rm], sx[sy == yy][:, :, f][smm]
            for nm, a, b in [("mean per week", rv.mean(), svv.mean()),
                             ("% of weeks non-zero", 100 * (rv > 0).mean(), 100 * (svv > 0).mean())]:
                gaps.append((abs(math.log((b + 0.05 * a + 0.01) / (a + 0.05 * a + 0.01))),
                             f"  * Y={yy} {FEATURES[f]} {nm}: real {_fmt(a)} vs yours {_fmt(b)}"))
        rz = ctx["zero_share"][yy] * 100
        sz = 100 * (sx[sy == yy].sum((1, 2)) == 0).mean()
        gaps.append((abs(math.log((sz + 1) / (rz + 1))),
                     f"  * Y={yy} students with no activity at all: real {rz:.0f}% vs yours {sz:.0f}%"))
    gaps.sort(key=lambda g: -g[0])
    lines += [g[1] for g in gaps[:5] if g[0] > 0.15]
    return "\n".join(lines)


def fairness_feedback(ctx, d2, li, sx, sm, sS, sy, tol):
    if abs(d2 - 0.5) <= tol:
        return (f"- FAIRNESS: good (D2 AUC {d2:.2f}, target 0.50). Students with the same Y "
                f"must keep looking the same for S=0 and S=1.")
    f, st = _stat_of_vec(li)
    lines = [f"- FAIRNESS: gender can still be predicted from your students "
             f"(D2 AUC {d2:.2f}, target 0.50)."]
    if f is not None:
        for yy in (0, 1):
            a = raw_stat(sx[(sS == 0) & (sy == yy)], sm[(sS == 0) & (sy == yy)], f, st)
            b = raw_stat(sx[(sS == 1) & (sy == yy)], sm[(sS == 1) & (sy == yy)], f, st)
            if len(a) and len(b):
                lines.append(f"  * Y={yy}: {FEATURES[f]} ({st}) is {_fmt(a.mean())} for S=0 "
                             f"but {_fmt(b.mean())} for S=1")
        lines.append(f"  Give S=0 and S=1 students with the same Y the same distribution "
                     f"of {FEATURES[f]}; S must not change any number.")
    return "\n".join(lines)


OPT_SYSTEM = ("You are a prompt optimizer in a GAN-style loop. A generator LLM receives "
              "GUIDELINES plus real example rows and writes synthetic student rows. A "
              "discriminator then tries to tell synthetic rows from real rows.")


def optimize_guidelines(gen, guidelines, history, feedback):
    hist = "\n".join(f"- round {h['round']}: D1 accuracy {h['D1_acc']:.2f}" for h in history[-3:])
    user = (f"Current guidelines:\n{guidelines or '(none yet)'}\n\nHistory (D1 accuracy, "
            f"0.50 is perfect):\n{hist or '(none)'}\n\nDiscriminator report on the latest "
            f"rows:\n{feedback}\n\nWrite improved GUIDELINES (bullet points, at most 150 words) "
            f"so the next rows are indistinguishable from real rows. Output only the guidelines.")
    return gen.chat_batch(OPT_SYSTEM, [user], 400)[0].strip()


# =============================================================================
# 6. PROMPTS
# =============================================================================
def system_prompt(cfg, ctx):
    if cfg.fmt == "temporal":
        fmt = ("FORMAT - one student per line:\n"
               "<S=s;Y=y;W=n> w0:hp,sp,oc,qz,sc | w1:hp,sp,oc,qz,sc | ... | "
               "w(n-1):hp,sp,oc,qz,sc </end>\n"
               "- Write exactly W weeks, numbered w0 .. w(W-1), five integers per week.\n")
    else:
        fmt = ("FORMAT - one student per line, as column=value pairs:\n"
               "<S=s;Y=y;W=n> hp_w0=.., hp_w1=.., ..., hp_w(n-1)=.., sp_w0=.., ..., "
               "sc_w(n-1)=.. </end>\n"
               "- Give every column hp,sp,oc,qz,sc for every week index 0 .. W-1.\n")
    rules = ("You generate synthetic records of Open University (OULAD) students for the first "
             "weeks of a course (early-warning window).\n" + fmt +
             "- S = gender code (0/1). Y = final outcome: 1 = withdrew or failed, 0 = passed.\n"
             "- hp = homepage clicks, sp = subpage clicks, oc = course-content clicks, "
             "qz = quiz interactions in that week (integers >= 0).\n"
             "- sc = mean score (0-100) of assessments submitted in that week, 0 if none.\n"
             "- Output ONLY the student lines: no numbering, no explanations, no code fences.\n"
             "- Every student must be NEW and different; never copy or lightly edit an example.\n")
    if cfg.mode == "full":
        rules += ("- FAIRNESS: S must not influence the numbers. Students with the same Y must "
                  "look statistically identical for S=0 and S=1.\n")
    if cfg.use_data_card:
        rules += "\n" + ctx["data_card"] + "\n"
    return rules


def sample_shots(ctx, cfg, rng):
    d, tr = ctx["data"], ctx["tr"]
    idx = []
    for (s, yy), ix in ctx["cell_idx"].items():
        idx += list(tr[rng.choice(ix, cfg.shots_per_cell, replace=False)])
    rng.shuffle(idx)
    return [serialize(d["X"][i], d["mask"][i], d["S"][i], d["y"][i], cfg.fmt) for i in idx]


def user_prompt(shots, slots, guidance, cfg):
    g = ""
    if guidance:
        title = "GUIDELINES" if cfg.optimizer else "FEEDBACK FROM THE DISCRIMINATORS (apply it)"
        g = f"\n{title}:\n{guidance}\n"
    heads = "\n".join(f"<S={s};Y={y};W={w}>" for s, y, w in slots)
    return ("REAL EXAMPLE STUDENTS:\n" + "\n".join(shots) + "\n" + g +
            f"\nWrite {len(slots)} NEW students, one per line, using exactly these headers "
            f"in this order:\n{heads}")


def draw_slots(ctx, cfg, rng, n, need=None):
    """need=None: draw cells from the target distribution; otherwise from the
    remaining per-cell need (final generation balances itself)."""
    cells = list(ctx["cell_prob"].keys())
    if need is None:
        p = np.array([ctx["cell_prob"][c] for c in cells])
    else:
        p = np.array([max(need.get(c, 0), 0) for c in cells], float)
        if p.sum() == 0:
            p = np.array([ctx["cell_prob"][c] for c in cells])
    p = p / p.sum()
    out = []
    for k in rng.choice(len(cells), n, p=p):
        s, yy = cells[k]
        wd = ctx["W_marg"] if cfg.label_policy == "parity" else ctx["W_cell"][(s, yy)]
        ws, wp = list(wd.keys()), np.array(list(wd.values()))
        out.append((s, yy, int(rng.choice(ws, p=wp / wp.sum()))))
    return out


# =============================================================================
# 7. ONE GENERATION STEP (batched prompts -> parsed -> filtered rows)
# =============================================================================
def generate_step(ctx, cfg, gen, guidance, rng, novelty, rej, need=None, accept=None):
    users = []
    for _ in range(cfg.prompts_per_batch):
        users.append(user_prompt(sample_shots(ctx, cfg, rng),
                                 draw_slots(ctx, cfg, rng, cfg.students_per_prompt, need),
                                 guidance, cfg))
    raw = gen.chat_batch(ctx["system"], users, ctx["max_new_tokens"])
    with open(os.path.join(cfg.OUT_DIR, "raw_samples.txt"), "a") as fh:   # for debugging
        fh.write(f"\n===== {time.strftime('%H:%M:%S')} (first output of this batch) =====\n"
                 + raw[0][:3000] + "\n")
    rows = []
    for text in raw:
        parsed, reasons = parse_generation(text, cfg.fmt, ctx["allowed_W"], ctx["clip_max"])
        for r in reasons:
            rej[r] = rej.get(r, 0) + 1
        for x, m, s, y in parsed:
            why, v = novelty.check(x, m)
            if why is None and accept is not None:
                why = accept(x, m, s, y)
            if why:
                rej[why] = rej.get(why, 0) + 1
                continue
            novelty.add(v)
            rows.append((x, m, s, y))
    rej["requested"] = rej.get("requested", 0) + cfg.prompts_per_batch * cfg.students_per_prompt
    rej["accepted"] = rej.get("accepted", 0) + len(rows)
    return rows


def _row_json(x, m, s, y, **extra):
    return json.dumps(dict(x=np.asarray(x).tolist(), m=np.asarray(m).tolist(),
                           s=int(s), y=int(y), **extra))


def _row_load(l):
    r = json.loads(l)
    return (np.array(r["x"], np.float32), np.array(r["m"], np.int8), r["s"], r["y"]), r


def _stack(rows):
    return (np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]),
            np.array([r[2] for r in rows]), np.array([r[3] for r in rows]))


# =============================================================================
# 8. ADVERSARIAL PROMPT OPTIMISATION
# =============================================================================
def _p(cfg, name): return os.path.join(cfg.OUT_DIR, name)


def train_loop(ctx, cfg, gen, t_start):
    rng = np.random.default_rng(cfg.seed)
    st = json.load(open(_p(cfg, "train_state.json"))) if os.path.exists(_p(cfg, "train_state.json")) else None
    if st and st.get("finished"):
        _log(cfg, "[train] finished state found -> skipping the adversarial loop")
        return st
    rounds = {}
    if os.path.exists(_p(cfg, "round_rows.jsonl")):
        for l in open(_p(cfg, "round_rows.jsonl")):
            row, meta = _row_load(l); rounds.setdefault(meta["round"], []).append(row)
    if not st:
        st = dict(next_round=0, guidance=None, history=[], best_score=1e9, best_guidance=None,
                  best_round=None, streak=0, converged=False, finished=False, rejections={})
    else:
        _log(cfg, f"[train] RESUMING at round {st['next_round']}")
    novelty = NoveltyFilter(ctx)
    budget = cfg.time_budget_h * 3600 * cfg.train_budget_frac

    for r in range(st["next_round"], cfg.max_rounds):
        if time.time() - t_start > budget:
            _log(cfg, f"[train] training time budget used up at round {r}"); break
        t0 = time.time(); rej = {}
        rows = generate_step(ctx, cfg, gen, st["guidance"], rng, novelty, rej)
        rounds[r] = rows
        with open(_p(cfg, "round_rows.jsonl"), "a") as fh:
            for x, m, s, y in rows:
                fh.write(_row_json(x, m, s, y, round=r) + "\n")
        pool, k = [], r
        while k >= 0 and r - k < cfg.pool_rounds and (k == r or len(pool) < cfg.min_eval_rows):
            pool += rounds.get(k, []); k -= 1
        for kk, vv in rej.items():
            st["rejections"][kk] = st["rejections"].get(kk, 0) + vv
        if len(pool) < cfg.min_eval_rows:
            _log(cfg, f"[round {r}] accepted={len(rows)} pooled={len(pool)} < {cfg.min_eval_rows}"
                      f" -> collecting more  rej={rej}")
            st["next_round"] = r + 1; json.dump(st, open(_p(cfg, "train_state.json"), "w"), indent=2)
            continue
        sx, sm, sS, sy = _stack(pool)
        sv = build_matrix(sx, sm)
        d1, top = score_D1(ctx, sv, sS, sy, cfg.seed + r)
        d2, li = score_D2(sv, sS, sy, cfg.seed + r)
        use_fair = cfg.mode == "full"
        score = abs(d1 - 0.5) + cfg.fairness_weight * abs(d2 - 0.5)
        ok = abs(d1 - 0.5) <= cfg.tol_d1 and (not use_fair or abs(d2 - 0.5) <= cfg.tol_d2)
        st["streak"] = st["streak"] + 1 if ok else 0
        h = dict(round=r, accepted=len(rows), pooled=len(pool), D1_acc=d1, D2_auc=d2,
                 label_SPD=spd(sy, sS), leak=vec_name(li), D1_top=[vec_name(i) for i, _ in top],
                 score=score, in_band=ok, rejections=rej, minutes=(time.time() - t0) / 60,
                 tok_per_s=gen.tokens_out / max(gen.seconds, 1e-9))
        st["history"].append(h)
        _log(cfg, f"[round {r}] acc={len(rows)}/{rej['requested']} pooled={len(pool)} "
                  f"D1={d1:.3f} D2={d2:.3f} SPD={h['label_SPD']:+.3f} leak={h['leak']} "
                  f"score={score:.3f} {'IN-BAND' if ok else ''} ({h['minutes']:.1f} min, "
                  f"{h['tok_per_s']:.0f} tok/s)")
        if score < st["best_score"]:     # guidance that PRODUCED this round's data
            st.update(best_score=score, best_guidance=st["guidance"], best_round=r)
        fb = realism_feedback(ctx, d1, top, sx, sm, sy, cfg.tol_d1)
        if use_fair:
            fb += "\n" + fairness_feedback(ctx, d2, li, sx, sm, sS, sy, cfg.tol_d2)
        n_copy = rej.get("copy_of_real", 0) + rej.get("duplicate_synthetic", 0)
        if n_copy:
            fb += f"\n- DIVERSITY: {n_copy} of your students were copies/duplicates and were discarded. Write new, varied students."
        st["guidance"] = optimize_guidelines(gen, st["guidance"], st["history"], fb) if cfg.optimizer else fb
        st["next_round"] = r + 1
        json.dump(st, open(_p(cfg, "train_state.json"), "w"), indent=2)
        if st["streak"] >= cfg.patience and r + 1 >= cfg.min_rounds:
            st["converged"] = True
            _log(cfg, f"[train] CONVERGED at round {r} ({cfg.patience} consecutive rounds in band)")
            break
    if not st["converged"]:
        _log(cfg, f"[train] NOT converged. Using best guidance from round {st['best_round']} "
                  f"(score {st['best_score']:.3f}). Report this honestly.")
    st["finished"] = True
    json.dump(st, open(_p(cfg, "train_state.json"), "w"), indent=2)
    return st


# =============================================================================
# 9. FINAL GENERATION (balanced, filtered, resumable, time-boxed)
# =============================================================================
def final_generation(ctx, cfg, gen, guidance, t_start):
    rng = np.random.default_rng(cfg.seed + 999)
    path = _p(cfg, "synthetic_rows.jsonl")
    target = {c: int(round(cfg.n_synthetic * p)) for c, p in ctx["cell_prob"].items()}
    have = {c: 0 for c in target}
    zero_have = {0: 0, 1: 0}
    y_target = {yy: sum(v for (s, y2), v in target.items() if y2 == yy) for yy in (0, 1)}
    zero_cap = {yy: int(math.ceil((ctx["zero_share"][yy] + cfg.zero_cap_margin) * y_target[yy]))
                for yy in (0, 1)}
    novelty = NoveltyFilter(ctx)
    if os.path.exists(path):
        for l in open(path):
            (x, m, s, y), _ = _row_load(l)
            have[(s, y)] += 1; zero_have[y] += int(x.sum() == 0)
            novelty.add(flat_log(x[None], m[None])[0])
    rej = json.load(open(_p(cfg, "rejections_final.json"))) if os.path.exists(_p(cfg, "rejections_final.json")) else {}
    _log(cfg, f"[final] target per cell {target}, already {have}, zero cap {zero_cap}")

    def accept(x, m, s, y):          # last check; counts the row when accepted
        if have[(s, y)] >= target[(s, y)]:
            return "cell_full"
        if x.sum() == 0 and zero_have[y] >= zero_cap[y]:
            return "zero_cap"
        have[(s, y)] += 1; zero_have[y] += int(x.sum() == 0)
        return None

    stalls = 0
    while sum(have.values()) < sum(target.values()):
        left = cfg.time_budget_h * 3600 - (time.time() - t_start)
        if left < 15 * 60:
            _log(cfg, "[final] time budget reached -> stopping with what we have"); break
        need = {c: target[c] - have[c] for c in target}
        rows = generate_step(ctx, cfg, gen, guidance, rng, novelty, rej, need=need, accept=accept)
        with open(path, "a") as fh:
            for x, m, s, y in rows:
                fh.write(_row_json(x, m, s, y) + "\n")
        json.dump(rej, open(_p(cfg, "rejections_final.json"), "w"), indent=2)
        stalls = stalls + 1 if not rows else 0
        if stalls >= 5:
            _log(cfg, "[final] 5 empty batches in a row -> stopping"); break
        _log(cfg, f"[final] {sum(have.values())}/{sum(target.values())}  (+{len(rows)}; "
                  f"{gen.tokens_out / max(gen.seconds, 1e-9):.0f} tok/s)")
    return finalize(ctx, cfg)


def finalize(ctx, cfg):
    recs = [_row_load(l)[0] for l in open(_p(cfg, "synthetic_rows.jsonl"))]
    SX, SM, SS, SY = _stack(recs)
    SM = SM.astype(np.int8); SS = SS.astype(np.int8); SY = SY.astype(np.int8)
    for n, a in [("synth_X_seq", SX), ("synth_mask", SM), ("synth_S_gender", SS), ("synth_y", SY)]:
        np.save(_p(cfg, f"{n}.npy"), a)
    rows = []
    for i in range(len(SX)):
        for w in np.where(SM[i] == 1)[0]:
            r = dict(synth_id=i, week=int(w), S_gender=int(SS[i]), label=int(SY[i]))
            r.update({f: float(SX[i, w, j]) for j, f in enumerate(FEATURES)})
            rows.append(r)
    pd.DataFrame(rows).to_csv(_p(cfg, "synthetic_weekly_long.csv"), index=False)
    # final check: D1 / D2 on the delivered data set
    sv = build_matrix(SX, SM)
    d1, _ = score_D1(ctx, sv, SS, SY, cfg.seed)
    d2, li = score_D2(sv, SS, SY, cfg.seed)
    d = ctx["data"]; tr = ctx["tr"]
    chk = dict(mode=cfg.mode, cohort=cfg.cohort, n=len(SX), D1_acc=d1, D2_auc=d2,
               D2_leak=vec_name(li), real_D2_auc=ctx["real_D2_auc"],
               label_SPD_synth=spd(SY, SS), label_SPD_real=spd(d["y"][tr], d["S"][tr]),
               female_rate=float(SS.mean()), pos_rate=float(SY.mean()),
               pos_rate_F=float(SY[SS == 1].mean()), pos_rate_M=float(SY[SS == 0].mean()),
               weeks=dict(zip(*[a.tolist() for a in np.unique(SM.sum(1), return_counts=True)])),
               all_zero_share={yy: float((SX[SY == yy].sum((1, 2)) == 0).mean()) for yy in (0, 1)})
    save_json(chk, _p(cfg, "final_check.json"))
    _log(cfg, f"[final] saved {len(SX)} students. check: {json.dumps(chk, default=str)}")
    return dict(X=SX, mask=SM, S=SS, y=SY)


# =============================================================================
# MAIN
# =============================================================================
def run(cfg: Config):
    cfg.resolve()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    save_json(asdict(cfg), _p(cfg, "config.json"))
    t_start = time.time()
    np.random.seed(cfg.seed)
    ctx = build_context(cfg)
    gen = make_generator(cfg, ctx)
    ctx["system"] = system_prompt(cfg, ctx)
    # output length: tokens of a long real student x students per prompt (+25%)
    d = ctx["data"]
    longest = [serialize(d["X"][i], d["mask"][i], 0, 0, cfg.fmt)
               for i in ctx["tr"][np.argsort(-d["X"][ctx["tr"]].sum((1, 2)))[:20]]]
    per = max(gen.n_tokens(s) for s in longest)
    ctx["max_new_tokens"] = int(per * cfg.students_per_prompt * 1.25) + 64
    _log(cfg, f"[run] mode={cfg.mode} fmt={cfg.fmt} optimizer={cfg.optimizer} "
              f"max_new_tokens={ctx['max_new_tokens']}")
    _log(cfg, "\n==== ADVERSARIAL PROMPT OPTIMISATION ====")
    st = train_loop(ctx, cfg, gen, t_start)
    _log(cfg, "\n==== FINAL GENERATION ====")
    synth = final_generation(ctx, cfg, gen, st["best_guidance"], t_start)
    _log(cfg, f"[run] done in {(time.time() - t_start) / 3600:.2f} h")
    return synth, st


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    for k, v in asdict(Config()).items():
        if k in ("fmt", "optimizer"):
            continue
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else
                        (lambda s: s.lower() in ("1", "true", "yes")), default=v)
    run(Config(**vars(ap.parse_args())))
