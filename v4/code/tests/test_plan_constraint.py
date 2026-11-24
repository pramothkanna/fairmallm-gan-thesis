import random, torch, numpy as np
from fair_mallm_gan_v4 import PlanConstraint, parse_generation, plan_recall
random.seed(0)
V = list("0123456789,:;=<>|wSYW /end\n") + [", ", ",\n", " | ", "<S=", " </end>\n"]
idx = {t: i for i, t in enumerate(V)}
dec = lambda ids: "".join(V[i] for i in ids)
ids = dict(d0=[idx["0"]], d19=[idx[c] for c in "123456789"], digits=[idx[c] for c in "0123456789"],
           comma=[i for t, i in idx.items() if t.startswith(",")])

class Writer:
    """Plan-ignoring fake LLM: writes headers in order, random values in every field
    (zero or not at random). Keeps a structured cursor and accepts forced tokens."""
    def __init__(self, heads):
        self.heads, self.k, self.queue, self.w, self.f, self.cur = heads, 0, [], 0, 0, ""
        self.start_student()
    def start_student(self):
        s, y, W = self.heads[self.k]
        self.queue = ["<S="] + list(f"{s};Y={y};W={W}> w0:"); self.w = 0; self.f = 0; self.cur = ""
        self.target = str(random.choice([0, 0, random.randint(1, 120)]))
    def want(self):
        if self.queue: return self.queue[0]
        if self.k >= len(self.heads): return None
        if len(self.cur) < len(self.target): return self.target[len(self.cur)]
        return "," if self.f < 4 else (" | " if self.w + 1 < self.heads[self.k][2] else " </end>\n")
    def accept(self, tok):
        if self.queue:
            assert tok == self.queue[0], (tok, self.queue[0]); self.queue.pop(0); return
        if tok.isdigit(): self.cur += tok; return
        if tok.startswith(","):
            self.f += 1; self.cur = ""; self.target = str(random.choice([0, random.randint(1, 120)])); return
        if tok == " | ":
            self.w += 1; self.f = 0; self.cur = ""; self.target = str(random.choice([0, random.randint(1, 60)]))
            self.queue = list(f"w{self.w}:"); return
        if tok == " </end>\n":
            self.k += 1
            if self.k < len(self.heads): self.start_student()
            return
        raise ValueError(repr(tok))

ok = tot = 0; rec = []
for trial in range(200):
    heads = [(random.randint(0, 1), random.randint(0, 1), random.choice([11, 12])) for _ in range(4)]
    plans = []
    for _, _, W in heads:
        q = [w for w in range(W) if random.random() < 0.3]; a = [w for w in range(W) if random.random() < 0.25]
        plans.append((q, a, {w: random.choice([1, 2, 2, 3]) for w in q}, {w: random.choice([1, 2, 2, 2, 3]) for w in a}))
    pc = PlanConstraint(dec, [plans], 0, ids); wr = Writer(heads); gen = []
    while True:
        want = wr.want()
        if want is None: break
        sc = torch.randn(1, len(V)) * 0.1
        if not wr.queue and wr.k < len(heads):   # realistic 2nd choice: end the current field
            sep = "," if wr.f < 4 else (" | " if wr.w + 1 < heads[wr.k][2] else " </end>\n")
            sc[0, idx[sep]] = 5.0
        sc[0, idx[want]] = 10.0
        sc = pc(torch.tensor([gen], dtype=torch.long).reshape(1, -1), sc)
        tok = V[int(sc[0].argmax())]; wr.accept(tok); gen.append(idx[tok])
    text = dec(gen)
    rows, rs = parse_generation(text, "temporal", [11, 12], np.full(5, 1e9))
    # note: parse clips sc at 100, so 3-digit scores > 100 would be clipped -> use raw lengths
    assert len(rows) == 4, (rs, text[:300])
    for x, m, s, y, k in rows:
        tot += 1
        q, a, qn, an = plans[k]
        ok += all((x[w, 3] > 0) == (w in q) and (x[w, 4] > 0) == (w in a)
                  and (w not in q or len(str(int(x[w, 3]))) == qn[w])
                  and (w not in a or len(str(int(x[w, 4]))) == an[w]) for w in range(int(m.sum())))
        assert x[:, 4].max() <= 999
print(f"students {tot}, follow plan exactly: {ok}, forced steps last trial {pc.stats}")
print(text[:330])
