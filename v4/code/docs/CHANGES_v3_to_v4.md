# Changes v3 -> v4 (21 Oct 2026): realism of quiz and assessment weeks

> **Outcome (22 Oct. 2026).** v4.3 was run in both modes on Kaggle (proposed 4.9 h, without D2 4.3 h) and
> evaluated jointly with v3, MALLM-GAN and CTGAN (5 seeds). Quiz/assessment weeks now match the real data
> (18–20 % vs 19 %), D1 0.68 (v3 0.83), TSTR AUC 0.721 / 0.737 (v3 0.686 / 0.692; CTGAN 0.724), gender leakage
> 0.60 with D2 vs 0.72 without. Results: `results/kaggle_runs/fairmallm-evaluate-joint/eval_joint/report.md`,
> analysis: `results/analysis_v4/`, smoke tests: `results/kaggle_runs/fairmallm-v4-smoke/SMOKE_TESTS.md`.

## Why
In the October v3 runs the LLM wrote far too few quiz and assessment weeks
(`results/analysis/A_activity_profile.csv`):

| data | quiz active weeks | assessment active weeks | all-zero students |
|---|---|---|---|
| Real (train) | 19.0 % | 18.5 % | 2.3 % |
| LLM + D2 (v3) | 7.2 % | 5.5 % | 5.2 % |
| LLM without D2 (v3) | 6.6 % | 5.2 % | 5.6 % |

The data card already told the model the real rates and the D1 feedback named the gap,
but Qwen2.5-7B did not act on it. This is the main reason for the low D1 realism (≈0.83)
and the lower GRU AUC (0.68–0.69 vs 0.72 CTGAN).

## v4.1 (same day, after the first smoke test)
Smoke test 1 (plan given as text before each header): plan recall only 0.52, 84 of 192
students rejected, quiz weeks 5 % (real 19 %), assessment weeks 17 % (real 18 %).
Qwen swapped the qz and sc columns and sometimes copied the plan into the header.
v4.1 gives the plan as a fill-in **template** per student instead, e.g.
`<S=1;Y=0;W=12> w0:_,_,_,0,0 | w1:_,_,_,Q,S | ...`: the model only replaces `_` (clicks),
`Q` (quiz interactions >= 1) and `S` (score 1-100). Few-shot examples are plain real
students (no plan text). Everything else as below.

## v4.3 (current default, used for the full runs)
Smoke test 3 (v4.2) worked: plan recall 1.00, 97 % of students kept, quiz weeks 19.5 % /
assessment weeks 18.6 % (real 19.0 / 18.4), D1 0.64, D2 0.48 (see
`results/kaggle_runs/fairmallm-v4-smoke/SMOKE_TESTS.md`). Remaining flaw: when forced to write a non-zero
value Qwen wrote the smallest one (qz mostly 1; real mean 9.6 per week for Y=0).
v4.3: the plan also carries the number of digits of the donor's qz / sc value in each
planned week (order of magnitude only); the decoder enforces that length, Qwen chooses
the digits. Test: 800/800 students follow plan and digit lengths exactly.

## v4.2 (after smoke test 2)
Smoke test 2 (templates): plan recall 0.49, Qwen set subpage/content clicks to 0 almost
everywhere, and the longer prompts caused a CUDA out-of-memory (batch 8 -> 4, slower).
Conclusion: a 7B model does not reliably follow positional instructions in the prompt.

v4.2 enforces the plan **during decoding** (`plan_mode="constrained"`, class
`PlanConstraint`, a Hugging Face logits processor). The prompt is the short v3 prompt.
While Qwen writes week w of student k, the 4th value (qz) must be exactly 0 if w is not a
planned quiz week and must start with a digit 1-9 if it is; the same for the 5th value
(sc, at most 3 digits). hp, sp, oc and the actual quiz/score values remain Qwen's choice.
This is standard grammar-constrained ("structured") decoding; nothing is edited after
generation. `tests/test_plan_constraint.py`: a fake model that deliberately ignores the
plans generates 800 students through the processor -> 800/800 follow the plan exactly.

## v4 / v4.1 details (kept for the record)
1. **Activity plan per requested student.** Each header comes with the quiz (`qz`) and
   assessment (`sc`) weeks of a randomly drawn *real* training student with the same
   outcome Y, e.g. `(plan: qz in w1,w3; sc in w3,w7) <S=0;Y=0;W=12>`. W is taken from the
   same donor. The LLM still writes every number.
   * mode `full`: donor drawn regardless of gender → the plan carries no gender signal.
   * mode `no_fair`: donor from the same S × Y cell (real mix, as the ablation should be).
   * mode `mallm_flat`: never used (baseline unchanged).
2. **Plan check:** a student that fills < 50 % of its planned qz/sc weeks is rejected
   (`plan_mismatch`, `Config.plan_min_recall`).
3. **Few-shot examples** show their own plan; half of them (`active_shot_frac`) have
   quiz/assessment activity.
4. **Feedback** always contains real vs synthetic qz/sc active-week rates and plan
   compliance; the data card adds "% with quiz activity by week".
5. `zero_cap_margin` 0.03 → 0.01.
6. Logs/`final_check.json` add `qz_active_weeks_pct`, `sc_active_weeks_pct`,
   `plan_recall_final`, `plan_mismatch_final`.

## Methodological note (for the thesis / supervisors)
The plan conditions the generator on the *binary* quiz/assessment calendar of real
students (not their counts). Like the gender × outcome headers in v3, it is conditioning,
not copying: the novelty filter and DCR still check for near-copies. In the proposed
model the plan is drawn independently of gender, so it cannot add gender leakage.
This is an addition to the proposal and should be listed under "differences from the
proposal text".

## Runs (Kaggle, self-contained notebooks, input only `oulad-processed`)
| notebook | purpose | approx. GPU time |
|---|---|---|
| `kaggle/fairmallm-v4-smoke.ipynb` | 128 students, 2 rounds: does Qwen follow plans? | ≈ 40 min |
| `kaggle/fairmallm-v4-full.ipynb` | proposed model, 2,000 students + GRU eval (3 seeds) | ≈ 7–9 h |
| `kaggle/fairmallm-v4-nofair.ipynb` | ablation without D2, same changes | ≈ 7–9 h |

Baselines (CTGAN, MALLM-GAN) and the real-data GRU references are **not** rerun.
Tested locally with the mock generator (both modes) and `evaluate_v3.py` on the output.
