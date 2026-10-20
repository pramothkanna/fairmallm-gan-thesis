# Progress log

| Date | Step | Result |
|---|---|---|
| before Sept. 2026 | Proposal and presentation; OULAD preprocessing (weekly binning, tensors, text panels) | finished |
| late Sept. 2026 | First generator (v2), first full Kaggle run | 400 students in ≈ 10 h; not converged; label gap −0.20; too few quiz/assessment weeks |
| 16 Oct. 2026 | Code review → v3 (cohort, grouped split, conditioned slots, novelty filter, batched generation, new evaluation); EDA; smoke test | ≈ 5× faster generation, all checks passed |
| 16–17 Oct. 2026 | Proposed model, ablation without D2, CTGAN, MALLM-GAN baseline; joint evaluation (5 seeds) | see `results/kaggle_runs/fairmallm-v3-evaluate-all/eval_all/report.md` |
| 19 Oct. 2026 | Analysis of the runs; size-matched comparison at 623 students; module-relative feature pilot | see `results/analysis/`, `results/size_matched_623/` |

## Current status per research question (19 Oct. 2026)
| RQ | Done | Open |
|---|---|---|
| RQ1 fairness | D2 0.58 vs 0.77 real; label SPD 0.000; ablation; D2 stability check | downstream SPD −0.09 (module proxy); second generator seed |
| RQ2 temporal fidelity | TSTR vs size-matched TRTR, fidelity metrics, activity profile | realism (D1 0.83), quiz/assessment weeks, diversity |
| RQ3 trade-off | all four generators, full size and 623 students | improved generator; MALLM-GAN to 2,000 if GPU time allows |
| RQ4 ETB | preprocessing, LightGBM, CTGAN baseline | LLM generation (data licence) |

## Main findings so far
1. The fairness feedback works on the data: gender leakage 0.58 (real 0.77), equal dropout rates, cost 0.006 AUC.
2. Realism is the bottleneck: quiz/assessment activity in 5–7 % of weeks instead of 19 %; D1 0.83; no convergence.
3. Fair data did not give fair predictions: women are concentrated in low-click modules, so click-driven
   models flag them more (overall SPD −0.09 to −0.14, within modules men are flagged more).
4. At equal size all generators are about equally useful; the LLM data stop adding information beyond
   a few hundred students (diversity).

## Planned changes (in this order)
1. Fairness reported overall and within modules; module-relative click features (pilot: SPD −0.067 → −0.012).
2. Feature-major weekly serialisation for sparse features.
3. Module-conditioned generation with the module's assessment weeks in the prompt.
4. Less noisy feedback (200 students per round, running best, 15 rounds).
5. Activity-rate matching in the final filter; more diverse prompts.
6. Final runs + two extra seeds; MALLM-GAN to 2,000 if GPU time allows; ETB LLM run.
