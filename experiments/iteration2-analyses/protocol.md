# Protocol: study iteration 2 — deficit-heuristic and statistical analyses

Status: **locked BEFORE running. Commit timestamp is the pre-registration.**

Extends the iteration-1 results with a training-free reference and tightened
statistics. Experiments 2/3/4 are analyses of already-released CSVs (zero
training); Experiment 1 is a new **training-free** baseline evaluation;
Experiment 5 is a statistical statement computed from released per-trio
values. No new deep-learning training runs; run audit unchanged.

## Experiment 1 — Largest-pressure-deficit heuristic baseline
`src/heuristic_baseline.py`. Design identical to the 15-trio sweep
(`cross_many_trios.py`): trio t uses placement seeds {20+3t, 21+3t, 22+3t};
train/test placement subsets from `default_rng(1000+seed)`; scenario split
seed = test placement seed; Experiment B (held-out test scenarios through
the unseen placement). No model is trained: each observed node is scored by
minus its mean training-standardized pressure over the window
(z-scoring with the test placement's train-scenario statistics); the
prediction is the argmax over observed nodes only (same k/9 observability
ceiling as the identity-adjacency GCN). Loc@1 and Loc@3 over ALL leak test
windows, per trio × k ∈ {7,5,3}. Output: `data/results/pressure_deficit_heuristic.csv`.
Paired per-trio comparisons vs CNN and GCN(real) from `cross_many_trios.csv`
with 95% t-intervals.

**Prediction.** The heuristic will transfer (it is placement-invariant by
construction) but sit well below the learned GCN variants — if it does NOT,
the paper's node-indexed-transfer claim must be requalified ("learned"
models add little over a hand rule). Either outcome reported as found.

## Experiment 2 — Within-density overlap→transfer regression
`src/overlap_transfer.py`, reading `cross_many_trios.csv` (per-trio
mean train–test overlap = mean of the recorded pair; n=15 per model×k).
Per (model, k): OLS slope of Experiment-B Loc@1 on overlap with 95% t-CI,
plus Spearman rho. Output: `data/results/overlap_transfer.csv`.

**Prediction.** Positive slopes within density for the channel CNN
(channel-semantics account predicts overlap matters at fixed k); flatter
slopes for node-indexed GCN variants (their input is node-indexed, so
train–test channel identity should not matter). If CNN slopes are flat,
weaken the overlap wording to "co-occurs with density" and say so.

## Experiment 3 — Size × density detection grid
`analysis_v2.py::size_density_analysis()` on `leak_size.csv` (4 arch × 4 k ×
6 subseed × 3 sizes): detection AUC (and F1) per (arch, size, k) with
mean ± SD; table `tab_sizedensity.tex`; stats section `size_density`.
Manuscript: §4.3 gains the grid; abstract/conclusion detection claims gain a
size qualifier; read against the 0.669 time-position floor.

**Prediction (from v1's own numbers).** Small-leak AUC at k ≤ 5 sits at or
below the time-position floor; large-leak AUC stays above it at every k.

## Experiment 4 — Variance-share bootstrap
`analysis_v2.py::variance_bootstrap_analysis()`: recompute per-condition
train/place/split variance shares (16 conditions, AUC and Loc@1), then
bootstrap over conditions (B = 10,000) for a 95% CI on the mean split share;
report the fraction of clipped per-cell components. Stats section
`variance_bootstrap`; §4.11 and Appendix C updated with the interval.

## Experiment 5 — Multiplicity / confirmatory labeling
From `cross_many_trios.csv`: paired t-tests (n=15 trios) for the 9
GCN-variant-vs-CNN contrasts; Holm correction across the family; which
contrasts survive at 0.05 recorded in `stats.json` (`multiplicity`).
Manuscript §3.9 gains a confirmatory/exploratory policy paragraph (the
pre-registered contrasts = confirmatory; the rest exploratory, uncorrected,
headline claims rest on confirmatory ones); §4.9 states the Holm outcome.

## Framing discipline
All five report as found; any wording that the results refute gets rewritten,
not defended. The heuristic is a diagnostic baseline (no novelty claim).
