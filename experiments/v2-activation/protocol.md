# Protocol: v2 Experiment 1 — Two-tier sensor-activation scheme (RQ4)

Status: **locked BEFORE running. Commit timestamp is the pre-registration.**

## Origin

The v1 results established the asymmetry
(detection robust at k=3; localization placement/channel-bound). The scheme turns
that finding into a deployment strategy: **3 sensors stay active for detection;
on alarm, the remaining sensors wake and a dense localizer pinpoints the leak.**

Why v1's own mechanism results predict this can work: the localization failure
was shown to be a channel-semantics/representation problem, not missing
information (permutation test §4.7, transplant §4.8). Activating the dormant
sensors returns the localizer to a placement it was trained on — no placement
shift at all. The honest binding constraint is detector recall at k=3
(small-leak AUC 0.53–0.68; time-position floor 0.669).

## Design

Per subseed s ∈ {0..5} (six replicates, same scenario-grouped split convention
as the master benchmark, split seed = subseed):

- **Nested sensor subsets.** One permutation of the 9 junctions per subseed,
  drawn with `default_rng(4000+s)` (fresh, disjoint seed range). Tier-1 set =
  first 3 (sorted); activation variants = first m for m ∈ {3, 5, 7, 9} (sorted;
  each is a superset of the tier-1 set).
- **Tier-1 detectors** (always on, 3 sensors):
  (a) CNN k=3 trained under the master protocol (fresh per-run seed tag
  `actdet-cnn-3-{s}`); (b) Extra Trees (200 trees, `random_state=0`) on raw
  flattened 3-channel windows — the strongest v1 classical detector.
- **Tier-2 localizers:** CNN trained at m ∈ {3, 5, 7, 9} active sensors on the
  nested m-subsets. The m=9 runs use the **master per-run seed tags**
  (`cnn-9-{s}`) so they are bit-identical to released `benchmark.csv` rows —
  this doubles as the harness gate (below).
- **Trigger rule:** detection probability ≥ τ, swept over
  τ ∈ {0.1, 0.2, …, 0.9} ∪ {0.5} ∪ {Youden index per run}. On trigger, the
  m-sensor localizer scores the same window (same window indices, m channels,
  training-split standardization — no leakage).

## Run counts (extend the machine-emitted run audit)

- 6 CNN k=3 detectors (fresh seeds) + 6 × 3 CNN localizers at m ∈ {3,5,7}
  (fresh seeds) + 6 CNN m=9 localizers (master seeds) = **30 DL runs** (~5 min
  GPU). 6 Extra Trees fits (classical count 200 → 206). τ sweep and latency are
  evaluation-only (no training).

## Metrics (per detector × subseed × τ × m → `data/results/activation_scheme.csv`)

recall, precision, FPR, no-leak activation rate (duty-cycle proxy), mean active
sensors per window = 3 + (m−3)·activation-rate, **unconditional Loc@1/Loc@3**
(untriggered leak windows count as localization misses), conditional Loc@1/Loc@3
on triggered leak windows, dense reference Loc@1 (m=9 localizer on all leak
windows), censor count and mean/median **time-to-localization** (first trigger
window end minus leak onset hour, in hours, from `window_meta.json` schedules;
4-h window length included by construction).

## Gate (abort if failed)

The 6 master-seed m=9 retrains must reproduce `benchmark.csv` rows
(arch=cnn, k=9, subseed=s) to within 1e-12 on bal_acc, f1, auc, loc_acc, loc3,
loc5. Written to `data/results/activation_gate.json`; any mismatch aborts the
sweep.

## Predictions (pre-registered; either outcome reported as found)

- **P1.** Unconditional Loc@1 (CNN detector, τ=0.5, m=9) lands well above the
  always-3 floor (0.096-class), expected ≈ 0.2–0.3 ≈ recall(τ) × 0.393.
- **P2.** Direction of the selection effect (Loc@1 on *triggered* windows vs the
  dense 0.393) is genuinely unknown: detected leaks skew large/easier (↑), but
  triggered windows include marginal-overlap cases (↓). Reported as found.
- **P3.** No-leak activation rate at τ=0.5 is the duty-cycle cost; expected
  ≤ ≈0.3 for the CNN, lower for Extra Trees. Reported as found, and the
  frontier (Loc@1 vs activation rate over τ) is the primary operating
  characteristic.
- **P4.** The budget curve (Loc@1 vs m at fixed τ=0.5) is expected to rise with
  m; its shape (steep-then-flat vs monotone) is reported as found.

## Analysis plan

`analysis_v2.py::activation_analysis()` → `stats.json` section `activation`,
table `tab_activation.tex`, Python figure `fig_activation.pdf` (two panels:
activation-operating frontier; budget curve), paired per-subseed comparisons vs
the always-3 (0.096) and always-9 (0.393) references with 95% t-intervals, run
audit extended by 30 DL runs + 6 classical fits.

## Framing discipline

The scheme is an application of the benchmark's asymmetry result, not a novel
architecture; claims calibrated per the house table (v1 claim-calibration file).
Duty-cycle statements are scoped to this benchmark's window distribution —
operational leak priors differ, and the paper will say so.
