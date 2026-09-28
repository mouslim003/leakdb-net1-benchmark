# Bug Report: Silent Data-Generation Bugs Found During the Study

This file documents the two pipeline bugs found during the study, per the
reproducibility practice recommended for this benchmark paper. The paper's
Discussion (section "Two silent bugs, and what they cost") summarizes both; this
file is the exact record.

## Bug 1 — RNG seeding leak

- **Component:** first implementation of the training sweep (`src/benchmark.py`).
- **Description:** a single `torch.manual_seed(SEED)` executed at module
  import time, before every run, so all supposedly independent runs of a
  sweep trained identical models (identical initialization, identical data
  order). The CSV contained four duplicated rows wearing different labels.
- **Detection:** the first cross-condition comparison returned *identical*
  metrics (to all decimals) for runs that should have differed only in seed;
  a direct probe of the RNG state confirmed it was reset at import.
- **Fix:** every training run now derives its seed from a SHA-1 hash of its
  run identity (experiment family, architecture, density, replicate),
  applied to `torch`, `numpy`, and the CUDA RNG at run start
  (`src/benchmark.py::train_eval_subsets` and equivalently in every
  experiment script).
- **Affected results:** the entire first sweep. All affected results were
  discarded; no number from an affected run appears in the paper.
- **Verification:** the 96-run master benchmark was re-run after the fix and
  confirmed to reproduce the released CSV with row-level equality
  (`data/results/benchmark.csv`).

## Bug 2 — Leak-timing / label mismatch (v2 dataset)

- **Component:** dataset generator, v2 (`data_gen.py` before the v3 fix).
- **Description:** the leak emitter was hydraulically active for the
  *entire* 24-hour simulation while the window labels marked only an 8-hour
  window as "leak". Every labeled no-leak window of a leak scenario
  hydraulically contained the leak, and the "detection" task partially
  degenerated into time-of-day discrimination (the demand pattern is
  clock-aligned).
- **Detection:** found while verifying the *physical* size of the modelled
  leaks: a counterfactual simulation (identical demands with
  and without the leak) showed the leak node's pressure depressed outside
  the labeled leak window (`src/leak_physics.py` methodology).
- **Fix:** v3 dataset generator uses WNTR's timed leak mechanism
  (`node.add_leak` with `start_time`/`end_time`), the same mechanism as the
  official LeakDB generator, with per-scenario random start hours (integer
  hours uniform in [2, 14], fixed 8-hour duration) so that time-of-day
  carries no label information.
- **Affected results:** the entire v2 dataset and every number computed on
  it (master benchmark tables of the previous draft). All affected results
  were discarded and regenerated on the v3 dataset.
- **Quantified impact** (Table "What the label bug cost" in the paper,
  regenerated via `src/analysis_v2.py`):

  | Model | v2 AUC k=9 | v2 AUC k=3 | v3 AUC k=9 | v3 AUC k=3 |
  |-------|-----------|-----------|-----------|-----------|
  | CNN | 0.871 | 0.867 | 0.885 | 0.830 |
  | LSTM | 0.871 | 0.869 | 0.746 | 0.673 |
  | GRU | 0.870 | 0.870 | 0.798 | 0.745 |
  | Transformer | 0.870 | 0.867 | 0.869 | 0.821 |

  Under the buggy labels all architectures showed a *flat* AUC band of
  0.867–0.874 across densities — a "sparsity-robust detection" story that
  the corrected labels refute (architecture-dependent degradation).

- **Preserved artifacts:** the v2 (buggy) dataset is preserved as
  `data/X_v2_buggy.npy`, `data/y_det_v2_buggy.npy`,
  `data/y_loc_v2_buggy.npy`, `data/window_meta_v2_buggy.json` for exactness.

## Confirmation

All results reported in the paper were regenerated after both fixes; no
number from an affected run appears in any table or figure. The released
`run_all.ps1` / `run_all.sh` regenerate every number in the paper from
scratch, deterministically (see the reproducibility appendix of the paper
for seeds, environment, and checksums).
