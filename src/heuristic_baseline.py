"""Largest-pressure-deficit heuristic baseline (study iteration 2).

Training-free, placement-invariant by construction: each observed node is
scored by minus its mean training-standardized pressure over the window
(z-scoring with the TEST placement's train-scenario statistics); the
prediction is the argmax over observed nodes only (same k/9 observability
ceiling as the identity-adjacency GCN).

Design identical to the 15-trio sweep (cross_many_trios.py): trio t uses
placement seeds {20+3t, 21+3t, 22+3t}; per-k test subsets drawn exactly as
the models' (default_rng(1000+seed).choice); scenario split seed = test
placement seed; Experiment B (held-out test scenarios through the unseen
placement). Loc@1/Loc@3 over ALL leak test windows (unobserved true node =
miss). Output: data/results/pressure_deficit_heuristic.csv
"""
import csv
import os
import sys

import numpy as np

import benchmark

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

KS = [7, 5, 3]
N_TRIOS = 15


def main():
    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)

    rows = []
    for t in range(N_TRIOS):
        p_train = [20 + 3 * t, 21 + 3 * t]
        p_test = 22 + 3 * t
        tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=p_test)

        def gather(sids):
            return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

        tr_idx, te_idx = gather(tr_s), gather(te_s)
        yl_te = yl[te_idx]
        leak = np.where(yl_te >= 0)[0]
        rng = np.random.default_rng(0)

        for k in KS:
            sens_test = np.sort(np.random.default_rng(1000 + p_test).choice(n_j, k, replace=False))
            # training-scenario statistics for this placement (no labels used)
            mu, sd = benchmark.WindowDataset(X, yd, yl, sens_test, tr_idx).stats
            Z = ((X[te_idx][:, :, sens_test].astype(np.float32) - mu) / sd)

            # deficit score per window: minus mean standardized pressure per node
            score = -Z.mean(axis=1)                      # (n_win, k)
            order = np.argsort(-score, axis=1)
            shuf = rng.permuted(score, axis=1)
            order_shuf = np.argsort(-shuf, axis=1)

            # true node position within the sorted observed subset; -1 = unobserved
            pos_of = {int(n): i for i, n in enumerate(sens_test)}
            true_pos = np.array([pos_of.get(int(n), -1) for n in yl_te])

            ov = [len(set(sens_test.tolist()) &
                      set(np.sort(np.random.default_rng(1000 + s)
                                  .choice(n_j, k, replace=False)).tolist()))
                  for s in p_train]

            row = {"trio": t, "k": k, "sensor_overlap": ov,
                   "B_loc1": float(np.mean([true_pos[i] in order[i, :1] for i in leak])),
                   "B_loc3": float(np.mean([true_pos[i] in order[i, :3] for i in leak])),
                   "B_loc1_shuffled": float(np.mean([true_pos[i] in order_shuf[i, :1]
                                                     for i in leak])),
                   "n_leak": int(len(leak)),
                   "frac_leak_observed": float(np.mean(true_pos[leak] >= 0))}
            rows.append(row)
        print(f"trio {t:2d}: " + "  ".join(
            f"k{k} Loc@1={rows[-3 + KS.index(k)]['B_loc1']:.3f}" for k in KS))

    fn = "data/results/pressure_deficit_heuristic.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
