"""POST-HOC in-distribution evaluation of the largest-pressure-deficit heuristic.

Not part of the locked protocol (experiments/iteration2-analyses/protocol.md
locks Experiment B only). This descriptive addition evaluates the SAME rule
on the MASTER benchmark's in-distribution splits (subseed s: split seed = s,
subset = default_rng(1000+s), k in {9,7,5,3}), so the learned models'
in-distribution advantage can be stated against the heuristic's in-dist
level. Labeled post hoc wherever used. No training; deterministic.

Output: data/results/pressure_deficit_heuristic_indist.csv
"""
import csv
import os
import sys

import numpy as np

import benchmark

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

KS = [9, 7, 5, 3]
SUBSEEDS = [0, 1, 2, 3, 4, 5]


def main():
    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)

    rows = []
    for s in SUBSEEDS:
        tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=s)

        def gather(sids):
            return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

        tr_idx, te_idx = gather(tr_s), gather(te_s)
        yl_te = yl[te_idx]
        leak = np.where(yl_te >= 0)[0]
        for k in KS:
            sens = np.sort(np.random.default_rng(1000 + s).choice(n_j, k, replace=False))
            mu, sd = benchmark.WindowDataset(X, yd, yl, sens, tr_idx).stats
            Z = ((X[te_idx][:, :, sens].astype(np.float32) - mu) / sd)
            order = np.argsort(-(-Z.mean(axis=1)), axis=1)
            pos_of = {int(n): i for i, n in enumerate(sens)}
            true_pos = np.array([pos_of.get(int(n), -1) for n in yl_te])
            rows.append({"subseed": s, "k": k,
                         "loc1": float(np.mean([true_pos[i] in order[i, :1] for i in leak])),
                         "loc3": float(np.mean([true_pos[i] in order[i, :3] for i in leak])),
                         "n_leak": int(len(leak))})

    fn = "data/results/pressure_deficit_heuristic_indist.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for k in KS:
        v = [r["loc1"] for r in rows if r["k"] == k]
        print(f"in-dist heuristic k={k}: Loc@1 {np.mean(v):.3f} +- {np.std(v, ddof=1):.3f}")
    print(f"Saved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
