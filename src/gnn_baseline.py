"""GNN baseline under the master benchmark protocol (in-distribution).

Trains the lightweight GCN (models.GNNLeakModel) on the same 6 placements x
4 densities as the master benchmark, with the same scenario-grouped splits,
per-run deterministic seeding, and metrics (incl. Loc@1/3/5). The GNN is
placement-agnostic by construction (unobserved nodes receive a learned
'missing' embedding), so this is the spatially-aware reference against which
the channel-based temporal encoders are compared.
"""
import csv
import hashlib
import os
import sys
import time

import numpy as np
import torch

import benchmark
import models

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
    adj, _ = models.get_net1_adjacency()

    rows = []
    for k in KS:
        for subseed in SUBSEEDS:
            rs = int(hashlib.sha1(f"gnn-{k}-{subseed}".encode()).hexdigest()[:8], 16)
            torch.manual_seed(rs)
            np.random.seed(rs)
            torch.cuda.manual_seed(rs)

            tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=subseed)

            def gather(s):
                return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

            tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
            sensors = np.sort(np.random.default_rng(1000 + subseed).choice(n_j, k, replace=False))

            model = models.build_gnn_model(adj).to(benchmark.DEVICE)
            model.set_sensors(sensors)
            tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
            t0 = time.time()
            model = benchmark.train(model, tr, va)
            wall = time.time() - t0
            m = benchmark.evaluate(model, te)
            row = {"arch": "gnn", "k": k, "subseed": subseed,
                   "sensors": sensors.tolist(), "n_params": model.param_count(),
                   "wall_min": round(wall / 60, 2), **m}
            rows.append(row)
            print(row)

    fn = "data/results/gnn_benchmark.csv"
    flds = ["arch", "k", "subseed", "sensors", "n_params", "wall_min",
            "bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
