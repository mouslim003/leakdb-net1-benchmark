"""Sensor-identity (set-encoder) models under the MASTER benchmark protocol.

In-distribution counterpart to the cross-sensor identity experiment: trains
models.SetLeakModel (node-identity-aware, permutation-invariant input encoder
+ temporal encoder) on the same 6 placements x 4 densities with the same
splits, per-run seeds, and metrics as the master benchmark. Establishes
whether the identity encoder pays an in-distribution accuracy price for its
placement invariance.
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
ARCHS = ["cnn", "lstm", "gru", "transformer"]


def main():
    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)

    rows = []
    for arch in ARCHS:
        for k in KS:
            for subseed in SUBSEEDS:
                rs = int(hashlib.sha1(f"setid-{arch}-{k}-{subseed}".encode()).hexdigest()[:8], 16)
                torch.manual_seed(rs)
                np.random.seed(rs)
                torch.cuda.manual_seed(rs)

                tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=subseed)

                def gather(s):
                    return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

                tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
                sensors = np.sort(np.random.default_rng(1000 + subseed).choice(n_j, k, replace=False))

                model = models.build_set_model(arch).to(benchmark.DEVICE)
                model.set_sensors(sensors)
                tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
                t0 = time.time()
                model = benchmark.train(model, tr, va)
                wall = time.time() - t0
                m = benchmark.evaluate(model, te)
                row = {"arch": arch, "k": k, "subseed": subseed,
                       "sensors": sensors.tolist(), "n_params": model.param_count(),
                       "wall_min": round(wall / 60, 2), **m}
                rows.append(row)
                print(row)

    fn = "data/results/set_identity_benchmark.csv"
    flds = ["arch", "k", "subseed", "sensors", "n_params", "wall_min",
            "bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
