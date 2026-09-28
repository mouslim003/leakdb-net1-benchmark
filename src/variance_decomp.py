"""Variance decomposition: placement vs split vs training stochasticity.

Motivation: the master benchmark varies placement and split together
(subseed determines both), so run-to-run variance cannot be attributed. This
script runs three controlled replication families (all with per-run seeds):

  V-place : split FIXED at seed 0; placement seeds {0..5}  -> Var_placement
  V-split : placement FIXED at seed 0; split seeds {0..5}  -> Var_split
  V-train : placement AND split FIXED at 0; 6 different torch seeds -> Var_train

Reported decomposition: sigma^2_place ~ Var(V-place) - sigma^2_train, etc.
4 archs x 4 densities x 6 replicates per family = 288 runs.
"""
import csv
import hashlib
import os
import sys

import numpy as np
import torch

import benchmark
import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

KS = [9, 7, 5, 3]
ARCHS = ["cnn", "lstm", "gru", "transformer"]


def run_one(arch, k, place_seed, split_seed, train_seed):
    rs = int(hashlib.sha1(f"vardep-{arch}-{k}-p{place_seed}-s{split_seed}-t{train_seed}"
                          .encode()).hexdigest()[:8], 16)
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)

    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=split_seed)

    def gather(s):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
    sensors = np.sort(np.random.default_rng(1000 + place_seed).choice(n_j, k, replace=False))

    model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
    tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
    model = benchmark.train(model, tr, va)
    m = benchmark.evaluate(model, te)
    return m


def main():
    rows = []
    families = {
        "place": [(p, 0, 0) for p in range(6)],   # placement varies
        "split": [(0, s, 0) for s in range(6)],   # split varies
        "train": [(0, 0, t) for t in range(6)],   # training seed varies
    }
    for arch in ARCHS:
        for k in KS:
            for fam, cells in families.items():
                for place_seed, split_seed, train_seed in cells:
                    m = run_one(arch, k, place_seed, split_seed, train_seed)
                    row = {"arch": arch, "k": k, "family": fam,
                           "place_seed": place_seed, "split_seed": split_seed,
                           "train_seed": train_seed, **m}
                    rows.append(row)
                    print(arch, k, fam, f"auc={m['auc']:.3f} loc={m['loc_acc']:.3f}")

    fn = "data/results/variance_decomp.csv"
    flds = ["arch", "k", "family", "place_seed", "split_seed", "train_seed",
            "bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
