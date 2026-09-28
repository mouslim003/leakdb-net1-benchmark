"""Window-labeling sensitivity: minimum leak-duration thresholds.

Motivation: the any-overlap rule labels a window "leak" even when
only a few of its 48 timesteps are inside the leak interval (e.g., 5 minutes
of leak inside 4 hours of normal operation). This script rebuilds the labels
with a minimum active-duration threshold m (in timesteps of 5 min) and reruns
the master benchmark protocol per labeling variant:

  m = 1  : any overlap (the master labeling; identical rows exist in
           benchmark.csv, but we rerun through the same code path)
  m = 6  : >= 30 min of leak inside the window
  m = 12 : >= 60 min
  m = 24 : >= 120 min (majority of the 4h window)
  m = 48 : fully leak-active window

Labels are recomputed geometrically from the leak-active interval stored in
window_meta.json (hours 8-16). The m=1 recomputation is verified against the
shipped y_det.npy. 4 archs x 4 k x 3 subseeds per variant.
"""
import csv
import hashlib
import json
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
SUBSEEDS = [0, 1, 2]
ARCHS = ["cnn", "lstm", "gru", "transformer"]
MS = [1, 6, 12, 24, 48]
WINDOW = 48
STRIDE = 8


def relabel(meta, m):
    """Return (yd, yl) with leak label iff #active steps in window >= m."""
    nums = benchmark.compute_scenario_bounds(meta)
    full_len = 24 * 3600 // 300
    yd, yl = [], []
    for sched in meta["schedules"]:
        starts = range(0, full_len - WINDOW + 1, STRIDE)
        if sched.get("type") != "leak":
            for _ in starts:
                yd.append(0)
                yl.append(-1)
            continue
        a = int(sched["active_start_h"] * 3600 // 300)
        b = int(sched["active_end_h"] * 3600 // 300)
        node = sched["leak_node"]
        j_idx = {j: i for i, j in enumerate(meta["junction_ids"])}
        loc = j_idx[node]
        for s in starts:
            ov = max(0, min(s + WINDOW, b) - max(s, a))
            leak = 1 if ov >= m else 0
            yd.append(leak)
            yl.append(loc if leak else -1)
    return np.array(yd, np.int8), np.array(yl, np.int8)


def main():
    X, yd0, yl0, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)

    # sanity: m=1 recomputation must match shipped labels
    yd1, yl1 = relabel(meta, 1)
    assert (yd1 == yd0).all(), "m=1 relabeling mismatch with shipped y_det"
    assert (yl1 == yl0).all(), "m=1 relabeling mismatch with shipped y_loc"
    print("m=1 relabeling verified against shipped labels.")

    rows = []
    for m in MS:
        yd, yl = (yd0, yl0) if m == 1 else relabel(meta, m)
        n_leak = int((yd == 1).sum())
        print(f"\n=== m={m} ({m*5} min min-duration): {n_leak}/{len(yd)} leak windows ===")
        for arch in ARCHS:
            for k in KS:
                for subseed in SUBSEEDS:
                    rs = int(hashlib.sha1(f"winlab-{m}-{arch}-{k}-{subseed}".encode())
                             .hexdigest()[:8], 16)
                    torch.manual_seed(rs)
                    np.random.seed(rs)
                    torch.cuda.manual_seed(rs)

                    tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=subseed)

                    def gather(s):
                        return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

                    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
                    sensors = np.sort(np.random.default_rng(1000 + subseed)
                                      .choice(n_j, k, replace=False))
                    model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN) \
                        .to(benchmark.DEVICE)
                    tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
                    model = benchmark.train(model, tr, va)
                    mm = benchmark.evaluate(model, te)
                    row = {"m": m, "arch": arch, "k": k, "subseed": subseed,
                           "n_leak_windows": n_leak, **mm}
                    rows.append(row)
                    print(f"m={m} {arch} k={k} s={subseed}: auc={mm['auc']:.3f} "
                          f"loc={mm['loc_acc']:.3f}")

    fn = "data/results/window_labeling.csv"
    flds = ["m", "arch", "k", "subseed", "n_leak_windows",
            "bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
