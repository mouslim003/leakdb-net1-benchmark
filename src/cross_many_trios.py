"""Many-trios placement-transfer sweep (15 placement trios).

The v2 claim "node-indexed models restore transfer" rested on n=3 placement
trios with wide intervals. This script runs the deployment-transfer
experiment (Experiment B) over 15 independently drawn placement trios:

  Trio t: placement seeds {20+3t, 21+3t, 22+3t} (fresh, disjoint from the
  master subseeds 0-5); train on placements (a, b); test on placement c.
  Scenario split seed = test placement seed (protocol-consistent with
  cross_sensor.py).

Models per trio and density k in {7, 5, 3}:
  cnn        channel-based reference (expected: chance)
  gnn_real   GCN over the Net1 junction adjacency
  gnn_ident  GCN with identity adjacency (no propagation)
  gnn_rand   GCN with a per-trio degree-preserving randomized adjacency
             (rewire seed 3000+t; transfer is thus tested over 15 distinct
             rewirings)

4 models x 3 densities x 15 trios = 180 training runs. Every run receives
an explicit deterministic per-run seed. The same trio/split/placements are
shared across models, so PAIRED per-trio differences (GCN variant - CNN)
are valid and are reported with t-CIs in the analysis.
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
from random_graph_gcn import randomize_adjacency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

N_TRIOS = 15
KS = [7, 5, 3]


def run_trio(model_name, k, t, real_adj, X, yd, yl, meta, nums, ranges):
    tag = f"manytrios-{model_name}-{k}-t{t}"
    rs = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16)
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)

    p_train = [20 + 3 * t, 21 + 3 * t]
    p_test = 22 + 3 * t
    n_j = X.shape[2]
    tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=p_test)

    def gather(sids):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)

    sens_train = [np.sort(np.random.default_rng(1000 + s).choice(n_j, k, replace=False))
                  for s in p_train]
    sens_test = np.sort(np.random.default_rng(1000 + p_test).choice(n_j, k, replace=False))

    if model_name == "cnn":
        model = models.build_model("cnn", in_ch=k, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
    else:
        if model_name == "gnn_real":
            adj = real_adj
        elif model_name == "gnn_ident":
            adj = np.eye(real_adj.shape[0], dtype=np.float32)
        elif model_name == "gnn_rand":
            adj = randomize_adjacency(real_adj, seed=3000 + t)
        else:
            raise ValueError(model_name)
        model = models.build_gnn_model(adj).to(benchmark.DEVICE)
        model.set_sensors(sens_train[0])

    tr_parts, va_parts = [], []
    for s in sens_train:
        tr_parts.append(benchmark.WindowDataset(X, yd, yl, s, tr_idx))
        va_parts.append(benchmark.WindowDataset(X, yd, yl, s, va_idx, *tr_parts[-1].stats))
    tr_ds = torch.utils.data.ConcatDataset(tr_parts)
    va_ds = torch.utils.data.ConcatDataset(va_parts)
    model = benchmark.train(model, tr_ds, va_ds)

    if model_name != "cnn":
        model.set_sensors(sens_test)
    stats_src = benchmark.WindowDataset(X, yd, yl, sens_test, tr_idx)
    mu, sd = stats_src.stats
    te_B = benchmark.WindowDataset(X, yd, yl, sens_test, te_idx, mu, sd)
    mB = benchmark.evaluate(model, te_B)

    overlap = [len(set(s.tolist()) & set(sens_test.tolist())) for s in sens_train]
    return {"model": model_name, "k": k, "trio": t,
            "p_train": p_train, "p_test": p_test,
            "sensor_overlap": overlap,
            "B_bal_acc": mB["bal_acc"], "B_f1": mB["f1"], "B_auc": mB["auc"],
            "B_loc_acc": mB["loc_acc"], "B_loc3": mB["loc3"], "B_loc5": mB["loc5"]}


def main():
    X, yd, yl, meta = benchmark.load_data()
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    real_adj, _ = models.get_net1_adjacency()

    rows = []
    for t in range(N_TRIOS):
        for k in KS:
            for model_name in ["cnn", "gnn_real", "gnn_ident", "gnn_rand"]:
                try:
                    row = run_trio(model_name, k, t, real_adj, X, yd, yl, meta, nums, ranges)
                    rows.append(row)
                    print(f"trio {t:2d} k={k} {model_name:9s}: "
                          f"B_loc={row['B_loc_acc']:.3f} B_loc3={row['B_loc3']:.3f} "
                          f"B_auc={row['B_auc']:.3f}")
                except Exception as e:
                    print(f"ERROR trio={t} k={k} {model_name}: {e}")
                    import traceback
                    traceback.print_exc()

    fn = "data/results/cross_many_trios.csv"
    flds = ["model", "k", "trio", "p_train", "p_test", "sensor_overlap",
            "B_bal_acc", "B_f1", "B_auc", "B_loc_acc", "B_loc3", "B_loc5"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in rows:
            w.writerow({kk: (json.dumps(r[kk]) if isinstance(r[kk], (list, tuple)) else r[kk])
                        for kk in flds})
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
