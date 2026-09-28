"""Cross-sensor generalization v2: placement transfer with full diagnostics.

Protocol notes:
- Every run receives an explicit deterministic per-run seed (fixes the
  "seeds once at import" inconsistency of the first sweep).
- TWO distinct transfer experiments, evaluated on the SAME trained model:
  * Experiment A (scenario-controlled placement shift): evaluate on the
    TRAINING scenarios, but through the UNSEEN test placement. The model has
    seen these hydraulic scenarios through placements {p0, p1}; any
    localization failure is attributable to the placement change alone,
    not to scenario novelty.
  * Experiment B (deployment transfer): evaluate on held-out TEST scenarios
    through the UNSEEN test placement. Neither the scenarios nor the placement
    were seen during training.
- Full predictions (true node, predicted node, detection prob) are saved for
  confusion-matrix and predicted-node-distribution analysis, plus Loc@1/3/5.
- Three independent placement trios (train placements -> test placement) for
  replication; sensor-overlap statistics between train and test placements
  are recorded.
- --model {std, set} selects channel-based encoders or the sensor-identity
  set encoder (models.SetLeakModel) to test the identity hypothesis.
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

KS = [7, 5, 3]
TRIOS = [((0, 1), 2), ((3, 4), 5), ((2, 5), 0)]   # (train placements, test placement)
ARCHS = ["cnn", "lstm", "gru", "transformer"]


def run_seed(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def run(arch, k, train_placements, test_placement, model_kind="std"):
    tag = f"cross2-{model_kind}-{arch}-{k}-{train_placements}-{test_placement}"
    rs = run_seed(tag)
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)

    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    # scenario split derived from the TEST placement subseed (held fixed per trio)
    tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=test_placement)

    def gather(sids):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)

    sens_train = [np.sort(np.random.default_rng(1000 + s).choice(n_j, k, replace=False))
                  for s in train_placements]
    sens_test = np.sort(np.random.default_rng(1000 + test_placement).choice(n_j, k, replace=False))

    if model_kind == "set":
        model = models.build_set_model(arch).to(benchmark.DEVICE)
        model.set_sensors(sens_train[0])
    elif model_kind == "gnn":
        adj, _ = models.get_net1_adjacency()
        model = models.build_gnn_model(adj).to(benchmark.DEVICE)
        model.set_sensors(sens_train[0])
    else:
        model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)

    # per-placement standardized datasets (train stats of the SAME placement)
    tr_parts, va_parts = [], []
    for s in sens_train:
        tr_parts.append(benchmark.WindowDataset(X, yd, yl, s, tr_idx))
        va_parts.append(benchmark.WindowDataset(X, yd, yl, s, va_idx, *tr_parts[-1].stats))
    tr_ds = torch.utils.data.ConcatDataset(tr_parts)
    va_ds = torch.utils.data.ConcatDataset(va_parts)
    model = benchmark.train(model, tr_ds, va_ds)

    out = {"arch": arch, "k": k, "model_kind": model_kind,
           "train_placements": list(train_placements),
           "test_placement": test_placement,
           "train_sensors": [s.tolist() for s in sens_train],
           "test_sensors": sens_test.tolist()}
    overlap = [len(set(s.tolist()) & set(sens_test.tolist())) for s in sens_train]
    out["sensor_overlap"] = overlap

    if model_kind in ("set", "gnn"):
        model.set_sensors(sens_test)
    # test datasets at the unseen placement, standardized with that
    # placement's TRAIN-scenario statistics (no test leakage)
    stats_src = benchmark.WindowDataset(X, yd, yl, sens_test, tr_idx)
    mu, sd = stats_src.stats
    # Experiment A: TRAIN scenarios through the unseen placement.
    te_A = benchmark.WindowDataset(X, yd, yl, sens_test, tr_idx, mu, sd)
    mA = benchmark.evaluate(model, te_A, return_preds=True)
    # Experiment B: TEST scenarios (unseen) through the unseen placement.
    te_B = benchmark.WindowDataset(X, yd, yl, sens_test, te_idx, mu, sd)
    mB = benchmark.evaluate(model, te_B, return_preds=True)

    for exp, m in (("A", mA), ("B", mB)):
        preds = m.pop("_preds")
        out[f"{exp}_bal_acc"] = m["bal_acc"]
        out[f"{exp}_f1"] = m["f1"]
        out[f"{exp}_auc"] = m["auc"]
        out[f"{exp}_loc_acc"] = m["loc_acc"]
        out[f"{exp}_loc3"] = m["loc3"]
        out[f"{exp}_loc5"] = m["loc5"]
        out[f"{exp}_n_leak"] = int(preds["leak_mask"].sum())
        # predicted-node distribution + per-(true,pred) confusion on leak windows
        pred_nodes = preds["pred_loc"][preds["leak_mask"]] - 1   # 0..8
        true_nodes = preds["true_loc_node"]
        out[f"{exp}_pred_node_hist"] = np.bincount(
            np.clip(pred_nodes, 0, 8), minlength=9).tolist()
        cm = np.zeros((9, 9), dtype=int)
        for t, p in zip(true_nodes, np.clip(pred_nodes, 0, 8)):
            cm[t, p] += 1
        out[f"{exp}_confusion"] = cm.tolist()
        out[f"{exp}_true_nodes"] = np.unique(true_nodes).tolist()
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["std", "set", "gnn"], default="std")
    args = parser.parse_args()

    archs = ARCHS if args.model in ("std", "set") else ["gnn"]

    os.makedirs("data/results", exist_ok=True)
    rows = []
    for arch in archs:
        for k in KS:
            for train_p, test_p in TRIOS:
                try:
                    row = run(arch, k, train_p, test_p, model_kind=args.model)
                    rows.append(row)
                    print(f"{args.model} {arch} k={k} tr={train_p} te={test_p}: "
                          f"A_loc={row['A_loc_acc']:.3f} B_loc={row['B_loc_acc']:.3f} "
                          f"B_loc3={row['B_loc3']:.3f}")
                except Exception as e:
                    print(f"ERROR {arch} k={k} {train_p}->{test_p}: {e}")
                    import traceback
                    traceback.print_exc()

    fn = f"data/results/cross_sensor_v2_{args.model}.csv"
    flds = ["arch", "k", "model_kind", "train_placements", "test_placement",
            "train_sensors", "test_sensors", "sensor_overlap",
            "A_bal_acc", "A_f1", "A_auc", "A_loc_acc", "A_loc3", "A_loc5",
            "A_n_leak", "A_pred_node_hist", "A_confusion", "A_true_nodes",
            "B_bal_acc", "B_f1", "B_auc", "B_loc_acc", "B_loc3", "B_loc5",
            "B_n_leak", "B_pred_node_hist", "B_confusion", "B_true_nodes"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in rows:
            w.writerow({kk: json.dumps(r[kk]) if isinstance(r[kk], (list, tuple)) else r[kk]
                        for kk in flds})
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
