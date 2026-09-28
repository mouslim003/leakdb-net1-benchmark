"""Leak-size sensitivity: stratify test performance by leak size class.

Motivation: which architecture is most robust to SMALL leaks? A leak with a tiny
emitter coefficient produces a weak pressure signature that is easy to miss, and
robustness to small leaks is the practical requirement for a low-instrumentation
network (early/incipient leaks).

Protocol:
- Uses the SAME data, scenario-grouped split, sensor selection (seed=1000+subseed),
  and training procedure as benchmark.py. Here we fix subseed=0 and k in {9,7,5,3}.
- Each leak scenario in window_meta.json has a size_k in {0,1,2} = {small,medium,large}
  (LEAK_SIZES = [0.02, 0.06, 0.15] emitter coefficients).
- Test windows are mapped back to their source scenario to recover leak size.
- Metrics are computed per (arch, k, size_class): detection balanced-accuracy, F1,
  AUC computed over windows whose ground-truth class is that size's leak (positive)
  vs normal (negative); localization top-1 acc over that size's leak windows only.
- Small-leak robustness score = (small-leak F1 + small-leak AUC)/2 vs the large-leak
  value -> a drop measure.
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import benchmark
import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SEED = 42
KS = [9, 7, 5, 3]
SUBSEEDS = [0, 1, 2, 3, 4, 5]   # all six placement/split replicates (was: 0 only)
SIZE_NAMES = {0: "small", 1: "medium", 2: "large"}


def scenario_size_map(meta):
    """window-index -> leak size class (or None for normal scenarios)."""
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    w2size = {}
    for i, sched in enumerate(meta["schedules"]):
        for widx in range(ranges[i][0], ranges[i][1]):
            w2size[widx] = sched.get("size_k") if sched.get("type") == "leak" else None
    return w2size


def evaluate_stratified(model, te, w2size, te_idx):
    model.eval()
    loader = torch.utils.data.DataLoader(te, batch_size=benchmark.BATCH)
    ds, ls, ys, ms = [], [], [], []
    with torch.no_grad():
        for xb, ydb, ylb, mb in loader:
            xb = xb.to(benchmark.DEVICE)
            dlogit, loc = model(xb)
            ds.append(torch.sigmoid(dlogit).cpu())
            ls.append(loc.argmax(1).cpu())
            ys.append(ydb.cpu())
            ms.append(mb.cpu())
    ds = torch.cat(ds).numpy()
    ls = torch.cat(ls).numpy()
    ys = torch.cat(ys).numpy()
    ms = torch.cat(ms).numpy()
    true_loc = yl_of(te, ms)

    results = {}
    # window indexes in te order
    leak_mask = ms > 0
    leak_idx_all = np.where(leak_mask)[0]      # global positions of leak windows
    true_loc_leak = true_loc[:len(leak_idx_all)]  # in leak-window order (collected via mb>0)
    # include NORMAL test windows as negatives for detection curve
    normal_mask = ys == 0
    widx = np.array(te_idx, dtype=int)
    for size_k, sname in SIZE_NAMES.items():
        # positives: leak windows whose source scenario has this size
        pos = np.array([w2size.get(int(w)) == size_k for w in widx])
        pos_all = pos | normal_mask           # detection eval: this size's leaks vs normals
        if pos.sum() == 0:
            results[sname] = {"bal_acc": float("nan"), "f1": float("nan"),
                              "auc": float("nan"), "loc_acc": float("nan"),
                              "n_windows": int(pos.sum())}
            continue
        y_sub = ys[pos_all]
        d_sub = ds[pos_all]
        pred = (d_sub >= 0.5).astype(int)
        bal = balanced_accuracy_score(y_sub, pred)
        f1 = f1_score(y_sub, pred)
        try:
            auc = roc_auc_score(y_sub, d_sub)
        except ValueError:
            auc = float("nan")
        # localization on this size's leak windows only (align over leak windows)
        sizemask = pos[leak_mask]
        loc_sub = ls[leak_idx_all[sizemask]]
        tl_sub = true_loc_leak[sizemask]
        loc_acc = float(np.mean(loc_sub == tl_sub)) if len(tl_sub) else float("nan")
        results[sname] = {"bal_acc": bal, "f1": f1, "auc": auc, "loc_acc": loc_acc,
                          "n_windows": int(pos.sum())}
    return results


def yl_of(te, mask):
    true_loc = []
    for xb, ydb, ylb, mb in te:
        ylb_np = ylb.numpy(); mb_np = mb.numpy()
        true_loc.extend(ylb_np[mb_np > 0])
    return np.array(true_loc)


def run(arch, k, subseed):
    # deterministic per-run seeding for independence & reproducibility
    import hashlib
    run_seed = int(hashlib.sha1(f"leaksize-{arch}-{k}-{subseed}".encode()).hexdigest()[:8], 16)
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    torch.cuda.manual_seed(run_seed)

    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    n_scen = len(meta["schedules"])
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    tr_s, va_s, te_s = benchmark.grouped_split(n_scen, seed=subseed)

    def gather(sids):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
    w2size = scenario_size_map(meta)

    rng = np.random.default_rng(1000 + subseed)
    sensors = np.sort(rng.choice(n_j, k, replace=False))

    model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
    tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
    model = benchmark.train(model, tr, va)
    per_size = evaluate_stratified(model, te, w2size, te_idx)
    return per_size, sensors


def main():
    os.makedirs("data/results", exist_ok=True)
    rows = []
    for arch in ["cnn", "lstm", "gru", "transformer"]:
        for k in KS:
            for subseed in SUBSEEDS:
                per_size, sensors = run(arch, k, subseed)
                for sname, m in per_size.items():
                    rows.append({"arch": arch, "k": k, "subseed": subseed,
                                 "size": sname, "sensors": sensors.tolist(), **m})
                print(arch, k, subseed,
                      {s: round(m["f1"], 3) for s, m in per_size.items()})

    flds = ["arch", "k", "subseed", "size", "sensors", "n_windows",
            "bal_acc", "f1", "auc", "loc_acc"]
    with open("data/results/leak_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved {len(rows)} rows -> data/results/leak_size.csv")


if __name__ == "__main__":
    main()
