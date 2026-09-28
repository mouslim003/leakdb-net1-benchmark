"""Benchmark: train & evaluate lightweight leak-detection models across sensor densities.

Protocol (locked BEFORE results, see git):
- Data: X (n_windows, W, 9) pressure windows; y_det (binary); y_loc (9 leak nodes + none).
- Split is BY SCENARIO (grouped), not by random window, to avoid leakage between
  overlapping windows of the same simulation trace.
- Sensor subsets: keep k of the 9 junction channels, for k in {9,7,5,3}, repeated over
  several random subset seeds. Train & test on the SAME subset (sensor-specific model).
- Metrics: detection balanced accuracy, detection F1 (positive=leak), AUC,
  localization top-1 accuracy computed on leak-present windows only.
- For each (arch, k, seed) we train one model with both heads (joint objective).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # anchor all relative data/src paths to the repo root
sys.path.insert(0, os.path.join(ROOT, "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = 64
ARCHS = ["cnn", "lstm", "gru", "transformer"]
SENSOR_KS = [9, 7, 5, 3]
SUBSEED = [0, 1, 2]          # sensor-subset random seeds -> 3 placements per (k)
EPOCHS = 60
BATCH = 128
LR = 2e-3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def scenario_id_per_window(meta, n_windows):
    """Map each window index to its scenario index."""
    sc = []
    for i, s in enumerate(meta["schedules"]):
        # windows from scenario i: this scenario contributes a contiguous block,
        # but extraction is uniform so we reconstruct lengths by re-extracting logic
        # We stored schedules in generation order = concatenation order.
        pass
    return None


def load_data():
    X = np.load("data/X.npy")
    yd = np.load("data/y_det.npy")
    yl = np.load("data/y_loc.npy")
    with open("data/window_meta.json") as f:
        meta = json.load(f)
    return X, yd, yl, meta


def compute_scenario_bounds(meta, T=48, stride=8):
    """Recompute how many windows each scenario produces (same logic as data_gen)."""
    full_len = (24) * 3600 // 300  # recorded length in steps
    n_scen = len(meta["schedules"])
    nums = []
    for _ in range(n_scen):
        starts = range(0, full_len - T + 1, stride)
        nums.append(len(list(starts)))
    return nums


def grouped_split(n_scen, tr_frac=0.7, val_frac=0.15, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(n_scen)
    n_tr = int(n_scen * tr_frac)
    n_va = int(n_scen * val_frac)
    return set(idx[:n_tr]), set(idx[n_tr:n_tr + n_va]), set(idx[n_tr + n_va:])


def window_index_ranges(nums):
    """Return list of (start, end) window indices per scenario."""
    ranges, acc = [], 0
    for n in nums:
        ranges.append((acc, acc + n))
        acc += n
    return ranges


class WindowDataset(torch.utils.data.Dataset):
    """Per-channel standardized window dataset.

    If (mu, sd) are not given they are computed from THIS dataset's windows
    (callers pass the training dataset's stats to val/test to avoid leakage).
    """

    def __init__(self, X, yd, yl, sensor_idx, win_idx, mu=None, sd=None):
        Xi = X[win_idx][:, :, sensor_idx].astype(np.float32)
        if mu is None:
            mu = Xi.mean(axis=(0, 1))
            sd = Xi.std(axis=(0, 1)) + 1e-8
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.asarray(sd, dtype=np.float32)
        self.X = torch.tensor((Xi - self.mu) / self.sd, dtype=torch.float32)
        self.yd = torch.tensor(yd[win_idx], dtype=torch.long)
        self.yl = torch.tensor(yl[win_idx] + 1, dtype=torch.long)  # shift -1->0
        self.mask = torch.tensor((yl[win_idx] >= 0), dtype=torch.float32)

    @property
    def stats(self):
        return self.mu, self.sd

    def __len__(self):
        return len(self.yd)

    def __getitem__(self, i):
        return self.X[i], self.yd[i], self.yl[i], self.mask[i]


def make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx):
    """Train/val/test datasets standardized with TRAIN-split statistics."""
    tr = WindowDataset(X, yd, yl, sensors, tr_idx)
    va = WindowDataset(X, yd, yl, sensors, va_idx, *tr.stats)
    te = WindowDataset(X, yd, yl, sensors, te_idx, *tr.stats)
    return tr, va, te


def _labels_of(ds, attr):
    if isinstance(ds, torch.utils.data.ConcatDataset):
        return np.concatenate([np.array(getattr(d, attr)) for d in ds.datasets])
    return np.array(getattr(ds, attr))


def train(model, tr, va, epochs=EPOCHS, bs=BATCH, lr=LR):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # class-weighted losses: the dataset is ~78% no-leak windows; without
    # weighting the harder (correctly labeled) task collapses to majority class
    ytr = _labels_of(tr, "yd")
    n_pos = max(float((ytr == 1).sum()), 1.0)
    n_neg = max(float((ytr == 0).sum()), 1.0)
    det_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos).to(DEVICE))
    yl_all = _labels_of(tr, "yl")
    counts = np.bincount(yl_all, minlength=10).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (10.0 * counts)
    loc_loss = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(DEVICE))
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va, batch_size=bs)
    best = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, ydb, ylb, mb in tr_loader:
            xb, ydb, ylb, mb = xb.to(DEVICE), ydb.float().to(DEVICE), ylb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            dlogit, loc = model(xb)
            loss = det_loss(dlogit, ydb) + loc_loss(loc, ylb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            ds, ls, ys, ms = [], [], [], []
            for xb, ydb, ylb, mb in va_loader:
                xb, ydb = xb.to(DEVICE), ydb.float().to(DEVICE)
                dlogit, loc = model(xb)
                ds.append(torch.sigmoid(dlogit).cpu())
                ls.append(loc.argmax(1).cpu())
                ys.append(ydb.cpu())
                ms.append(mb.cpu())
            ds = torch.cat(ds).numpy()
            ls = torch.cat(ls).numpy()
            ys = torch.cat(ys).numpy()
            ms = torch.cat(ms).numpy()
            det_pred = (ds >= 0.5).astype(int)
            bal = balanced_accuracy_score(ys, det_pred)
        if bal > best:
            best = bal
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def evaluate(model, te, return_preds=False):
    """Detection + localization metrics on a test WindowDataset.

    Localization metrics (computed on leak windows only, denominator = ALL
    leak windows in the test split):
      loc_acc (Loc@1): top-1 node hit-rate.
      loc3 / loc5 (Loc@3 / Loc@5): true node within the k highest-scoring
      node classes. The no-leak class (index 0 after the +1 shift) is excluded
      from the node ranking: localization ranks candidate *nodes* only.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(te, batch_size=BATCH)
    ds, ls, ys, ms = [], [], [], []
    with torch.no_grad():
        for xb, ydb, ylb, mb in loader:
            xb = xb.to(DEVICE)
            dlogit, loc = model(xb)
            ds.append(torch.sigmoid(dlogit).cpu())
            ls.append(loc.cpu())
            ys.append(ydb.cpu())
            ms.append(mb.cpu())
    ds = torch.cat(ds).numpy()
    logits = torch.cat(ls)
    ls = logits.argmax(1).numpy()
    ys = torch.cat(ys).numpy()
    ms = torch.cat(ms).numpy()
    det_pred = (ds >= 0.5).astype(int)
    bal = balanced_accuracy_score(ys, det_pred)
    f1 = f1_score(ys, det_pred)
    try:
        auc = roc_auc_score(ys, ds)
    except ValueError:
        auc = float("nan")
    # localization: among leak-present windows, top-1 hit (+ top-k node hits)
    leak_mask = ms > 0
    true_loc = []
    for xb, ydb, ylb, mb in te:
        ylb_np = ylb.numpy()
        mb_np = mb.numpy()
        true_loc.extend(ylb_np[mb_np > 0])
    true_loc = np.array(true_loc)
    est = ls[leak_mask]
    loc_acc = float(np.mean(est == true_loc)) if len(true_loc) else float("nan")
    # top-k over node classes only (drop class 0 = no-leak)
    node_logits = logits.numpy()[leak_mask][:, 1:]
    tl = true_loc - 1  # back to 0..8 node indices
    if len(tl):
        order = np.argsort(-node_logits, axis=1)
        loc3 = float(np.mean([tl[i] in order[i, :3] for i in range(len(tl))]))
        loc5 = float(np.mean([tl[i] in order[i, :5] for i in range(len(tl))]))
    else:
        loc3 = loc5 = float("nan")
    out = {"bal_acc": bal, "f1": f1, "auc": auc, "loc_acc": loc_acc,
           "loc3": loc3, "loc5": loc5}
    if return_preds:
        out["_preds"] = {"det_prob": ds, "det_pred": det_pred, "pred_loc": ls,
                         "true_loc": true_loc, "y_det": ys, "leak_mask": leak_mask,
                         "node_logits": node_logits, "true_loc_node": tl}
    return out


def train_eval_subsets(arch, k, seed):
    # deterministic per-run seeding for independence & reproducibility
    import hashlib
    run_seed = int(hashlib.sha1(f"{arch}-{k}-{seed}".encode()).hexdigest()[:8], 16)
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    torch.cuda.manual_seed(run_seed)

    X, yd, yl, meta = load_data()
    n_j = X.shape[2]
    n_scen = len(meta["schedules"])
    nums = compute_scenario_bounds(meta)
    ranges = window_index_ranges(nums)
    tr_s, va_s, te_s = grouped_split(n_scen, seed=seed)
    # window indices
    def gather(sids):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])
    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)

    rng = np.random.default_rng(1000 + seed)
    sensors = rng.choice(n_j, k, replace=False)
    sensors = np.sort(sensors)

    model = models.build_model(arch, in_ch=k, hidden=HIDDEN).to(DEVICE)
    tr, va, te = make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
    t0 = time.time()
    model = train(model, tr, va)
    wall = time.time() - t0
    metrics = evaluate(model, te)
    n_params = model.param_count()
    return metrics, sensors, wall, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="*", default=ARCHS)
    parser.add_argument("--ks", nargs="*", type=int, default=SENSOR_KS)
    parser.add_argument("--subseed", nargs="*", type=int, default=SUBSEED)
    args = parser.parse_args()

    os.makedirs("data/results", exist_ok=True)
    rows = []
    for arch in args.archs:
        for k in args.ks:
            for seed in args.subseed:
                try:
                    metrics, sensors, wall, nparams = train_eval_subsets(arch, k, seed)
                    row = {"arch": arch, "k": k, "subseed": seed,
                           "sensors": sensors.tolist(),
                           "n_params": nparams,
                           "wall_min": round(wall / 60, 2),
                           **metrics}
                    rows.append(row)
                    print(row)
                except Exception as e:
                    print(f"ERROR arch={arch} k={k} seed={seed}: {e}")
                    import traceback; traceback.print_exc()
    # save
    import csv
    fieldnames = ["arch", "k", "subseed", "sensors", "n_params", "wall_min",
                  "bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]
    fn = "data/results/benchmark.csv"
    if os.path.exists(fn):
        os.replace(fn, fn + ".bak")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
