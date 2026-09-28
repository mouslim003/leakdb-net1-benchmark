"""Two-stage localization baseline: detection model -> localization model.

Purpose: justify the joint multitask design by comparing it with the
natural alternative --
  Stage 1: detection-only model (binary head, BCE loss).
  Stage 2: 9-class localization-only model trained on TRAIN leak windows only
           (pure node classifier, no no-leak class).
At test time, stage 2 is applied only to windows stage 1 flags as leaks.

Metrics on the test split:
  loc_uncond (Loc@1, unconditional): (# leak windows correctly detected AND
      correctly localized) / (# leak windows). Directly comparable to the
      multitask Loc@1, which also counts undetected leaks as misses.
  loc_cond: localization accuracy among correctly-detected leak windows.
  det_*: stage-1 detection metrics.
Same 6 placements x 4 densities, scenario-grouped splits, per-run seeds.
"""
import csv
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import benchmark
import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

KS = [9, 7, 5, 3]
SUBSEEDS = [0, 1, 2, 3, 4, 5]
ARCHS = ["cnn", "lstm", "gru", "transformer"]


def train_det(model, tr, va):
    return benchmark.train(model, tr, va)


def train_loc(model, tr_loader_idx, X, yd, yl, sensors, tr_idx, va_idx):
    """Train a 9-class node classifier on (standardized) leak windows only."""
    # leak windows in train: yl >= 0
    leak_tr = tr_idx[yl[tr_idx] >= 0]
    leak_va = va_idx[yl[va_idx] >= 0]
    mu = X[tr_idx][:, :, sensors].mean(axis=(0, 1))
    sd = X[tr_idx][:, :, sensors].std(axis=(0, 1)) + 1e-8
    Xt = torch.tensor((X[leak_tr][:, :, sensors] - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(yl[leak_tr], dtype=torch.long)
    Xv = torch.tensor((X[leak_va][:, :, sensors] - mu) / sd, dtype=torch.float32)
    yv = torch.tensor(yl[leak_va], dtype=torch.long)
    tr_ds = torch.utils.data.TensorDataset(Xt, yt)
    va_ds = torch.utils.data.TensorDataset(Xv, yv)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=benchmark.BATCH, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=benchmark.BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=benchmark.LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for ep in range(benchmark.EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(benchmark.DEVICE), yb.to(benchmark.DEVICE)
            opt.zero_grad()
            _, loc = model(xb)
            loss = ce(loc, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for xb, yb in va_loader:
                xb, yb = xb.to(benchmark.DEVICE), yb.to(benchmark.DEVICE)
                _, loc = model(xb)
                correct += (loc.argmax(1) == yb).sum().item()
                total += len(yb)
            acc = correct / max(total, 1)
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)

    rows = []
    for arch in ARCHS:
        for k in KS:
            for subseed in SUBSEEDS:
                rs = int(hashlib.sha1(f"twostage-{arch}-{k}-{subseed}".encode()).hexdigest()[:8], 16)
                torch.manual_seed(rs)
                np.random.seed(rs)
                torch.cuda.manual_seed(rs)

                tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=subseed)

                def gather(s):
                    return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

                tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
                sensors = np.sort(np.random.default_rng(1000 + subseed).choice(n_j, k, replace=False))

                # stage 1: detection-only (binary head; loc head unused in loss?
                # build_model has both heads -- train with detection loss only)
                det_model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
                tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
                # detection-only training: custom loop, BCE only
                opt = torch.optim.AdamW(det_model.parameters(), lr=benchmark.LR, weight_decay=1e-4)
                bce = nn.BCEWithLogitsLoss()
                tr_loader = torch.utils.data.DataLoader(tr, batch_size=benchmark.BATCH, shuffle=True)
                va_loader = torch.utils.data.DataLoader(va, batch_size=benchmark.BATCH)
                best, best_state = -1.0, None
                for ep in range(benchmark.EPOCHS):
                    det_model.train()
                    for xb, ydb, ylb, mb in tr_loader:
                        xb, ydb = xb.to(benchmark.DEVICE), ydb.float().to(benchmark.DEVICE)
                        opt.zero_grad()
                        dlogit, _ = det_model(xb)
                        loss = bce(dlogit, ydb)
                        loss.backward()
                        opt.step()
                    det_model.eval()
                    with torch.no_grad():
                        ds, ys = [], []
                        for xb, ydb, ylb, mb in va_loader:
                            xb = xb.to(benchmark.DEVICE)
                            dlogit, _ = det_model(xb)
                            ds.append(torch.sigmoid(dlogit).cpu())
                            ys.append(ydb)
                        ds = torch.cat(ds).numpy()
                        ys = torch.cat(ys).numpy()
                        bal = balanced_accuracy_score(ys, (ds >= 0.5).astype(int))
                    if bal > best:
                        best = bal
                        best_state = {kk: v.detach().clone() for kk, v in det_model.state_dict().items()}
                det_model.load_state_dict(best_state)

                # stage 1 metrics on test
                m = benchmark.evaluate(det_model, te)
                preds = m  # detection numbers

                # stage 2: 9-class localization on train leak windows
                loc_model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
                loc_model = train_loc(loc_model, None, X, yd, yl, sensors, tr_idx, va_idx)

                # evaluate two-stage on test
                te_loader = torch.utils.data.DataLoader(te, batch_size=benchmark.BATCH)
                det_probs, loc_pred, true_loc_all, y_true = [], [], [], []
                loc_model.eval()
                det_model.eval()
                with torch.no_grad():
                    for xb, ydb, ylb, mb in te_loader:
                        xb = xb.to(benchmark.DEVICE)
                        dlogit, _ = det_model(xb)
                        _, loc = loc_model(xb)
                        det_probs.append(torch.sigmoid(dlogit).cpu())
                        loc_pred.append(loc.argmax(1).cpu())
                        y_true.append(ydb)
                        true_loc_all.append(ylb)
                det_probs = torch.cat(det_probs).numpy()
                loc_pred = torch.cat(loc_pred).numpy()
                y_true = torch.cat(y_true).numpy()
                true_loc_all = torch.cat(true_loc_all).numpy()

                det_flag = det_probs >= 0.5
                leak_mask = true_loc_all >= 0
                n_leak = int(leak_mask.sum())
                detected_leak = leak_mask & det_flag
                correct = (loc_pred[detected_leak] == true_loc_all[detected_leak]).sum() if detected_leak.sum() else 0
                loc_uncond = correct / n_leak if n_leak else float("nan")
                loc_cond = correct / int(detected_leak.sum()) if detected_leak.sum() else float("nan")

                row = {"arch": arch, "k": k, "subseed": subseed,
                       "det_bal_acc": m["bal_acc"], "det_f1": m["f1"], "det_auc": m["auc"],
                       "loc_uncond": loc_uncond, "loc_cond": loc_cond,
                       "n_leak": n_leak, "n_detected_leak": int(detected_leak.sum())}
                rows.append(row)
                print(row)

    fn = "data/results/two_stage.csv"
    flds = ["arch", "k", "subseed", "det_bal_acc", "det_f1", "det_auc",
            "loc_uncond", "loc_cond", "n_leak", "n_detected_leak"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
