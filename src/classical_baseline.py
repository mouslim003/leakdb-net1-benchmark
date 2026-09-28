"""Classical ML baseline + leak-size stratification for the reproducibility framing.

Baselines:
- Isolation Forest (unsupervised, on pressure residuals) — the classic non-DL method
  that many leak-detection papers use.
- XGBoost (supervised classifier on flattened window features) — strong classical baseline.

We evaluate on the same scenario-grouped split and the same sensor subsets as the DL
benchmark (k in {9,7,5,3}), comparing detection balanced-acc/F1/AUC.

Leak-size stratification: report detection AUC per leak size class (small/medium/large)
for XGBoost, to show sensitivity to leak severity.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import benchmark  # for load_data, split helpers

SEED = 7
np.random.seed(SEED)


def get_sensor_k(X, sensors):
    # X full: (n, W, 9); return windowed flattened features for chosen sensors
    return X[:, :, sensors]


def run_detection(X, yd, sensors, tr_idx, te_idx):
    Xtr = get_sensor_k(X, sensors)[tr_idx]
    ytr = yd[tr_idx]
    Xte = get_sensor_k(X, sensors)[te_idx]
    yte = yd[te_idx]
    n = Xtr.shape[0]
    Xtr_f = Xtr.reshape(n, -1)
    Xte_f = Xte.reshape(Xte.shape[0], -1)
    return Xtr_f, ytr, Xte_f, yte


def run_isof(X, yd, sensors, tr_idx, te_idx):
    Xtr_f, ytr, Xte_f, yte = run_detection(X, yd, sensors, tr_idx, te_idx)
    model = IsolationForest(contamination="auto", random_state=SEED, n_estimators=200)
    model.fit(Xtr_f[ytr == 0])  # unsupervised: fit on NO-LEAK training windows only
    scores = model.score_samples(Xte_f)   # higher = more normal
    # leak windows should be anomalous (lower score)
    # anomaly score
    y_score = -scores
    try:
        auc = roc_auc_score(yte, y_score)
    except ValueError:
        auc = float("nan")
    # threshold at training p90-normal to get a decision
    thr = np.percentile(-model.score_samples(Xtr_f[ytr == 0]), 90)
    pred = (y_score >= thr).astype(int)
    bal = balanced_accuracy_score(yte, pred)
    f1 = f1_score(yte, pred)
    return {"bal_acc": bal, "f1": f1, "auc": auc}


def run_xgb(X, yd, sensors, tr_idx, te_idx):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    Xtr_f, ytr, Xte_f, yte = run_detection(X, yd, sensors, tr_idx, te_idx)
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          eval_metric="logloss", use_label_encoder=False,
                          subsample=0.8, colsample_bytree=0.8, random_state=SEED)
    model.fit(Xtr_f, ytr)
    prob = model.predict_proba(Xte_f)[:, 1]
    pred = (prob >= 0.5).astype(int)
    bal = balanced_accuracy_score(yte, pred)
    f1 = f1_score(yte, pred)
    auc = roc_auc_score(yte, prob)
    return {"bal_acc": bal, "f1": f1, "auc": auc}


def main():
    try:
        from xgboost import XGBClassifier  # noqa: F401
    except ImportError:
        raise SystemExit("xgboost is required for the classical baselines: pip install xgboost")

    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)

    rows = []
    for name, fn in [("isforest", run_isof), ("xgboost", run_xgb)]:
        for k in [9, 7, 5, 3]:
            # same 6 placement/split replicates as the master benchmark
            accs = {"bal_acc": [], "f1": [], "auc": []}
            for sd in range(6):
                tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=sd)
                ranges_l = ranges

                def gather(s, ranges_l=ranges_l):
                    return np.concatenate([np.arange(*ranges_l[i]) for i in sorted(s)])

                tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
                sensors = np.sort(np.random.default_rng(1000 + sd).choice(n_j, k, replace=False))
                m = fn(X, yd, sensors, tr_idx, te_idx)
                if m is None:
                    break
                for kk in accs:
                    accs[kk].append(m[kk])
            if not accs["bal_acc"]:
                continue
            row = {"method": name, "k": k,
                   "bal_acc": float(np.mean(accs["bal_acc"])),
                   "bal_acc_sd": float(np.std(accs["bal_acc"], ddof=1)),
                   "f1": float(np.mean(accs["f1"])),
                   "f1_sd": float(np.std(accs["f1"], ddof=1)),
                   "auc": float(np.mean(accs["auc"])),
                   "auc_sd": float(np.std(accs["auc"], ddof=1))}
            rows.append(row)
            print(row)

    import csv
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/classical_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "k", "bal_acc", "bal_acc_sd",
                                          "f1", "f1_sd", "auc", "auc_sd"])
        w.writeheader()
        w.writerows(rows)
    print("\nSaved data/results/classical_baseline.csv")


if __name__ == "__main__":
    main()
