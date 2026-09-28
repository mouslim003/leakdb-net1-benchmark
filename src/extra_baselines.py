"""Chance-level and additional simple-ML baselines under the master protocol.

Purpose:
1. Chance baselines so that 0.0 localization has context:
   - Random detection (AUC 0.5 by construction; balanced acc / F1 of a coin flip).
   - Majority-class detection (always "no leak").
   - Random localization: uniform over the 9 candidate nodes -> Loc@1 = 1/9,
     Loc@3 = 3/9, Loc@5 = 5/9 (empirically estimated over many draws).
   - Majority-node localization: always predict the most frequent TRAIN leak node.
2. Additional simple supervised baselines on the same flattened windows and the
   same scenario-grouped splits / sensor placements as the deep models:
   - Logistic Regression (linear reference)
   - Random Forest
   - Extra Trees
   - HistGradientBoosting (scikit-learn's LightGBM-style learner)

All baselines use the SAME 6 subseeds (placement+split pairs) and k in {9,7,5,3}
as the master benchmark, so mean +/- SD is directly comparable to Table 1.
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import benchmark

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SUBSEEDS = [0, 1, 2, 3, 4, 5]
KS = [9, 7, 5, 3]
N_NODES = 9
RNG = np.random.default_rng(12345)


def prep(X, yd, sensors, tr_idx, te_idx):
    Xtr = X[tr_idx][:, :, sensors].reshape(len(tr_idx), -1)
    Xte = X[te_idx][:, :, sensors].reshape(len(te_idx), -1)
    return Xtr, yd[tr_idx], Xte, yd[te_idx]


def det_metrics(yte, score, pred):
    bal = balanced_accuracy_score(yte, pred)
    f1 = f1_score(yte, pred)
    try:
        auc = roc_auc_score(yte, score)
    except ValueError:
        auc = float("nan")
    return {"bal_acc": bal, "f1": f1, "auc": auc}


def run_ml(name, X, yd, yl, meta, k, subseed, make_model):
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=subseed)

    def gather(s):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

    tr_idx, te_idx = gather(tr_s), gather(te_s)
    sensors = np.sort(np.random.default_rng(1000 + subseed).choice(N_NODES, k, replace=False))
    Xtr, ytr, Xte, yte = prep(X, yd, sensors, tr_idx, te_idx)
    model = make_model()
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(Xte)[:, 1]
    else:  # logreg pipeline
        score = model.decision_function(Xte)
    pred = (score >= 0.5).astype(int) if hasattr(model, "predict_proba") else model.predict(Xte)
    m = det_metrics(yte, score, pred)
    m.update({"method": name, "k": k, "subseed": subseed,
              "sensors": sensors.tolist(), "n_test_leak": int(yte.sum())})
    return m


def chance_rows(X, yd, yl, meta, k, subseed):
    """Chance / majority baselines for one (k, subseed) split."""
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=subseed)

    def gather(s):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

    tr_idx, te_idx = gather(tr_s), gather(te_s)
    sensors = np.sort(np.random.default_rng(1000 + subseed).choice(N_NODES, k, replace=False))
    ytr, yte = yd[tr_idx], yd[te_idx]
    # leak windows + true nodes in train and test (for localization baselines)
    tr_leak_nodes, te_leak_nodes = yl[tr_idx], yl[te_idx]
    tr_leak_nodes = tr_leak_nodes[tr_leak_nodes >= 0]
    te_leak_nodes = te_leak_nodes[te_leak_nodes >= 0]
    rows = []

    # --- random detection: score = U(0,1) ---
    scores = RNG.uniform(0, 1, size=(200, len(yte)))
    preds = scores >= 0.5
    aucs, bals, f1s = [], [], []
    for i in range(200):
        m = det_metrics(yte, scores[i], preds[i])
        aucs.append(m["auc"]); bals.append(m["bal_acc"]); f1s.append(m["f1"])
    rows.append({"method": "random_det", "k": k, "subseed": subseed,
                 "bal_acc": float(np.mean(bals)), "f1": float(np.mean(f1s)),
                 "auc": float(np.mean(aucs)),
                 "sensors": sensors.tolist(), "n_test_leak": int(yte.sum())})

    # --- majority-class detection: always no-leak ---
    pred = np.zeros_like(yte)
    score = np.zeros_like(yte, dtype=float)
    m = det_metrics(yte, score, pred)
    rows.append({"method": "majority_det", "k": k, "subseed": subseed, **m,
                 "sensors": sensors.tolist(), "n_test_leak": int(yte.sum())})

    # --- random localization: uniform over 9 nodes, 200 draws ---
    draws = RNG.integers(0, N_NODES, size=(200, len(te_leak_nodes)))
    loc1 = float(np.mean(draws == te_leak_nodes[None, :]))
    order = np.argsort(-RNG.uniform(size=(200, len(te_leak_nodes), N_NODES)), axis=2)
    in3 = [te_leak_nodes[i] in order[d, i, :3] for d in range(200) for i in range(len(te_leak_nodes))]
    in5 = [te_leak_nodes[i] in order[d, i, :5] for d in range(200) for i in range(len(te_leak_nodes))]
    rows.append({"method": "random_loc", "k": k, "subseed": subseed,
                 "loc_acc": loc1, "loc3": float(np.mean(in3)), "loc5": float(np.mean(in5)),
                 "sensors": sensors.tolist(), "n_test_leak": int(yte.sum())})

    # --- majority-node localization: most frequent TRAIN leak node ---
    counts = np.bincount(tr_leak_nodes, minlength=N_NODES)
    top_node = int(np.argmax(counts))
    rows.append({"method": "majority_loc", "k": k, "subseed": subseed,
                 "loc_acc": float(np.mean(te_leak_nodes == top_node)),
                 "loc3": float(np.mean(np.isin(te_leak_nodes, np.argsort(-counts)[:3]))),
                 "loc5": float(np.mean(np.isin(te_leak_nodes, np.argsort(-counts)[:5]))),
                 "sensors": sensors.tolist(), "n_test_leak": int(yte.sum())})
    return rows


def main():
    X, yd, yl, meta = benchmark.load_data()
    rows = []
    for subseed in SUBSEEDS:
        for k in KS:
            rows.extend(chance_rows(X, yd, yl, meta, k, subseed))
            print(f"chance done k={k} subseed={subseed}")

    ml_makers = {
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=0)),
        "rf": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "extratrees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "hgb": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    for name, maker in ml_makers.items():
        for subseed in SUBSEEDS:
            for k in KS:
                m = run_ml(name, X, yd, yl, meta, k, subseed, maker)
                rows.append(m)
                print(f"{name} k={k} subseed={subseed}: "
                      f"auc={m['auc']:.3f} f1={m['f1']:.3f} bal={m['bal_acc']:.3f}")

    os.makedirs("data/results", exist_ok=True)
    flds = ["method", "k", "subseed", "sensors", "n_test_leak",
            "bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]
    fn = "data/results/extra_baselines.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in rows:
            w.writerow({kk: r.get(kk, "") for kk in flds})
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
