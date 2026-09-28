"""Time-only confound baseline (no pressure, no demand features).

Leak start hours are uniform on integers [2, 14] with a fixed 8-hour
duration, so the leak-active intervals are restricted to hours [2, 22] and
the leak-label distribution over window start positions is NON-uniform:
mid-day windows overlap some leak interval with high probability, early
and late windows rarely do. The v2 claim "time-of-day carries no label
information" is therefore not literally true. This script quantifies the
confound:

  - empirical window-start-position -> leak-rate profile;
  - detection classifiers given ONLY the window's start position
    (position index + sin/cos of the corresponding hour): logistic
    regression and a small MLP, under the six scenario-grouped splits;
  - localization from time features only (expected chance: leak start is
    drawn independently of the leak node).

No pressure or demand features are used.
"""
import csv
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

import benchmark

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

STRIDE = 8
REPORT_TS = 300  # seconds
WINDOW = 48


def build_time_features(meta, nums, ranges):
    """Per-window start-position features (position within the scenario day)."""
    feats, labels_det, labels_loc, scen = [], [], [], []
    n_scen = len(meta["schedules"])
    for i in range(n_scen):
        a, b = ranges[i]
        n_win = b - a
        for w in range(n_win):
            start_step = w * STRIDE
            hour = start_step * REPORT_TS / 3600.0
            ang = 2 * np.pi * hour / 24.0
            feats.append([start_step, np.sin(ang), np.cos(ang)])
            scen.append(i)
    return np.array(feats, dtype=np.float64), scen


def train_mlp(Xtr, ytr, seed, d=32, epochs=400, lr=2e-3):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def main():
    X, yd, yl, meta = benchmark.load_data()
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)

    feats, scen = build_time_features(meta, nums, ranges)
    scen = np.array(scen)
    yd = np.asarray(yd)
    yl = np.asarray(yl)

    # empirical leak-rate by window position (for the text)
    positions = feats[:, 0].astype(int)
    prof = {}
    for p in np.unique(positions):
        m = positions == p
        prof[int(p)] = float(yd[m].mean())
    stats = {"leak_rate_by_position": prof,
             "min": float(min(prof.values())), "max": float(max(prof.values()))}
    print("window-position leak-rate profile: min %.3f max %.3f" %
          (stats["min"], stats["max"]))

    rows = []
    for subseed in range(6):
        rs = int(hashlib.sha1(f"timeonly-{subseed}".encode()).hexdigest()[:8], 16)
        tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=subseed)
        tr_mask = np.isin(scen, list(tr_s))
        te_mask = np.isin(scen, list(te_s))
        mu, sd = feats[tr_mask].mean(0), feats[tr_mask].std(0) + 1e-8
        Z = (feats - mu) / sd

        row = {"subseed": subseed}
        # logistic regression, detection
        clf = LogisticRegression(max_iter=5000)
        clf.fit(Z[tr_mask], yd[tr_mask])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yd[te_mask], clf.predict_proba(Z[te_mask])[:, 1])
        row["logreg_auc"] = auc
        # MLP, detection
        mlp = train_mlp(Z[tr_mask], yd[tr_mask], rs)
        with torch.no_grad():
            p = torch.softmax(mlp(torch.tensor(Z[te_mask], dtype=torch.float32)), 1)[:, 1].numpy()
        row["mlp_auc"] = roc_auc_score(yd[te_mask], p)
        # time -> leak node (localization; expected chance)
        leak_tr = tr_mask & (yl >= 0)
        leak_te = te_mask & (yl >= 0)
        if leak_te.sum():
            from sklearn.linear_model import LogisticRegression as LR
            cl = LR(max_iter=5000)
            cl.fit(Z[leak_tr], yl[leak_tr])
            row["logreg_loc1"] = float(np.mean(
                cl.predict(Z[leak_te]) == yl[leak_te]))
        rows.append(row)
        print(f"subseed {subseed}: logreg AUC={row['logreg_auc']:.3f} "
              f"MLP AUC={row['mlp_auc']:.3f} "
              f"loc1={row.get('logreg_loc1', float('nan')):.3f}")

    fn = "data/results/time_only.csv"
    flds = ["subseed", "logreg_auc", "mlp_auc", "logreg_loc1"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")
    with open("data/results/time_only_profile.json", "w") as f:
        import json
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
