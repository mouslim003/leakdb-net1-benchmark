"""Demand-only scenario memorization baseline (mechanism probe).

Hypothesis under test: the apparent partial localization transfer in
Experiment A (same scenarios, unseen placement) is scenario memorization --
each scenario's stochastic demand realization is a unique fingerprint that
suffices to recall the leak node of a SEEN scenario, with no spatial
reasoning and no leak signal at all.

Direct test: reconstruct each scenario's demand realization deterministically
from the stored demand seed (window_meta.json) -- 24 hourly pattern
multipliers + 9 per-junction base-demand multipliers = 33 features, NO
pressure data -- and train a fingerprint -> leak-node classifier on the
training scenarios of each scenario-grouped split. Evaluate:

  seen  : Loc@1 on the TRAIN scenarios themselves (memorization condition).
          If far above chance, the fingerprint alone explains Experiment A.
  held  : Loc@1 on the held-out TEST scenarios (transfer condition).
          If at chance, fingerprints carry no leak-node information, as
          expected from the data-generation process (demand seeds are drawn
          independently of the leak node).

Models: logistic regression (linear reference), a small MLP (flexible), and
1-NN (pure memorization probe; the self-match on seen scenarios is excluded
by leave-one-out, so 'seen' for 1-NN means LOO over train scenarios and
'held' means nearest train neighbor for test scenarios).

Splits: the six scenario-grouped splits of the master benchmark
(split seed = subseed in {0..5}), covering the three cross-placement trio
splits (2, 5, 0) among them.
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import benchmark

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SUBSEEDS = [0, 1, 2, 3, 4, 5]
FEAT_DIM = 24 + 9


def demand_features_from_seed(demand_seed, n_junctions=9):
    """Re-extract the demand realization exactly as data_gen.make_wn does."""
    rng = np.random.default_rng(demand_seed)
    pat = 1.0 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 24) +
                             rng.uniform(0, 2 * np.pi))
    pat = np.clip(pat, 0.55, 1.45)
    mults = rng.uniform(0.7, 1.3, size=n_junctions)
    return np.concatenate([pat, mults]).astype(np.float64)


def build_feature_matrix():
    with open("data/window_meta.json") as f:
        meta = json.load(f)
    j_idx = {j: i for i, j in enumerate(meta["junction_ids"])}
    X, y, n_leak_windows = [], [], []
    for sc in meta["schedules"]:
        feats = demand_features_from_seed(sc["seed"])
        X.append(feats)
        if sc["type"] == "leak":
            y.append(j_idx[sc["leak_node"]])
        else:
            y.append(-1)
        n_leak_windows.append(0)
    # leak window counts per scenario: recompute window labeling
    yd = np.load("data/y_det.npy")
    yl = np.load("data/y_loc.npy")
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    for i, (a, b) in enumerate(ranges):
        n_leak_windows[i] = int((yd[a:b] == 1).sum())
        if y[i] < 0 and n_leak_windows[i] > 0:
            y[i] = int(yl[a:b][yd[a:b] == 1][0])
    return np.stack(X), np.array(y), np.array(n_leak_windows), ranges


def train_mlp(Xtr, ytr, seed, d=64, epochs=300, lr=2e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nn.Sequential(
        nn.Linear(FEAT_DIM, d), nn.ReLU(),
        nn.Linear(d, d), nn.ReLU(),
        nn.Linear(d, 9))
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
    Xf, y, n_leak_win, ranges = build_feature_matrix()
    leak_scen = np.where(y >= 0)[0]
    print(f"{len(leak_scen)} leak scenarios; feature dim {Xf.shape[1]}")

    rows = []
    for subseed in SUBSEEDS:
        rs = int(hashlib.sha1(f"demandonly-{subseed}".encode()).hexdigest()[:8], 16)
        tr_s, va_s, te_s = benchmark.grouped_split(len(y), seed=subseed)
        tr_leak = sorted([i for i in leak_scen if i in tr_s])
        te_leak = sorted([i for i in leak_scen if i in te_s])

        # standardize with train statistics
        mu, sd = Xf[tr_leak].mean(0), Xf[tr_leak].std(0) + 1e-8
        Xz = (Xf - mu) / sd

        def loc1(preds, scen_ids):
            # window-weighted Loc@1 (each scenario's leak windows share the prediction)
            num = sum(n_leak_win[i] for i in scen_ids if preds[i] == y[i])
            den = sum(n_leak_win[i] for i in scen_ids)
            return num / max(den, 1)

        out = {"subseed": subseed, "n_train_leak": len(tr_leak),
               "n_test_leak": len(te_leak)}

        # --- logistic regression (linear reference)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=5000, multi_class="multinomial")
        clf.fit(Xz[tr_leak], y[tr_leak])
        pred = np.full(len(y), -1)
        pred[leak_scen] = clf.predict(Xz[leak_scen])
        out["logreg_seen_loc1"] = loc1(pred, tr_leak)
        out["logreg_held_loc1"] = loc1(pred, te_leak)

        # --- small MLP (flexible)
        mlp = train_mlp(Xz[tr_leak], y[tr_leak], rs)
        with torch.no_grad():
            p = mlp(torch.tensor(Xz[leak_scen], dtype=torch.float32)).argmax(1).numpy()
        pred = np.full(len(y), -1)
        pred[leak_scen] = p
        out["mlp_seen_loc1"] = loc1(pred, tr_leak)
        out["mlp_held_loc1"] = loc1(pred, te_leak)

        # --- 1-NN (pure memorization probe; LOO on train, nearest-train for test)
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(Xz[tr_leak], y[tr_leak])
        # seen (LOO): nearest OTHER train scenario
        D = np.linalg.norm(Xz[tr_leak][:, None] - Xz[tr_leak][None], axis=2)
        np.fill_diagonal(D, np.inf)
        loo_pred = y[tr_leak][D.argmin(1)]
        loo_correct_w = sum(n_leak_win[i] for i, pr in zip(tr_leak, loo_pred) if pr == y[i])
        loo_total_w = sum(n_leak_win[i] for i in tr_leak)
        out["knn_seen_loc1"] = loo_correct_w / max(loo_total_w, 1)
        pred = np.full(len(y), -1)
        pred[te_leak] = knn.predict(Xz[te_leak])
        out["knn_held_loc1"] = loc1(pred, te_leak)

        rows.append(out)
        print(f"subseed {subseed}: logreg seen/held = "
              f"{out['logreg_seen_loc1']:.3f}/{out['logreg_held_loc1']:.3f}  "
              f"MLP = {out['mlp_seen_loc1']:.3f}/{out['mlp_held_loc1']:.3f}  "
              f"1NN(LOO) = {out['knn_seen_loc1']:.3f}/{out['knn_held_loc1']:.3f}")

    fn = "data/results/demand_only.csv"
    flds = ["subseed", "n_train_leak", "n_test_leak",
            "logreg_seen_loc1", "logreg_held_loc1",
            "mlp_seen_loc1", "mlp_held_loc1",
            "knn_seen_loc1", "knn_held_loc1"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")

    # summary
    for model in ("logreg", "mlp", "knn"):
        for cond in ("seen", "held"):
            vals = [r[f"{model}_{cond}_loc1"] for r in rows]
            print(f"{model} {cond}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")


if __name__ == "__main__":
    main()
