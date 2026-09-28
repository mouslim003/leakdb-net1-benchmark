"""Demand–leak transplant intervention (mechanism probe).

The demand-only fingerprint baseline (v2) shows that Experiment A PERMITS
memorization: a classifier given the exact scenario fingerprint can recall
seen scenarios' leak nodes. It does not show that the trained channel
models actually USE the fingerprint. This script is the matched
intervention: keep the demand realization (and leak size and start hour)
IDENTICAL, move the leak to a DIFFERENT node, and ask where the trained
model localizes — both through the TRAINING placement (in-distribution)
and through an UNSEEN placement (the Experiment-A condition, where channel
semantics are broken and any placement-invariant shortcut, such as the
demand fingerprint embedded in the pressure temporal pattern, is the only
remaining signal).

Design (k=5, CNN, master seeds -> bit-identical to the master benchmark):
  - For each of the 6 master replicates: train CNN at k=5 on placement
    (subseed s); unseen placement = placement of subseed (s+1) mod 6.
  - Select the leak scenarios of the TRAINING split (up to 8 per replicate).
  - For each, simulate two counterfactuals: same demand seed, same orifice
    diameter, same start hour, leak node moved to (idx+4) mod 9 and
    (idx+7) mod 9.
  - Evaluate leak-window predictions:
      in-dist   : through the training placement, training stats
      placement : through the UNSEEN placement, that placement's
                  train-scenario stats (cross_sensor protocol)
    recording P(pred = original training node n) [fingerprint use],
    P(pred = counterfactual true node m) [spatial use], P(other), plus
    the control P(pred = n) on the original scenario windows.
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
from data_gen import make_wn, run_sim, extract_windows, REPORT_TS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAX_SCEN = 8
OFFSETS = [4, 7]


def train_cnn_master(subseed, X, yd, yl, meta, nums, ranges):
    """Train the k=K CNN exactly as the master benchmark (same seed)."""
    rs = int(hashlib.sha1(f"cnn-{K}-{subseed}".encode()).hexdigest()[:8], 16)
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)
    tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=subseed)

    def gather(sids):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
    n_j = X.shape[2]
    sensors = np.sort(np.random.default_rng(1000 + subseed).choice(n_j, K, replace=False))
    model = models.build_model("cnn", in_ch=K, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
    tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
    model = benchmark.train(model, tr, va)
    return model, sensors, tr, tr_s, tr_idx


def simulate_scenario(junction_ids, demand_seed, leak_node_idx, diameter, start_h):
    """Run one counterfactual simulation; return windows + labels."""
    lnode = junction_ids[leak_node_idx]
    end_h = start_h + 8
    wn = make_wn(junction_ids, demand_seed)
    P, _ = run_sim(wn, leak_node=lnode, leak_diameter=diameter,
                   start_h=start_h, end_h=end_h)
    n_steps = P.shape[0]
    times_h = np.arange(n_steps) * (REPORT_TS / 3600.0)
    mask = ((times_h >= start_h) & (times_h < end_h)).astype(np.int8)
    Xw, ydw, ylw = extract_windows(P, mask, leak_node_idx)
    return Xw, ydw, ylw


def eval_windows(model, sensors, mu, sd, Xw, ydw, ylw, n_orig):
    """Evaluate the trained model on (possibly counterfactual) windows."""
    from sklearn.metrics import roc_auc_score
    Xi = Xw[:, :, sensors].astype(np.float32)
    Xz = (Xi - np.asarray(mu)) / np.asarray(sd)
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(Xz, dtype=torch.float32).to(benchmark.DEVICE)
        dlogit, loc = model(xt)
        probs = torch.sigmoid(dlogit).cpu().numpy()
        preds = loc.argmax(1).cpu().numpy()
    leak_mask = ylw >= 0
    n_leak = int(leak_mask.sum())
    out = {"n_leak_windows": n_leak, "auc": float("nan"), "p_orig": float("nan"),
           "p_true": float("nan"), "p_other": float("nan")}
    if n_leak:
        pl = np.clip(preds[leak_mask] - 1, -1, 8)  # node indices (drop no-leak)
        yl_true = ylw[leak_mask]
        out["p_orig"] = float(np.mean(pl == n_orig))
        out["p_true"] = float(np.mean(pl == yl_true))
        out["p_other"] = float(np.mean((pl != n_orig) & (pl != yl_true)))
    try:
        out["auc"] = float(roc_auc_score(ydw, probs))
    except ValueError:
        pass
    return out


def main():
    X, yd, yl, meta = benchmark.load_data()
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    junction_ids = meta["junction_ids"]

    rows = []
    for subseed in range(6):
        model, sensors, tr_ds, tr_s, tr_idx = train_cnn_master(
            subseed, X, yd, yl, meta, nums, ranges)
        mu_tr, sd_tr = tr_ds.stats

        # unseen placement: subseed (s+1) mod 6, standardized with that
        # placement's TRAIN-scenario statistics (cross_sensor protocol)
        s_unseen = (subseed + 1) % 6
        sensors_u = np.sort(np.random.default_rng(1000 + s_unseen).choice(
            X.shape[2], K, replace=False))
        stats_src = benchmark.WindowDataset(X, yd, yl, sensors_u, tr_idx)
        mu_u, sd_u = stats_src.stats

        leak_scen = sorted([i for i, s in enumerate(meta["schedules"])
                            if s["type"] == "leak" and i in tr_s])[:MAX_SCEN]

        for i in leak_scen:
            sc = meta["schedules"][i]
            n_orig = junction_ids.index(sc["leak_node"])
            d = sc["leak_diameter_m"]
            start_h = sc["active_start_h"]

            # control: original scenario windows (training data)
            Xo = X[ranges[i][0]:ranges[i][1]]
            ylo = yl[ranges[i][0]:ranges[i][1]]
            ydo = yd[ranges[i][0]:ranges[i][1]]
            for cond, sens, mu, sd in [("original_indist", sensors, mu_tr, sd_tr),
                                       ("original_unseen", sensors_u, mu_u, sd_u)]:
                r = eval_windows(model, sens, mu, sd, Xo, ydo, ylo, n_orig)
                rows.append({"subseed": subseed, "scenario": i, "condition": cond,
                             "orig_node": n_orig, "true_node": n_orig, **r})

            # counterfactuals: same demand seed/size/start, moved leak
            for off in OFFSETS:
                m = (n_orig + off) % len(junction_ids)
                Xw, ydw, ylw = simulate_scenario(junction_ids, sc["seed"], m, d, start_h)
                for cond, sens, mu, sd in [("cf+{}_indist".format(off), sensors, mu_tr, sd_tr),
                                           ("cf+{}_unseen".format(off), sensors_u, mu_u, sd_u)]:
                    r = eval_windows(model, sens, mu, sd, Xw, ydw, ylw, n_orig)
                    rows.append({"subseed": subseed, "scenario": i, "condition": cond,
                                 "orig_node": n_orig, "true_node": m, **r})
            r1 = [r for r in rows if r["scenario"] == i and r["subseed"] == subseed]
            ind = [r for r in r1 if "cf+4_indist" in r["condition"]][0]
            un = [r for r in r1 if "cf+4_unseen" in r["condition"]][0]
            print(f"s{subseed} scen{i} n{n_orig}: cf in-dist P(n)={ind['p_orig']:.2f} "
                  f"P(m)={ind['p_true']:.2f} | cf unseen P(n)={un['p_orig']:.2f} "
                  f"P(m)={un['p_true']:.2f}")

    fn = f"data/results/demand_transplant_k{K}.csv"
    flds = ["subseed", "scenario", "condition", "orig_node", "true_node",
            "n_leak_windows", "p_orig", "p_true", "p_other", "auc"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
