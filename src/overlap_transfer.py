"""Within-density overlap->transfer regression (study iteration 2).

Reads the released 15-trio sweep (data/results/cross_many_trios.csv) and, per
(model, k), regresses per-trio Experiment-B Loc@1 on the mean train-test
sensor overlap (n=15 trios; the recorded per-placement pair averaged).
Reports the OLS slope with 95% t-CI and Spearman rho. Zero training runs.

Output: data/results/overlap_transfer.csv
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MODELS = ["cnn", "gnn_real", "gnn_ident", "gnn_rand"]
KS = [7, 5, 3]


def main():
    df = pd.read_csv("data/results/cross_many_trios.csv")
    df["overlap_mean"] = df["sensor_overlap"].apply(
        lambda s: float(np.mean(eval(s))) if isinstance(s, str) else float(np.mean(s)))
    rows = []
    for model in MODELS:
        for k in KS:
            d = df[(df.model == model) & (df.k == k)]
            x = d["overlap_mean"].values
            y = d["B_loc_acc"].values
            res = sps.linregress(x, y)
            t_crit = sps.t.ppf(0.975, len(x) - 2)
            rho, rho_p = sps.spearmanr(x, y)
            rows.append({"model": model, "k": k, "n": len(x),
                         "overlap_min": float(x.min()), "overlap_max": float(x.max()),
                         "loc1_mean": float(y.mean()),
                         "slope": float(res.slope),
                         "slope_ci_lo": float(res.slope - t_crit * res.stderr),
                         "slope_ci_hi": float(res.slope + t_crit * res.stderr),
                         "slope_p": float(res.pvalue),
                         "spearman": float(rho), "spearman_p": float(rho_p),
                         "r2": float(res.rvalue ** 2)})
            print(f"{model:9s} k={k}: slope={res.slope:+.3f} "
                  f"[{rows[-1]['slope_ci_lo']:+.3f},{rows[-1]['slope_ci_hi']:+.3f}] "
                  f"p={res.pvalue:.4f} rho={rho:+.2f}")

    fn = "data/results/overlap_transfer.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
