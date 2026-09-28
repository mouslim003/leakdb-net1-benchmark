"""Two-tier sensor-activation scheme (v2, RQ4; protocol: experiments/v2-activation/protocol.md).

Tier 1: 3 sensors stay active; a detector (CNN k=3 or Extra Trees) runs
continuously. Tier 2: on trigger (detection probability >= tau) the remaining
sensors wake; a CNN localizer trained at m active sensors (nested subsets,
m in {3,5,7,9}) scores the same window. The m=9 localizer retrains with the
MASTER per-run seed tags, so it must be bit-identical to the released
benchmark.csv rows (gate; abort otherwise).

Outputs:
  data/results/activation_scheme.csv   (one row per detector x subseed x tau x m)
  data/results/activation_gate.json    (bit-identity gate vs benchmark.csv)
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve

import benchmark
import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SUBSEEDS = [0, 1, 2, 3, 4, 5]
M_ACTIVE = [3, 5, 7, 9]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STRIDE = 8
WIN = 48          # steps of 5 min
GATE_TOL = 1e-12


def run_seed(tag):
    return int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16)


def seed_all(rs):
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)


def load_master_rows():
    """Released master-benchmark rows for the gate (arch=cnn, k=9)."""
    rows = {}
    with open("data/results/benchmark.csv") as f:
        for r in csv.DictReader(f):
            if r["arch"] == "cnn" and int(r["k"]) == 9:
                rows[int(r["subseed"])] = r
    return rows


def main():
    t_start = time.time()
    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    n_scen = len(meta["schedules"])
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    master = load_master_rows()

    gate = {"per_subseed": {}, "passed": True}
    rows = []

    for s in SUBSEEDS:
        tr_s, va_s, te_s = benchmark.grouped_split(n_scen, seed=s)

        def gather(sids):
            return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

        tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)

        # nested subsets from one permutation (fresh seed range 4000+)
        perm = np.random.default_rng(4000 + s).permutation(n_j)
        sensors = {m: np.sort(perm[:m]) for m in M_ACTIVE}

        # ---------- tier-1 detectors ----------
        det = {}
        # (a) CNN k=3, fresh per-run seed
        rs = run_seed(f"actdet-cnn-3-{s}")
        seed_all(rs)
        model = models.build_model("cnn", in_ch=3, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
        tr3 = benchmark.WindowDataset(X, yd, yl, sensors[3], tr_idx)
        va3 = benchmark.WindowDataset(X, yd, yl, sensors[3], va_idx, *tr3.stats)
        model = benchmark.train(model, tr3, va3)
        te3 = benchmark.WindowDataset(X, yd, yl, sensors[3], te_idx, *tr3.stats)
        det["cnn"] = benchmark.evaluate(model, te3, return_preds=True)["_preds"]["det_prob"]
        # (b) Extra Trees on raw flattened 3-channel windows (v1 recipe)
        Xtr_flat = X[tr_idx][:, :, sensors[3]].reshape(len(tr_idx), -1)
        Xte_flat = X[te_idx][:, :, sensors[3]].reshape(len(te_idx), -1)
        et = ExtraTreesClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        et.fit(Xtr_flat, yd[tr_idx])
        det["et"] = et.predict_proba(Xte_flat)[:, 1]

        # ---------- tier-2 localizers (one per m) ----------
        loc = {}
        leak_mask = None
        for m in M_ACTIVE:
            tag = f"cnn-9-{s}" if m == 9 else f"actloc-cnn-{m}-{s}"
            rs = run_seed(tag)
            seed_all(rs)
            model = models.build_model("cnn", in_ch=m, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
            trm = benchmark.WindowDataset(X, yd, yl, sensors[m], tr_idx)
            vam = benchmark.WindowDataset(X, yd, yl, sensors[m], va_idx, *trm.stats)
            model = benchmark.train(model, trm, vam)
            tem = benchmark.WindowDataset(X, yd, yl, sensors[m], te_idx, *trm.stats)
            out = benchmark.evaluate(model, tem, return_preds=True)
            if m == 9:  # gate: bit-identity with released master row
                mrow = master[s]
                diffs = {k: abs(float(out[k]) - float(mrow[k]))
                         for k in ["bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]}
                ok = max(diffs.values()) <= GATE_TOL
                gate["per_subseed"][s] = {"max_abs_diff": max(diffs.values()),
                                          "diffs": diffs, "pass": ok}
                if not ok:
                    gate["passed"] = False
                    print(f"GATE FAIL subseed {s}: {diffs}")
            preds = out["_preds"]
            if leak_mask is None:
                leak_mask = preds["leak_mask"]
            nl = preds["node_logits"]
            # Loc@1 follows the paper's definition (Eq. 1 / benchmark.evaluate):
            # full 10-class argmax; a no-leak prediction on a leak window is a miss.
            top1 = preds["pred_loc"][leak_mask] == preds["true_loc"]
            tl = preds["true_loc_node"]
            order = np.argsort(-nl, axis=1)
            loc[m] = {
                "top1": top1,
                "top3": np.array([tl[i] in order[i, :3] for i in range(len(tl))]),
            }
            print(f"subseed {s}: localizer m={m} trained "
                  f"(loc1={loc[m]['top1'].mean():.3f})")

        # window blocks per test scenario (for latency), te order = sorted scenarios
        blocks = []
        row0 = 0
        for i in sorted(te_s):
            a, b = ranges[i]
            blocks.append((i, row0, b - a))
            row0 += b - a

        def latency(trigger):
            delays, censored = [], 0
            for i, st, n in blocks:
                sch = meta["schedules"][i]
                if sch.get("type") != "leak":
                    continue
                onset = sch["active_start_h"]
                found = False
                for w in range(n):
                    end_h = (w * STRIDE + WIN) / 12.0
                    if end_h <= onset:
                        continue
                    if trigger[st + w]:
                        delays.append(end_h - onset)
                        found = True
                        break
                if not found:
                    censored += 1
            return delays, censored

        leak_idx = np.where(leak_mask)[0]
        n_leak = len(leak_idx)
        n_leak_idx = np.where(~leak_mask)[0]

        for dname, prob in det.items():
            fpr_c, tpr_c, thr = roc_curve(leak_mask.astype(int), prob)
            youden = float(thr[int(np.argmax(tpr_c - fpr_c))]) if len(thr) else 0.5
            taus = sorted({round(t, 4) for t in TAUS + [0.5, youden]})
            for tau in taus:
                trigger = prob >= tau
                trig_leak = trigger[leak_idx]
                n_trig_leak = int(trig_leak.sum())
                n_trig_noleak = int(trigger[n_leak_idx].sum())
                n_trig = n_trig_leak + n_trig_noleak
                recall = n_trig_leak / max(n_leak, 1)
                precision = n_trig_leak / max(n_trig, 1)
                fprate = n_trig_noleak / max(len(n_leak_idx), 1)
                act_rate = float(trigger.mean())
                delays, censored = latency(trigger)
                dense1 = float(loc[9]["top1"].mean())
                for m in M_ACTIVE:
                    h1, h3 = loc[m]["top1"], loc[m]["top3"]
                    rows.append({
                        "detector": dname, "subseed": s, "tau": tau, "m": m,
                        "sensors": sensors[m].tolist(),
                        "n_leak": n_leak, "n_trig_leak": n_trig_leak,
                        "n_trig_noleak": n_trig_noleak,
                        "recall": recall, "precision": precision, "fpr": fprate,
                        "act_rate": act_rate,
                        "mean_active_sensors": 3 + (m - 3) * act_rate,
                        "loc1_uncond": float((h1 & trig_leak).sum() / max(n_leak, 1)),
                        "loc3_uncond": float((h3 & trig_leak).sum() / max(n_leak, 1)),
                        "loc1_cond": float((h1 & trig_leak).sum() / max(n_trig_leak, 1))
                                     if n_trig_leak else float("nan"),
                        "loc3_cond": float((h3 & trig_leak).sum() / max(n_trig_leak, 1))
                                     if n_trig_leak else float("nan"),
                        "loc1_dense": dense1,
                        "loc1_dense_m": float(h1.mean()),
                        "loc3_dense_m": float(h3.mean()),
                        "n_censored": censored,
                        "mean_delay_h": float(np.mean(delays)) if delays else float("nan"),
                        "median_delay_h": float(np.median(delays)) if delays else float("nan"),
                    })
        print(f"subseed {s}: detectors evaluated "
              f"(cnn auc-row ok={gate['per_subseed'].get(s, {}).get('pass')})")

    with open("data/results/activation_gate.json", "w") as f:
        json.dump(gate, f, indent=1)
    if not gate["passed"]:
        print("GATE FAILED - master-seed m=9 retrains differ from benchmark.csv. Aborting.")
        sys.exit(1)

    fn = "data/results/activation_scheme.csv"
    flds = list(rows[0].keys())
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nGate passed for all {len(SUBSEEDS)} subseeds.")
    print(f"Saved {len(rows)} rows -> {fn} ({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
