"""Demand-stochasticity sensitivity: is detection trivial without demand noise?

Motivation: the claim "without demand stochasticity, leak detection
is trivial" was asserted, not demonstrated. This script demonstrates it: it
regenerates the 67-scenario dataset at four demand-noise scales
  s = 0.0 : deterministic demands (all scenarios share one demand profile)
  s = 0.5 : half the master noise
  s = 1.0 : the master noise (reproduces the shipped dataset)
  s = 2.0 : double the master noise
using the SAME timed-orifice leak mechanism and per-scenario leak windows as
data_gen.py v3, and trains all four architectures (k in {9,3}, 3 placement/
split replicates) on each. The demand seeds are identical across scales, so
the only change is the noise amplitude. Detection AUC vs s is the result.
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import wntr

import benchmark
import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

NET1_INP = "datasets/LeakDB/CCWI-WDSA2018/Dataset_Generator_Py3/networks/Net1.inp"
REPORT_TS = 300
HOURS = 24
WARMUP_H = 2
WINDOW = 48
STRIDE = 8
RNG_SEED = 42
N_NORMAL = 40
N_LEAK = 27
LEAK_DIAMS = [0.02, 0.03, 0.05]
LEAK_DISCHARGE = 0.75
LEAK_DURATION_H = 8
SCALES = [0.0, 0.5, 1.0, 2.0]
KS = [9, 3]
SUBSEEDS = [0, 1, 2]
ARCHS = ["cnn", "lstm", "gru", "transformer"]


def make_wn(junction_ids, demand_seed, scale):
    wn = wntr.network.WaterNetworkModel(NET1_INP)
    wn.options.time.duration = (HOURS + WARMUP_H) * 3600
    wn.options.time.report_timestep = REPORT_TS
    wn.options.time.pattern_timestep = 3600
    wn.options.hydraulic.demand_model = "PDD"
    rng = np.random.default_rng(demand_seed)
    pat = 1.0 + 0.4 * scale * np.sin(np.linspace(0, 4 * np.pi, 24) +
                                     rng.uniform(0, 2 * np.pi))
    pat = np.clip(pat, 1 - 0.45 * scale, 1 + 0.45 * scale)
    name = f"randpat{int(demand_seed)}"
    wn.add_pattern(name, pattern=pat.tolist())
    for jn in junction_ids:
        node = wn.get_node(jn)
        mult = float(1.0 + scale * rng.uniform(-0.3, 0.3))
        node.demand_timeseries_list[0].base_value = node.base_demand * mult
        node.demand_timeseries_list[0].pattern_name = name
    return wn


def run_sim(wn, leak_node=None, leak_diameter=None, start_h=None, end_h=None):
    if leak_node is not None:
        area = np.pi * (leak_diameter / 2) ** 2
        wn.get_node(leak_node).add_leak(
            wn, area=area, discharge_coeff=LEAK_DISCHARGE,
            start_time=int((WARMUP_H + start_h) * 3600),
            end_time=int((WARMUP_H + end_h) * 3600))
    res = wntr.sim.WNTRSimulator(wn).run_sim()
    press = res.node["pressure"]
    warm_rows = int(WARMUP_H * 3600 / REPORT_TS)
    press = press.iloc[warm_rows:, :]
    junction = wn.junction_name_list
    return press[junction].astype(np.float32).values


def extract_windows(P, leak_steps, y_loc):
    T = P.shape[0]
    X, yd, yl = [], [], []
    for s in range(0, T - WINDOW + 1, STRIDE):
        seg = P[s:s + WINDOW]
        active = leak_steps[s:s + WINDOW].mean() > 0.0
        X.append(seg)
        yd.append(int(active))
        yl.append(y_loc if active else -1)
    return np.stack(X), np.array(yd, np.int8), np.array(yl, np.int8)


def gen_dataset(scale, outdir):
    os.makedirs(outdir, exist_ok=True)
    wn0 = wntr.network.WaterNetworkModel(NET1_INP)
    junction_ids = sorted(wn0.junction_name_list)
    j_idx = {j: i for i, j in enumerate(junction_ids)}
    rng = np.random.default_rng(RNG_SEED)
    demand_seeds = rng.integers(0, 10_000_000, size=N_NORMAL + N_LEAK)
    leak_starts = rng.integers(2, 15, size=N_LEAK)

    Xs, yds, yls, schedules = [], [], [], []
    for s in range(N_NORMAL):
        wn = make_wn(junction_ids, demand_seeds[s], scale)
        P = run_sim(wn)
        X, yd, yl = extract_windows(P, np.zeros(P.shape[0], np.int8), -1)
        Xs.append(X); yds.append(yd); yls.append(yl)
        schedules.append({"type": "normal", "seed": int(demand_seeds[s])})
    leak_events = [(jn, C) for jn in junction_ids for C in LEAK_DIAMS]
    for k, (lnode, d) in enumerate(leak_events):
        seed = demand_seeds[N_NORMAL + k]
        start_h = int(leak_starts[k])
        wn = make_wn(junction_ids, seed, scale)
        P = run_sim(wn, leak_node=lnode, leak_diameter=d,
                    start_h=start_h, end_h=start_h + LEAK_DURATION_H)
        n_steps = P.shape[0]
        times_h = np.arange(n_steps) * (REPORT_TS / 3600.0)
        mask = ((times_h >= start_h) &
                (times_h < start_h + LEAK_DURATION_H)).astype(np.int8)
        X, yd, yl = extract_windows(P, mask, j_idx[lnode])
        Xs.append(X); yds.append(yd); yls.append(yl)
        schedules.append({"type": "leak", "seed": int(seed), "leak_node": lnode,
                          "leak_diameter_m": d,
                          "active_start_h": start_h,
                          "active_end_h": start_h + LEAK_DURATION_H})
    X = np.concatenate(Xs, 0)
    yd = np.concatenate(yds, 0)
    yl = np.concatenate(yls, 0)
    np.save(os.path.join(outdir, "X.npy"), X)
    np.save(os.path.join(outdir, "y_det.npy"), yd)
    np.save(os.path.join(outdir, "y_loc.npy"), yl)
    with open(os.path.join(outdir, "window_meta.json"), "w") as f:
        json.dump({"junction_ids": junction_ids, "window": WINDOW,
                   "stride": STRIDE, "report_ts": REPORT_TS,
                   "schedules": schedules}, f)
    print(f"scale={scale}: X {X.shape}, leak windows {int(yd.sum())}")
    return X, yd, yl


def main():
    rows = []
    for scale in SCALES:
        outdir = f"data/noise_sweep/s{scale}"
        if os.path.exists(os.path.join(outdir, "X.npy")):
            X = np.load(os.path.join(outdir, "X.npy"))
            yd = np.load(os.path.join(outdir, "y_det.npy"))
            yl = np.load(os.path.join(outdir, "y_loc.npy"))
        else:
            t0 = time.time()
            X, yd, yl = gen_dataset(scale, outdir)
            print(f"  (generated in {time.time()-t0:.0f}s)")

        n_j = X.shape[2]
        with open(os.path.join(outdir, "window_meta.json")) as f:
            meta = json.load(f)
        nums = benchmark.compute_scenario_bounds(meta)
        ranges = benchmark.window_index_ranges(nums)

        for arch in ARCHS:
            for k in KS:
                for subseed in SUBSEEDS:
                    rs = int(hashlib.sha1(f"noise-{scale}-{arch}-{k}-{subseed}".encode())
                             .hexdigest()[:8], 16)
                    torch.manual_seed(rs)
                    np.random.seed(rs)
                    torch.cuda.manual_seed(rs)
                    tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=subseed)

                    def gather(s):
                        return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

                    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
                    sensors = np.sort(np.random.default_rng(1000 + subseed)
                                      .choice(n_j, k, replace=False))
                    model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN) \
                        .to(benchmark.DEVICE)
                    tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
                    model = benchmark.train(model, tr, va)
                    m = benchmark.evaluate(model, te)
                    row = {"scale": scale, "arch": arch, "k": k, "subseed": subseed, **m}
                    rows.append(row)
                    print(f"s={scale} {arch} k={k} sub={subseed}: "
                          f"auc={m['auc']:.3f} f1={m['f1']:.3f} loc={m['loc_acc']:.3f}")

    fn = "data/results/demand_noise.csv"
    flds = ["scale", "arch", "k", "subseed", "bal_acc", "f1", "auc",
            "loc_acc", "loc3", "loc5"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
