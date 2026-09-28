"""Data generation v3: TIMED orifice leaks on LeakDB Net1 (WNTRSimulator, PDD).

v3 fixes a critical label-generation bug found during the study (documented in
the paper): in v2 the leak emitter was hydraulically active for the ENTIRE
24h simulation while the labels marked only an 8h window as "leak"; labeled
no-leak windows of leak scenarios therefore contained the leak signal, and
the detection task was partially confounded with time-of-day. v3 uses:

  * WNTR's timed leak mechanism (``node.add_leak`` with ``start_time`` /
    ``end_time``) -- the same mechanism as the official LeakDB dataset
    generator -- driven by the WNTRSimulator in PDD mode;
  * per-scenario RANDOM leak start times (integer hours uniform in [2, 14],
    fixed 8h duration), so that time-of-day carries no label information;
  * leak sizes as orifice hole diameters d in {0.02, 0.03, 0.05} m
    (discharge coeff 0.75), spanning ~14% / ~32% / ~85% of average total
    network demand (~235 m3/h) on this network.

Everything else is unchanged from v2: 40 normal + 27 leak scenarios
(9 nodes x 3 sizes), stochastic per-junction demands, 24h recorded at 5-min
resolution after a 2h warm-up, overlapping 4h windows (48 steps, stride 8),
and identical RNG seeding so the demand realizations match v2.
"""
import json
import os
import time

import numpy as np
import wntr

NET1_INP = "datasets/LeakDB/CCWI-WDSA2018/Dataset_Generator_Py3/networks/Net1.inp"
REPORT_TS = 300          # 5 min
HOURS = 24               # recorded length
WARMUP_H = 2
WINDOW = 48              # 4h window (steps)
STRIDE = 8               # window stride (steps) -> overlap

RNG_SEED = 42
N_NORMAL = 40            # normal (no-leak) scenarios
N_LEAK = 9 * 3           # 9 nodes x 3 sizes
LEAK_DIAMS = [0.02, 0.03, 0.05]   # orifice hole diameters [m]
LEAK_DISCHARGE = 0.75    # orifice discharge coefficient
LEAK_DURATION_H = 8      # active leak duration (hours)
LEAK_START_MIN_H = 2     # random start uniform in [2, 14] -> leak inside [2,22]
LEAK_START_MAX_H = 14


def make_wn(junction_ids, demand_seed):
    """Build a WaterNetworkModel with stochastic per-junction demand."""
    wn = wntr.network.WaterNetworkModel(NET1_INP)
    wn.options.time.duration = (HOURS + WARMUP_H) * 3600
    wn.options.time.report_timestep = REPORT_TS
    wn.options.time.pattern_timestep = 3600
    wn.options.hydraulic.demand_model = "PDD"

    rng = np.random.default_rng(demand_seed)
    # hourly pattern (24 multipliers), smooth-ish random walk in [0.6, 1.4]
    pat = 1.0 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 24) +
                             rng.uniform(0, 2 * np.pi))
    pat = np.clip(pat, 0.55, 1.45)
    name = f"randpat{int(demand_seed)}"
    wn.add_pattern(name, pattern=pat.tolist())
    for jn in junction_ids:
        node = wn.get_node(jn)
        mult = float(rng.uniform(0.7, 1.3))
        node.demand_timeseries_list[0].base_value = node.base_demand * mult
        node.demand_timeseries_list[0].pattern_name = name
    return wn


def run_sim(wn, leak_node=None, leak_diameter=None, start_h=None, end_h=None):
    """Run PDD simulation; return junction pressures and leak demand."""
    if leak_node is not None:
        area = np.pi * (leak_diameter / 2) ** 2
        wn.get_node(leak_node).add_leak(
            wn, area=area, discharge_coeff=LEAK_DISCHARGE,
            start_time=int((WARMUP_H + start_h) * 3600),
            end_time=int((WARMUP_H + end_h) * 3600))
    res = wntr.sim.WNTRSimulator(wn).run_sim()
    press = res.node["pressure"]
    leak_dem = res.node["leak_demand"]
    warm_rows = int(WARMUP_H * 3600 / REPORT_TS)
    press = press.iloc[warm_rows:, :]
    leak_dem = leak_dem.iloc[warm_rows:, :]
    junction = wn.junction_name_list
    P = press[junction].astype(np.float32).values          # (T, n_junctions)
    L = leak_dem[junction].astype(np.float32).values       # (T, n_junctions)
    return P, L


def extract_windows(P, leak_steps, y_loc):
    """Slice (T,n) into overlapping windows -> (nW, W, n). Label: any overlap."""
    T = P.shape[0]
    X, yd, yl = [], [], []
    for s in range(0, T - WINDOW + 1, STRIDE):
        seg = P[s:s + WINDOW]
        m = leak_steps[s:s + WINDOW]
        active = m.mean() > 0.0
        X.append(seg)
        yd.append(int(active))
        yl.append(y_loc if active else -1)
    return (np.stack(X), np.array(yd, np.int8), np.array(yl, np.int8))


def main():
    os.makedirs("data", exist_ok=True)
    wn0 = wntr.network.WaterNetworkModel(NET1_INP)
    junction_ids = sorted(wn0.junction_name_list)
    n_j = len(junction_ids)
    j_idx = {j: i for i, j in enumerate(junction_ids)}
    t0 = time.time()

    Xs, yds, yls = [], [], []
    schedules = []
    rng = np.random.default_rng(RNG_SEED)
    demand_seeds = rng.integers(0, 10_000_000, size=N_NORMAL + N_LEAK)
    # per-scenario leak start hours (drawn once, deterministic order)
    leak_starts = rng.integers(LEAK_START_MIN_H, LEAK_START_MAX_H + 1, size=N_LEAK)

    # --- Normal scenarios ---
    print("Simulating normal scenarios...")
    for s in range(N_NORMAL):
        wn = make_wn(junction_ids, demand_seeds[s])
        P, _ = run_sim(wn)
        T = P.shape[0]
        mask_zero = np.zeros(T, np.int8)
        X, yd, yl = extract_windows(P, mask_zero, -1)
        Xs.append(X); yds.append(yd); yls.append(yl)
        schedules.append({"type": "normal", "seed": int(demand_seeds[s])})

    # --- Leak scenarios: node x size, timed window over random demand ---
    print("Simulating leak scenarios...")
    leak_events = []
    for jidx_ in range(n_j):
        for size_k, d in enumerate(LEAK_DIAMS):
            leak_events.append((junction_ids[jidx_], size_k, d))
    for k, (lnode, size_k, d) in enumerate(leak_events):
        seed = demand_seeds[N_NORMAL + k]
        start_h = int(leak_starts[k])
        end_h = start_h + LEAK_DURATION_H
        wn = make_wn(junction_ids, seed)
        P, L = run_sim(wn, leak_node=lnode, leak_diameter=d,
                       start_h=start_h, end_h=end_h)
        n_steps = P.shape[0]
        times_h = np.arange(n_steps) * (REPORT_TS / 3600.0)
        mask = ((times_h >= start_h) & (times_h < end_h)).astype(np.int8)
        y_loc = j_idx[lnode]
        X, yd, yl = extract_windows(P, mask, y_loc)
        Xs.append(X); yds.append(yd); yls.append(yl)
        leak_q = float(L.sum(axis=1)[mask > 0].mean()) if mask.sum() else 0.0
        schedules.append({"type": "leak", "seed": int(seed), "leak_node": lnode,
                          "size_k": size_k, "leak_diameter_m": d,
                          "leak_flow_mean_m3s": leak_q,
                          "active_start_h": start_h, "active_end_h": end_h})

    X = np.concatenate(Xs, 0)
    yd = np.concatenate(yds, 0)
    yl = np.concatenate(yls, 0)
    print(f"\nX: {X.shape}  y_det classes: {np.bincount(yd)}  "
          f"y_loc unique: {np.unique(yl)}")
    np.save("data/X.npy", X)
    np.save("data/y_det.npy", yd)
    np.save("data/y_loc.npy", yl)
    with open("data/window_meta.json", "w") as f:
        json.dump({"junction_ids": junction_ids, "window": WINDOW, "stride": STRIDE,
                   "report_ts": REPORT_TS, "leak_model": "orifice_add_leak",
                   "leak_diameters_m": LEAK_DIAMS,
                   "leak_discharge_coeff": LEAK_DISCHARGE,
                   "schedules": schedules}, f, indent=2)
    with open("data/junction_ids.json", "w") as f:
        json.dump(junction_ids, f, indent=2)

    print("\nPer-window detection label counts:", np.bincount(yd))
    print("Per-window localization label counts (non--1):",
          np.unique(yl[yl >= 0], return_counts=True)[1])
    # leak start hour distribution (confound check)
    starts = [s["active_start_h"] for s in schedules if s["type"] == "leak"]
    print("Leak start hours:", sorted(starts))
    flows = [s["leak_flow_mean_m3s"] for s in schedules if s["type"] == "leak"]
    print(f"Leak flows (m3/s) by size: "
          f"{[round(float(np.mean([f for f, s in zip(flows, [x for x in schedules if x['type']=='leak']) if s['size_k']==k])), 4) for k in range(3)]}")
    print("\nDone in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
