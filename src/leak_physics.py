"""Physical characterization of the leak model (orifice holes, timed).

Purpose: quantify what the leak sizes mean physically. For each
hole diameter we measure, with matched demand realizations (counterfactual
twin simulations):
  - the mean emitter/orifice leak discharge (L/s, m3/h, % of total demand),
  - the pressure drop at the leak node and the network-wide mean drop,
during the leak-active hours of a fixed mid-day window.
Units: WNTR reports pressures in m (SI) and flows in m3/s on this network.
"""
import json
import os
import sys

import numpy as np
import wntr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

NET1_INP = "datasets/LeakDB/CCWI-WDSA2018/Dataset_Generator_Py3/networks/Net1.inp"
REPORT_TS = 300
HOURS = 24
WARMUP_H = 2
LEAK_DIAMS = [0.02, 0.03, 0.05]
LEAK_DISCHARGE = 0.75
SEED = 1234
START_H, END_H = 8, 16


def make_wn(junction_ids, demand_seed):
    wn = wntr.network.WaterNetworkModel(NET1_INP)
    wn.options.time.duration = (HOURS + WARMUP_H) * 3600
    wn.options.time.report_timestep = REPORT_TS
    wn.options.time.pattern_timestep = 3600
    wn.options.hydraulic.demand_model = "PDD"
    rng = np.random.default_rng(demand_seed)
    pat = 1.0 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 24) + rng.uniform(0, 2 * np.pi))
    pat = np.clip(pat, 0.55, 1.45)
    name = f"randpat{int(demand_seed)}"
    wn.add_pattern(name, pattern=pat.tolist())
    for jn in junction_ids:
        node = wn.get_node(jn)
        node.demand_timeseries_list[0].base_value = node.base_demand * \
            float(rng.uniform(0.7, 1.3))
        node.demand_timeseries_list[0].pattern_name = name
    return wn


def run(wn):
    res = wntr.sim.WNTRSimulator(wn).run_sim()
    press = res.node["pressure"]
    leak = res.node["leak_demand"]
    dem = res.node["demand"]
    warm = int(WARMUP_H * 3600 / REPORT_TS)
    return (press.iloc[warm:, :], leak.iloc[warm:, :], dem.iloc[warm:, :])


def main():
    wn0 = wntr.network.WaterNetworkModel(NET1_INP)
    junction_ids = sorted(wn0.junction_name_list)
    out = {"leak_model": "orifice (WNTR add_leak), Cd=0.75",
           "leak_node": "21", "window_h": [START_H, END_H], "results": []}

    # reference (no-leak twin)
    wn = make_wn(junction_ids, SEED)
    P0, _, D0 = run(wn)
    P0 = P0[junction_ids]
    tot_dem = D0[junction_ids].values.sum(axis=1).mean()
    out["total_demand_m3s"] = float(tot_dem)

    leak_node = "21"
    for d in LEAK_DIAMS:
        wn = make_wn(junction_ids, SEED)
        area = np.pi * (d / 2) ** 2
        wn.get_node(leak_node).add_leak(
            wn, area=area, discharge_coeff=LEAK_DISCHARGE,
            start_time=int((WARMUP_H + START_H) * 3600),
            end_time=int((WARMUP_H + END_H) * 3600))
        P1, L1, _ = run(wn)
        P1 = P1[junction_ids]
        L1 = L1[junction_ids]
        steps = np.arange(P1.shape[0]) * (REPORT_TS / 3600.0)
        mask = (steps >= START_H) & (steps < END_H)
        q = float(L1.values[mask].sum(axis=1).mean())
        drop = (P0.values - P1.values)[mask]
        res = {
            "diameter_m": d,
            "leak_flow_m3s": q,
            "leak_flow_Ls": q * 1000,
            "leak_flow_m3h": q * 3600,
            "pct_of_demand": 100 * q / tot_dem,
            "drop_at_leak_node_mean": float(drop[:, junction_ids.index(leak_node)].mean()),
            "drop_network_mean": float(drop.mean()),
            "drop_at_leak_node_max": float(drop[:, junction_ids.index(leak_node)].max()),
        }
        out["results"].append(res)
        print(f"d={d*1000:.0f}mm: q={q*1000:.1f} L/s ({q*3600:.1f} m3/h, "
              f"{100*q/tot_dem:.0f}% of demand), drop@leak "
              f"{res['drop_at_leak_node_mean']:.1f} m (max {res['drop_at_leak_node_max']:.1f}), "
              f"network drop {res['drop_network_mean']:.2f} m")

    out["network_pressure_mean"] = float(P0.values.mean())
    out["network_pressure_std"] = float(P0.values.std())
    print(f"network pressure mean {out['network_pressure_mean']:.1f} m, "
          f"std {out['network_pressure_std']:.1f} m")

    with open("data/results/leak_physics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved -> data/results/leak_physics.json")


if __name__ == "__main__":
    main()
