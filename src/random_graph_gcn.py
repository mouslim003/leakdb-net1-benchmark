"""Random-graph GCN ablation: is it the PHYSICAL graph or just GCN capacity?

Motivation: the GCN restores placement transfer, but the
comparison changes both the architecture and the representation. This ablation
holds the GCN architecture, capacity, input encoding, and training protocol
fixed and replaces only the adjacency: the real Net1 junction graph vs. a
degree-preserving randomized graph (double-edge-swap rewiring, seeded,
connectivity-checked; the degree sequence is exactly preserved, so the
normalization D^-1/2 (A+I) D^-1/2 acts on an identical degree profile).

Protocol:
- In-distribution benchmark: 4 densities x 6 replicates = 24 runs.
- Cross-placement Experiments A/B: 3 trios x 3 densities = 9 runs
  (identical trio structure to cross_sensor.py --model gnn).
- Results are compared against the real-graph GCN runs already released
  (data/results/gnn_benchmark.csv, cross_sensor_v2_gnn.csv).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

KS = [9, 7, 5, 3]
SUBSEEDS = [0, 1, 2, 3, 4, 5]
XKS = [7, 5, 3]
TRIOS = [((0, 1), 2), ((3, 4), 5), ((2, 5), 0)]


def randomize_adjacency(adj, seed, n_swaps=200):
    """Degree-preserving double-edge-swap randomization of an undirected graph.

    Repeatedly picks two edges (a,b), (c,d) with four distinct endpoints and
    rewires them to (a,d), (c,b). Degrees are exactly preserved; self-loops
    and multi-edges are rejected. Falls back to additional attempts if the
    result is disconnected.
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    edges = [tuple(sorted((int(a), int(b))))
             for a, b in zip(*np.nonzero(np.triu(adj)))]
    edges = list(set(edges))

    def connected(es):
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in es:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        return len({find(i) for i in range(n)}) == 1

    best = None
    for attempt in range(20):
        es = list(edges)
        for _ in range(n_swaps):
            for _ in range(100):  # rejection sampling for a valid swap
                i, j = rng.integers(0, len(es), 2)
                if i == j:
                    continue
                (a, b), (c, d) = es[i], es[j]
                if len({a, b, c, d}) < 4:
                    continue
                e1, e2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
                if e1 in es or e2 in es:
                    continue
                es[i], es[j] = e1, e2
                break
        if connected(es):
            best = es
            break
        if best is None:
            best = es
    out = np.zeros((n, n), dtype=np.float32)
    for a, b in best:
        out[a, b] = 1.0
        out[b, a] = 1.0
    return out


def run_indist(k, subseed, adj, graph_seed=None):
    tag = f"randgnn-{k}-{subseed}" + (f"-g{graph_seed}" if graph_seed else "")
    rs = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16)
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)

    X, yd, yl, meta = benchmark.load_data()
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=subseed)

    def gather(sids):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
    n_j = X.shape[2]
    sensors = np.sort(np.random.default_rng(1000 + subseed).choice(n_j, k, replace=False))

    model = models.build_gnn_model(adj).to(benchmark.DEVICE)
    model.set_sensors(sensors)
    tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
    model = benchmark.train(model, tr, va)
    m = benchmark.evaluate(model, te)
    return {"k": k, "subseed": subseed, "n_params": model.param_count(), **m}


def run_cross(k, train_placements, test_placement, adj):
    tag = f"randgnn-cross-{k}-{train_placements}-{test_placement}"
    rs = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16)
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)

    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=test_placement)

    def gather(sids):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
    sens_train = [np.sort(np.random.default_rng(1000 + s).choice(n_j, k, replace=False))
                  for s in train_placements]
    sens_test = np.sort(np.random.default_rng(1000 + test_placement).choice(n_j, k, replace=False))

    model = models.build_gnn_model(adj).to(benchmark.DEVICE)
    model.set_sensors(sens_train[0])

    tr_parts, va_parts = [], []
    for s in sens_train:
        tr_parts.append(benchmark.WindowDataset(X, yd, yl, s, tr_idx))
        va_parts.append(benchmark.WindowDataset(X, yd, yl, s, va_idx, *tr_parts[-1].stats))
    tr_ds = torch.utils.data.ConcatDataset(tr_parts)
    va_ds = torch.utils.data.ConcatDataset(va_parts)
    model = benchmark.train(model, tr_ds, va_ds)

    model.set_sensors(sens_test)
    stats_src = benchmark.WindowDataset(X, yd, yl, sens_test, tr_idx)
    mu, sd = stats_src.stats
    te_A = benchmark.WindowDataset(X, yd, yl, sens_test, tr_idx, mu, sd)
    te_B = benchmark.WindowDataset(X, yd, yl, sens_test, te_idx, mu, sd)
    mA = benchmark.evaluate(model, te_A)
    mB = benchmark.evaluate(model, te_B)

    out = {"k": k, "train_placements": list(train_placements),
           "test_placement": test_placement}
    for exp, m in (("A", mA), ("B", mB)):
        for kk in ("bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"):
            out[f"{exp}_{kk}"] = m[kk]
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", choices=["random", "identity", "multi-random"],
                        default="random")
    args = parser.parse_args()

    real_adj, junctions = models.get_net1_adjacency()
    os.makedirs("data/results", exist_ok=True)

    # ---- multi-random: 4 additional independent rewirings, in-distribution
    # only (transfer side is covered by cross_many_trios.py, which uses a
    # distinct rewiring per trio). The released 2026 graph makes 5 total.
    if args.adj == "multi-random":
        rows = []
        for gseed in [2027, 2028, 2029, 2030]:
            adj = randomize_adjacency(real_adj, seed=gseed)
            deg_real = real_adj.sum(1).astype(int).tolist()
            deg_rand = adj.sum(1).astype(int).tolist()
            assert deg_real == deg_rand, "degree sequence not preserved"
            for k in KS:
                for subseed in SUBSEEDS:
                    row = run_indist(k, subseed, adj, graph_seed=gseed)
                    rows.append({"family": "indist", "adj_kind": "random",
                                 "graph_seed": gseed, **row})
                    print(f"randgnn g{gseed} indist k={k} s={subseed}: "
                          f"loc={row['loc_acc']:.3f} loc3={row['loc3']:.3f} "
                          f"auc={row['auc']:.3f}")
        fn = "data/results/random_graph_gcn.csv"
        flds = ["family", "adj_kind", "graph_seed", "k", "subseed",
                "train_placements", "test_placement", "n_params", "bal_acc",
                "f1", "auc", "loc_acc", "loc3", "loc5",
                "A_bal_acc", "A_f1", "A_auc", "A_loc_acc", "A_loc3", "A_loc5",
                "B_bal_acc", "B_f1", "B_auc", "B_loc_acc", "B_loc3", "B_loc5"]
        existing = []
        if os.path.exists(fn):
            with open(fn) as f:
                import csv as _csv
                for r in _csv.DictReader(f):
                    if not r.get("adj_kind"):
                        r["adj_kind"] = "random"
                    r.setdefault("graph_seed", "2026" if r["adj_kind"] == "random" else "")
                    existing.append(r)
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            for r in existing:
                w.writerow({kk: r.get(kk, "") for kk in flds})
            for r in rows:
                w.writerow({kk: (json.dumps(r[kk]) if isinstance(r.get(kk), (list, tuple)) else r.get(kk))
                            for kk in flds})
        print(f"\nSaved {len(rows)} new rows (multi-random) -> {fn}")
        return

    if args.adj == "random":
        adj = randomize_adjacency(real_adj, seed=2026)
        deg_real = real_adj.sum(1).astype(int).tolist()
        deg_rand = adj.sum(1).astype(int).tolist()
        assert deg_real == deg_rand, "degree sequence not preserved"
        n_edges = int(real_adj.sum() / 2)
        print(f"real graph: {n_edges} edges, degrees {deg_real}")
        print(f"randomized: {int(adj.sum() / 2)} edges, degrees {deg_rand}")
    else:
        # identity adjacency: no neighbor mixing at all; the model becomes a
        # per-node temporal MLP with shared weights and per-node scores
        adj = np.eye(real_adj.shape[0], dtype=np.float32)
        print("identity adjacency (self-loops only; no propagation)")

    rows = []
    for k in KS:
        for subseed in SUBSEEDS:
            row = run_indist(k, subseed, adj)
            rows.append({"family": "indist", "adj_kind": args.adj, **row})
            print(f"{args.adj}gnn indist k={k} s={subseed}: loc={row['loc_acc']:.3f} "
                  f"loc3={row['loc3']:.3f} auc={row['auc']:.3f}")

    cross_rows = []
    for k in XKS:
        for train_p, test_p in TRIOS:
            row = run_cross(k, train_p, test_p, adj)
            cross_rows.append({"adj_kind": args.adj, **row})
            print(f"{args.adj}gnn cross k={k} tr={train_p} te={test_p}: "
                  f"A_loc={row['A_loc_acc']:.3f} B_loc={row['B_loc_acc']:.3f}")

    fn = "data/results/random_graph_gcn.csv"
    flds = ["family", "adj_kind", "graph_seed", "k", "subseed",
            "train_placements", "test_placement", "n_params", "bal_acc",
            "f1", "auc", "loc_acc", "loc3", "loc5",
            "A_bal_acc", "A_f1", "A_auc", "A_loc_acc", "A_loc3", "A_loc5",
            "B_bal_acc", "B_f1", "B_auc", "B_loc_acc", "B_loc3", "B_loc5"]
    existing = []
    if os.path.exists(fn):
        with open(fn) as f:
            import csv as _csv
            for r in _csv.DictReader(f):
                if not r.get("adj_kind"):
                    r["adj_kind"] = "random"  # rows from the pre-adj_kind run
                r.setdefault("graph_seed", "2026" if r["adj_kind"] == "random" else "")
                if r.get("adj_kind") != args.adj:
                    existing.append(r)
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in existing:
            w.writerow({kk: r.get(kk, "") for kk in flds})
        for r in rows + cross_rows:
            w.writerow({kk: (json.dumps(r[kk]) if isinstance(r.get(kk), (list, tuple)) else r.get(kk))
                        for kk in flds})
    print(f"\nSaved {len(rows) + len(cross_rows)} new rows ({args.adj}) -> {fn}")
    if args.adj == "random":
        np.save("data/results/random_graph_adjacency.npy", adj)


if __name__ == "__main__":
    main()
