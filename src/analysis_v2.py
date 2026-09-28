"""Analysis v2: all tables, figures, and statistics for the paper.

Generates:
  paper/figures/*.pdf   (vector figures, colorblind-safe Okabe-Ito palette)
  paper/tables/*.tex    (LaTeX table bodies, \\input-ready)
  paper/stats.json      (every inline number with mean/SD/CI for the text)

Statistical reporting policy:
  - mean +/- SD over the six placement/split replicates, PLUS 95% t-CIs
  - AUC now reported with SD/CI like every other metric
  - independent experimental units (scenarios, placements) stated explicitly
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

FIG = "paper/figures"
TAB = "paper/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Okabe-Ito colorblind-safe palette
C = {"cnn": "#0072B2", "lstm": "#E69F00", "gru": "#009E73",
     "transformer": "#CC79A7", "gnn": "#D55E00", "set": "#56B4E9",
     "random": "#999999"}
ARCH_LABEL = {"cnn": "CNN", "lstm": "LSTM", "gru": "GRU",
              "transformer": "Transformer", "gnn": "GCN"}
ARCHS = ["cnn", "lstm", "gru", "transformer"]
KS = [9, 7, 5, 3]

stats_out = {}


def msci(vals, conf=0.95):
    """mean, sd, half-width of 95% t-CI over the independent replicates."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(vals.mean())
    s = float(vals.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = float(sps.t.ppf(0.5 + conf / 2, n - 1) * s / np.sqrt(n))
    else:
        h = float("nan")
    return m, s, h


def fmt(m, s=None, ci=None, prec=3):
    out = f"{m:.{prec}f}"
    if s is not None:
        out += f" \\pm {s:.{prec}f}"
    if ci is not None and not np.isnan(ci):
        out += f" [{m-ci:.{prec}f}, {m+ci:.{prec}f}]"
    return out


def save_table(name, body, header=None):
    with open(os.path.join(TAB, name), "w") as f:
        if header:
            f.write(header)
        f.write(body)
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {TAB}/{name}")


# ============================================================
# 1j. Paired replicate-level architecture comparison
# ============================================================
def paired_arch_comparison():
    """Paired comparison over the six shared replicates (same subseed =
    same placement AND same split), replacing invalid 'overlapping CIs'
    reasoning. Reports the paired difference with a 95% t-CI."""
    df = pd.read_csv("data/results/benchmark.csv")
    stats_out.setdefault("paired", {})
    for a, b, k, metric in [("gru", "cnn", 3, "loc_acc"),
                            ("cnn", "lstm", 3, "auc"),
                            ("extratrees", "cnn", None, "auc")]:
        if a == "extratrees":
            continue
        pa = df[(df.arch == a) & (df.k == k)].sort_values("subseed")[metric].values
        pb = df[(df.arch == b) & (df.k == k)].sort_values("subseed")[metric].values
        d = pa - pb
        m, s, ci = msci(d)
        stats_out["paired"][f"{a}_minus_{b}_k{k}_{metric}"] = {
            "mean": m, "sd": s, "ci95": ci,
            "ci_excludes_zero": bool(m - ci > 0 or m + ci < 0)}
    # Extra Trees vs CNN, density-averaged, paired by (subseed)
    et = pd.read_csv("data/results/extra_baselines.csv")
    et_k = et[et.method == "extratrees"].pivot_table(index="subseed", columns="k",
                                                     values="auc")
    cnn_k = df[df.arch == "cnn"].pivot_table(index="subseed", columns="k", values="auc")
    d = et_k.mean(axis=1).values - cnn_k.mean(axis=1).values
    m, s, ci = msci(d)
    stats_out["paired"]["extratrees_minus_cnn_densityavg_auc"] = {
        "mean": m, "sd": s, "ci95": ci,
        "ci_excludes_zero": bool(m - ci > 0 or m + ci < 0)}


# ============================================================
# 1. Dataset overview
# ============================================================
def dataset_overview():
    X = np.load("data/X.npy")
    yd = np.load("data/y_det.npy")
    yl = np.load("data/y_loc.npy")
    with open("data/window_meta.json") as f:
        meta = json.load(f)
    n_leak_scen = sum(1 for s in meta["schedules"] if s["type"] == "leak")
    n_leak_win = int((yd == 1).sum())
    assert n_leak_win == 461, f"leak window count changed: {n_leak_win}"
    body = r"""\begin{tabular}{lr}
\toprule
\textbf{Level} & \textbf{Number} \\
\midrule
Normal scenarios & 40 \\
Leak scenarios & 27 \\
Leak locations (distinct nodes) & 9 \\
Leak sizes (orifice diameters) & 3 \\
\midrule
Total scenarios (independent hydraulic units) & 67 \\
Windows (48 steps, stride 8; overlapping) & 2{,}077 \\
\quad no-leak windows & 1{,}616 \\
\quad leak windows & 461 \\
Windows per scenario & 31 \\
Sensor placements per density (independently sampled subsets) & 6 \\
Sensor densities $k$ & 4 \\
DL training runs in the master benchmark & 96 \\"""
    save_table("tab_dataset.tex", body)
    stats_out["dataset"] = {
        "n_scenarios": 67, "n_normal": 40, "n_leak": 27,
        "n_windows": int(X.shape[0]), "n_leak_windows": n_leak_win,
        "windows_per_scenario": int(X.shape[0] // 67)}


# ============================================================
# 1b. Run-count audit (one verified number for the whole paper)
# ============================================================
def run_count_audit():
    """Count exact DL training runs and classical model fits across all
    released result CSVs. Derived/analysis-only CSVs are excluded."""
    import csv as _csv

    def nrows(fn):
        with open(fn) as f:
            return sum(1 for _ in _csv.DictReader(f))

    dl_runs = {
        "master benchmark": nrows("data/results/benchmark.csv"),
        "identity set-encoder benchmark": nrows("data/results/set_identity_benchmark.csv"),
        "GCN benchmark": nrows("data/results/gnn_benchmark.csv"),
        "cross-placement std": nrows("data/results/cross_sensor_v2_std.csv"),
        "cross-placement set": nrows("data/results/cross_sensor_v2_set.csv"),
        "cross-placement GCN": nrows("data/results/cross_sensor_v2_gnn.csv"),
        "demand-noise sweep": nrows("data/results/demand_noise.csv"),
        "window labeling": nrows("data/results/window_labeling.csv"),
        "two-stage (2 models/row)": 2 * nrows("data/results/two_stage.csv"),
        "variance decomposition": nrows("data/results/variance_decomp.csv"),
        "measurement-noise (trained models)": len(pd.read_csv("data/results/sensor_noise.csv")
                                                  [["arch", "k", "subseed", "train_alpha"]]
                                                  .drop_duplicates()),
    }
    try:
        dl_runs["channel permutation"] = len(pd.read_csv("data/results/channel_permutation.csv")
                                             [["arch", "k", "subseed"]].drop_duplicates())
    except Exception:
        pass
    try:
        rg = pd.read_csv("data/results/random_graph_gcn.csv")
        dl_runs["random-graph GCN (single)"] = int(len(rg[rg.family != "indist"]) +
                                                   len(rg[(rg.family == "indist") &
                                                          (rg.adj_kind != "random")]))
        dl_runs["multi-rewiring GCN"] = int(len(rg[(rg.family == "indist") &
                                                   (rg.adj_kind == "random") &
                                                   (rg.graph_seed != 2026)]))
    except Exception:
        pass
    try:
        dl_runs["many-trios transfer sweep"] = len(pd.read_csv("data/results/cross_many_trios.csv")
                                                   [["model", "k", "trio"]].drop_duplicates())
    except Exception:
        pass
    try:
        dl_runs["transplant CNN retrains"] = 12
    except Exception:
        pass
    try:
        to = pd.read_csv("data/results/time_only.csv")
        dl_runs["time-only baselines"] = 2 * len(to)
    except Exception:
        pass
    classical_fits = {
        "extra baselines": nrows("data/results/extra_baselines.csv"),
        "classical baseline (xgb/isoforest)": nrows("data/results/classical_baseline.csv"),
    }
    try:
        # v2 activation scheme: 6 CNN k=3 detectors + 24 nested-subset CNN
        # localizers (m in {3,5,7,9}; m=9 retrains use master seeds)
        dl_runs["activation scheme"] = 30
        classical_fits["activation ET detectors"] = 6
    except Exception:
        pass
    # (data/results/classical.csv and cross_sensor.csv are superseded first-pass
    # sweeps that feed no table or figure in the paper; they are preserved in
    # the repository but excluded from the audit.)
    total_dl = int(sum(dl_runs.values()))
    total_classical = int(sum(classical_fits.values()))
    stats_out["run_audit"] = {"dl_training_runs": dl_runs, "total_dl": total_dl,
                              "classical_fits": classical_fits,
                              "total_classical": total_classical,
                              "grand_total": total_dl + total_classical}
    print("run audit:", json.dumps(stats_out["run_audit"], indent=1))


# ============================================================
# 1c. Bootstrap CI sensitivity (robustness check)
# ============================================================
def bootstrap_ci_analysis():
    """Percentile bootstrap 95% CIs over the six replicates of every master
    benchmark cell; compared against the Student-t CIs. With n=6 the t-CI is
    the headline interval; the bootstrap is a distribution-free sensitivity
    check. Agreement is judged against the chance reference of each metric
    (1/9 for localization metrics, 0.5 for detection metrics): the two
    interval types agree when both include or both exclude that reference."""
    rng = np.random.default_rng(2026)
    df = pd.read_csv("data/results/benchmark.csv")
    stats_out.setdefault("bootstrap", {})
    n_disagree = 0
    n_cells = 0
    disagreements = []
    widths_t, widths_b = [], []
    for arch in ARCHS:
        for k in KS:
            d = df[(df.arch == arch) & (df.k == k)]
            for col in ["bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]:
                vals = d[col].dropna().values
                if len(vals) < 2:
                    continue
                m, s, ci_t = msci(vals)
                B = 10000
                idx = rng.integers(0, len(vals), size=(B, len(vals)))
                means = vals[idx].mean(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                ref = 1.0 / 9.0 if col.startswith("loc") else 0.5
                t_excl = (m - ci_t > ref) or (m + ci_t < ref)
                b_excl = (lo > ref) or (hi < ref)
                n_cells += 1
                if t_excl != b_excl:
                    n_disagree += 1
                    disagreements.append(f"{arch}_k{k}_{col}")
                widths_t.append(2 * ci_t)
                widths_b.append(hi - lo)
                stats_out["bootstrap"][f"{arch}_k{k}_{col}"] = {
                    "mean": m, "t_ci": [m - ci_t, m + ci_t],
                    "boot_ci": [float(lo), float(hi)]}
    stats_out["bootstrap"]["summary"] = {
        "n_cells": n_cells,
        "n_chance_reference_disagreements": n_disagree,
        "disagreements": disagreements,
        "mean_t_width": float(np.mean(widths_t)),
        "mean_boot_width": float(np.mean(widths_b))}
    print("bootstrap:", stats_out["bootstrap"]["summary"])


# ============================================================
# 1d. Channel permutation analysis
# ============================================================
def channel_permutation_analysis():
    df = pd.read_csv("data/results/channel_permutation.csv")
    stats_out.setdefault("permutation", {})
    ident = df[df.perm == "identity"]
    perm = df[df.perm != "identity"]
    body = ""
    for arch in ARCHS:
        for k in KS:
            di = ident[(ident.arch == arch) & (ident.k == k)]
            dp = perm[(perm.arch == arch) & (perm.k == k)]
            cells = []
            for dfr, col in [(di, "loc_acc"), (dp, "loc_acc"), (dp, "loc3"),
                             (di, "auc"), (dp, "auc")]:
                m, s, ci = msci(dfr[col])
                cells.append(f"${m:.3f} \\pm {s:.3f}$")
                stats_out["permutation"][f"{arch}_k{k}_{col}_{'id' if dfr is di else 'perm'}"] = {
                    "mean": m, "sd": s, "ci95": ci}
            if k == KS[0]:
                label = "\\multirow{4}{*}{" + ARCH_LABEL[arch] + "}"
            else:
                label = ""
            body += (label + f" & {k} & " + " & ".join(cells) + " \\\\\n")
            if k == KS[-1]:
                body += "\\addlinespace\n"
    header = r"""\begin{tabular}{lcccccc}
\toprule
 & & \multicolumn{3}{c}{\textbf{Localization}} & \multicolumn{2}{c}{\textbf{Detection AUC}} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
\textbf{Architecture} & $k$ & \textbf{Loc@1 (identity)} & \textbf{Loc@1 (permuted)} & \textbf{Loc@3 (permuted)} & \textbf{identity} & \textbf{permuted} \\
\midrule
\small
"""
    save_table("tab_permutation.tex", body, header)

    # summary + figure
    rel = []
    for arch in ARCHS:
        for k in KS:
            mi = ident[(ident.arch == arch) & (ident.k == k)]["loc_acc"].mean()
            mp = perm[(perm.arch == arch) & (perm.k == k)]["loc_acc"].mean()
            rel.append((arch, k, mi, mp))
    stats_out["permutation"]["mean_id_loc1"] = float(np.mean([r[2] for r in rel]))
    stats_out["permutation"]["mean_perm_loc1"] = float(np.mean([r[3] for r in rel]))
    stats_out["permutation"]["mean_id_auc"] = float(ident["auc"].mean())
    stats_out["permutation"]["mean_perm_auc"] = float(perm["auc"].mean())
    # reference: Experiment B mean across channel encoders
    try:
        dx = pd.read_csv("data/results/cross_sensor_v2_std.csv")
        stats_out["permutation"]["mean_expB_loc1"] = float(dx["B_loc_acc"].mean())
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, name in zip(axes, ["loc_acc", "auc"], ["Loc@1", "AUC"]):
        x = np.arange(len(KS))
        w = 0.36
        mi = [ident[ident.k == k][col].mean() for k in KS]
        mp = [perm[perm.k == k][col].mean() for k in KS]
        ax.bar(x - w / 2, mi, w, label="identity (in-distribution)", color="#0072B2", alpha=.85)
        ax.bar(x + w / 2, mp, w, label="permuted channels", color="#D55E00", alpha=.85)
        if col == "loc_acc":
            try:
                dx = pd.read_csv("data/results/cross_sensor_v2_std.csv")
                bvals = [dx[dx.k == k]["B_loc_acc"].mean() for k in [7, 5, 3]]
                ax.bar(np.arange(1, 4) + w / 2, bvals, 0.18,
                       label="Experiment B (unseen placement)", color="#999999", alpha=.9)
            except Exception:
                pass
            ax.axhline(1 / 9, ls="--", c="#999999", lw=1)
            ax.text(3.4, 1 / 9 + 0.012, "chance", fontsize=7, color="#666666", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$k$={k}" for k in KS], fontsize=8)
        ax.set_ylabel(name)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_permutation.pdf"))
    plt.close(fig)
    print("wrote fig_permutation.pdf")


# ============================================================
# 1e. Random-graph GCN ablation
# ============================================================
def random_graph_analysis():
    dr = pd.read_csv("data/results/random_graph_gcn.csv")
    dg = pd.read_csv("data/results/gnn_benchmark.csv")
    dxr = pd.read_csv("data/results/cross_sensor_v2_gnn.csv")
    stats_out.setdefault("random_graph", {})

    def indist(name, k):
        if name == "real":
            return dg[dg.k == k]
        d = dr[(dr.adj_kind == name) & (dr.family == "indist") & (dr.k == k)]
        return d

    # randomized adjacency: aggregate over the 5 rewired graphs
    n_graphs = dr[(dr.adj_kind == "random")]["graph_seed"].nunique()
    per_graph = {}
    for k in KS:
        d = dr[(dr.adj_kind == "random") & (dr.family == "indist") & (dr.k == k)]
        per_graph[k] = d.groupby("graph_seed").loc_acc.mean().to_dict()
    stats_out["random_graph"]["n_rewired_graphs"] = int(n_graphs)
    stats_out["random_graph"]["per_graph_loc1"] = {
        str(k): {str(g): float(v) for g, v in per_graph[k].items()} for k in KS}

    labels = {"real": "GCN, real Net1 adjacency",
              "random": "GCN, randomized adjacency (degree-preserving)",
              "identity": "GCN, identity adjacency (no propagation)"}
    body = ""
    for name in ["real", "random", "identity"]:
        for k in KS:
            d = indist(name, k)
            m1, s1, _ = msci(d["loc_acc"])
            m3, s3, _ = msci(d["loc3"])
            mA, sA, _ = msci(d["auc"])
            stats_out["random_graph"][f"{name}_k{k}"] = {
                "loc1": m1, "loc1_sd": s1, "loc3": m3, "auc": mA, "auc_sd": sA}
            cells = [f"${m1:.3f} \\pm {s1:.3f}$", f"${m3:.3f} \\pm {s3:.3f}$",
                     f"${mA:.3f} \\pm {sA:.3f}$"]
            if k == KS[0]:
                body += f"\\multirow{{4}}{{*}}{{{labels[name]}}} & {k} & " + \
                        " & ".join(cells) + " \\\\\n"
            else:
                body += f"& {k} & " + " & ".join(cells) + " \\\\\n"
        body += "\\addlinespace\n"
    header = r"""\begin{tabular}{lcccc}
\toprule
\textbf{Adjacency} & $k$ & \textbf{Loc@1 (in-dist.)} & \textbf{Loc@3 (in-dist.)} & \textbf{AUC (in-dist.)} \\
\midrule
\small
"""
    save_table("tab_randomgraph.tex", body, header)


# ============================================================
# 1g. Many-trios transfer sweep
# ============================================================
def many_trios_analysis():
    df = pd.read_csv("data/results/cross_many_trios.csv")
    stats_out.setdefault("many_trios", {})
    chance = 1.0 / 9.0
    models_list = [("cnn", "CNN (channel-based)"),
                   ("gnn_real", "GCN, real adjacency"),
                   ("gnn_ident", "GCN, identity adjacency"),
                   ("gnn_rand", "GCN, randomized adjacency")]
    piv = df.pivot_table(index=["k", "trio"], columns="model", values="B_loc_acc")

    body = ""
    for key, label in models_list:
        for k in [7, 5, 3]:
            d = df[(df.model == key) & (df.k == k)]
            m, s, ci = msci(d["B_loc_acc"])
            m3, s3, _ = msci(d["B_loc3"])
            above = "^{\\dagger}" if m - ci > chance else ""
            cells = [f"${m:.3f} \\pm {s:.3f}{above}$", f"${m3:.3f} \\pm {s3:.3f}$"]
            if key != "cnn":
                dd = (piv.loc[k][key] - piv.loc[k]["cnn"]).values
                dm, ds, dci = msci(dd)
                cells.append(f"${dm:+.3f}$ [{dm-dci:+.3f}, {dm+dci:+.3f}]"
                             + (r"$^{*}$" if dm - dci > 0 else ""))
                stats_out["many_trios"][f"{key}_k{k}"] = {
                    "loc1": m, "loc1_sd": s, "loc1_ci": ci,
                    "loc3": m3, "ci_excludes_chance": bool(m - ci > chance),
                    "paired_vs_cnn": dm, "paired_ci": [dm - dci, dm + dci],
                    "paired_significant": bool(dm - dci > 0)}
            else:
                cells.append("--")
                stats_out["many_trios"][f"cnn_k{k}"] = {
                    "loc1": m, "loc1_sd": s, "loc1_ci": ci, "loc3": m3,
                    "ci_excludes_chance": bool(m - ci > chance)}
            row_label = label if k == 7 else ""
            body += f"{row_label} & {k} & " + " & ".join(cells) + " \\\\\n"
        body += "\\addlinespace\n"
    header = r"""\begin{tabular}{llccc}
\toprule
\textbf{Model} & $k$ & \textbf{Exp.~B Loc@1} & \textbf{Exp.~B Loc@3} & \textbf{Paired $\Delta$Loc@1 vs.\ CNN} \\
\midrule
\small
"""
    save_table("tab_manytrios.tex", body, header)

    # figure: box plot per model x k + paired differences
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    positions, data = [], []
    x = 0
    for key, label in models_list:
        for k in [7, 5, 3]:
            data.append(df[(df.model == key) & (df.k == k)]["B_loc_acc"].values)
            positions.append(x)
            x += 1
    bp = ax.boxplot(data, positions=positions, widths=0.7, showmeans=True,
                    patch_artist=True, medianprops=dict(color="black"))
    cols = {"cnn": "#999999", "gnn_real": "#D55E00",
            "gnn_ident": "#0072B2", "gnn_rand": "#009E73"}
    for patch, key in zip(bp["boxes"], [k for k, _ in models_list for _ in range(3)]):
        patch.set_facecolor(cols[key])
        patch.set_alpha(0.6)
    ax.axhline(1 / 9, ls="--", c="#999999", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for _ in models_list for k in [7, 5, 3]], fontsize=7)
    ax.set_xlabel("density $k$ (groups: CNN, real, identity, random)")
    ax.set_ylabel("Exp. B Loc@1 (15 trios)")
    ax = axes[1]
    for i, (key, label) in enumerate(models_list[1:]):
        for k in [7, 5, 3]:
            dd = (piv.loc[k][key] - piv.loc[k]["cnn"]).values
            xx = i * 3 + [7, 5, 3].index(k) + 1
            ax.scatter([xx] * len(dd), dd, s=12, alpha=0.6, color=cols[key])
            dm, ds, dci = msci(dd)
            ax.errorbar([xx], [dm], yerr=[[dci], [dci]], fmt="_",
                        color="black", capsize=3, markersize=12, lw=1.5)
    ax.axhline(0, ls="-", c="#999999", lw=0.8)
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels(["7", "5", "3"] * 3, fontsize=8)
    ax.set_xlabel("paired $\\Delta$Loc@1 vs CNN (real | identity | random), by $k$")
    ax.set_ylabel("$\\Delta$Loc@1 (GCN $-$ CNN)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_manytrios.pdf"))
    plt.close(fig)
    print("wrote fig_manytrios.pdf")

    stats_out["many_trios"]["n_trios"] = int(df.trio.nunique())
    stats_out["many_trios"]["cnn_loc1_sd_max"] = float(
        max(df[(df.model == "cnn") & (df.k == kk)]["B_loc_acc"].std(ddof=1)
            for kk in [7, 5, 3]))


# ============================================================
# 1h. Demand-leak transplant intervention
# ============================================================
def transplant_analysis():
    stats_out.setdefault("transplant", {})
    body = ""
    for K in [5, 3]:
        df = pd.read_csv(f"data/results/demand_transplant_k{K}.csv")

        def sel(kind, place):
            if kind == "original":
                return df[df.condition == f"original_{place}"]
            return df[df.condition.str.startswith("cf") & df.condition.str.endswith(place)]

        conds = [("original", "indist", "Original scenario, training placement"),
                 ("original", "unseen", "Original scenario, unseen placement"),
                 ("cf", "indist", "Counterfactual (moved leak), training placement"),
                 ("cf", "unseen", "Counterfactual (moved leak), unseen placement")]
        first = True
        for kind, place, label in conds:
            d = sel(kind, place)
            w = d.n_leak_windows
            p_orig = float((d.p_orig * w).sum() / w.sum())
            p_true = float((d.p_true * w).sum() / w.sum())
            p_other = float((d.p_other * w).sum() / w.sum())
            auc = float(d.auc.mean())
            stats_out["transplant"][f"k{K}_{kind}_{place}"] = {
                "p_orig": p_orig, "p_true": p_true, "p_other": p_other, "auc": auc}
            tag = f"\\multirow{{4}}{{*}}{{$k$={K}}}" if first else ""
            first = False
            body += (f"{tag} & {label} & ${p_orig:.3f}$ & ${p_true:.3f}$ & "
                     f"${p_other:.3f}$ & ${auc:.3f}$ \\\\\n")
        body += "\\addlinespace\n"
    header = r"""\begin{tabular}{llcccc}
\toprule
 & \textbf{Condition} & $\mathbf{P(\hat{y}{=}n)}$ & $\mathbf{P(\hat{y}{=}m)}$ & $\mathbf{P(\text{other})}$ & \textbf{AUC} \\
\midrule
\small
"""
    save_table("tab_transplant.tex", body, header)


# ============================================================
# 1i. Time-only confound baseline
# ============================================================
def time_only_analysis():
    df = pd.read_csv("data/results/time_only.csv")
    stats_out.setdefault("time_only", {})
    m, s, ci = msci(df["logreg_auc"])
    m2, s2, _ = msci(df["mlp_auc"])
    ml, sl, _ = msci(df["logreg_loc1"])
    stats_out["time_only"] = {"logreg_auc": {"mean": m, "sd": s, "ci95": ci},
                              "mlp_auc": {"mean": m2, "sd": s2},
                              "logreg_loc1": {"mean": ml, "sd": sl}}
    try:
        with open("data/results/time_only_profile.json") as f:
            stats_out["time_only"]["profile"] = json.load(f)
    except Exception:
        pass
    body = (f"Time features only (logistic regression) & ${m:.3f} \\pm {s:.3f}$ & "
            f"${ml:.3f} \\pm {sl:.3f}$ \\\\\n"
            f"Time features only (small MLP) & ${m2:.3f} \\pm {s2:.3f}$ & -- \\\\\n")
    header = r"""\begin{tabular}{lcc}
\toprule
\textbf{Detector input} & \textbf{Detection AUC} & \textbf{Loc@1} \\
\midrule
\small
"""
    save_table("tab_timeonly.tex", body, header)


# ============================================================
# 1f. Demand-only memorization analysis
# ============================================================
def demand_only_analysis():
    df = pd.read_csv("data/results/demand_only.csv")
    stats_out.setdefault("demand_only", {})
    body = ""
    for model, label in [("logreg", "Logistic regression (linear)"),
                         ("mlp", "Small MLP (flexible)"),
                         ("knn", "1-NN (LOO; pure memorization)")]:
        ms, mh, ss, sh = [], [], [], []
        for cond in ["seen", "held"]:
            m, s, ci = msci(df[f"{model}_{cond}_loc1"])
            if cond == "seen":
                ms, ss = m, s
            else:
                mh, sh = m, s
            stats_out["demand_only"][f"{model}_{cond}"] = {"mean": m, "sd": s, "ci95": ci}
        body += (f"{label} & ${ms:.3f} \\pm {ss:.3f}$ & ${mh:.3f} \\pm {sh:.3f}$ \\\\\n")
    header = r"""\begin{tabular}{lcc}
\toprule
\textbf{Fingerprint model} & \textbf{Seen scenarios} & \textbf{Held-out scenarios} \\
\midrule
\small
"""
    save_table("tab_demandonly.tex", body, header)


# ============================================================
# 2. Master benchmark with CIs
# ============================================================
def master_table():
    df = pd.read_csv("data/results/benchmark.csv")
    rows = []
    for arch in ARCHS:
        for k in KS:
            d = df[(df.arch == arch) & (df.k == k)]
            cells = {}
            for col in ["bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]:
                m, s, ci = msci(d[col])
                cells[col] = (m, s, ci)
            rows.append((arch, k, d.n_params.iloc[0], cells))
            stats_out.setdefault("master", {})[f"{arch}_k{k}"] = {
                c: {"mean": v[0], "sd": v[1], "ci95": v[2]}
                for c, v in cells.items()}

    best = {}  # best per column for bolding
    for col in ["bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]:
        for k in KS:
            vals = {a: [r for r in rows if r[0] == a and r[1] == k][0][3][col][0]
                     for a in ARCHS}
            best[(col, k)] = max(vals, key=vals.get)

    body = ""
    nl = "\n"
    for arch in ARCHS:
        n_par = None
        for arch2, k, params, cells in rows:
            if arch2 != arch:
                continue
            n_par = params
            bold = {c: best.get((c, k)) == arch for c in
                    ["bal_acc", "f1", "auc", "loc_acc"]}
            def cell(col, b=True):
                m, s, ci = cells[col]
                txt = f"${m:.3f} \\pm {s:.3f}$"
                if b and bold.get(col):
                    txt = f"\\textbf{{{txt}}}"
                return txt
            if k == KS[0]:
                label = "\\multirow{4}{*}{" + f"{ARCH_LABEL[arch]} ({n_par/1e3:.1f}k)" + "}"
            else:
                label = ""
            body += (f"{label} & {k} & {cell('bal_acc')} & {cell('f1')} & {cell('auc')} & "
                     f"{cell('loc_acc')} & {cell('loc3')} & {cell('loc5')} \\\\{nl}")
            if k == KS[-1]:
                body += "\\addlinespace\n"
    header = r"""\begin{tabular}{lccccccc}
\toprule
\textbf{Architecture} & $k$ & \textbf{Bal.\,Acc.\,$\uparrow$} & \textbf{F1 $\uparrow$} & \textbf{AUC $\uparrow$} & \textbf{Loc@1 $\uparrow$} & \textbf{Loc@3 $\uparrow$} & \textbf{Loc@5 $\uparrow$} \\
\midrule
"""
    save_table("tab_master.tex", body, header + "\\small\n")

    # ---- degradation figure (now with AUC bands) ----
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    metrics = [("bal_acc", "Balanced accuracy"), ("f1", "F1 (leak class)"),
               ("auc", "AUC"), ("loc_acc", "Top-1 localization")]
    for ax, (col, title) in zip(axes.flat, metrics):
        for arch in ARCHS:
            ms, sds = [], []
            for k in KS:
                d = df[(df.arch == arch) & (df.k == k)]
                m, s, _ = msci(d[col])
                ms.append(m); sds.append(s)
            ax.errorbar(KS, ms, yerr=sds, marker="o", capsize=3,
                        color=C[arch], label=ARCH_LABEL[arch])
        ax.set_xlabel("sensor count $k$")
        ax.set_title(title)
        ax.set_xticks(KS)
        ax.invert_xaxis()
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_degradation.pdf"))
    plt.close(fig)
    print("wrote fig_degradation.pdf")

    # ---- params vs performance frontier ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, k, title in zip(axes, [9, 3], [f"Loc@1 at $k$={9}", f"Loc@1 at $k$={3}"]):
        for arch in ARCHS:
            d = df[(df.arch == arch) & (df.k == k)]
            m, s, _ = msci(d["loc_acc"])
            p = d.n_params.iloc[0]
            ax.errorbar([p], [m], yerr=[s], marker="o", capsize=3,
                        color=C[arch], label=ARCH_LABEL[arch])
        try:
            dg = pd.read_csv("data/results/gnn_benchmark.csv")
            d = dg[(dg.k == k)]
            m, s, _ = msci(d["loc_acc"])
            ax.errorbar([d.n_params.iloc[0]], [m], yerr=[s], marker="s", capsize=3,
                        color=C["gnn"], label=ARCH_LABEL["gnn"])
        except Exception:
            pass
        ax.set_xscale("log")
        ax.set_xlabel("parameters")
        ax.set_ylabel("Loc@1")
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_params_frontier.pdf"))
    plt.close(fig)
    print("wrote fig_params_frontier.pdf")


# ============================================================
# 3. Leak-size table (all 6 placements)
# ============================================================
def leaksize_table():
    df = pd.read_csv("data/results/leak_size.csv")
    stats_out.setdefault("leak_size", {})
    body = ""
    for metric, name in [("bal_acc", "Balanced accuracy"), ("f1", "F1 (leak class)"),
                         ("auc", "AUC")]:
        body += f"\\multicolumn{{5}}{{l}}{{\\emph{{{name}}}}} \\\\\n"
        for arch in ARCHS:
            cells = []
            for size in ["small", "medium", "large"]:
                d = df[(df.arch == arch) & (df["size"] == size)]
                m, s, ci = msci(d[metric])
                cells.append(f"${m:.3f} \\pm {s:.3f}$")
                stats_out["leak_size"][f"{arch}_{size}_{metric}"] = {
                    "mean": m, "sd": s, "ci95": ci, "n": len(d)}
            if metric == "f1":
                small_f1 = df[(df.arch == arch) & (df["size"] == "small")]["f1"]
                large_f1 = df[(df.arch == arch) & (df["size"] == "large")]["f1"]
                mS, _, _ = msci(small_f1)
                mL, _, _ = msci(large_f1)
                cells.append(f"{mL - mS:+.3f}")
            else:
                cells.append("--")
            body += f"{ARCH_LABEL[arch]} & " + " & ".join(cells) + " \\\\\n"
        body += "\\addlinespace\n"
    header = r"""\begin{tabular}{lcccc}
\toprule
\textbf{Architecture} & \textbf{Small ($d{=}20$mm)} & \textbf{Medium ($d{=}30$mm)} & \textbf{Large ($d{=}50$mm)} & \textbf{F1 gap} \\
\midrule
"""
    save_table("tab_leaksize.tex", body, header)

    # localization by size
    stats_out["leak_size"]["loc"] = {}
    for arch in ARCHS:
        for size in ["small", "medium", "large"]:
            d = df[(df.arch == arch) & (df["size"] == size)]
            m, s, ci = msci(d["loc_acc"])
            stats_out["leak_size"]["loc"][f"{arch}_{size}"] = {
                "mean": m, "sd": s, "ci95": ci}


# ============================================================
# 4. Baselines: classical + simple ML + chance (per-density AUC)
# ============================================================
def baselines_table():
    df = pd.read_csv("data/results/extra_baselines.csv")
    stats_out.setdefault("baselines", {})
    body = ""
    # chance detection rows (placement-invariant by construction); the
    # chance-localization floors (Loc@1/3/5 = 1/9, 1/3, 5/9) are stated in
    # the caption and text rather than tabulated
    for method, label in [("random_det", "Random detection"),
                          ("majority_det", "Majority class (no-leak)")]:
        d = df[df.method == method]
        mB, sB, _ = msci(d["bal_acc"])
        mF, sF, _ = msci(d["f1"])
        mA, sA, _ = msci(d["auc"])
        stats_out["baselines"][method] = {
            "bal_acc": {"mean": mB, "sd": sB}, "f1": {"mean": mF, "sd": sF},
            "auc": {"mean": mA, "sd": sA}}
        body += (f"\\emph{{{label}}} & ${mB:.3f} \\pm {sB:.3f}$ & "
                 f"${mF:.3f} \\pm {sF:.3f}$ & ${mA:.3f} \\pm {sA:.3f}$ & "
                 "\\multicolumn{5}{c}{--} \\\\\n")
    body += "\\midrule\n"
    # chance-localization floors: stats only (constant by construction;
    # tabulating them would misalign the per-density columns)
    for method in ["random_loc", "majority_loc"]:
        d = df[df.method == method]
        stats_out["baselines"][method] = {
            "loc_acc": {"mean": float(d["loc_acc"].mean()),
                        "sd": float(d["loc_acc"].std(ddof=1))},
            "loc3": {"mean": float(d["loc3"].mean()),
                     "sd": float(d["loc3"].std(ddof=1))},
            "loc5": {"mean": float(d["loc5"].mean()),
                     "sd": float(d["loc5"].std(ddof=1))}}
    # supervised / classical baselines: AUC PER DENSITY (matched comparison)
    dfc = None
    try:
        dfc = pd.read_csv("data/results/classical_baseline.csv")
    except Exception:
        pass
    rows_sup = [("logreg", "Logistic regression"),
                ("rf", "Random Forest"),
                ("extratrees", "Extra Trees"),
                ("hgb", "HistGradientBoosting")]
    rows_cl = []
    if dfc is not None:
        rows_cl = [("isforest", "Isolation Forest"), ("xgboost", "XGBoost")]
    for src, method_list in ((df, rows_sup), (dfc, rows_cl)):
        if src is None:
            continue
        for method, label in method_list:
            d = src[src.method == method]
            cells, per_k = [], {}
            for k in KS:
                dk = d[d.k == k]
                m, s, _ = msci(dk["auc"])
                cells.append(f"${m:.3f} \\pm {s:.3f}$")
                per_k[k] = m
            mm, ss, _ = msci(d["auc"])
            stats_out["baselines"][method] = {
                "auc_per_k": {str(k): per_k[k] for k in KS},
                "auc_mean": mm, "auc_mean_sd": ss}
            body += f"{label} & \\multicolumn{{3}}{{c}}{{--}} & " + \
                    " & ".join(cells) + f" & ${mm:.3f}$ \\\\\n"
    # deep-model reference rows: CNN and GRU per-density AUC (matched)
    bench = pd.read_csv("data/results/benchmark.csv")
    for arch in ["cnn", "gru"]:
        cells, per_k = [], {}
        for k in KS:
            d = bench[(bench.arch == arch) & (bench.k == k)]
            m, s, _ = msci(d["auc"])
            cells.append(f"${m:.3f} \\pm {s:.3f}$")
            per_k[k] = m
        # density-averaged AUC per replicate -> matched mean/SD
        pv = bench[bench.arch == arch].pivot_table(index="subseed", columns="k",
                                                   values="auc")
        mm, ss, _ = msci(pv.mean(axis=1))
        stats_out["baselines"][f"{arch}_auc_per_k"] = {str(k): per_k[k] for k in KS}
        stats_out["baselines"][f"{arch}_density_averaged"] = {"mean": mm, "sd": ss}
        body += f"\\emph{{{ARCH_LABEL[arch]} (deep)}} & \\multicolumn{{3}}{{c}}{{--}} & " + \
                " & ".join(cells) + f" & ${mm:.3f}$ \\\\\n"
    header = r"""\begin{tabular}{lcccccccc}
\toprule
 & & & & \multicolumn{4}{c}{\textbf{AUC by density}} & \\
\cmidrule(lr){5-8}
\textbf{Method} & \textbf{Bal.\,Acc.} & \textbf{F1} & \textbf{AUC} & $k{=}9$ & $k{=}7$ & $k{=}5$ & $k{=}3$ & \textbf{mean} \\
\midrule
\small
"""
    save_table("tab_baselines.tex", body, header)


# ============================================================
# 5. Cross-sensor experiments A and B + confusion analysis
# ============================================================
def cross_sensor_analysis():
    df = pd.read_csv("data/results/cross_sensor_v2_std.csv")
    stats_out.setdefault("cross", {})
    # per-arch per-k means over trios
    for exp in ["A", "B"]:
        for arch in ARCHS:
            for k in [7, 5, 3]:
                d = df[(df.arch == arch) & (df.k == k)]
                for col in ["loc_acc", "loc3", "loc5", "auc", "f1", "bal_acc"]:
                    m, s, ci = msci(d[f"{exp}_{col}"])
                    stats_out["cross"][f"{exp}_{arch}_k{k}_{col}"] = {
                        "mean": m, "sd": s, "ci95": ci}

    # ---- money figure: in-dist vs A vs B ----
    bench = pd.read_csv("data/results/benchmark.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, k in zip(axes, [7, 5, 3]):
        x = np.arange(len(ARCHS))
        w = 0.27
        ind = []
        for arch in ARCHS:
            m, _, _ = msci(bench[(bench.arch == arch) & (bench.k == k)]["loc_acc"])
            ind.append(m)
        a_m = [msci(df[(df.arch == a) & (df.k == k)]["A_loc_acc"])[0] for a in ARCHS]
        b_m = [msci(df[(df.arch == a) & (df.k == k)]["B_loc_acc"])[0] for a in ARCHS]
        ax.bar(x - w, ind, w, label="in-distribution", color="#0072B2", alpha=.85)
        ax.bar(x, a_m, w, label="Exp.\\ A (seen scenarios, unseen placement)",
               color="#E69F00", alpha=.85)
        ax.bar(x + w, b_m, w,
               label="Exp.\\ B (unseen scenarios + placement)", color="#D55E00", alpha=.85)
        ax.axhline(1 / 9, ls="--", c="#999999", lw=1)
        ax.text(len(ARCHS) - 0.45, 1 / 9 + 0.01, "chance (1/9)", fontsize=7,
                color="#666666", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHS], fontsize=8)
        ax.set_title(f"$k$={k}")
        ax.set_ylabel("Loc@1")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_cross.pdf"))
    plt.close(fig)
    print("wrote fig_cross.pdf")

    # ---- confusion matrices (Exp B, aggregated over archs and trios, per k) ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, k in zip(axes, [7, 5, 3]):
        cm = np.zeros((9, 9))
        for _, r in df[df.k == k].iterrows():
            cm += np.array(json.loads(r["B_confusion"]))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels([f"n{i+1}" for i in range(9)], fontsize=6)
        ax.set_yticklabels([f"n{i+1}" for i in range(9)], fontsize=6)
        ax.set_xlabel("predicted node")
        ax.set_ylabel("true node")
        ax.set_title(f"$k$={k}")
        for i in range(9):
            for j in range(9):
                if cm[i, j] > 0:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            fontsize=5, color="white" if cm[i, j] > cm.max() / 2 else "black")
        stats_out["cross"][f"confusion_k{k}"] = cm.tolist()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_confusion.pdf"))
    plt.close(fig)
    print("wrote fig_confusion.pdf")

    # ---- predicted-node distribution (Exp B aggregated) ----
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.27
    for offset, k, col in [(-w, 7, "#0072B2"), (0, 5, "#E69F00"), (w, 3, "#D55E00")]:
        hist = np.zeros(9)
        n = 0
        for _, r in df[df.k == k].iterrows():
            hist += np.array(json.loads(r["B_pred_node_hist"]))
            n += r["B_n_leak"]
        hist = hist / max(n, 1)
        ax.bar(np.arange(9) + offset, hist, w, label=f"$k$={k}")
        stats_out["cross"][f"predhist_k{k}"] = hist.tolist()
    ax.axhline(1 / 9, ls="--", c="#999999", lw=1)
    ax.set_xlabel("predicted node index")
    ax.set_ylabel("fraction of leak windows")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_predhist.pdf"))
    plt.close(fig)
    print("wrote fig_predhist.pdf")

    # ---- tables A and B ----
    for exp, fname, caption in [
            ("A", "tab_crossA.tex", "Experiment A"),
            ("B", "tab_crossB.tex", "Experiment B")]:
        body = ""
        for arch in ARCHS:
            for k in [7, 5, 3]:
                d = df[(df.arch == arch) & (df.k == k)]
                cells = []
                for col in ["bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]:
                    m, s, ci = msci(d[f"{exp}_{col}"])
                    cells.append(f"${m:.3f} \\pm {s:.3f}$")
                if k == 7:
                    label = "\\multirow{3}{*}{" + ARCH_LABEL[arch] + "}"
                else:
                    label = ""
                body += (label + " & " + str(k) + " & " + " & ".join(cells) + " \\\\\n")
                if k == 3:
                    body += "\\addlinespace\n"
        header = r"""\begin{tabular}{lccccccc}
\toprule
\textbf{Architecture} & $k$ & \textbf{Bal.\,Acc.\,$\uparrow$} & \textbf{F1 $\uparrow$} & \textbf{AUC $\uparrow$} & \textbf{Loc@1 $\uparrow$} & \textbf{Loc@3 $\uparrow$} & \textbf{Loc@5 $\uparrow$} \\
\midrule
\small
"""
        save_table(fname, body, header)

    # sensor overlap stats
    ov = []
    for _, r in df.iterrows():
        ov.extend(json.loads(r["sensor_overlap"]))
    stats_out["cross"]["sensor_overlap_mean"] = float(np.mean(ov))
    stats_out["cross"]["sensor_overlap_by_k"] = {
        int(k): float(np.mean([np.mean(json.loads(r["sensor_overlap"]))
                               for _, r in df[df.k == k].iterrows()]))
        for k in [7, 5, 3]}


# ============================================================
# 6. Sensor-identity (set) + GNN analysis
# ============================================================
def identity_analysis():
    # in-distribution set-encoder benchmark
    df = pd.read_csv("data/results/set_identity_benchmark.csv")
    stats_out.setdefault("identity", {})
    body = ""
    for arch in ARCHS:
        for k in KS:
            d = df[(df.arch == arch) & (df.k == k)]
            cells = []
            for col in ["bal_acc", "f1", "auc", "loc_acc"]:
                m, s, ci = msci(d[col])
                cells.append(f"${m:.3f} \\pm {s:.3f}$")
                stats_out["identity"][f"set_{arch}_k{k}_{col}"] = {
                    "mean": m, "sd": s, "ci95": ci}
            if k == KS[0]:
                label = "\\multirow{4}{*}{" + ARCH_LABEL[arch] + "}"
            else:
                label = ""
            body += (label + " & " + str(k) + " & " + " & ".join(cells) +
                     f" & {d.n_params.iloc[0]:,} \\\\\n")
            if k == KS[-1]:
                body += "\\addlinespace\n"
    # GNN rows
    dg = pd.read_csv("data/results/gnn_benchmark.csv")
    for k in KS:
        d = dg[dg.k == k]
        cells = []
        for col in ["bal_acc", "f1", "auc", "loc_acc"]:
            m, s, ci = msci(d[col])
            cells.append(f"${m:.3f} \\pm {s:.3f}$")
            stats_out["identity"][f"gnn_k{k}_{col}"] = {"mean": m, "sd": s, "ci95": ci}
        body += ("\\multirow{4}{*}{GCN (8.7k)} & " + str(k) + " & "
                 + " & ".join(cells) + f" & {d.n_params.iloc[0]:,} \\\\\n")
        if k == KS[-1]:
            body += "\\addlinespace\n"
    header = r"""\begin{tabular}{lcccccc}
\toprule
\textbf{Architecture} & $k$ & \textbf{Bal.\,Acc.\,$\uparrow$} & \textbf{F1 $\uparrow$} & \textbf{AUC $\uparrow$} & \textbf{Loc@1 $\uparrow$} & \textbf{Params} \\
\midrule
\small
"""
    save_table("tab_identity.tex", body, header)

    # cross-sensor for set + gnn
    for kind in ["set", "gnn"]:
        try:
            dx = pd.read_csv(f"data/results/cross_sensor_v2_{kind}.csv")
        except Exception:
            print(f"cross_sensor_v2_{kind}.csv missing")
            continue
        for exp in ["A", "B"]:
            rows = dx if kind == "set" else dx
            archs = ARCHS if kind == "set" else ["gnn"]
            for arch in archs:
                for k in [7, 5, 3]:
                    d = dx[(dx.arch == arch) & (dx.k == k)]
                    for col in ["loc_acc", "loc3", "loc5", "auc"]:
                        m, s, ci = msci(d[f"{exp}_{col}"])
                        stats_out["identity"][f"cross_{kind}_{exp}_{arch}_k{k}_{col}"] = {
                            "mean": m, "sd": s, "ci95": ci}

    # identity comparison figure: Loc@1 on unseen placement (Exp B),
    # channel encoders vs set encoder vs GCN (real / identity adjacency)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    try:
        ds = pd.read_csv("data/results/cross_sensor_v2_set.csv")
        dst = pd.read_csv("data/results/cross_sensor_v2_std.csv")
        dgnn = pd.read_csv("data/results/cross_sensor_v2_gnn.csv")
        drg = pd.read_csv("data/results/random_graph_gcn.csv")
        drgi = drg[(drg.adj_kind == "identity") & (drg.family != "indist")]
        x = np.arange(3)
        w = 0.2

        def bars(dfr, archs, off, label, col):
            ms, sds = [], []
            has_arch = "arch" in dfr.columns
            for k in [7, 5, 3]:
                vals = []
                d0 = dfr[dfr.k == k] if not has_arch else None
                for arch in archs:
                    d = dfr[(dfr.arch == arch) & (dfr.k == k)] if has_arch else d0
                    if len(d):
                        vals.append(d["B_loc_acc"].mean())
                ms.append(np.mean(vals) if vals else 0.0)
                sds.append(np.std(vals) / max(np.sqrt(len(vals)), 1) if vals else 0.0)
            ax.bar(x + off, ms, w, yerr=sds, capsize=2, label=label, color=col, alpha=.9)

        bars(dst, ARCHS, -1.5 * w, "channel encoders", "#999999")
        bars(ds, ARCHS, -0.5 * w, "+ node identity (set encoder)", "#009E73")
        bars(dgnn, ["gnn"], 0.5 * w, "GCN (real adjacency)", "#D55E00")
        bars(drgi, ["gnn"], 1.5 * w, "GCN (identity adjacency)", "#0072B2")
        ax.axhline(1 / 9, ls="--", c="#999999", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["$k$=7", "$k$=5", "$k$=3"])
        ax.set_ylabel("Loc@1 on unseen placement (Exp. B)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_identity.pdf"))
        plt.close(fig)
        print("wrote fig_identity.pdf")
    except Exception as e:
        print("identity figure failed:", e)
        plt.close(fig)


# ============================================================
# 7. Two-stage vs multitask
# ============================================================
def two_stage_analysis():
    df = pd.read_csv("data/results/two_stage.csv")
    bench = pd.read_csv("data/results/benchmark.csv")
    stats_out.setdefault("two_stage", {})
    body = ""
    for arch in ARCHS:
        for k in KS:
            d = df[(df.arch == arch) & (df.k == k)]
            b = bench[(bench.arch == arch) & (bench.k == k)]
            mts = []
            for col in ["loc_uncond", "loc_cond"]:
                m, s, ci = msci(d[col])
                mts.append(f"${m:.3f} \\pm {s:.3f}$")
                stats_out["two_stage"][f"{arch}_k{k}_{col}"] = {
                    "mean": m, "sd": s, "ci95": ci}
            mm, ms_, ci = msci(b["loc_acc"])
            mts.append(f"${mm:.3f} \\pm {ms_:.3f}$")
            if k == KS[0]:
                label = "\\multirow{4}{*}{" + ARCH_LABEL[arch] + "}"
            else:
                label = ""
            body += (label + " & " + str(k) + " & " + " & ".join(mts) + " \\\\\n")
            if k == KS[-1]:
                body += "\\addlinespace\n"
    header = r"""\begin{tabular}{lcccc}
\toprule
\textbf{Architecture} & $k$ & \textbf{Two-stage Loc@1 (uncond.) $\uparrow$} & \textbf{Two-stage Loc@1 (cond.) $\uparrow$} & \textbf{Multitask Loc@1 $\uparrow$} \\
\midrule
\small
"""
    save_table("tab_twostage.tex", body, header)


# ============================================================
# 8b. Iteration-2 response analyses (deficit heuristic, overlap regression, Holm policy, variance shares)
# ============================================================
def iteration2_response_analysis():
    # ---------- heuristic baseline ----------
    h = pd.read_csv("data/results/pressure_deficit_heuristic.csv")
    m = pd.read_csv("data/results/cross_many_trios.csv")
    hi = pd.read_csv("data/results/pressure_deficit_heuristic_indist.csv")
    stats_out.setdefault("iteration2", {})
    KS_T = [7, 5, 3]
    for k in KS_T:
        hk = h[h.k == k].sort_values("trio")
        m_, s_, ci_ = msci(hk["B_loc1"])
        stats_out["iteration2"][f"heuristic_k{k}_loc1"] = {"mean": m_, "sd": s_, "ci95": ci_}
        m3, s3, ci3 = msci(hk["B_loc3"])
        stats_out["iteration2"][f"heuristic_k{k}_loc3"] = {"mean": m3, "sd": s3, "ci95": ci3}
        stats_out["iteration2"][f"heuristic_k{k}_shuffle"] = {
            "mean": float(hk["B_loc1_shuffled"].mean()),
            "sd": float(hk["B_loc1_shuffled"].std(ddof=1))}
        stats_out["iteration2"][f"heuristic_k{k}_obs_ceiling"] = {
            "mean": float(hk["frac_leak_observed"].mean())}
        for mod in ["cnn", "gnn_real", "gnn_ident", "gnn_rand"]:
            mk = m[(m.model == mod) & (m.k == k)].sort_values("trio")
            d = hk["B_loc1"].values - mk["B_loc_acc"].values
            md, sd_, hd = msci(d)
            stats_out["iteration2"][f"heuristic_minus_{mod}_k{k}"] = {
                "mean": md, "sd": sd_, "ci95": hd,
                "excludes_zero": bool(abs(md) > hd)}
    for k in [9, 7, 5, 3]:
        m_, s_, ci_ = msci(hi[hi.k == k]["loc1"])
        stats_out["iteration2"][f"heuristic_indist_k{k}_loc1"] = {
            "mean": m_, "sd": s_, "ci95": ci_, "posthoc": True}
    # table
    body = ""
    rows_def = [("Deficit heuristic (no training)", "heuristic"),
                ("Channel CNN", "cnn"), ("GCN (real adjacency)", "gnn_real"),
                ("GCN (identity adjacency)", "gnn_ident"), ("GCN (randomized)", "gnn_rand")]
    for lbl, key in rows_def:
        cells = []
        for k in KS_T:
            if key == "heuristic":
                v = stats_out["iteration2"][f"heuristic_k{k}_loc1"]
            else:
                mk = m[(m.model == key) & (m.k == k)]
                v = {"mean": mk["B_loc_acc"].mean(), "sd": mk["B_loc_acc"].std(ddof=1)}
            cells.append(f"${v['mean']:.3f} \\pm {v['sd']:.3f}$")
        body += f"{lbl} & " + " & ".join(cells) + " \\\\\n"
    body += "\\addlinespace\n"
    obs = " & ".join(f"${stats_out['iteration2'][f'heuristic_k{k}_obs_ceiling']['mean']:.3f}$"
                     for k in KS_T)
    body += f"\\emph{{Observed-leak ceiling (heuristic)}} & {obs} \\\\\n"
    header = r"""\begin{tabular}{lccc}
\toprule
 & \multicolumn{3}{c}{\textbf{Experiment-B Loc@1 (15 trios)}} \\
\cmidrule(lr){2-4}
\textbf{Method} & $k{=}7$ & $k{=}5$ & $k{=}3$ \\
\midrule
\small
"""
    save_table("tab_heuristic.tex", body, header)

    # ---------- overlap slopes ----------
    ov = pd.read_csv("data/results/overlap_transfer.csv")
    for _, r in ov.iterrows():
        stats_out["iteration2"][f"overlap_{r.model}_k{int(r.k)}"] = {
            "slope": r.slope, "ci": [r.slope_ci_lo, r.slope_ci_hi],
            "p": r.slope_p, "spearman": r.spearman}
    pos_sep = int(sum(1 for k_ in ov.itertuples()
                      if k_.slope_ci_lo > 0))
    stats_out["iteration2"]["overlap_positive_separated_count"] = pos_sep

    # ---------- Holm across the 9 GCN-vs-CNN paired contrasts ----------
    from scipy import stats as sps
    contrasts = []
    for mod in ["gnn_real", "gnn_ident", "gnn_rand"]:
        for k in KS_T:
            mk = m[(m.model == mod) & (m.k == k)].sort_values("trio")
            ck = m[(m.model == "cnn") & (m.k == k)].sort_values("trio")
            d = mk["B_loc_acc"].values - ck["B_loc_acc"].values
            t_, p_ = sps.ttest_1samp(d, 0.0)
            contrasts.append({"model": mod, "k": k, "mean": float(d.mean()), "p": float(p_)})
    order_idx = np.argsort([c["p"] for c in contrasts])
    mfam = len(contrasts)
    holm_max = 0.0
    for rank, idx in enumerate(order_idx):
        adj = min(1.0, (mfam - rank) * contrasts[idx]["p"])
        contrasts[idx]["p_holm"] = adj
        holm_max = max(holm_max, adj)
        contrasts[idx]["holm_significant"] = bool(adj < 0.05)
    for c in contrasts:
        stats_out["iteration2"][f"paired_{c['model']}_k{c['k']}"] = {
            "mean": c["mean"], "p": c["p"], "p_holm": c["p_holm"],
            "holm_significant": c["holm_significant"]}
    stats_out["iteration2"]["holm_survivors"] = [
        f"{c['model']}_k{c['k']}" for c in contrasts if c["holm_significant"]]

    # ---------- size x density detection grid ----------
    ls = pd.read_csv("data/results/leak_size.csv")
    stats_out.setdefault("size_density", {})
    body = ""
    for size in ["small", "medium", "large"]:
        cells = []
        for k in KS:
            cnn_v = ls[(ls.arch == "cnn") & (ls["size"] == size) & (ls.k == k)]["auc"]
            all_v = ls[(ls["size"] == size) & (ls.k == k)]["auc"]
            m_, s_, _ = msci(cnn_v)
            rng_lo, rng_hi = f"{all_v.min():.2f}", f"{all_v.max():.2f}"
            cells.append(f"${m_:.3f} \\pm {s_:.3f}$ [{rng_lo}--{rng_hi}]")
            stats_out["size_density"][f"cnn_{size}_k{k}_auc"] = {
                "mean": m_, "sd": s_,
                "min_all_arch": float(all_v.min()), "max_all_arch": float(all_v.max())}
        body += f"\\textbf {{{'S' if size=='small' else 'M' if size=='medium' else 'L'}}} & " + \
                " & ".join(cells) + " \\\\\n"
    header = r"""\begin{tabular}{lcccc}
\toprule
\textbf{Leak size} & \multicolumn{4}{c}{\textbf{Detection AUC} [range over architectures]} \\
\cmidrule(lr){2-5}
 & $k{=}9$ & $k{=}7$ & $k{=}5$ & $k{=}3$ \\
\midrule
\small
"""
    save_table("tab_sizedensity.tex", body, header)

    # ---------- bootstrap CI for the variance-share headline ----------
    # The paper's 74%/84% are ratios of MEAN per-condition variances
    # (sum split / sum of components over the 16 conditions); the bootstrap
    # resamples conditions and recomputes that same ratio-of-means.
    vd = pd.read_csv("data/results/variance_decomp.csv")
    comps = {"auc": [], "loc_acc": []}
    clipped = {"auc": 0, "loc_acc": 0, "total": 0}
    for arch in ARCHS:
        for k in KS:
            cells = vd[(vd.arch == arch) & (vd.k == k)]
            for metric in ["auc", "loc_acc"]:
                vt = float(cells[cells.family == "train"][metric].var(ddof=1))
                vp = max(float(cells[cells.family == "place"][metric].var(ddof=1)) - vt, 0.0)
                vs = max(float(cells[cells.family == "split"][metric].var(ddof=1)) - vt, 0.0)
                comps[metric].append((vt, vp, vs))
                clipped["total"] += 1
                if vp == 0.0 or vs == 0.0:
                    clipped[metric] += 1
    rng = np.random.default_rng(0)
    for metric in ["auc", "loc_acc"]:
        arr = np.array(comps[metric])            # (16, 3)
        n = len(arr)
        bs = []
        for _ in range(10000):
            idx = rng.integers(0, n, n)
            m_ = arr[idx].mean(axis=0)
            tot = m_.sum()
            bs.append(m_[2] / tot if tot > 0 else np.nan)
        lo, hi_ = np.percentile(bs, [2.5, 97.5])
        mean_row = arr.mean(axis=0)
        share = mean_row[2] / mean_row.sum()
        stats_out["iteration2"][f"varshare_split_{metric}"] = {
            "share": float(share), "boot_ci": [float(lo), float(hi_)],
            "n_conditions": int(n)}
        stats_out["iteration2"][f"varshare_place_{metric}"] = {
            "share": float(mean_row[1] / mean_row.sum())}
        stats_out["iteration2"][f"varshare_train_{metric}"] = {
            "share": float(mean_row[0] / mean_row.sum())}
    stats_out["iteration2"]["variance_clipped_cells"] = clipped
    print(f"iteration2: heuristic k3 {stats_out['iteration2']['heuristic_k3_loc1']['mean']:.3f} "
          f"vs cnn (paired {stats_out['iteration2']['heuristic_minus_cnn_k3']['mean']:+.3f}); "
          f"overlap pos-sep {pos_sep}/12; holm survivors "
          f"{stats_out['iteration2']['holm_survivors']}; split share AUC "
          f"{stats_out['iteration2']['varshare_split_auc']['share']:.2f} "
          f"CI {stats_out['iteration2']['varshare_split_auc']['boot_ci']}")


# ============================================================
# 8. Variance decomposition
# ============================================================
def activation_analysis():
    df = pd.read_csv("data/results/activation_scheme.csv")
    stats_out.setdefault("activation", {})
    dets = [("cnn", "CNN", C["cnn"]), ("et", "Extra Trees", C["gru"])]
    T0 = 0.5
    M_ACT = [3, 5, 7, 9]

    # headline cells at tau=0.5, m=9 + paired per-subseed comparisons
    for det, _lbl, _c in dets:
        d9 = df[(df.detector == det) & (df.tau == T0) & (df.m == 9)].sort_values("subseed")
        for col in ["recall", "precision", "fpr", "act_rate", "mean_active_sensors",
                    "loc1_uncond", "loc3_uncond", "loc1_cond", "loc3_cond",
                    "median_delay_h", "mean_delay_h", "n_censored"]:
            m, s, h = msci(d9[col])
            stats_out["activation"][f"{det}_tau05_m9_{col}"] = {
                "mean": m, "sd": s, "ci95": h}
        d3 = df[(df.detector == det) & (df.tau == T0) & (df.m == 3)].sort_values("subseed")
        pairs = [("uncond_minus_always3", d9["loc1_uncond"].values, d3["loc1_dense_m"].values),
                 ("uncond_minus_dense9", d9["loc1_uncond"].values, d9["loc1_dense"].values),
                 ("cond_minus_dense", d9["loc1_cond"].values, d9["loc1_dense"].values)]
        for name, a, b in pairs:
            d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
            m, s, h = msci(d)
            stats_out["activation"][f"{det}_paired_{name}"] = {
                "mean": m, "sd": s, "ci95": h,
                "excludes_zero": bool(not np.isnan(h) and abs(m) > h)}
        # activation-budget curve at tau=0.5
        for mm in M_ACT:
            dm = df[(df.detector == det) & (df.tau == T0) & (df.m == mm)]
            for key, col in [("uncond", "loc1_uncond"), ("alwayson", "loc1_dense_m"),
                             ("uncond3", "loc3_uncond")]:
                m, s, h = msci(dm[col])
                stats_out["activation"][f"{det}_budget_m{mm}_{key}"] = {
                    "mean": m, "sd": s, "ci95": h}
    # frontier over tau (m=9): activation rate vs unconditional Loc@1
    for det, _lbl, _c in dets:
        pts = []
        for tau in sorted(df[df.detector == det].tau.unique()):
            dm = df[(df.detector == det) & (np.isclose(df.tau, tau)) & (df.m == 9)]
            pts.append({"tau": float(tau), "act_rate": float(dm["act_rate"].mean()),
                        "loc1": float(dm["loc1_uncond"].mean()),
                        "recall": float(dm["recall"].mean())})
        stats_out["activation"][f"{det}_frontier"] = pts
    m, s, h = msci(df[(df.tau == T0) & (df.m == 9)]["loc1_dense"])
    stats_out["activation"]["dense_ref"] = {"mean": m, "sd": s, "ci95": h}
    m, s, h = msci(df[(df.tau == T0) & (df.m == 3)]["loc1_dense_m"])
    stats_out["activation"]["always3_ref"] = {"mean": m, "sd": s, "ci95": h}

    # ---- figure: frontier + budget curve ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    dense = stats_out["activation"]["dense_ref"]["mean"]
    for det, lbl, col in dets:
        pts = stats_out["activation"][f"{det}_frontier"]
        axes[0].plot([p["act_rate"] for p in pts], [p["loc1"] for p in pts],
                     "o-", color=col, label=f"{lbl} detector", ms=4)
        t05 = min(pts, key=lambda p: abs(p["tau"] - 0.5))
        axes[0].plot(t05["act_rate"], t05["loc1"], "*", color=col, ms=14)
    axes[0].axhline(1 / 9, color="gray", ls="--", lw=1)
    axes[0].axhline(dense, color="gray", ls=":", lw=1)
    axes[0].text(0.98, 1 / 9 + 0.012, "chance (1/9)", ha="right", fontsize=8, color="gray")
    axes[0].text(0.98, dense - 0.045, "always-9 dense", ha="right", fontsize=8, color="gray")
    axes[0].set_xlabel("window activation rate (fraction with tier-2 awake)")
    axes[0].set_ylabel("unconditional Loc@1")
    axes[0].set_title("(a) activation-operating frontier ($m{=}9$)")
    axes[0].legend(fontsize=8, loc="lower right")
    for det, lbl, col in dets:
        y = [stats_out["activation"][f"{det}_budget_m{mm}_uncond"]["mean"] for mm in M_ACT]
        axes[1].plot(M_ACT, y, "o-", color=col, label=f"{lbl} detector ($\\tau$ = 0.5)")
    yref = [stats_out["activation"][f"cnn_budget_m{mm}_alwayson"]["mean"] for mm in M_ACT]
    axes[1].plot(M_ACT, yref, "s--", color="#999999", label="always-$m$ (no activation)")
    axes[1].axhline(1 / 9, color="gray", ls="--", lw=1)
    axes[1].set_xticks(M_ACT)
    axes[1].set_xlabel("active sensors after alarm, $m$")
    axes[1].set_ylabel("unconditional Loc@1")
    axes[1].set_title("(b) activation-budget curve")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_activation.pdf"))
    plt.close(fig)

    # ---- table: tau=0.5 cells ----
    body = ""
    for det, lbl, _c in dets:
        for i, mm in enumerate(M_ACT):
            dm = df[(df.detector == det) & (df.tau == T0) & (df.m == mm)]
            r = "$" + fmt(*msci(dm["recall"])[:2]) + "$"
            a = "$" + fmt(*msci(dm["act_rate"])[:2]) + "$"
            act = "$" + fmt(*msci(dm["mean_active_sensors"])[:2], prec=2) + "$"
            u = "$" + fmt(*msci(dm["loc1_uncond"])[:2]) + "$"
            cd = "$" + fmt(*msci(dm["loc1_cond"])[:2]) + "$"
            ref = "$" + fmt(*msci(dm["loc1_dense_m"])[:2]) + "$"
            lab = f"\\multirow{{4}}{{*}}{{{lbl}}}" if i == 0 else ""
            body += (f"{lab} & {mm} & {r} & {a} & {act} & {u} & {cd} & {ref} \\\\\n")
        body += "\\addlinespace\n"
    header = r"""\begin{tabular}{llcccccc}
\toprule
\textbf{Detector} & $m$ & \textbf{Recall} & \textbf{Activ.\ rate} & \textbf{Mean active} & \textbf{Loc@1 uncond.} & \textbf{Loc@1 trig.} & \textbf{Always-$m$ ref.} \\
\midrule
\small
"""
    save_table("tab_activation.tex", body, header)
    p = stats_out["activation"]
    print(f"activation: cnn loc1_uncond={p['cnn_tau05_m9_loc1_uncond']['mean']:.3f} "
          f"(paired vs always3 {p['cnn_paired_uncond_minus_always3']['mean']:+.3f}, "
          f"excl0={p['cnn_paired_uncond_minus_always3']['excludes_zero']}), "
          f"et={p['et_tau05_m9_loc1_uncond']['mean']:.3f}")


# ============================================================
# 8. Variance decomposition
# ============================================================
def variance_analysis():
    df = pd.read_csv("data/results/variance_decomp.csv")
    stats_out.setdefault("variance", {})
    rows = []
    for arch in ARCHS:
        for k in KS:
            vp = df[(df.family == "place") & (df.arch == arch) & (df.k == k)]
            vs = df[(df.family == "split") & (df.arch == arch) & (df.k == k)]
            vt = df[(df.family == "train") & (df.arch == arch) & (df.k == k)]
            cells = []
            for col in ["auc", "loc_acc"]:
                var_p = float(np.var(vp[col], ddof=1))
                var_s = float(np.var(vs[col], ddof=1))
                var_t = float(np.var(vt[col], ddof=1))
                cells += [var_t, max(var_p - var_t, 0.0), max(var_s - var_t, 0.0)]
                stats_out["variance"][f"{arch}_k{k}_{col}"] = {
                    "var_train": var_t, "var_place": max(var_p - var_t, 0.0),
                    "var_split": max(var_s - var_t, 0.0)}
            rows.append((arch, k, cells))
    # summary: mean over archs and ks
    arr = np.array([r[2] for r in rows])
    mean_row = arr.mean(axis=0)
    stats_out["variance"]["summary"] = {
        "auc": {"var_train": mean_row[0], "var_place": mean_row[1],
                "var_split": mean_row[2]},
        "loc": {"var_train": mean_row[3], "var_place": mean_row[4],
                "var_split": mean_row[5]}}
    body = ""
    for arch, k, cells in rows:
        c = [f"{v:.5f}" for v in cells]
        body += (f"{ARCH_LABEL[arch]} & {k} & " + " & ".join(
            [c[0], c[1], c[2], c[3], c[4], c[5]]) + " \\\\\n")
    c = [f"{v:.5f}" for v in mean_row]
    body += "\\midrule\n\\emph{mean} & & " + " & ".join(c) + " \\\\\n"
    header = r"""\begin{tabular}{lccccccc}
\toprule
 & & \multicolumn{3}{c}{\textbf{AUC variance}} & \multicolumn{3}{c}{\textbf{Loc@1 variance}} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
\textbf{Arch.} & $k$ & train & place & split & train & place & split \\
\midrule
\small
"""
    save_table("tab_variance.tex", body, header)


# ============================================================
# 9. Window labeling sensitivity
# ============================================================
def window_labeling_analysis():
    df = pd.read_csv("data/results/window_labeling.csv")
    stats_out.setdefault("window_labeling", {})
    body = ""
    for m in [1, 6, 12, 24, 48]:
        d = df[df.m == m]
        cells = [f"{d.n_leak_windows.iloc[0]}"]
        for col in ["auc", "f1", "loc_acc", "loc3"]:
            m_, s, ci = msci(d[col])
            cells.append(f"${m_:.3f} \\pm {s:.3f}$")
            stats_out["window_labeling"][f"m{m}_{col}"] = {
                "mean": m_, "sd": s, "ci95": ci}
        body += f"{m} ({m*5}\\,min) & " + " & ".join(cells) + " \\\\\n"
    header = r"""\begin{tabular}{lccccc}
\toprule
\textbf{Min.\ leak duration $m$} & \textbf{\# leak windows} & \textbf{AUC $\uparrow$} & \textbf{F1 $\uparrow$} & \textbf{Loc@1 $\uparrow$} & \textbf{Loc@3 $\uparrow$} \\
\midrule
\small
"""
    save_table("tab_windowlab.tex", body, header)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, col, name in zip(axes, ["auc", "loc_acc"], ["AUC", "Loc@1"]):
        for arch in ARCHS:
            ms, sds = [], []
            for m in [1, 6, 12, 24, 48]:
                d = df[(df.m == m) & (df.arch == arch)]
                mm, s, _ = msci(d[col])
                ms.append(mm); sds.append(s)
            ax.errorbar([1, 6, 12, 24, 48], ms, yerr=sds, marker="o", capsize=3,
                        color=C[arch], label=ARCH_LABEL[arch])
        ax.set_xlabel("minimum leak duration $m$ (steps of 5 min)")
        ax.set_title(name)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_windowlab.pdf"))
    plt.close(fig)
    print("wrote fig_windowlab.pdf")


# ============================================================
# 10. Demand-noise sweep
# ============================================================
def demand_noise_analysis():
    df = pd.read_csv("data/results/demand_noise.csv")
    stats_out.setdefault("demand_noise", {})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, k in zip(axes, [9, 3]):
        for arch in ARCHS:
            ms, sds = [], []
            for scale in [0.0, 0.5, 1.0, 2.0]:
                d = df[(df.scale == scale) & (df.arch == arch) & (df.k == k)]
                m, s, _ = msci(d["auc"])
                ms.append(m); sds.append(s)
                stats_out["demand_noise"][f"s{scale}_{arch}_k{k}_auc"] = {
                    "mean": m, "sd": s}
            ax.errorbar([0.0, 0.5, 1.0, 2.0], ms, yerr=sds, marker="o", capsize=3,
                        color=C[arch], label=ARCH_LABEL[arch])
        ax.set_xlabel("demand-noise scale $s$")
        ax.set_ylabel("detection AUC")
        ax.set_title(f"$k$={k}")
        ax.set_ylim(0.45, 1.05)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_demand_noise.pdf"))
    plt.close(fig)
    print("wrote fig_demand_noise.pdf")

    # localization also
    for arch in ARCHS:
        for scale in [0.0, 0.5, 1.0, 2.0]:
            d = df[(df.scale == scale) & (df.arch == arch)]
            m, s, _ = msci(d["loc_acc"])
            stats_out["demand_noise"][f"s{scale}_{arch}_loc"] = {"mean": m, "sd": s}


# ============================================================
# 11. Sensor (measurement) noise
# ============================================================
def sensor_noise_analysis():
    df = pd.read_csv("data/results/sensor_noise.csv")
    stats_out.setdefault("sensor_noise", {})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    alphas = [0.0, 0.05, 0.1, 0.25, 0.5]
    for ax, col, name in zip(axes, ["auc", "loc_acc"], ["AUC", "Loc@1"]):
        for arch in ARCHS:
            for train_a, ls, alpha in [(0.0, "-", 0.6), (0.1, "--", 1.0)]:
                d0 = df[(df.train_alpha == train_a) & (df.arch == arch)]
                ms, sds = [], []
                for a in alphas:
                    d = d0[d0.eval_alpha == a]
                    m, s, _ = msci(d[col])
                    ms.append(m); sds.append(s)
                ax.errorbar(alphas, ms, yerr=sds, marker="o", capsize=3, ls=ls,
                            color=C[arch], alpha=alpha,
                            label=ARCH_LABEL[arch] + (" (noise-augm.)" if train_a else ""))
                for i, a in enumerate(alphas):
                    stats_out["sensor_noise"][f"{arch}_tr{train_a}_ev{a}_{col}"] = {
                        "mean": ms[i], "sd": sds[i]}
        ax.set_xlabel(r"test-time noise level $\alpha$ ($\times$ channel std)")
        ax.set_title(name)
        if col == "auc":
            ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sensor_noise.pdf"))
    plt.close(fig)
    print("wrote fig_sensor_noise.pdf")


# ============================================================
# 12. Leak physics
# ============================================================
def leak_physics_analysis():
    try:
        with open("data/results/leak_physics.json") as f:
            stats_out["leak_physics"] = json.load(f)
    except Exception as e:
        print("leak_physics.json missing:", e)


def main():
    dataset_overview()
    run_count_audit()
    bootstrap_ci_analysis()
    try:
        channel_permutation_analysis()
    except Exception as e:
        print("channel_permutation skipped:", e)
    try:
        random_graph_analysis()
    except Exception as e:
        print("random_graph skipped:", e)
    try:
        demand_only_analysis()
    except Exception as e:
        print("demand_only skipped:", e)
    try:
        many_trios_analysis()
    except Exception as e:
        print("many_trios skipped:", e)
    try:
        transplant_analysis()
    except Exception as e:
        print("transplant skipped:", e)
    try:
        time_only_analysis()
    except Exception as e:
        print("time_only skipped:", e)
    try:
        paired_arch_comparison()
    except Exception as e:
        print("paired comparison skipped:", e)
    master_table()
    try:
        leaksize_table()
    except Exception as e:
        print("leaksize skipped:", e)
    try:
        baselines_table()
    except Exception as e:
        print("baselines skipped:", e)
    try:
        cross_sensor_analysis()
    except Exception as e:
        print("cross_sensor skipped:", e)
    try:
        identity_analysis()
    except Exception as e:
        print("identity skipped:", e)
    try:
        two_stage_analysis()
    except Exception as e:
        print("two_stage skipped:", e)
    try:
        activation_analysis()
    except Exception as e:
        print("activation skipped:", e)
    try:
        iteration2_response_analysis()
    except Exception as e:
        print("iteration2 skipped:", e)
    try:
        variance_analysis()
    except Exception as e:
        print("variance skipped:", e)
    try:
        window_labeling_analysis()
    except Exception as e:
        print("window_labeling skipped:", e)
    try:
        demand_noise_analysis()
    except Exception as e:
        print("demand_noise skipped:", e)
    try:
        sensor_noise_analysis()
    except Exception as e:
        print("sensor_noise skipped:", e)
    leak_physics_analysis()

    with open("paper/stats.json", "w") as f:
        json.dump(stats_out, f, indent=1)
    print("\nwrote paper/stats.json")


if __name__ == "__main__":
    main()



