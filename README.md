# Sensor Sparsity and Placement Shift in Water Distribution Networks — Reproducible Benchmark

Code, data, and per-run results for the study:

> **Sensor Sparsity and Placement Shift in Water Distribution Networks: A
> Reproducible Benchmark of Leak Detection and Localisation**
> (submitted to *Water Supply*, IWA Publishing).

The study benchmarks four lightweight temporal architectures (1D-CNN, LSTM, GRU, small
Transformer encoder), a sensor-identity set encoder, and lightweight graph convolutional
models on the LeakDB Net1 network under controlled sensor sparsity
(k ∈ {9, 7, 5, 3} of 9 junctions), with six independently sampled sensor placements per
density, scenario-grouped splits, chance and classical baselines, cross-placement transfer
experiments, five mechanism probes, a training-free largest-pressure-deficit reference,
a full robustness suite, and a two-tier sensor-activation scheme.

## One-command reproduction

```bash
# Linux / macOS
bash run_all.sh

# Windows (PowerShell)
.\run_all.ps1
```

Both scripts regenerate the dataset, every experiment, and every per-run CSV from
scratch. On an RTX 4060 Laptop GPU (8 GB) the full pipeline takes about five hours of
wall time (the 96-run master benchmark alone takes about 16 minutes); on CPU only it
takes a few hours.

If you only want to recompute the paper's tables and figures from the released per-run
CSVs (no training):

```bash
python src/analysis_v2.py
```

## Environment

```bash
conda env create -f environment.yml && conda activate leakdb-benchmark
# or
pip install -r requirements.txt
```

A `Dockerfile` is provided (CPU-only reproduction; the data-generation stage
needs the LeakDB generator's `Net1.inp` mounted at the path below). The dataset generator needs the LeakDB generator's
`Net1.inp`, expected at
`datasets/LeakDB/CCWI-WDSA2018/Dataset_Generator_Py3/networks/Net1.inp`
(download the LeakDB generator from the
[LeakDB repository](https://github.com/KIOS-Research/LeakDB) or
[Zenodo record 1313116](https://doi.org/10.5281/zenodo.1313116)).

## Repository layout

```
src/                    all experiment code (see below)
data/                   released dataset (X.npy, y_det.npy, y_loc.npy, window_meta.json),
                        the preserved buggy-label dataset (*_v2_buggy.*), and
                        data/results/*.csv — per-run metrics for every experiment
experiments/            pre-registered protocols (committed before running)
bug_report.md           exact record of the two data-generation bugs found during the study
run_all.sh / run_all.ps1  one-command reproduction
```

### Source files

| File | Purpose |
|---|---|
| `src/data_gen.py` | Dataset generation: timed orifice leaks (WNTR `add_leak`), stochastic demands, windowing |
| `src/models.py` | CNN/LSTM/GRU/Transformer encoders, set encoder (sensor identity), GCN |
| `src/benchmark.py` | Master benchmark: 4 archs × 4 densities × 6 placements, shared training utilities |
| `src/leak_size.py` | Leak-size stratification over all 6 placements |
| `src/cross_sensor.py` | Cross-placement transfer, Experiments A and B (`--model std/set/gnn`) |
| `src/sensor_identity.py` | Set-encoder (identity-aware) master benchmark |
| `src/gnn_baseline.py` | GCN master benchmark (real adjacency) |
| `src/random_graph_gcn.py` | Adjacency ablation (`--adj random/identity/multi-random`) |
| `src/two_stage.py` | Two-stage (detect-then-localize) baseline |
| `src/extra_baselines.py` | Chance/majority + LogReg/RF/ExtraTrees/HGB baselines |
| `src/classical_baseline.py` | Isolation Forest / XGBoost baselines |
| `src/channel_permutation.py` | Channel-permutation test |
| `src/demand_only.py` | Demand-fingerprint (memorization-shortcut availability) probe |
| `src/demand_transplant.py` | Counterfactual demand–leak transplant intervention |
| `src/cross_many_trios.py` | Fifteen-trio deployment-transfer sweep |
| `src/time_only.py` | Time-position-only confound baseline |
| `src/heuristic_baseline.py` | Training-free largest-pressure-deficit reference (transfer) |
| `src/heuristic_indist_posthoc.py` | Deficit rule, in-distribution (post hoc) |
| `src/overlap_transfer.py` | Within-density overlap→transfer regression |
| `src/activation_scheme.py` | Two-tier sensor-activation scheme, with bit-identity gate |
| `src/variance_decomp.py` | Placement vs split vs training-seed variance decomposition |
| `src/window_labeling.py` | Minimum-leak-duration labeling sensitivity |
| `src/demand_noise.py` | Demand-stochasticity sweep |
| `src/sensor_noise.py` | Measurement-noise robustness |
| `src/leak_physics.py` | Physical characterization of the leak sizes |
| `src/analysis_v2.py` | All tables/figures/statistics from the released CSVs |

## Dataset

67 hydraulic scenarios (40 normal, 27 leak = 9 nodes × 3 orifice sizes) on LeakDB Net1,
simulated with WNTR's WNTRSimulator in pressure-driven-demand (PDD) mode with
stochastic per-junction demands. Leaks are timed orifices (hole diameters 20/30/50 mm,
Cd = 0.75, 8 h duration, random start hour) equivalent to roughly 14% / 32% / 85% of
average total network demand. 2,077 overlapping 4-hour windows (48 steps at 5-minute
resolution, stride 8). Splits are **scenario-grouped** (70/15/15) — windows from the
same scenario never straddle splits. The `*_v2_buggy.*` files preserve the buggy-label
dataset documented in `bug_report.md`.

## Determinism

Every training run derives its seed from a SHA-1 hash of its run identity
(experiment family, architecture, density, replicate), and
`torch.backends.cudnn.deterministic = True` is set throughout. Re-running any sweep
reproduces its CSV bit-for-bit on the same hardware/software stack.

## Software versions (verified at runtime)

Windows 11 (x64); Python 3.11.6; PyTorch 2.13.0+cu126; CUDA 12.6; WNTR 1.5.0;
NumPy 2.4.6; SciPy 1.15.1; scikit-learn 1.6.1; XGBoost 3.2.0; Matplotlib 3.11.1.

## License

Released under the MIT License (see `LICENSE`). The LeakDB benchmark and the WNTR
package are distributed under their own licenses; this repository neither includes nor
redistributes them.

## Citation

If you use this code or dataset, please cite the paper (citation block added on
acceptance) and the underlying benchmarks:

- Vrachimis, S.G., Kyriakou, M.S., Eliades, D.G. & Polycarpou, M.M. (2018) LeakDB: A
  benchmark dataset for leakage diagnosis in water distribution networks. *Proc. 1st
  International WDSA/CCWI Joint Conference*. doi.org/10.5281/zenodo.1313115
- Vrachimis, S.G., Eliades, D.G., Taormina, R., Kapelan, Z., Ostfeld, A., Liu, S.,
  Kyriakou, M., Pavlou, P., Qiu, M. & Polycarpou, M.M. (2022) Battle of the leakage
  detection and isolation methods. *Journal of Water Resources Planning and
  Management*, 148 (12), 04022068. doi.org/10.1061/(ASCE)WR.1943-5452.0001601
- Klise, K.A., Murray, R. & Haxton, T. (2017) An Overview of the Water Network Tool
  for Resilience (WNTR). Sandia National Laboratories, Albuquerque, NM, USA.
