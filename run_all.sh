#!/usr/bin/env bash
# Full reproduction pipeline (Linux/macOS)
# Usage: bash run_all.sh   (from the repository root)
set -e

echo "=== 1/8 Generating dataset (67 scenarios, WNTRSimulator PDD) ==="
python src/data_gen.py

echo "=== 2/8 Master benchmark (96 runs) ==="
python src/benchmark.py --subseed 0 1 2 3 4 5

echo "=== 3/8 Leak-size sweep (96 runs) ==="
python src/leak_size.py

echo "=== 4/8 Cross-placement transfer (std + set + GNN) ==="
python src/cross_sensor.py --model std
python src/cross_sensor.py --model set
python src/cross_sensor.py --model gnn

echo "=== 5/8 Identity + GNN in-distribution benchmarks ==="
python src/sensor_identity.py
python src/gnn_baseline.py

echo "=== 6/8 Baselines and two-stage ==="
python src/extra_baselines.py
python src/classical_baseline.py
python src/two_stage.py

echo "=== 7/10 Robustness suite (variance, labeling, noise) ==="
python src/variance_decomp.py
python src/window_labeling.py
python src/demand_noise.py
python src/sensor_noise.py
python src/leak_physics.py

echo "=== 8/10 Mechanism experiments (permutation, adjacency, demand-only) ==="
python src/channel_permutation.py
python src/random_graph_gcn.py --adj random
python src/random_graph_gcn.py --adj identity
python src/random_graph_gcn.py --adj multi-random
python src/demand_only.py
python src/cross_many_trios.py
python src/demand_transplant.py 5
python src/demand_transplant.py 3
python src/time_only.py
python src/activation_scheme.py

echo "=== 9/10 Deficit-heuristic reference + overlap regression (evaluation only) ==="
python src/heuristic_baseline.py
python src/heuristic_indist_posthoc.py
python src/overlap_transfer.py

echo "=== 10/10 Tables, figures, statistics ==="
python src/analysis_v2.py

echo "Done. See paper/tables/, paper/figures/, paper/stats.json."
