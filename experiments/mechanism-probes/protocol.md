# Protocol: Mechanism probes (study iteration 2)

Three mechanism experiments probe why channel-based localisation fails under
placement shift, plus statistical and reporting fixes. Protocol locked BEFORE running.

## Experiment 1 — Channel permutation test (`src/channel_permutation.py`)

**Hypothesis (P1):** Channel-based localizers depend on the semantic
channel-to-node mapping learned at training time. If we randomly permute the
channel order of the TEST inputs (same physical data, different semantic
mapping), localization collapses to the Experiment-B level even though the
hydraulic distribution is unchanged and the sensors are the SAME placement
the model was trained on.

**Design:** Train exactly as in the master benchmark (same per-run seeds:
4 archs x 4 densities x 6 replicates = 96 runs). Evaluate each trained model
on its own test split with (a) identity permutation (= master benchmark),
(b) 3 random channel permutations (seeded per run). Report Loc@1/Loc@3 and
detection AUC under permutation.

**Prediction:** permuted Loc@1 << in-distribution Loc@1, comparable to
Experiment B; detection AUC nearly unchanged (permutation-invariant task).

## Experiment 2 — Random-graph GCN ablation (`src/random_graph_gcn.py`)

**Hypothesis (P2):** The GCN's placement transfer is attributable to the
PHYSICAL network structure, not merely to its architecture/capacity.

**Design:** Replace the Net1 junction adjacency with a degree-preserving
randomized graph (double-edge-swap rewiring, seeded, connectivity-checked;
the degree sequence is exactly preserved). (a) In-distribution benchmark:
random-graph GCN over 4 densities x 6 replicates = 24 runs. (b) Cross-
placement Experiment A/B: 3 trios x 3 densities = 9 runs. Compare with the
real-graph GCN results.

**Prediction:** random-graph GCN fails to localize (in-distribution and
transfer), demonstrating that the real adjacency — not GCN capacity —
restores transfer.

## Experiment 3 — Demand-only memorization baseline (`src/demand_only.py`)

**Hypothesis (P3):** Experiment A's apparent partial transfer is scenario
memorization: each scenario's demand realization is a unique fingerprint
that lets a model recall the leak node of a SEEN scenario without any
spatial reasoning.

**Design:** Reconstruct each scenario's demand realization deterministically
from the stored demand seeds (9 per-junction base multipliers + 24 hourly
pattern multipliers = 33 features; no pressure data). Train a fingerprint
-> leak-node classifier (small MLP and logistic regression) on the training
scenarios of each of the 6 scenario splits (same grouped splits as the
master benchmark). Evaluate Loc@1 on (a) the TRAINING scenarios themselves
(memorization condition) and (b) the held-out TEST scenarios (transfer
condition).

**Prediction:** seen-scenario localization is far above chance (the MLP can
memorize the fingerprint->node mapping); held-out-scenario localization is
at chance (fingerprints carry no leak-node information). This provides
direct evidence for the memorization mechanism, previously inferred
indirectly.

## Statistical additions (`src/analysis_v2.py`)

- Bootstrap (percentile, B=10,000) 95% CIs for all master-benchmark cells,
  reported as a sensitivity check next to the Student-t CIs.
- Exact run-count audit across all result CSVs; one number in the paper.

## Reporting fixes (manuscript)

- Focused abstract; no exact run counts in abstract.
- RQ -> experiment map table.
- Move robustness/debug details to appendix; bug narrative shortened in main.
- All wording qualifications from the protocol checklist.

All runs use explicit per-run SHA-1-derived seeds, deterministic cuDNN.
