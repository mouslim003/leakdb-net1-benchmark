# Protocol: Transfer mechanism experiments (study iteration 2.1)

Strengthens the causal story where several central claims outran the evidence
at iteration 2. Protocol locked BEFORE running.

## Experiment 1 — Many-trios placement-transfer sweep (`src/cross_many_trios.py`)

**Gap addressed:** "node-indexed models restore transfer" rests on
n=3 trios with wide intervals (e.g., identity-adjacency k=5: 0.387±0.188).

**Design:** 15 independently drawn placement trios. Trio t uses placement
seeds {20+3t, 21+3t, 22+3t} (fresh seeds, disjoint from master subseeds
0-5); train on placements (a,b), test on c; scenario split seed = test
placement seed (protocol-consistent). Models: CNN (channel reference),
GCN real adjacency, GCN identity adjacency, GCN randomized adjacency
(rewire seed 3000+t, so transfer is tested over 15 distinct rewirings).
k in {7,5,3}; Experiment B only. 4 x 3 x 15 = 180 training runs.

**Analysis:** per (model, k): mean ± SD and 95% t-CI over the 15 trios;
PAIRED per-trio differences (GCN variant − CNN) with t-CIs (same trio,
same split, same placements => paired comparison is valid); per-trio
dot/box figure.

**Prediction:** node-indexed variants' transfer Loc@1 exceeds chance with
CI separation at k in {7,5}; k=3 may remain uncertain — report honestly.
Channel CNN stays at chance.

## Experiment 2 — Demand–leak transplant intervention (`src/demand_transplant.py`)

**Gap addressed:** the demand-only classifier shows Experiment A
PERMITS memorization (perfect seen-scenario recall is trivially learnable),
not that channel models USE it.

**Design (matched intervention):** for each of the 6 master replicates,
retrain the CNN with the master per-run seeds (bit-identical). Select the
training leak scenarios (up to 8 per replicate). For each, simulate
COUNTERFACTUAL scenarios with the IDENTICAL demand seed, leak size, and
start hour, but the leak moved to a different node (m1 = idx+4 mod 9,
m2 = idx+7 mod 9). Evaluate the trained model's leak-window predictions
through the TRAINING placement (in-distribution):
  - P(pred = original training node n)  -> fingerprint use
  - P(pred = counterfactual true node m) -> spatial evidence use
  - control: P(pred = n) on the original scenario windows.

**Prediction:** if the model exploits the demand fingerprint, predictions
follow the ORIGINAL node under the counterfactual; if it reads the
pressure-deficit location, predictions follow the NEW node. Either
outcome is reported as found.

## Experiment 3 — Time-only confound baseline (`src/time_only.py`)

**Gap addressed:** leak starts uniform on [2,14] + 8h duration make
the leak-label distribution over the day non-uniform; "time-of-day carries
no label information" is not literally true.

**Design:** per-window time features ONLY (start hour + sin/cos encoding;
no pressure, no demand). Logistic regression + small MLP under the six
scenario-grouped splits; detection AUC. Also report the empirical
window-start-hour -> leak-rate profile. Revise the claim per the result.

## Experiment 4 — Multi-rewiring adjacency ablation (`src/random_graph_gcn.py --adj multi-random`)

**Gap addressed:** one randomized graph cannot support "physical
adjacency is not the active ingredient."

**Design:** 4 additional independent degree-preserving rewirings (seeds
2027-2030; the released 2026 graph makes 5 total), in-distribution
benchmark only: 4 graphs x 4 densities x 6 replicates = 96 runs. Transfer
side is covered by Experiment 1 (15 distinct rewirings).

## Statistical/reporting fixes (no new experiments)

- Per-density baseline table (matched comparison vs deep models).
- "Best at every density" corrected (identity adjacency is NOT best
  in-dist at k=3: random 0.151 / GRU 0.137 vs identity 0.124).
- Loc@3 "above chance" softened to point-estimate trend + cell bootstrap.
- No non-significance inference from overlapping CIs; add paired
  replicate-level comparison (GRU vs CNN at k=3).
- AUC is threshold-free (remove "fixed 0.5 threshold").
- Title narrowed to channel-based localisation.
- Author/repository details finalized before submission (repository URL/DOI
  and author names).

All runs use explicit per-run SHA-1-derived seeds, deterministic cuDNN.
