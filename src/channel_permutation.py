"""Channel permutation test: does the channel-to-node semantic mapping matter?

Design: take a trained channel model and randomly
permute the channel-to-node mapping at test time. The pressure DATA are
unchanged, the placement is the SAME one the model was trained on, and only
the semantic association between channel index and physical node is
scrambled. If localization collapses to the Experiment-B level while
detection is preserved, the model demonstrably depends on channel semantics
rather than on placement-invariant spatial evidence -- isolating semantic
channel permutation from hydraulic distribution shift.

Protocol:
- Train exactly as the master benchmark (same per-run SHA-1 seeds):
  4 archs x 4 densities x 6 replicates = 96 runs.
- Evaluate each trained model on its own test split under:
    identity  : unpermuted channels (= master benchmark reference)
    perm1..3  : three seeded random channel permutations
- Record Loc@1/Loc@3/Loc@5 and detection AUC/F1/Bal.Acc for each condition.
"""
import csv
import hashlib
import os
import sys

import numpy as np
import torch

import benchmark
import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

ARCHS = ["cnn", "lstm", "gru", "transformer"]
KS = [9, 7, 5, 3]
SUBSEEDS = [0, 1, 2, 3, 4, 5]
N_PERMS = 3


class PermutedDataset(torch.utils.data.Dataset):
    """Wraps a WindowDataset, permuting the channel axis with a fixed permutation."""

    def __init__(self, base, perm):
        self.base = base
        self.perm = perm

    def __len__(self):
        return len(self.base)

    @property
    def yd(self):
        return self.base.yd

    @property
    def yl(self):
        return self.base.yl

    def __getitem__(self, i):
        x, yd, yl, m = self.base[i]
        return x[:, self.perm], yd, yl, m


def run(arch, k, subseed):
    # SAME per-run seed derivation as benchmark.train_eval_subsets, so the
    # trained model is identical to the corresponding master-benchmark run
    # and the identity row reproduces the released master CSV bit-for-bit.
    rs = int(hashlib.sha1(f"{arch}-{k}-{subseed}".encode()).hexdigest()[:8], 16)
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)

    X, yd, yl, meta = benchmark.load_data()
    n_j = X.shape[2]
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)
    tr_s, va_s, te_s = benchmark.grouped_split(len(meta["schedules"]), seed=subseed)

    def gather(sids):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(sids)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)

    sensors = np.sort(np.random.default_rng(1000 + subseed).choice(n_j, k, replace=False))

    model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
    tr, va, te = benchmark.make_split_datasets(X, yd, yl, sensors, tr_idx, va_idx, te_idx)
    model = benchmark.train(model, tr, va)

    rows = []
    # identity (reference = master benchmark value for this run)
    m = benchmark.evaluate(model, te)
    rows.append({"arch": arch, "k": k, "subseed": subseed, "perm": "identity",
                 "perm_idx": list(range(k)), **m})
    # seeded random permutations
    for p in range(N_PERMS):
        prng = np.random.default_rng(rs + 1 + p)
        perm = prng.permutation(k)
        rows.append({"arch": arch, "k": k, "subseed": subseed,
                     "perm": f"perm{p + 1}", "perm_idx": perm.tolist(),
                     **benchmark.evaluate(model, PermutedDataset(te, perm))})
    return rows


def main():
    os.makedirs("data/results", exist_ok=True)
    rows = []
    for arch in ARCHS:
        for k in KS:
            for subseed in SUBSEEDS:
                try:
                    rs = run(arch, k, subseed)
                    rows.extend(rs)
                    idm = rs[0]
                    pm = np.mean([r["loc_acc"] for r in rs[1:]])
                    print(f"{arch} k={k} s={subseed}: Loc@1 id={idm['loc_acc']:.3f} "
                          f"perm={pm:.3f} (AUC id={idm['auc']:.3f})")
                except Exception as e:
                    print(f"ERROR {arch} k={k} s={subseed}: {e}")
                    import traceback
                    traceback.print_exc()

    fn = "data/results/channel_permutation.csv"
    flds = ["arch", "k", "subseed", "perm", "perm_idx",
            "bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in rows:
            w.writerow({kk: (json.dumps(r[kk]) if isinstance(r[kk], list) else r[kk])
                        for kk in flds})
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    import json
    main()
