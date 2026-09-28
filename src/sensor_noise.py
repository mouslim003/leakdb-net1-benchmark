"""Measurement-noise robustness: Gaussian pressure-sensor noise at test time.

Motivation: real pressure sensors add noise, drift, and dropouts;
the master benchmark has no measurement noise. This script quantifies
robustness: models are trained on clean data (or with noise augmentation) and
evaluated with additive zero-mean Gaussian noise on the test inputs, with
sigma = alpha * per-channel std of the TRAINING data.

Conditions:
  eval-clean    : train clean, eval clean (reference)
  eval-noisy    : train clean, eval at alpha in {0.05, 0.1, 0.25, 0.5}
  train-noisy   : train WITH alpha=0.1 noise augmentation, eval at all alphas
4 archs x k in {9,5} x 3 subseeds.
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

KS = [9, 5]
SUBSEEDS = [0, 1, 2]
ARCHS = ["cnn", "lstm", "gru", "transformer"]
ALPHAS = [0.0, 0.05, 0.1, 0.25, 0.5]
TRAIN_ALPHA = 0.1


class NoisyDataset(torch.utils.data.Dataset):
    """Wraps a standardized WindowDataset, adding Gaussian noise in z-space.

    The base dataset is z-scored per channel with train statistics, so a
    noise level alpha corresponds to alpha x the per-channel training std.
    """

    def __init__(self, base, sigma, gen):
        self.base = base
        self.sigma = sigma.reshape(1, -1)   # (1,k), broadcasts over (T,k)
        self.gen = gen

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
        if self.sigma.max() > 0:
            x = x + torch.from_numpy(
                self.gen.normal(0.0, 1.0, size=tuple(x.shape)).astype(np.float32)) * \
                torch.from_numpy(self.sigma.astype(np.float32))
        return x, yd, yl, m


def run_family(arch, k, subseed, train_alpha, X, yd, yl, meta, nums, ranges):
    rs = int(hashlib.sha1(f"measnoise-{arch}-{k}-{subseed}-a{train_alpha}".encode())
             .hexdigest()[:8], 16)
    torch.manual_seed(rs)
    np.random.seed(rs)
    torch.cuda.manual_seed(rs)
    tr_s, va_s, te_s = benchmark.grouped_split(len(nums), seed=subseed)

    def gather(s):
        return np.concatenate([np.arange(*ranges[i]) for i in sorted(s)])

    tr_idx, va_idx, te_idx = gather(tr_s), gather(va_s), gather(te_s)
    n_j = X.shape[2]
    sensors = np.sort(np.random.default_rng(1000 + subseed).choice(n_j, k, replace=False))
    gen = np.random.default_rng(rs)

    model = models.build_model(arch, in_ch=k, hidden=benchmark.HIDDEN).to(benchmark.DEVICE)
    tr_base, va_base, te_base = benchmark.make_split_datasets(
        X, yd, yl, sensors, tr_idx, va_idx, te_idx)
    # sigma is expressed in z-space: alpha = fraction of the per-channel train std
    sigma = np.full(k, train_alpha, dtype=np.float32)
    tr = NoisyDataset(tr_base, sigma, gen) if train_alpha > 0 else tr_base
    va = NoisyDataset(va_base, sigma, gen) if train_alpha > 0 else va_base
    model = benchmark.train(model, tr, va)

    rows = []
    for alpha in ALPHAS:
        te = NoisyDataset(te_base, np.full(k, alpha, dtype=np.float32), gen)
        m = benchmark.evaluate(model, te)
        rows.append({"arch": arch, "k": k, "subseed": subseed,
                     "train_alpha": train_alpha, "eval_alpha": alpha, **m})
        print(f"{arch} k={k} s={subseed} train_a={train_alpha} eval_a={alpha}: "
              f"auc={m['auc']:.3f} bal={m['bal_acc']:.3f} loc={m['loc_acc']:.3f}")
    return rows


def main():
    X, yd, yl, meta = benchmark.load_data()
    nums = benchmark.compute_scenario_bounds(meta)
    ranges = benchmark.window_index_ranges(nums)

    rows = []
    for arch in ARCHS:
        for k in KS:
            for subseed in SUBSEEDS:
                rows.extend(run_family(arch, k, subseed, 0.0, X, yd, yl, meta, nums, ranges))
                rows.extend(run_family(arch, k, subseed, TRAIN_ALPHA, X, yd, yl, meta, nums, ranges))

    fn = "data/results/sensor_noise.csv"
    flds = ["arch", "k", "subseed", "train_alpha", "eval_alpha",
            "bal_acc", "f1", "auc", "loc_acc", "loc3", "loc5"]
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {fn}")


if __name__ == "__main__":
    main()
