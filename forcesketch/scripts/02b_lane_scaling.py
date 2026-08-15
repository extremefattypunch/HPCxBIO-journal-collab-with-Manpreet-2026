#!/usr/bin/env python
"""Lane-scaling microbenchmark on the real committee (plan Task 2.3).

The decisive early measurement: how does reverse-pass latency scale with the
number of head-space cotangent lanes L? Exact force UQ needs L = M = 8 (one mean
lane + M-1 centered directions); ForceSketch(K) needs L = K+1. The ratio
T(L=8)/T(L=K+1) is the ceiling on the incremental UQ speedup, and spec SS50 kills
the project if it stays under 1.2x.

Implementation note (measured, not assumed): `is_grads_batched=True` is NOT
usable on this MACE + e3nn 0.4.4 stack. It succeeds for the first call or two in
a fresh process and then raises

    RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

from inside the TorchScript interpreter, because e3nn's optimized autograd nodes
cannot accept vmap's BatchedTensor. Disabling jit_script_fx, optimize_einsums and
specialized_code does not reliably fix it, and neither does rebuilding the model
unscripted. So the strongest AVAILABLE exact baseline here is the serial loop over
centered directions -- which is also exactly what the reference implementation
does in `get_outputs_committee`. Spec SS17 items 3 and 4 are therefore reported as
unavailable on this stack, with the reason, rather than silently skipped.

Per spec SS44: CUDA events, per-iteration timing, >=100 warmup, median + IQR.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from forcesketch.adapters.mace_data import load_frames, make_loader
from forcesketch.adapters.mace_mhc import MaceMHCAdapter
from forcesketch.utils.reproducibility import git_commit, pin_numerics

CKPT = "models/zenodo/3BPA/trainset_100/multihead-disjoint/multihead_committee_stagetwo.model"


def timed(fn, iters: int, warmup: int) -> tuple[float, float, float]:
    """Median, IQR and min in ms, using one CUDA event pair per iteration and a
    single synchronize at the end (spec SS44: sync only at timing boundaries)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in evs)
    q1, q3 = t[len(t) // 4], t[3 * len(t) // 4]
    return st.median(t), q3 - q1, t[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default="results/raw/lane_scaling.jsonl")
    args = ap.parse_args()

    pin_numerics()
    adapter = MaceMHCAdapter.from_checkpoint(CKPT, dtype=torch.float32)
    M = adapter.num_heads
    frames = load_frames("data/3bpa/test_1200K_ref.xyz", limit=max(args.batch_sizes))
    sha, dirty = git_commit()
    records = []

    print(f"3BPA committee, M={M}, {len(frames[0])} atoms/structure")
    print(f"{args.iters} measured iters, {args.warmup} warmup, CUDA events, serial VJP\n")
    print(f"{'B':>3} {'atoms':>6} | " + "  ".join(f"L={L}" for L in range(1, M + 1)))

    for B in args.batch_sizes:
        batch = adapter.prepare(
            next(iter(make_loader(frames[:B], adapter.model, batch_size=B))).to_dict()
        )
        n_atoms = int(batch["positions"].shape[0])
        row, med_by_L = [], {}
        for L in range(1, M + 1):
            seeds = torch.randn(L, B, M, device=adapter.device, dtype=adapter.dtype)

            def step(seeds=seeds):
                return adapter.vjp_for_seeds(batch, seeds, batched=False)

            med, iqr, _ = timed(step, args.iters, args.warmup)
            med_by_L[L] = med
            row.append(f"{med:6.2f}")
            records.append({
                "experiment_id": "lane_scaling", "git_commit": sha, "git_dirty": dirty,
                "system": "3bpa", "method": "serial_vjp", "lanes": L, "batch_size": B,
                "num_atoms": n_atoms, "num_heads": M, "precision": "fp32",
                "gpu": torch.cuda.get_device_name(0),
                "median_ms": med, "iqr_ms": iqr,
            })
        print(f"{B:>3} {n_atoms:>6} | " + "  ".join(row))
        ceil3 = med_by_L[M] / med_by_L[4]
        ceil2 = med_by_L[M] / med_by_L[3]
        verdict = ("KILL (<1.2x)" if ceil3 < 1.2 else
                   "weak (<1.5x)" if ceil3 < 1.5 else
                   "H4 min" if ceil3 < 2.0 else "H4 STRONG")
        print(f"{'':>10} | ceiling K=3: {ceil3:.2f}x   K=2: {ceil2:.2f}x   -> {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {args.out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
