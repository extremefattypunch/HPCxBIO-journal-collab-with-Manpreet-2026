#!/usr/bin/env python
"""Profiler analysis (plan Task 7.1; spec SS46, SS43).

Answers the questions spec SS46 asks:
  * which kernels dominate the exact VJP;
  * whether sketching reduces repeated trunk execution;
  * how occupancy and memory scale with lane count;
  * whether postprocessing matters -- i.e. spec SS43's decision rule, is
    T_postprocess > 10% of ForceSketch runtime?

NVTX ranges are emitted around forward, each backward lane, and the score
reduction, so `nsys stats --report nvtx_sum` attributes GPU time to phases rather
than to bare kernel names.

Reuses the module-hook idiom from tensornet2_profiling/harness/run_infer.py:67.
The load-bearing detail there is that the hooks MUST return None -- nvtx.range_push
returns an int (the stack depth), and a forward hook returning non-None REPLACES
the module's output, silently corrupting the model.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

from forcesketch.adapters.mace_data import load_frames, make_loader
from forcesketch.adapters.mace_mhc import MaceMHCAdapter
from forcesketch.estimators.scores import uncertainty_scores
from forcesketch.exact.centered_basis import exact_seed_bundle
from forcesketch.sketches.registry import make_sketch_seeds
from forcesketch.utils.reproducibility import git_commit, pin_numerics

CKPT = "models/zenodo/3BPA/trainset_100/multihead-disjoint/multihead_committee_stagetwo.model"
DATA = "data/3bpa/test_1200K_ref.xyz"


def add_module_nvtx_hooks(model: torch.nn.Module) -> None:
    """Wrap every named submodule in an NVTX range.

    Both hooks return None. nvtx.range_push returns an int; a forward_pre_hook
    returning non-None replaces the module's args, and a forward_hook returning
    non-None replaces its output. Returning the push depth would corrupt the model
    while still running -- the exact failure the source harness documents.
    """
    skipped = 0
    for name, mod in model.named_modules():
        if not name or getattr(mod, "_fs_nvtx", False):
            continue
        if isinstance(mod, torch.jit.ScriptModule):
            # e3nn ships compiled submodules and TorchScript rejects hooks on them.
            # Their time still appears in the enclosing range, so attribution is
            # coarser inside the tensor products but not lost.
            skipped += 1
            continue

        def pre(_m, _inp, _name=name):
            nvtx.range_push(_name)
            return None

        def post(_m, _inp, _out):
            nvtx.range_pop()
            return None

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        mod._fs_nvtx = True
    if skipped:
        print(f"[nvtx] {skipped} ScriptModules cannot take hooks (e3nn compiled code); "
              "their time is attributed to the enclosing range")


def timed_phases(adapter, batch, bundle, *, iters: int, warmup: int,
                 nvtx_on: bool = True) -> dict:
    """Per-phase CUDA-event timing with matching NVTX ranges (spec SS45)."""
    bidx = adapter.batch_index(batch)
    B = adapter.num_structures(batch)

    def forward_only():
        nvtx.range_push("forward") if nvtx_on else None
        e = adapter.energies(batch)
        nvtx.range_pop() if nvtx_on else None
        return e

    def full():
        nvtx.range_push("vjp") if nvtx_on else None
        lanes = adapter.vjp_for_seeds(batch, bundle.seeds, batched=False)
        nvtx.range_pop() if nvtx_on else None
        nvtx.range_push("reduce") if nvtx_on else None
        sc = uncertainty_scores(lanes, bundle, batch_index=bidx, n_structures=B,
                                reduce_dtype=None)
        nvtx.range_pop() if nvtx_on else None
        return sc

    def reduce_only(lanes=None):
        nvtx.range_push("reduce_only") if nvtx_on else None
        sc = uncertainty_scores(lanes, bundle, batch_index=bidx, n_structures=B,
                                reduce_dtype=None)
        nvtx.range_pop() if nvtx_on else None
        return sc

    def med(fn) -> tuple[float, float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
        for s, e in evs:
            s.record(); fn(); e.record()
        torch.cuda.synchronize()
        t = sorted(s.elapsed_time(e) for s, e in evs)
        return st.median(t), t[3 * len(t) // 4] - t[len(t) // 4]

    lanes = adapter.vjp_for_seeds(batch, bundle.seeds, batched=False).detach()
    torch.cuda.reset_peak_memory_stats()
    t_total, iqr_total = med(full)
    peak = torch.cuda.max_memory_allocated()
    t_fwd, _ = med(forward_only)
    t_red, _ = med(lambda: reduce_only(lanes))
    return {
        "total_ms": t_total, "total_iqr_ms": iqr_total,
        "forward_ms": t_fwd, "reduce_ms": t_red,
        "vjp_ms": t_total - t_red,
        "reduce_frac": t_red / t_total,
        "peak_memory_bytes": peak,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--nvtx-modules", action="store_true",
                    help="add per-submodule NVTX ranges (for nsys, not for timing)")
    ap.add_argument("--ncu", action="store_true",
                    help="run exactly ONE measured step inside one NVTX range")
    ap.add_argument("--out", default="results/raw/08_profile.jsonl")
    args = ap.parse_args()

    pin_numerics()
    adapter = MaceMHCAdapter.from_checkpoint(CKPT, dtype=torch.float32)
    M = adapter.num_heads
    frames = load_frames(DATA, limit=args.batch_size)
    batch = adapter.prepare(
        next(iter(make_loader(frames, adapter.model, batch_size=args.batch_size))).to_dict())
    if args.nvtx_modules:
        add_module_nvtx_hooks(adapter.model)

    configs = [("exact", exact_seed_bundle(M, args.batch_size, dtype=torch.float32,
                                           device=adapter.device))]
    for K in (2, 3, 4):
        configs.append((f"haar_K{K}",
                        make_sketch_seeds("haar", M=M, K=K, batch_size=args.batch_size,
                                          seed=1000003, dtype=torch.float32,
                                          device=adapter.device)))

    if args.ncu:
        # One instance for Nsight Compute to replay.
        name, bundle = configs[0]
        nvtx.range_push("model_step")
        adapter.vjp_for_seeds(batch, bundle.seeds, batched=False)
        nvtx.range_pop()
        torch.cuda.synchronize()
        return 0

    sha, dirty = git_commit()
    records = []
    print(f"3BPA disjoint, B={args.batch_size} ({args.batch_size * len(frames[0])} atoms), "
          f"fp32, {args.iters} iters\n")
    print(f"{'config':<12}{'lanes':>6}{'total ms':>10}{'forward':>9}{'vjp':>9}"
          f"{'reduce':>9}{'reduce %':>10}{'peak MB':>9}")
    print("-" * 74)
    for name, bundle in configs:
        m = timed_phases(adapter, batch, bundle, iters=args.iters, warmup=args.warmup,
                         nvtx_on=args.nvtx_modules)
        m.update({"experiment_id": "08_profile", "git_commit": sha, "git_dirty": dirty,
                  "dataset": "3bpa", "config": name, "lanes": bundle.K,
                  "batch_size": args.batch_size, "num_heads": M, "precision": "fp32",
                  "gpu": torch.cuda.get_device_name(0)})
        records.append(m)
        print(f"{name:<12}{bundle.K:>6}{m['total_ms']:>10.2f}{m['forward_ms']:>9.2f}"
              f"{m['vjp_ms']:>9.2f}{m['reduce_ms']:>9.3f}{100*m['reduce_frac']:>9.1f}%"
              f"{m['peak_memory_bytes']/2**20:>9.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    # --- spec SS43 decision rule ------------------------------------------
    sketch = [r for r in records if r["config"] == "haar_K3"][0]
    frac = sketch["reduce_frac"]
    print(f"\nSS43 decision rule: postprocessing is {100*frac:.1f}% of ForceSketch(K=3) runtime")
    print(f"  -> {'ABOVE' if frac > 0.10 else 'BELOW'} the 10% threshold; "
          f"Triton {'may be' if frac > 0.10 else 'is NOT'} worth implementing")
    print(f"\nwrote {out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
