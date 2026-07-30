#!/usr/bin/env python3
"""Quantify CUDA-graph capture for the LAUNCH-BOUND small-molecule regime, and batching.

We measured (ANALYSIS.md §4) a ~24 ms/step floor at small N — launch/dispatch overhead, not compute
(nsys: ~1 us kernel exec). torchmd-net *already supports* CUDA graphs (calculators.py `External`,
static_shapes design); this script BENCHMARKS that capability (we quantify, not invent) by mirroring
its capture pattern (calculators.py:117-128): warmup on a side stream, then capture the full
forward+forces, then replay.

Run: conda run -n tn2prof python experiments/cuda_graphs.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "harness")
sys.path.insert(0, HARNESS)

import torch  # noqa: E402
from model_build import build_model  # noqa: E402
from workloads import gen_system  # noqa: E402


def time_ms(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench_eager(model, z, pos, batch, iters=30):
    def step():
        model(z, pos.detach().clone(), batch=batch)
    return time_ms(step, iters)


def bench_graph(model, z, pos, batch, iters=30, warmup=12):
    """Mirror calculators.py _init_cuda_graph: capture full forward+forces, then replay."""
    pos_static = pos.clone().detach().requires_grad_(True)
    stream = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            energy, forces = model(z, pos_static, batch=batch)
        with torch.cuda.graph(g):
            energy, forces = model(z, pos_static, batch=batch)
    torch.cuda.current_stream().wait_stream(stream)

    def step():
        with torch.no_grad():
            pos_static.copy_(pos)   # new coords in, static buffer
        g.replay()
    ms = time_ms(step, iters)
    return ms, torch.isfinite(forces).all().item()


def main():
    dev = "cuda"
    print("=" * 74)
    print("CUDA-GRAPH capture vs eager — launch-bound small-molecule regime")
    print("=" * 74)
    print(f"{'N':>6} {'eager ms/step':>14} {'graph ms/step':>14} {'speedup':>9} {'graph steps/s':>14}")
    for n in (21, 42, 250):
        model, info = build_model(coulomb_cutoff=None, device=dev, dtype=torch.float32)
        model.eval()
        z, pos, batch = gen_system(n, seed=1, device=dev, dtype=torch.float32)
        eager = bench_eager(model, z, pos, batch)
        try:
            graph, ok = bench_graph(model, z, pos, batch)
            print(f"{n:>6} {eager:>14.3f} {graph:>14.3f} {eager/graph:>8.2f}x {1000/graph:>14.1f}"
                  + ("" if ok else "  [forces non-finite!]"))
        except Exception as ex:  # noqa: BLE001
            print(f"{n:>6} {eager:>14.3f} {'CAPTURE FAILED':>14}   ({type(ex).__name__}: {str(ex)[:60]})")

    print("\n" + "=" * 74)
    print("BATCHING — many small molecules per forward (amortize launch overhead)")
    print("=" * 74)
    base_n = 42
    print(f"{'#mols':>6} {'atoms':>6} {'ms/step':>10} {'molecules/s':>12}")
    for m in (1, 8, 32):
        model, info = build_model(coulomb_cutoff=None, device=dev, dtype=torch.float32)
        model.eval()
        # concatenate m independent copies -> batch index identifies each molecule
        zs, ps, bs = [], [], []
        for i in range(m):
            zi, pi, _ = gen_system(base_n, seed=i, device=dev, dtype=torch.float32)
            zs.append(zi); ps.append(pi); bs.append(torch.full((base_n,), i, dtype=torch.long, device=dev))
        z = torch.cat(zs); pos = torch.cat(ps); batch = torch.cat(bs)
        ms = bench_eager(model, z, pos, batch, iters=20)
        print(f"{m:>6} {m*base_n:>6} {ms:>10.3f} {1000*m/ms:>12.1f}")


if __name__ == "__main__":
    main()
