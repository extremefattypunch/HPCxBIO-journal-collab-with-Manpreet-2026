#!/usr/bin/env python3
"""TensorNet2 MD inference+forces profiling harness (RTX 5070 Laptop, sm_120, 8 GB).

Single entry point used by all three profilers:
  * bare timing:        python harness/run_infer.py --workload B --n 1000 --steps 50
  * PyTorch Profiler:   python harness/run_infer.py --workload B --n 1000 --profile
  * Nsight Systems:     nsys profile ... python harness/run_infer.py --workload B --n 1000
  * Nsight Compute:     ncu --nvtx --nvtx-include "coulomb/" ... python harness/run_infer.py --ncu ...

NVTX ranges (for nsys/ncu targeting): per-submodule (via forward hooks) plus explicit
'charge_equilibration' (ChargePredict.qeq) and 'coulomb' (ScalarPlusWeightedCoulomb.pre_reduce),
which are the NEW TensorNet2 ops absent from the v1 study.

Forces are produced inside TorchMD_Net (derivative=True): model(z,pos,batch) -> (energy, forces).
We must NOT use torch.no_grad()/inference_mode() (the internal autograd.grad needs the graph).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")

import torch  # noqa: E402
from model_build import build_model  # noqa: E402
from workloads import gen_system, PRESETS  # noqa: E402


def _wrap_nvtx_methods():
    """Monkeypatch the two NEW v2 ops with NVTX ranges (no source edits)."""
    import torch.cuda.nvtx as nvtx
    try:
        from torchmdnet.models.tensornet2 import ChargePredict
        if not getattr(ChargePredict.qeq, "_nvtx", False):
            _orig = ChargePredict.qeq

            def qeq(self, *a, **k):
                nvtx.range_push("charge_equilibration")
                try:
                    return _orig(self, *a, **k)
                finally:
                    nvtx.range_pop()
            qeq._nvtx = True
            ChargePredict.qeq = qeq
    except Exception as e:  # noqa: BLE001
        print(f"[nvtx] could not wrap ChargePredict.qeq: {e}")
    try:
        from torchmdnet.models.output_modules import ScalarPlusWeightedCoulomb as C
        if not getattr(C.pre_reduce, "_nvtx", False):
            _orig2 = C.pre_reduce

            def pre_reduce(self, *a, **k):
                nvtx.range_push("coulomb")
                try:
                    return _orig2(self, *a, **k)
                finally:
                    nvtx.range_pop()
            pre_reduce._nvtx = True
            C.pre_reduce = pre_reduce
    except Exception as e:  # noqa: BLE001
        print(f"[nvtx] could not wrap Coulomb.pre_reduce: {e}")


def _add_module_nvtx_hooks(model):
    """Per-submodule NVTX ranges so nsys/ncu can target any stage by module path.

    IMPORTANT: nvtx.range_push/pop return an int (stack depth). A forward_pre_hook that
    returns non-None REPLACES the module args, and a forward_hook that returns non-None
    REPLACES the output. So the hooks MUST return None.
    """
    import torch.cuda.nvtx as nvtx

    def make(nm):
        def pre(mod, inp):
            nvtx.range_push(nm)          # returns int -> must NOT propagate
        def post(mod, inp, out):
            nvtx.range_pop()             # returns int -> must NOT propagate
        return pre, post

    for name, m in model.named_modules():
        if not name:
            continue
        short = name.split(".")[-1] if name.count(".") > 2 else name
        pre, post = make(short)
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)


def make_inputs(args, device):
    if args.workload == "A":
        n = PRESETS[args.preset]
    else:
        n = args.n
    z, pos, batch = gen_system(n, seed=args.seed, device=device, dtype=args.torch_dtype)
    return z, pos, batch, n


def one_step(model, z, pos, batch):
    # fresh leaf each step so autograd graph is clean; forces returned by the model
    p = pos.detach().clone()
    energy, forces = model(z, p, batch=batch)
    return energy, forces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["A", "B"], default="B")
    ap.add_argument("--preset", choices=list(PRESETS), default="aspirin_like")
    ap.add_argument("--n", type=int, default=1000, help="atoms (workload B)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--coulomb-cutoff", default="none",
                    help="'none' => all-to-all O(N^2); or a float in Angstrom")
    ap.add_argument("--no-coulomb", action="store_true",
                    help="ablation: plain Scalar head, no Coulomb energy term")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--q-dim", type=int, default=16)
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("--force-eager", action="store_true",
                    help="force pure-PyTorch backbone (OPT=False) instead of Warp")
    ap.add_argument("--nvtx", action="store_true", help="add per-module NVTX hooks")
    ap.add_argument("--profile", action="store_true", help="run under torch.profiler")
    ap.add_argument("--ncu", action="store_true",
                    help="single measured step inside NVTX ranges (for Nsight Compute)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.torch_dtype = torch.float32 if args.dtype == "fp32" else torch.float64
    cc = None if str(args.coulomb_cutoff).lower() == "none" else float(args.coulomb_cutoff)

    if not torch.cuda.is_available():
        sys.exit("CUDA not available — see env/verify_env.py")
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    _wrap_nvtx_methods()
    # The pure-PyTorch (non-Warp) backbone has a static-shapes dummy-atom off-by-one in
    # TensorNet2; use dynamic shapes for the eager path (Warp path keeps static_shapes=True).
    static = False if args.force_eager else None
    model, info = build_model(
        hidden_channels=args.hidden, q_dim=args.q_dim, num_layers=args.num_layers,
        coulomb_cutoff=cc, device=device, dtype=args.torch_dtype,
        force_eager=args.force_eager, static_shapes=static,
        use_coulomb=not args.no_coulomb,
    )
    model.eval()  # eval mode => 1st-order forces (create_graph=False); NOT torch.no_grad
    if args.nvtx or args.ncu:
        _add_module_nvtx_hooks(model)

    z, pos, batch, n = make_inputs(args, device)
    print(f"[config] N={n} {info} dtype={args.dtype} coulomb_cutoff={cc}")

    # warmup (Warp/Triton JIT, cuBLAS, allocator)
    for _ in range(args.warmup):
        e, f = one_step(model, z, pos, batch)
    torch.cuda.synchronize()
    finite = torch.isfinite(e).all().item() and torch.isfinite(f).all().item()
    print(f"[warmup done] energy={float(e.reshape(-1)[0]):.4f} forces_finite={finite} "
          f"peakVRAM={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    if not finite:
        sys.exit("non-finite energy/forces — check geometry/dtype")

    if args.ncu:
        # ONE measured instance for Nsight Compute to replay
        import torch.cuda.nvtx as nvtx
        nvtx.range_push("model_step")
        one_step(model, z, pos, batch)
        nvtx.range_pop()
        torch.cuda.synchronize()
        print("[ncu] single step done")
        return

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        os.makedirs(RESULTS, exist_ok=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True, with_stack=False) as prof:
            for _ in range(max(5, args.steps // 5)):
                one_step(model, z, pos, batch)
                torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        tag = f"{args.workload}_n{n}_{'eager' if args.force_eager else 'warp'}"
        trace = os.path.join(RESULTS, f"torchprof_{tag}.json")
        prof.export_chrome_trace(trace)
        print(f"[profile] chrome trace -> {trace}  (open at https://ui.perfetto.dev/)")
        return

    # bare timing with CUDA events
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(args.steps):
        one_step(model, z, pos, batch)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.steps
    print(f"[timing] {ms:.3f} ms/step  ({1000/ms:.1f} steps/s)  "
          f"peakVRAM={torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
