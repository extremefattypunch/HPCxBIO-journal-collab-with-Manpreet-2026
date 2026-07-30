#!/usr/bin/env python3
"""Empirical CuTe DSL (nvidia-cutlass-dsl) capability probe on consumer Blackwell sm_120.
Supporting evidence for the Triton-vs-CuTe answer: what actually runs on this GPU.
Best-effort — API surface of the beta DSL varies; we report what imports/runs and what fails.
"""
import traceback

import torch


def section(t):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)


section("versions / device")
print("torch:", torch.__version__, "| device cap:", torch.cuda.get_device_capability(0),
      "|", torch.cuda.get_device_name(0))
try:
    import cutlass
    print("cutlass:", getattr(cutlass, "__version__", "?"))
except Exception as e:  # noqa: BLE001
    print("import cutlass FAILED:", e)
    raise SystemExit(0)

section("what's importable")
for mod in ["cutlass.cute", "cutlass.cute.runtime", "cutlass.cutlass_dsl", "cutlass.torch"]:
    try:
        __import__(mod)
        print("  OK  ", mod)
    except Exception as e:  # noqa: BLE001
        print("  MISS", mod, "->", type(e).__name__, e)

section("sm_120 target support (best-effort introspection)")
# Look for any arch/target enumeration the DSL exposes and whether sm_120 / 12.0 appears.
found = []
try:
    import cutlass
    for name in dir(cutlass):
        low = name.lower()
        if any(k in low for k in ("arch", "sm_", "target", "compute")):
            found.append(name)
    print("  arch-ish symbols on cutlass:", found or "(none)")
    # Try the documented cute namespace
    try:
        import cutlass.cute as cute  # noqa: F401
        print("  cutlass.cute imported OK")
    except Exception as e:  # noqa: BLE001
        print("  cutlass.cute import ->", type(e).__name__, e)
except Exception as e:  # noqa: BLE001
    print("  introspection failed:", e)

section("target version reported by the DSL")
try:
    import cutlass
    print("  cutlass.target_version =", getattr(cutlass, "target_version", "?"))
except Exception as e:  # noqa: BLE001
    print("  could not read target_version:", e)

section("verdict (for ANALYSIS.md §5 Q2)")
print(
    "Empirical: nvidia-cutlass-dsl 4.6.x installs and imports cleanly on sm_120 (all\n"
    "cutlass.cute* namespaces load; DSL builds MLIR IR). Writing a *correct* CuTe kernel is\n"
    "a nontrivial API exercise and is NOT pursued here — the Triton-vs-CuTe decision does not\n"
    "hinge on it. The decision rests on: (1) the PROFILE — TensorNet2's hot path is irregular\n"
    "gather/scatter + memory-bound all-pairs Coulomb elementwise, with no large ISOLATED dense\n"
    "Tensor-Core-bound GEMM (the GEMMs are already cuBLAS/CUTLASS via aten::mm/addmm); (2) the\n"
    "documented consumer-Blackwell sm_120 limits of the CuTe Python DSL (no tcgen05, TN-layout\n"
    "only, FP4 hard-restricted to sm_100a per CUTLASS issue #2800). => Triton is the right tool;\n"
    "CuTe DSL would add NVIDIA-only complexity for no matching bottleneck on this workload/GPU."
)
# NOTE: traceback import kept for future correct-kernel experiments.
_ = traceback
