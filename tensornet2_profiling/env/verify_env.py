#!/usr/bin/env python3
"""Environment gate for the TensorNet2 profiling study (RTX 5070 Laptop, sm_120, 8 GB).

Run INSIDE the conda env:  conda run -n tn2prof python env/verify_env.py
Exits non-zero if any CORE check fails (torch CUDA on sm_120). Nsight/CuTe/torchmd-net
are reported but treated as non-fatal so this can be run before every piece is installed.
"""
import shutil
import subprocess
import sys

OK, WARN, BAD = "\033[92mOK\033[0m", "\033[93mWARN\033[0m", "\033[91mFAIL\033[0m"
core_failed = False


def line(status, msg):
    print(f"[{status}] {msg}")


print("=" * 70)
print("TensorNet2 profiling — environment verification")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
if sys.version_info[:2] == (3, 14):
    line(WARN, "Python 3.14 has no CUDA torch wheels (PyTorch #169929) — use 3.12")

# ---- torch + CUDA (CORE) ----------------------------------------------------
try:
    import torch

    line(OK, f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        line(BAD, "torch.cuda.is_available() == False")
        core_failed = True
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        line(OK, f"device: {name}, capability sm_{cap[0]}{cap[1]}")
        if cap != (12, 0):
            line(WARN, f"expected (12, 0) for RTX 5070 Blackwell, got {cap}")
        line(OK if any('sm_120' in a or '120' in a for a in torch.cuda.get_arch_list())
             else WARN,
             f"arch_list={torch.cuda.get_arch_list()} "
             "(no sm_120 is fine IF a real op below runs via PTX-JIT)")
        # The decisive test: does a real kernel actually execute on this GPU?
        try:
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda")
            c = (a @ b).sum()
            torch.cuda.synchronize()
            line(OK, f"real CUDA matmul executed (checksum finite: {torch.isfinite(c).item()})")
        except Exception as e:  # noqa: BLE001
            line(BAD, f"CUDA matmul FAILED (likely 'no kernel image' on sm_120): {e}")
            core_failed = True
except Exception as e:  # noqa: BLE001
    line(BAD, f"import torch failed: {e}")
    core_failed = True

# ---- triton (needed only if we later lower via torch.compile) ---------------
try:
    import triton

    line(OK, f"triton {triton.__version__} "
             "(sm_120 codegen is fragile — see plan Risks; set "
             "TRITON_PTXAS_PATH=/opt/cuda/bin/ptxas, TORCH_CUDA_ARCH_LIST=12.0)")
except Exception as e:  # noqa: BLE001
    line(WARN, f"triton not importable: {e}")

# ---- torchmd-net (the model under study) ------------------------------------
try:
    import torchmdnet
    from torchmdnet.models import model as tmn_model  # noqa: F401

    line(OK, f"torchmdnet {getattr(torchmdnet, '__version__', '?')} importable")
except Exception as e:  # noqa: BLE001
    line(WARN, f"torchmdnet not importable yet: {e}")

# ---- CuTe DSL (installed only to demonstrate the Triton-vs-CuTe verdict) -----
try:
    import cutlass  # noqa: F401

    line(OK, "nvidia-cutlass-dsl importable (expect sm_120 restrictions; CUTLASS #2800)")
except Exception as e:  # noqa: BLE001
    line(WARN, f"cutlass (CuTe DSL) not importable: {e}")

# ---- profilers on PATH ------------------------------------------------------
for tool, why in (("ncu", "Nsight Compute (pacman -S nsight-compute)"),
                  ("nsys", "Nsight Systems (pacman -S nsight-systems)")):
    path = shutil.which(tool)
    line(OK if path else WARN, f"{tool}: {path or 'NOT on PATH — ' + why}")

# ---- perf-counter permission (required for ncu HW counters) -----------------
try:
    with open("/proc/driver/nvidia/params") as f:
        params = f.read()
    val = next((l.split(":")[1].strip() for l in params.splitlines()
                if "RmProfilingAdminOnly" in l), "?")
    if val == "0":
        line(OK, "RmProfilingAdminOnly=0 — ncu HW counters allowed for non-root")
    else:
        line(WARN, f"RmProfilingAdminOnly={val} — ncu will ERR_NVGPUCTRPERM unless run "
                   "as root or the modprobe fix is applied (see env/setup.sh step 1e)")
except OSError:
    line(WARN, "could not read /proc/driver/nvidia/params")

print("=" * 70)
if core_failed:
    line(BAD, "CORE checks failed — fix torch/CUDA before profiling.")
    sys.exit(1)
line(OK, "Core torch/CUDA checks passed.")
