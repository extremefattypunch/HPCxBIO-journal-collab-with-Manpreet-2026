#!/usr/bin/env bash
# Environment setup for the TensorNet2 profiling study.
# Verified machine (2026-07-30): Arch Linux, RTX 5070 Laptop (Blackwell, sm_120, 8 GB),
# driver 610.43.02 / CUDA-UMD 13.3, CUDA toolkit 13.3 at /opt/cuda, conda 26.1.1.
#
# Sections 1-2 are NON-sudo and are run by the setup (conda env `tn2prof`, Python 3.12).
# Sections 3-4 require sudo + a REBOOT and MUST be run by you.  Section 6 re-verifies.
set -euo pipefail
ENV=tn2prof

echo "== 1. conda env + PyTorch (cu128 wheels ship native sm_120 cubin) =="
# conda create -y -n "$ENV" -c conda-forge python=3.12
# conda run -n "$ENV" pip install numpy
# conda run -n "$ENV" pip install torch --index-url https://download.pytorch.org/whl/cu128
#   -> verified: torch 2.11.0+cu128, arch_list includes sm_120, triton 3.6.0, real matmul OK.
#   If a future torch drops sm_120 cubin, fall back to the cu130 index or nightly.

echo "== 2. torchmd-net (from source main; TensorNet2 may post-date the PyPI release) + CuTe DSL =="
# git clone --depth 1 https://github.com/torchmd/torchmd-net third_party/torchmd-net
# conda run -n "$ENV" pip install -e third_party/torchmd-net   # NO nvcc build; Warp JITs at runtime
# conda run -n "$ENV" pip install nvidia-cutlass-dsl           # CuTe DSL: install only to DEMONSTRATE
#   the Triton-vs-CuTe verdict; expect consumer sm_120 restrictions (CUTLASS #2800), do not rely on it.

echo "== 3. [SUDO] Nsight tools (ncu not bundled in /opt/cuda; the Arch extra repo has 2026.2.x) =="
echo "   sudo pacman -S nsight-compute nsight-systems"
echo "   ncu --version   # expect >= 2026.2 (supports Blackwell sm_120)"

echo "== 4. [SUDO + REBOOT] perf-counter permission (else ncu => ERR_NVGPUCTRPERM) =="
echo "   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf"
echo "   sudo mkinitcpio -P    # nvidia is early-loaded on Arch; rebuild initramfs"
echo "   sudo reboot"
echo "   # after reboot: grep RmProfilingAdminOnly /proc/driver/nvidia/params  -> must print 0"
echo "   # ALTERNATIVE (no reboot): prefix every profile with 'sudo' (set NCU_BIN='sudo ncu')."

echo "== 5. Triton sm_120 workaround (only needed if torch.compile / custom Triton is used) =="
echo "   export TRITON_PTXAS_PATH=/opt/cuda/bin/ptxas   # CUDA 13.3 ptxas (verified present)"
echo '   export TORCH_CUDA_ARCH_LIST="12.0"'

echo "== 6. verify =="
echo "   conda run -n $ENV python env/verify_env.py"
