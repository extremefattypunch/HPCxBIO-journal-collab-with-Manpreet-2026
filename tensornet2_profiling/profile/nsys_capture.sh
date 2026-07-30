#!/usr/bin/env bash
# Stage-1 whole-run timeline with Nsight Systems (launch overhead, CPU gaps, CUDA-graph effect).
# Usage: profile/nsys_capture.sh <tag> -- <python args...>
#   e.g. profile/nsys_capture.sh workloadA -- --workload A --steps 100
# Produces results/timeline_<tag>.nsys-rep (open with `nsys-ui`) and a .sqlite stats export.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: nsys_capture.sh <tag> -- <python args...>}"; shift
[[ "${1:-}" == "--" ]] && shift

OUT="results/timeline_${TAG}"
# --trace: cuda kernels+memops, nvtx ranges, python + OS runtime so we see launch gaps.
# --cuda-graph-trace=node so CUDA-graph-captured launches show as individual nodes.
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=node \
  --python-backtrace=cuda \
  --force-overwrite true \
  -o "${OUT}" \
  python harness/run_infer.py "$@"

echo "== nsys stats (top CUDA kernels + summary) =="
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "${OUT}.nsys-rep" || true
echo "Timeline: ${OUT}.nsys-rep  (nsys-ui ${OUT}.nsys-rep)"
