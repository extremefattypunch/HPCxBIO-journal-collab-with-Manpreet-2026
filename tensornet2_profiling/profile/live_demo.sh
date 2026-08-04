#!/usr/bin/env bash
# Guided, watch-it-live profiling demo for TensorNet2.
#
#   Top tmux pane : nvtop  (live GPU utilisation + VRAM)
#   Bottom pane   : a narrated sequence you step through with Enter
#   At the end    : opens the Nsight Systems + Nsight Compute GUIs
#
# Usage:   bash profile/live_demo.sh
#   (Detach the tmux view any time with:  Ctrl-b then d .  Reattach:  tmux attach -t tn2demo)
#
# Standalone (no tmux, two terminals yourself):
#   term 1:  nvtop
#   term 2:  bash profile/live_demo.sh --steps
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
ENV=tn2prof
PY="/opt/miniconda3/envs/${ENV}/bin/python"          # env interpreter (no activation needed)
[[ -x "$PY" ]] || PY="conda run -n ${ENV} --no-capture-output python"
SESSION=tn2demo
cd "$PROJ"
mkdir -p results

# ----------------------------------------------------------------------------- launcher
if [[ "${1:-}" != "--steps" ]]; then
  if ! command -v tmux >/dev/null; then
    echo "tmux not found — run the two-terminal way:"
    echo "  term 1:  nvtop"
    echo "  term 2:  bash profile/live_demo.sh --steps"
    exit 1
  fi
  command -v nvtop >/dev/null || { echo "nvtop not found (sudo pacman -S nvtop)"; exit 1; }
  tmux kill-session -t "$SESSION" 2>/dev/null
  # pane 0 (top) = live GPU monitor; pane 1 (bottom, 72%) = the guided steps
  tmux new-session -d -s "$SESSION" "nvtop || (echo 'nvtop exited'; sleep 5)"
  tmux split-window -v -l 72% -t "$SESSION" \
    "bash '$PROJ/profile/live_demo.sh' --steps; echo; echo '[demo done — GUI windows stay open]'; echo '[press any key to close this pane]'; read -n1"
  tmux set -g mouse on 2>/dev/null
  tmux select-pane -t "$SESSION".1
  exec tmux attach -t "$SESSION"
fi

# ----------------------------------------------------------------------------- steps
bold(){ printf '\033[1;36m%s\033[0m\n' "$*"; }
dim(){  printf '\033[2m%s\033[0m\n' "$*"; }
pause(){ printf '\033[1;33m\n>>> %s\033[0m'  "$1"; read -r _ ; }
# keep Warp/CUDA banner noise down but let real output through
FILT='Warp 1.15|CUDA Toolkit 12.9|Devices:|"cpu"|"cuda:0"|Kernel cache:|warp/1.15|Module .* load on device|SyntaxWarning|invalid escape|UserWarning|Consider using|run_backward'

clear
bold "==================================================================="
bold " TensorNet2 live profiling demo  —  watch the TOP pane (nvtop)"
bold "==================================================================="
dim  " GPU: RTX 5070 Laptop (Blackwell sm_120, 8 GB).  The top pane shows"
dim  " live GPU utilisation + VRAM.  Each step below prints its own timing."

pause "STEP 1/6 — baseline fp32 at N=2000. Watch VRAM in the top pane climb toward ~6.9 GB."
$PY harness/run_infer.py --precision fp32 --workload B --n 2000 --steps 20 --warmup 6 2>&1 | grep -vE "$FILT"

pause "STEP 2/6 — selective mixed BF16 Coulomb, same N=2000. Watch VRAM stay ~5.2 GB (-25%)."
$PY harness/run_infer.py --precision mixed_bf16 --workload B --n 2000 --steps 20 --warmup 6 2>&1 | grep -vE "$FILT"

pause "STEP 3/6 — CUDA graphs vs eager on small molecules (launch-bound). Expect 12-18x, tiny VRAM."
$PY experiments/cuda_graphs.py 2>&1 | grep -vE "$FILT"

pause "STEP 4/6 — Nsight SYSTEMS timeline capture (N=1000). Produces a .nsys-rep."
nsys profile --trace=cuda,nvtx,osrt --force-overwrite true -o results/demo_timeline \
  $PY harness/run_infer.py --workload B --n 1000 --steps 12 --warmup 6 --nvtx 2>&1 \
  | grep -vE "$FILT" | grep -iE "timing|generated|report|\.nsys-rep" | head
dim  "  (also printing the top GPU kernels from the trace:)"
nsys stats --force-export=true --report cuda_gpu_kern_sum results/demo_timeline.nsys-rep 2>/dev/null | sed -n '5,16p'

pause "STEP 5/6 — Nsight COMPUTE roofline capture of the Coulomb kernels. Watch the 'N passes' replay."
ncu --nvtx --nvtx-include "coulomb/" --set basic --section MemoryWorkloadAnalysis \
    --clock-control none -c 10 -f -o results/demo_coulomb --target-processes all \
    $PY harness/run_infer.py --ncu --workload B --n 1000 2>&1 \
  | grep -iE "==PROF==|error|ERR_NVGPUCTRPERM" | grep -vE "$FILT" | head -14

pause "STEP 6/6 — open the Nsight GUIs (windows appear under Hyprland)."
bold " launching nsys-ui  (timeline)  and  ncu-ui  (roofline/memory)…"
( nsys-ui results/demo_timeline.nsys-rep >/dev/null 2>&1 & )
( ncu-ui  results/demo_coulomb.ncu-rep  >/dev/null 2>&1 & )
dim  " In ncu-ui: open the 'GPU Speed Of Light' section to see 91% DRAM / ~2% SM on the Coulomb kernel."
dim  " In nsys-ui: zoom the NVTX row to see the 'coulomb' range and the ~1 us kernels."
echo
bold " Reports saved: results/demo_timeline.nsys-rep , results/demo_coulomb.ncu-rep"
