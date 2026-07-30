#!/usr/bin/env bash
# Stage-2 per-kernel hardware counters with Nsight Compute, targeted + VRAM-safe (8 GB).
#
# ncu replays each kernel several times; --set full is minutes/kernel and can OOM on 8 GB, so:
#   * target ONE stage by NVTX range (--nvtx --nvtx-include "<range>/")
#   * profile ONE instance (--launch-count 1)
#   * lock clocks for comparability (--clock-control base)
#   * start with --set basic; escalate to --set full only on confirmed bottlenecks
#
# If RmProfilingAdminOnly=1 (see verify_env.py), prefix with sudo OR apply the modprobe fix.
#
# Usage:
#   profile/ncu_sections.sh <range> <tag> [basic|full] -- <python args...>
# Examples:
#   profile/ncu_sections.sh message_passing mp basic -- --workload B --n 1000
#   profile/ncu_sections.sh coulomb coulomb full     -- --workload B --n 1000
#   profile/ncu_sections.sh charge_equilibration qeq basic -- --workload B --n 1000
set -euo pipefail
cd "$(dirname "$0")/.."

RANGE="${1:?usage: ncu_sections.sh <nvtx_range> <tag> [basic|full] -- <python args...>}"
TAG="${2:?missing <tag>}"
SET="${3:-basic}"
shift 3
[[ "${1:-}" == "--" ]] && shift

SECTIONS=(--section SpeedOfLight --section Occupancy)
if [[ "$SET" == "full" ]]; then
  SECTIONS=(--section SpeedOfLight --section MemoryWorkloadAnalysis \
            --section Occupancy --section SchedulerStats --section WarpStateStats \
            --section ComputeWorkloadAnalysis)
fi

OUT="results/${TAG}_${SET}"
NCU_BIN="${NCU_BIN:-ncu}"   # set NCU_BIN="sudo ncu" if the permission fix isn't applied

set -x
$NCU_BIN --set "$SET" "${SECTIONS[@]}" \
  --nvtx --nvtx-include "${RANGE}/" \
  --launch-count 1 \
  --clock-control base \
  --target-processes all \
  -f -o "$OUT" \
  python harness/run_infer.py --ncu "$@"
set +x

echo "Report: ${OUT}.ncu-rep   (open: ncu-ui ${OUT}.ncu-rep ; CSV: ncu -i ${OUT}.ncu-rep --csv --page raw > ${OUT}.csv)"
