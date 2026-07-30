# TensorNet2 GPU Profiling (PyTorch Profiler + Nsight Systems + Nsight Compute)

A profiling **study** of TensorNet2 (torchmd-net) on an RTX 5070 Laptop (Blackwell, `sm_120`,
8 GB), extending Manpreet Singh's Triton/TensorNet-v1 paper (`../literature/32_*`) with
**hardware-counter evidence** and answering: *where to accelerate* and *Triton vs CuTe DSL*.
Scope this round: **profiling + analysis only** (no custom kernels); path = **MD inference +
autograd forces**. See `../.claude/plans/...` for the full plan.

## Layout

- `env/` — `setup.sh` (install runbook), `verify_env.py` (gate; run first)
- `harness/` — `model_build.py` (TensorNet2 + Coulomb, forces), `workloads.py` (VRAM-safe systems),
  `run_infer.py` (single entry for all profilers; adds NVTX ranges incl. the NEW `charge_equilibration`
  - `coulomb` ops)
- `profile/` — `nsys_capture.sh`, `ncu_sections.sh`, `run_torch_profiler.py`
- `results/` — reports (git-ignored)
- `ANALYSIS.md` — the deliverable; fill with measured numbers + the two answers

## Run order

```bash
# 0. one-time env (see env/setup.sh; sections 3-4 need sudo + reboot for ncu counters)
conda run -n tn2prof python env/verify_env.py            # must end "Core checks passed"
conda run -n tn2prof python harness/model_build.py        # smoke test: finite energy+forces

# 1. Stage-1 whole-run timeline (launch overhead / CUDA-graph story)
conda run -n tn2prof bash profile/nsys_capture.sh B1000 -- --workload B --n 1000 --nvtx

# 2. Stage-1 operator attribution (v2 analogue of the paper's Table 1) — eager vs warp
conda run -n tn2prof python harness/run_infer.py --profile --workload B --n 1000
conda run -n tn2prof python harness/run_infer.py --profile --workload B --n 1000 --force-eager

# 3. Stage-2 per-kernel hardware counters (Nsight Compute) on the NEW ops
conda run -n tn2prof bash profile/ncu_sections.sh charge_equilibration qeq     basic -- --workload B --n 1000
conda run -n tn2prof bash profile/ncu_sections.sh coulomb              coulomb full  -- --workload B --n 1000
conda run -n tn2prof bash profile/ncu_sections.sh message_passing      mp      basic -- --workload B --n 1000
#   If RmProfilingAdminOnly=1: prefix with sudo, i.e. NCU_BIN="sudo ncu" (env var) or run as root.

# 4. Scaling + ablation
for N in 250 500 1000; do conda run -n tn2prof python harness/run_infer.py --workload B --n $N; done
conda run -n tn2prof python harness/run_infer.py --workload B --n 1000 --coulomb-cutoff 9.0   # cutoff vs all-to-all
```

## Key knobs (`run_infer.py`)

`--workload {A,B}` · `--n` atoms · `--coulomb-cutoff none|<Å>` (none = **O(N²) all-to-all**) ·
`--force-eager` (pure-PyTorch backbone vs Warp) · `--profile` / `--ncu` / `--nvtx` · `--dtype fp32|fp64`.

## Notes

- `model(z,pos,batch)` returns `(energy, forces)`; forces are 1st-order (`eval()` ⇒ `create_graph=False`).
  Do **not** wrap in `torch.no_grad()` — the model's internal `autograd.grad` needs the graph.
- `warp_opt_active` (printed in `[config]`) tells you whether the Warp backbone or the naive
  PyTorch fallback ran — record it in `ANALYSIS.md`.
- 8 GB VRAM: watch `peakVRAM` in the output; back off `--n` on OOM.
