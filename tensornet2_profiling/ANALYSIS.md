# TensorNet2 GPU Profiling — Analysis & Findings

Extends Manpreet Singh's Triton/TensorNet-v1 paper (`../literature/32_*`) to **TensorNet2** with
**hardware-grounded** profiling. Hardware: **RTX 5070 Laptop (Blackwell, sm_120, 8 GB)**. Path:
**MD inference + first-order autograd forces** (`F = −dE/dx`). Scope: profiling + analysis (no
custom kernels this round).

## 0. Reproducibility

- torchmd-net `main` @ **`833c07708e`** · torch **2.13.0+cu130** · triton **3.7.1** ·
  nvidia-cutlass-dsl **4.6.1** · **Warp 1.15.0** · numpy 2.5.1 · Python 3.12.13 (conda env `tn2prof`).
- GPU cap **(12, 0)**; driver 610.43.02 / CUDA-UMD 13.3; CUDA toolkit 13.3 at `/opt/cuda`.
- **`warp_opt_active = True`** — Warp 1.15 JIT-compiles the TensorNet2 backbone kernels on sm_120
  (neighbors_brute, graph_transform, message_passing, de/compose_tensor, tensor_matmul_o3/so3,
  tensor_norm3, + their `_bwd`). So the Warp-optimized backbone is the *default* on this GPU.
- Model: `TensorNet2(hidden=128, q_dim=16, num_layers=2, num_rbf=32, cutoff=4.5, output_charges=True)`
  - `ScalarPlusWeightedCoulomb(coulomb_cutoff=None → all-to-all O(N²))` + `TorchMD_Net(derivative=True)`.
- **All three tools ran**: PyTorch Profiler + CUDA-event timing (§2–4), **Nsight Systems 2026.1.3**
  (§4/§7 timeline), **Nsight Compute 2026.2.1** (§7 per-kernel counters). Perf counters unlocked
  (`RmProfilingAdminOnly=0`).

## 1. Workloads (measured; synthetic jittered-lattice at ~0.09 atoms/Å³, batch 1)

| ID | N atoms | Coulomb | ms/step | steps/s | peak VRAM |
|----|--------:|---------|--------:|--------:|----------:|
| A  | 21 | all-to-all | 24.5 | 40.7 | 0.11 GB |
| A  | 42 | all-to-all | 28.1 | 35.6 | 0.15 GB |
| B  | 250 | all-to-all | 24.9 | 40.2 | 0.54 GB |
| B  | 500 | all-to-all | 28.7 | 34.8 | 1.12 GB |
| B  | 1000 | all-to-all | 49.8 | 20.1 | 2.62 GB |
| B  | 1500 | all-to-all | 86.3 | 11.6 | 4.54 GB |
| B  | 2000 | all-to-all | 130.6 | 7.7 | 6.92 GB |
| B  | 500 | all-to-all, **eager** (no Warp) | 29.8 | 33.5 | 0.76 GB |
| B  | 1000 | **scalar-only (no Coulomb)** | 35.1 | 28.5 | 1.84 GB |
| B  | 2000 | **scalar-only (no Coulomb)** | 66.5 | 15.0 | 3.61 GB |

**Max safe size = N≈2000** (6.92 GB of 8 GB). N=1000 is the primary profiling point.

## 2. Stage 1 — operator attribution (PyTorch Profiler)  [v2 analogue of the paper's Table 1]

### 2a. Warp path, all-to-all Coulomb, N=1000 (Self CUDA time; total 463.8 ms / 10 steps ≈ 46 ms/step)

| Kernel / op | Self CUDA % | note |
|-------------|------------:|------|
| `aten::mul` | **30.8** | Coulomb all-pairs elementwise (`q_i*q_j`, `fc*q_ij/d_ij`, `*qweights`); 20.6 GB transient, 1280 calls |
| `vectorized_elementwise_kernel` | 15.3 | elementwise (Coulomb / activation / damping) |
| `elementwise_kernel<128,2>` | 12.8 | elementwise |
| `aten::addmm` | 10.6 | **GEMM** (charge/output MLPs) |
| `aten::mm` | 9.3 | **GEMM** |
| `magma_sgemm` | 9.1 | **GEMM** backend |
| `aten::_index_put_impl_` | 7.4 | **scatter** (Coulomb `index_add`) |
| `aten::silu_backward` / `aten::silu` | 7.1 / 4.8 | activation |
| `aten::div` | 5.8 | Coulomb `1/d_ij` |
| `aten::sum` | 5.6 | reductions (Coulomb channel-mean, qeq `mol_sum`) |
| `cutlass_80_simt_sgemm` | 4.1 | **GEMM** |
| `vectorized_gather` / `aten::index` | 3.7 / 2.9 | **gather** (`charges[edge]`, `pos[edge]`) |
| `indexing_backward_kernel` (×2) | 3.5 + 3.3 | autograd of gather/scatter |
| `tensornet::radial_message_passing_bwd` (**Warp**) | **2.5** | the only backbone kernel in the top-25 |

**Category rollup (Warp path):** GEMM (MLPs) ≈ **33%** · Coulomb-dominated elementwise+div+sum ≈
**≥42%** · gather/scatter+their backward ≈ **21%** · activation ≈ **12%** · **Warp backbone ≈ 2.5%**.

### 2b. Eager path (Warp OFF), N=1000 — reproduces the v1 paper's regime

With the Warp backbone disabled, the **message-passing index ops resurface**: `aten::_index_put_impl_`
8.6%, `vectorized_gather` 5.1%, `aten::index` 3.4%, `aten::index_add` 3.3%, `indexFuncLargeIndex`
3.2%, `indexing_backward` 4.1+3.7% → backbone gather/scatter ≈ **15–20%** (vs **2.5%** with Warp).
These are the paper's `indexSelectLarge` / `indexFuncLargeIndex`. **`aten::mul` (Coulomb) is still #1
at 35.2%** — the new term dominates regardless of the backbone path.

**Takeaway:** torchmd-net's **Warp kernels already fuse the v1 message-passing bottleneck** (36% of
v1 time → ~2.5% here). The dominant cost has **moved to the new v2 all-pairs Coulomb** + the GEMM MLPs.

## 3. Scaling & ablation — the O(N²) Coulomb is isolated and quadratic

| N | total ms | scalar-only ms | **Coulomb Δ (ms)** | Coulomb share |
|---|--------:|---------------:|-------------------:|--------------:|
| 1000 | 49.8 | 35.1 | **14.7** | **30%** |
| 2000 | 130.6 | 66.5 | **64.1** | **49%** |

- **Coulomb term:** 14.7 → 64.1 ms = **4.4× for 2×N ≈ O(N²)**.
- **Local (scalar-only) part:** 35.1 → 66.5 ms = **1.9× for 2×N ≈ O(N)** (message passing + MLPs).
- ⇒ `T(N) ≈ T_local(N) + T_coulomb(N²)`; the Coulomb crossover makes it the **dominant** cost by N≈2000.
- The cutoff variant (`coulomb_cutoff=9.0`, N=1000) measured **292 ms/step** — *slower*, because a 9 Å
  cutoff in this dense synthetic box captures ~275 neighbours/atom **and rebuilds the neighbour list
  every step** (uncached), unlike the all-to-all triu indices (cached once). Not a like-for-like
  "Coulomb off" — use the scalar-only ablation above for that.

## 4. Small-molecule regime — launch-bound (CUDA-graph opportunity)

N=21 (24.5 ms) ≈ N=250 (24.9 ms): a **flat ~24 ms/step floor** independent of size ⇒ dominated by
**kernel-launch/Python-dispatch overhead**, not compute (many small kernels). This is the regime where
CUDA-graph capture (AceFF's route to >100 steps/s) pays off; our harness does not capture graphs.

## 5. ANSWERS

### Q1 — Where & how do we implement GPU acceleration?

Ranked by measured impact (N≥1000, MD inference+forces on sm_120):

1. **The new all-pairs Coulomb energy (`ScalarPlusWeightedCoulomb`, `coulomb_cutoff=None`)** — #1 op
   (`aten::mul` 30–35%), grows to ~**49% of step time at N=2000**, ~O(N²). It is a chain of
   **memory-bound elementwise (mul/div) + gather + reduction + scatter over ~N²/2 × 48-channel
   tensors** (20 GB of transient allocations/10 steps). **Nsight Compute confirms it is
   bandwidth-bound**: the dominant kernel runs at **91% DRAM peak but only 1.8% SM** (§7b) — it moves
   intermediates through DRAM rather than computing. *How:* two complementary levers —
   (a) **fuse** the gather→`q_i*q_j`→damp→`/d_ij`→channel-mean→scatter chain into one kernel to kill
   the huge intermediates (a **Triton** tiled pairwise kernel); (b) the **larger** win is
   **algorithmic** — replace all-pairs with cutoff+reaction-field / PME / FMM to change the N²
   asymptote. The profile justifies (a); scaling (§3) justifies (b).
2. **Charge equilibration (`ChargePredict.qeq` / `mol_sum`)** — new per-layer segmented reductions +
   elementwise; contributes to the `aten::sum`/elementwise share. *How:* fuse the
   `mol_sum → residual → redistribute` into one segmented-reduction kernel (Triton). (Nsight Compute
   will confirm whether it is atomic-contention- or launch-bound — §7.)
3. **Do NOT re-optimize the backbone.** torchmd-net's **Warp kernels already handle it** (message
   passing ≈ 2.5%). The v1 Triton contribution is largely **superseded by Warp on sm_120**.
4. **Do NOT rewrite the GEMMs** (~33%): they are already cuBLAS/CUTLASS (`aten::mm`/`addmm`/
   `magma`/`cutlass_sgemm`) — the same "leave GEMM alone" conclusion as the v1 paper.
5. **Small systems:** the win is **CUDA-graph capture** (kill the ~24 ms launch floor), not kernels.

### Q2 — Triton or CuTe DSL?

**Triton.** The hot path is **irregular, memory-bound gather/scatter + all-pairs Coulomb elementwise**
— Triton's exact sweet spot — with **no large isolated dense Tensor-Core-bound GEMM** left to justify
CuTe (the GEMMs are already optimal via cuBLAS/CUTLASS, and individually ≤11%). **Nsight Compute
settles it**: every hot kernel is memory-bound (Coulomb 91% DRAM / **1.8% SM**; backbone 86–93% DRAM;
qeq 7–8% occupancy) — there is **no compute-/Tensor-Core-bound kernel** for CuTe to accelerate. Empirically,
`nvidia-cutlass-dsl 4.6.1` **installs/imports on sm_120**, but consumer Blackwell `sm_120` is
**capability-restricted for the CuTe Python DSL** (no `tcgen05`, TN-layout only, FP4 hard-blocked to
`sm_100a` — CUTLASS #2800), and it is NVIDIA-only. CuTe would add complexity with **no matching
bottleneck** on this workload/GPU. *Reserve CuTe only if* a future change introduces a single large
dense TC-bound GEMM (none here). Escalation ladder outcome on sm_120: **eager/Warp → (torch.compile
fragile on sm_120) → Triton**; stop at Triton.

## 6. Why Nsight Compute adds value over the v1 (PyTorch-only) study — realized

Stage-1 localized *what/how much*; Nsight Compute supplied the *why* the v1 paper lacked, and it
**changed the conclusion per stage**: the three hot regions have **three different hardware
signatures** (memory-saturated / near-optimal / launch-bound), which alone dictate three different
(non-obvious) remedies. Effective-bandwidth estimates like the v1 paper's could not have separated
these. Details in §7.

## 7. Nsight Compute + Nsight Systems — RESULTS (N=1000, one MD step, sm_120)

### 7a. Nsight Systems timeline

- **Workload A (21 atoms): launch-bound.** Per-kernel GPU *execution* is ~1 µs (`KMed` ≈ 1.0–1.8 µs)
  while wall time is dominated by `cudaLaunchKernel` + host syncs (the static-shapes/all-to-all path
  emits `_assert_async` / `compare_scalar` / `reduce<bool>` sync kernels each step). The GPU is idle
  most of the step ⇒ **CUDA-graph capture** is the fix, not kernel tuning.
- **Workload B (N=1000):** NVTX grouping attributes the biggest single kernels to `coulomb/`
  (gather ≈ 628 µs, elementwise ≈ 573 µs, `q·q` mul ≈ 815 µs avg); Warp `radial_message_passing_bwd`
  ≈ 2.5%. Reports: `results/timeline_A21.nsys-rep`, `results/timeline_B1000.nsys-rep`.

### 7b. Nsight Compute per-kernel counters (roofline)

| Stage / kernel | dur (µs) | **DRAM %** | SM % | Occ % | Regime → remedy |
|---|---:|---:|---:|---:|---|
| **Coulomb** `q·q`/damp elementwise | **769** | **91.2** | 1.8 | 86 | **memory-bound, DRAM-saturated** → fuse to eliminate [~500k×48] intermediates; lower precision; fewer pairs |
| **Coulomb** gather `charges[edge]`,`pos[edge]` | 582 / 549 | 43 / 42 | 14 / 15 | 74 / **31** | memory-bound (one low-occupancy) → fuse gather into a tiled pairwise kernel |
| **Coulomb** scatter (`index_add`) | 28–31 | **80–81** | 12–44 | 74 | memory-bound → fuse into the same kernel |
| **Coulomb** `triu_indices` (pair gen) | 95 | 0.25 | **84** | 93 | compute-bound but small; don't rebuild each step (cache/persistent) |
| **Backbone** `message_passing_fwd` (Warp) | 150 | **90.4** | 25 | 87 | memory-bound **but near-peak & fused already → leave alone** |
| **Backbone** `message_passing_bwd` (Warp) | 271 | **86.4** | 23 | 58 | near-peak → leave alone |
| **Charge-equilib** `mol_sum` reduce / elementwise | 7–9 | 4–5 | 0–2.5 | **7–8** | **latency/occupancy-bound (tiny)** → fuse the small reductions (matters most for *batched* small molecules) |

**Reading:** the O(N²) Coulomb is emphatically **memory-bandwidth-bound** — its dominant kernel does
~2% compute at **91% of DRAM peak**, i.e. it is moving the huge pairwise intermediates through DRAM,
not computing. That is the exact profile where **fusion** (keep pair data in registers/SRAM, write
once) and **reduced precision** help, and where compute/Tensor-Core work (CuTe) would not. The Warp
backbone is *also* memory-bound but already saturates bandwidth at good occupancy — confirming it is
not worth re-optimizing. Charge equilibration is neither compute- nor bandwidth-bound — it is simply
too small to fill the GPU (7–8% occupancy) ⇒ a launch/fusion problem, amplified under batching.
Reports: `results/ncu_coulomb.ncu-rep`, `ncu_mp.ncu-rep`, `ncu_qeq.ncu-rep`.

## 8. Follow-on (out of scope this round)

1. **Triton fused pairwise-Coulomb kernel** (gather→charge-product→damp→channel-mean→scatter) —
   expect large intermediate-memory + launch reductions; validate vs the 30–49% budget.
2. **Algorithmic Coulomb** (cutoff+reaction-field / PME / FMM) to break the O(N²) asymptote — the
   bigger win at MD-relevant N.
3. **CUDA-graph capture** for the small-molecule launch-bound regime (§4).
