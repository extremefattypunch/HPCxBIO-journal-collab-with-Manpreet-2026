# What Actually Bottlenecks TensorNet2? A Hardware-Grounded Study on Consumer Blackwell, and Selective Mixed-Precision Acceleration of Learned-Charge Electrostatics

*Working draft for the EurIPS/NeurIPS SimBioChem (AI4Science) workshop. Companion to Singh (2025),
"Accelerating Molecular Simulations with Triton: Fused GPU Kernels for TensorNet Neural Potentials."*

## Abstract

The prior work accelerated **TensorNet v1** by Triton **kernel fusion** of its message-passing
backbone. We revisit acceleration for **TensorNet2** (the AceFF-2 architecture, which adds learned
per-atom charges, neutral-charge equilibration, and an explicit Coulomb energy) on a **consumer
Blackwell GPU (RTX 5070 Laptop, `sm_120`, 8 GB)** using the first **hardware-counter** study of the
model (PyTorch Profiler + Nsight Systems + **Nsight Compute** roofline). Our characterization
overturns the natural sequel: the v1 fusion target is **already superseded** by torchmd-net's NVIDIA
Warp kernels (message passing ≈ 2.5% of step, 86–93% of DRAM peak). The cost has **moved**, and it is
**regime-dependent**: (i) an **O(N²) all-pairs Coulomb** that is memory-bandwidth-saturated (91% DRAM,
1.8% SM) and grows from 30% → 49% of step time as N goes 1000 → 2000; (ii) **launch-bound** small
molecules (~24 ms/step floor, ~1 µs kernel execution); (iii) an **occupancy-bound** charge-equilibration
(7–8%). We turn this into a **regime→remedy taxonomy** and validate the non-fusion remedies each regime
calls for. As the central new technique, we introduce **selective mixed precision** for the
bandwidth-bound Coulomb — BF16 for the per-edge charge intermediates, fp32 for geometry and
accumulation — reaching **~15% end-to-end speedup and −25% VRAM at N=2000** while **preserving rotational
equivariance** (7e-5) at ~0.5% relative force error; we show *uniform* low precision is not even
supported (the Warp neighbor kernel rejects fp16), so selectivity is necessary, and that the win only
materializes once charges are cast **before** the gather (an Nsight-guided fix). We further quantify
that CUDA-graph capture removes the launch floor (**12–18× at N=21–42**, 517–650 steps/s) and that
batching amortizes it (8 molecules at the wall-time of 1). We situate algorithmic long-range
electrostatics (PME / Ewald message passing / latent-Ewald) as the large-N path and analyze when it
pays off. All results are reproducible on an 8 GB laptop with shared Nsight reports.

## 1. Contributions (each is novel relative to the v1 fusion paper)

1. **First hardware-grounded characterization of TensorNet2/AceFF on consumer Blackwell** (PyTorch
   Profiler → nsys timeline → **ncu roofline/occupancy/DRAM**), extending v1's profiler-only method —
   and it *changes the conclusion*: v1's message-passing fusion is superseded by Warp.
2. **A regime→remedy taxonomy** for equivariant-MLFF MD (launch-bound → CUDA graphs; memory-bound
   Coulomb → fewer bytes/fewer pairs; occupancy-bound qeq → batching), each backed by counters.
3. **Selective mixed-precision (BF16) for the memory-bound Coulomb** — a new technique (torchmd-net
   has only *uniform* dtype; Warp has no bf16 kernels) — with a force-accuracy / rotational-equivariance
   / precision analysis and an Nsight-guided implementation insight.
4. **Quantified systems results** (benchmarking existing torchmd-net capability): CUDA-graph speedups
   in the launch-bound regime and batched throughput for high-throughput screening.
5. **A scaling analysis** placing algorithmic long-range electrostatics as the large-N path, with prior
   art delineated so it is future work, not an overclaim.

## 2. Setup

- **GPU** RTX 5070 Laptop (Blackwell, `sm_120`, 8 GB); driver 610.43.02 / CUDA-UMD 13.3.
- **Software** torchmd-net `main`@`833c077`; torch 2.13.0+cu130 (native `sm_120`); Warp 1.15
  (`warp_opt_active=True`); Python 3.12. Tools: PyTorch Profiler, Nsight Systems 2026.1.3, Nsight
  Compute 2026.2.1 (perf counters via `NVreg_RestrictProfilingToAdminUsers=0`).
- **Model** `TensorNet2(hidden=128, q_dim=16, num_layers=2, num_rbf=32, cutoff=4.5, output_charges=True)`
  - `ScalarPlusWeightedCoulomb(coulomb_cutoff=None → all-to-all O(N²))` + `TorchMD_Net(derivative=True)`;
  path = MD **inference + first-order autograd forces**. Harness: `tensornet2_profiling/`.
- **Workloads** jittered-lattice systems at realistic density (~0.09 atoms/Å³), N=21…2000, batch 1;
  fp64 used as the precision reference. (Precision study uses random-init weights → numbers are
  *numerical-precision* magnitudes; a trained AceFF-2 checkpoint would refine absolute values.)

## 3. Characterization (contributions 1–2)  [details: `ANALYSIS.md`]

**Operator attribution (N=1000):** Coulomb elementwise (`aten::mul`) is #1 at 30.8%; GEMM MLPs ~33%
(already cuBLAS/CUTLASS); the Warp backbone kernel is 2.5%. **Scaling+ablation:** the Coulomb term is
14.7 ms (30%) at N=1000 and 64.1 ms (49%) at N=2000 — 4.4× for 2×N (≈O(N²)) — while the rest is ~linear.
**Nsight Compute roofline:**

| Stage | dur (µs) | **DRAM %** | SM % | Occ % | Regime |
|---|---:|---:|---:|---:|---|
| Coulomb `q·q`/damp elementwise | 769 | **91** | 1.8 | 86 | memory-bandwidth-saturated |
| Coulomb gather (`charges[edge]`) | 582/549 | 43/42 | 14/15 | 74/31 | memory-bound |
| Warp `message_passing` fwd/bwd | 150/271 | **90/86** | 25/23 | 87/58 | memory-bound **but near-peak (leave alone)** |
| charge-equilibration `mol_sum` | 7–9 | 4–5 | ~1 | **7–8** | latency/occupancy-bound |

**nsys** confirms small-N is launch-bound (~1 µs kernel exec; ~24 ms wall floor).

### Regime → remedy taxonomy (the paper's organizing table)

| Regime | Signature | Remedy (this paper) |
|---|---|---|
| Small N (screening) | launch-bound, ~24 ms floor | **CUDA graphs** (§5) + **batching** (§5) |
| Large N | O(N²) Coulomb, 91% DRAM | **selective mixed precision** (§4); algorithmic long-range (§6, future) |
| Per-layer charges | qeq 7–8% occupancy | **batching** (§5) |
| Backbone / GEMM | already near-peak / cuBLAS | **leave alone** (Warp/cuBLAS) |

## 4. Selective mixed precision for the memory-bound Coulomb (contribution 3, NEW)

**Design.** Cast the bandwidth-heavy per-edge tensors (gathered charges, `q_i·q_j`, damping, `e_ij`;
shape [E≈N²/2, C=48]) to **BF16**, keep **positions/distances fp32** (geometry → equivariance) and the
`index_add` **accumulation fp32** (stable reduction). Implemented as a drop-in patch of
`ScalarPlusWeightedCoulomb.pre_reduce` (`harness/mixed_precision.py`); a port check (BF16→fp32) matches
stock fp32 to **1e-6**.

**Why selective (not uniform).** Uniform fp16 **fails**: torchmd-net's Warp neighbor-list kernel raises
"Unsupported dtype" for fp16 positions. Low precision must therefore be applied *selectively* to the
Coulomb math while geometry stays fp32 — which also protects equivariance.

**Nsight-guided implementation insight.** A first attempt cast charges *after* the gather → the
(memory-bound) gather still moved fp32 bytes and extra cast kernels appeared; ncu showed **no speedup**
(Coulomb range 2365→2673 µs). Casting charges **once, before** the gather captured the win.

**Speed + memory (fixed impl):**

| N | fp32 ms | mixed_bf16 ms | speedup | fp32 VRAM | mixed VRAM |
|---|---:|---:|---:|---:|---:|
| 1000 | 50.0 | 44.9 | 1.11× | 2.62 GB | 2.18 GB (−17%) |
| 2000 | 131.4 | 112.1 | 1.17× | 6.92 GB | 5.20 GB (**−25%**) |

**ncu (Coulomb range, N=1000):** total 2365 → **1923 µs (−19%)**; the dominant elementwise **768 → 356 µs**
(halved bytes on a DRAM-bound op); gather **DRAM 43% → 20%** (bytes halved). *Secondary finding:* once
halved, the gathers become **latency-bound** (DRAM 20%) — the next win requires **fusion** (future work),
i.e. profiling reveals the successor bottleneck.

**Accuracy / equivariance (vs fp64, same weights):**

| precision | rel force err | max abs force err | rel energy err | equivariance err |
|---|---:|---:|---:|---:|
| fp32 | 2e-6 | 1e-4 | ~5e-6 | 4e-6 |
| **mixed_bf16** | **~5e-3** | ~0.1 | **1–3e-2** | **7e-5** |

Forces tolerate BF16 better than the energy sum (forces differentiate through the fp32 positions);
**equivariance is preserved** (7e-5) because geometry stays fp32. Honest trade-off: ~0.5% force error and
1–3% energy error for ~15% speed and −25% VRAM at N=2000.

## 5. Systems remedies for the launch/occupancy regimes (contribution 4, quantified)

CUDA graphs are supported by torchmd-net (`calculators.py`); we **benchmark** them.

| N | eager ms | graph ms | speedup | graph steps/s |
|---|---:|---:|---:|---:|
| 21 | 27.6 | 1.54 | **17.9×** | 650 |
| 42 | 24.4 | 1.94 | **12.6×** | 517 |
| 250 | 27.9 | 8.50 | 3.3× | 118 |

Graphs collapse the ~24 ms launch floor → 500–650 steps/s at ligand sizes (past AceFF's >100 steps/s
bar); the speedup shrinks as compute grows (regime crossover). **Batching** amortizes launch overhead:
8 molecules (336 atoms) run at the wall-time of 1 (24.7 vs 25.9 ms) → **8× throughput**; 32 → 669 mol/s —
the remedy for the occupancy-bound qeq in high-throughput screening.

## 6. Scaling analysis & algorithmic long-range (contribution 5, positioned as future work)

Our N-sweep shows the O(N²) Coulomb overtakes the linear remainder by N≈2000. Beyond precision, the
asymptotic fix is algorithmic — cutoff+reaction-field (**already in torchmd-net**, not novel), or
**PME O(N log N)** / **Ewald message passing** (Kosmala et al., ICML 2023) / **latent-Ewald** (Cheng,
npj Comput. Mater. 2025). We do **not** claim these: differentiable **PME with Cartesian-tensor message
passing** was published in 2026, and RF is shipped. We frame them as the large-N path and note they lie
partly outside AceFF's small-molecule design regime (where §5 dominates).

## 7. Related work & novelty defense

- **vs v1 (Singh 2025):** different bottleneck (its fusion target is Warp-handled now), different GPU
  generation (A100→consumer Blackwell), different levers (measurement + precision + systems, not fusion).
- **vs AceFF-2 (Farr et al. 2026):** they introduced the architecture + Coulomb and cite Warp/CUDA
  graphs; we give the first **hardware-counter** cost breakdown on consumer Blackwell and a new
  precision technique.
- **CUDA graphs / reaction-field / uniform dtype are existing** → we *quantify/cite*, never claim.
  The new contributions are the **characterization**, the **taxonomy**, and **selective mixed precision
  with the equivariance/precision analysis**.
- **vs long-range MLFF literature** (DeePMD, 4G-HDNNP, SpookyNet, AIMNet2, PME/LES) → cited as the
  large-N future path, not competed with.

## 8. Limitations

Random-init weights for the precision study (magnitudes indicative, not physical — a trained AceFF-2
checkpoint would sharpen the accuracy table and enable a real NVE energy-drift test); single consumer
GPU (no multi-GPU); non-periodic all-pairs Coulomb regime; mixed-precision energy error (1–3%) is at the
high end and motivates the fusion follow-on the ncu data points to.

## 9. Reproducibility

Harness, scripts, and Nsight reports in `tensornet2_profiling/` (`ANALYSIS.md`,
`results/{precision_accuracy.txt, cuda_graphs.txt, ncu_coulomb*.ncu-rep, timeline_*.nsys-rep}`); each
number ties to an exact CLI (`README.md`). Env: `env/setup.sh` + `env/verify_env.py`.

## Key references

Singh (2025, EurIPS SimBioChem, v1 fusion) · Farr et al. (2026, AceFF-2 / TensorNet2, arXiv:2601.00581) ·
Kosmala et al. (2023, ICML, Ewald message passing, arXiv:2303.04791) · Cheng (2025, npj Comput. Mater.,
latent Ewald, arXiv:2408.15165) · differentiable PME + Cartesian-tensor MP (2026, arXiv:2606.01598) ·
Micikevicius et al. (2018, mixed-precision training, arXiv:1710.03740) · Doerr et al. (2021, TorchMD).
