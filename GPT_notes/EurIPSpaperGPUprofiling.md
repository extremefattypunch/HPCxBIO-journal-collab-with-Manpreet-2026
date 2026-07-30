## Overview

The paper used **profiling-driven optimization**: first measure which CUDA kernels dominate an ordinary TensorNet forward pass, then inspect the corresponding PyTorch operations, write fused Triton replacements, and benchmark each replacement before integrating it.

The profiling itself was performed with **PyTorch’s built-in profiler** on an NVIDIA A100, rather than with NVIDIA Nsight Systems or Nsight Compute. The paper reports kernel execution times, percentages of total CUDA time, and invocation counts, but does not report low-level hardware counters such as cache-hit rates, achieved occupancy, warp stalls, or DRAM transaction counts.

---

# 1. The baseline being profiled

The authors deliberately profiled the normal, uncompiled PyTorch implementation:

* **Model:** TensorNet as implemented in TorchMD-NET
* **Execution mode:** PyTorch eager mode
* **No `torch.compile`:** this was excluded from the baseline
* **GPU:** NVIDIA A100-SXM4 with 80 GB memory
* **CPU:** AMD EPYC 7763
* **System memory:** 512 GB
* **PyTorch:** 2.7.1
* **CUDA:** 12.6
* **Triton:** 2.0

Using eager mode matters because each PyTorch operator normally becomes one or more separately launched CUDA kernels. That exposes exactly the launch fragmentation and intermediate-memory traffic that Triton fusion is intended to eliminate.

The paper argues that `torch.compile` was not used because its general-purpose compiler did not effectively fuse the graph-neural-network and index-based patterns appearing in TensorNet.

---

# 2. The molecular workload used for profiling

The primary profiler run used a synthetic or representative molecular graph with:

* **4,096 atoms**
* **Batch size:** 1
* **Cutoff radius:** 5.0 Å
* **Average neighbors per atom:** 32
* **Total graph edges:** 131,072

The edge count follows directly from

[
4096 \times 32 = 131072.
]

This is important because TensorNet’s computational cost is controlled not only by atom count (N), but also by the number of neighbor-list edges (E). Message passing executes over edges, while the aggregated outputs are written to atoms.

The authors describe 4,096 atoms as representative of a medium-sized protein. It is large enough that GPU computation and memory movement are meaningful, while still being small enough to profile conveniently.

---

# 3. What the PyTorch profiler recorded

The paper says it used the **PyTorch built-in profiler** and reports a table of the top CUDA kernels. For each kernel, it extracted:

1. **CUDA kernel name**
2. **Accumulated CUDA execution time**
3. **Percentage of total execution time**
4. **Number of kernel calls**

The reported top entries were:

| CUDA kernel family                   |         Time |      Share | Calls |
| ------------------------------------ | -----------: | ---------: | ----: |
| `elementwise_kernel<128,2,...>`      |     8.498 ms |     24.80% |    35 |
| `indexFuncLargeIndex<float,...>`     |     6.772 ms |     19.77% |    12 |
| `indexSelectLarge<float,...>`        |     5.571 ms |     16.26% |     9 |
| `gemmSN_NN_kernel<float,...>`        |     4.247 ms |     12.39% |     9 |
| `volta_sgemm_128x64_tn`              |     3.336 ms |      9.74% |    14 |
| Other vectorized elementwise kernels |     2.142 ms |      6.26% |    50 |
| Other kernels in the top ten         | about 1.8 ms | about 5.3% |    71 |

The top ten kernel entries accounted for:

* **32.379 ms**
* **94.5% of reported CUDA execution time**
* **200 kernel invocations**

This concentration is what made targeted optimization practical: the authors did not need to optimize every operation in TensorNet. A relatively small number of kernel families accounted for nearly all measured GPU time.

---

# 4. How raw CUDA kernel names were mapped to TensorNet operations

The profiler initially returns implementation-level names such as `indexSelectLarge` and `elementwise_kernel`, not scientific operations such as “tensor decomposition.”

The authors grouped these kernels into higher-level categories by tracing them back to the PyTorch operations in the TensorNet forward pass.

Their category-level breakdown was:

* **Element-wise operations:** 24.8%
* **`index_add` or scatter aggregation:** 19.8%
* **`index_select` or gather:** 16.3%
* **GEMM/matrix multiplication:** 22.1%
* **Other operations:** 17.0%

The three principal **non-GEMM** categories therefore account for approximately

[
24.8 + 19.8 + 16.3 = 60.9%.
]

Those were the main targets for Triton optimization.

The reason GEMMs were not targeted is that matrix multiplications were already being executed by highly optimized NVIDIA/cuBLAS kernels. Reimplementing those in Triton was unlikely to provide the same return as eliminating fragmented element-wise and index-based operations.

---

# 5. What the profiler revealed about the bottleneck

The important result was not simply that individual arithmetic operations were slow. Rather, the profile showed that TensorNet was executing **many small or intermediate CUDA kernels**.

For example, a TensorNet tensor transformation might conceptually perform:

[
v
\rightarrow v \otimes v
\rightarrow \operatorname{trace}(v\otimes v)
\rightarrow \text{symmetrization}
\rightarrow \text{trace subtraction}.
]

In eager PyTorch, these steps can involve:

1. Launch kernel 1.
2. Read inputs from GPU memory.
3. Write an intermediate tensor to GPU memory.
4. Launch kernel 2.
5. Read the intermediate tensor again.
6. Write another intermediate result.
7. Continue for several operations.

The profiler’s large call counts—35 calls for the largest element-wise family, for example—showed this fragmentation.

The authors interpreted this as primarily an **IO and launch-overhead problem**, rather than a shortage of floating-point throughput:

* repeated CUDA kernel launches;
* repeated reading and writing of intermediate tensors;
* index-based gathers and scatters;
* poor data reuse across separate kernels;
* atomic write conflicts during message aggregation.

That diagnosis motivated **kernel fusion**, where several operations are executed within one Triton kernel while temporary values remain in registers, SRAM, or cache rather than being materialized repeatedly in global memory.

---

# 6. Profiling the message-passing path

The message-passing operation has the general form

[
h_i^{(l+1)}
===========

\sum_{j\in\mathcal{N}(i)}
w_{ij}h_j^{(l)}.
]

A conventional PyTorch execution involves several stages:

1. **Gather:** use source-node indices to obtain (h_j).
2. **Element-wise computation:** multiply gathered features by (w_{ij}).
3. **Initialize output:** allocate or zero an output tensor.
4. **Scatter-add:** aggregate edge messages into destination atoms with `index_add`.

These stages appear in the profiler as combinations of:

* `indexSelectLarge`
* element-wise kernels
* initialization kernels
* `indexFuncLargeIndex` or related index-add kernels

Together, the gather and scatter categories consumed approximately 36% of measured time:

[
19.8% + 16.3% \approx 36.1%.
]

The authors therefore combined gathering, weighting, and aggregation into a custom Triton message-passing kernel. Atomic additions were retained where multiple edges wrote to the same destination atom, but unnecessary intermediate arrays and launches were removed.

---

# 7. Profiling tensor construction and decomposition

The other major target was TensorNet’s Cartesian tensor manipulation.

For example, converting a vector (v\in\mathbb{R}^3) to a symmetric traceless tensor involves:

[
T = v\otimes v,
]

[
\bar{t}
=======

\frac{T_{00}+T_{11}+T_{22}}{3},
]

[
S
=

\frac{1}{2}(T+T^\top)-\bar{t}I.
]

In ordinary PyTorch, the outer product, transpose, addition, scalar multiplication, trace computation, identity construction, and subtraction can be broken into several CUDA kernels.

The paper says the vector-to-symmetric-tensor sequence required approximately **five launches** in the original implementation. The Triton version performed the sequence in one launch.

The same logic was applied to:

* vector-to-skew-tensor conversion;
* tensor decomposition into isotropic, antisymmetric, and symmetric components;
* cutoff-function evaluation;
* message weighting;
* cutoff plus message-passing fusion.

---

# 8. The optimization loop following profiling

The paper describes the workflow as:

[
\boxed{
\text{Profile}
\rightarrow
\text{identify bottlenecks}
\rightarrow
\text{write fused Triton kernel}
\rightarrow
\text{benchmark against eager PyTorch}
\rightarrow
\text{keep only genuine improvements}
}
]

That final step is important. The authors did not replace every PyTorch operation with Triton.

They found several cases where Triton was slower:

* heavily contended 4D scatter-add operations;
* small molecular systems;
* simple reductions such as norms and sums.

Consequently, profiling was also used to support a **hybrid implementation**: use Triton where fusion reduces memory traffic, but retain PyTorch’s optimized primitives where its reductions or atomics are superior.

---

# 9. How the optimized kernels were timed

The paper’s subsequent microbenchmark procedure was more explicit than its initial profiler procedure.

For each operation and system size:

* **20 warm-up iterations**
* **100 timed iterations**
* repeated for **5 runs**
* report the **median time**

The tested system sizes were:

* 1,000 atoms
* 4,000 atoms
* 16,000 atoms
* 64,000 atoms

The warm-up iterations are necessary because the first executions may include:

* Triton JIT compilation;
* CUDA context initialization;
* allocator initialization;
* cold cache effects;
* one-time PyTorch dispatch overhead.

Taking a median over five runs makes the result less sensitive to transient GPU scheduling or system noise.

The paper does not explicitly state whether each individual timed iteration used CUDA events, `torch.utils.benchmark`, profiler timing, or synchronized wall-clock timing. It only gives the iteration protocol and resulting medians.

---

# 10. End-to-end profiling and benchmarking

After microbenchmarking individual operations, the authors measured the complete TensorNet forward pass on:

* MD17 aspirin, 21 atoms;
* MD22 Ac-Ala3-NHMe, 42 atoms;
* batch sizes 1 and 32.

The resulting end-to-end speedups ranged from 2.54× to 2.96×, with an average of 2.82×.

The microbenchmark average was higher, around 3.14×, because the full forward pass still contains work that was not optimized:

* cuBLAS GEMMs;
* Python dispatch;
* miscellaneous framework operations;
* operations for which PyTorch remained faster.

In other words, the profiler was used to estimate the optimizable fraction, while the end-to-end benchmark measured the actual application-level gain.

This is broadly consistent with an Amdahl’s-law interpretation: even a large acceleration of the 60.9% memory-bound section cannot accelerate the unchanged remainder by the same amount.

---

# 11. Kernel-launch profiling after fusion

The authors also compared the number of launches before and after optimization:

| Operation                 | PyTorch launches | Triton launches | Reduction |
| ------------------------- | ---------------: | --------------: | --------: |
| Vector → symmetric tensor |                5 |               1 |       80% |
| Tensor decomposition      |                6 |               1 |       83% |
| Message passing           |                4 |               1 |       75% |
| Cutoff + message passing  |                8 |               1 |       88% |

This is the most direct evidence supporting the fusion mechanism. Rather than merely showing lower wall-clock time, the authors showed that the number of launches associated with each compound operation decreased substantially.

However, the paper does not say whether these counts came directly from another `torch.profiler` trace, manual inspection of the operator sequence, or both.

---

# 12. Memory-bandwidth analysis

The paper reports “memory bandwidth utilization” values such as:

* message passing at 4K atoms:

  * PyTorch: 27.35 GB/s
  * Triton: 95.91 GB/s
* fused cutoff plus message passing at 4K atoms:

  * PyTorch: 20.46 GB/s
  * Triton: 100.02 GB/s
* vector-to-symmetric-tensor at 64K atoms:

  * PyTorch: 36.92 GB/s
  * Triton: 112.42 GB/s

The authors attribute the increase to:

1. contiguous or coalesced memory accesses;
2. fewer global-memory round trips;
3. reuse of values within a fused kernel;
4. better use of L1/L2 cache.

An important caveat is that the paper does **not** specify how these GB/s figures were calculated. It does not say whether they were:

* effective bandwidth calculated as estimated bytes divided by elapsed time;
* profiler-reported throughput;
* Nsight Compute DRAM counters;
* another measurement.

Because the reported bandwidth improvement closely matches the reported speedup—for example, 95.91/27.35 ≈ 3.51—it appears likely that these are **effective-bandwidth estimates based on fixed byte counts and runtime**, rather than direct measurement of physical DRAM traffic. The paper does not make this explicit, so that interpretation should be treated as an inference.

---

# 13. What the paper does not disclose about profiling

Although the overall profiling strategy is understandable, several details necessary for exact replication are missing.

The paper does not specify:

* the exact `torch.profiler.profile(...)` configuration;
* whether both CPU and CUDA activities were enabled;
* whether the table shows `self_cuda_time_total` or `cuda_time_total`;
* whether tensor shapes or memory allocation were recorded;
* whether stack traces were enabled;
* how many forward passes were included in the initial profile;
* the profiler scheduling parameters;
* whether gradients were enabled;
* whether profiling was under `torch.no_grad()` or `torch.inference_mode()`;
* whether CUDA synchronization occurred immediately before and after measurement;
* whether neighbor-list construction was included;
* whether data transfer from CPU to GPU was excluded;
* whether Triton compilation time was excluded from every measurement;
* how bandwidth figures were derived;
* any Nsight Systems timeline;
* occupancy, cache-hit, or warp-stall measurements;
* error bars or timing distributions.

The paper itself acknowledges that it reports medians over five runs but does not include confidence intervals or error bars.

Therefore, the profiling methodology is sufficient to explain **how bottlenecks were selected**, but not sufficient to reproduce every number exactly without consulting the accompanying code.

---

# 14. A plausible reconstruction of the initial profile

The paper’s initial profiler run was probably structurally similar to the following, although this is a reconstruction rather than code printed in the paper:

```python
import torch
from torch.profiler import ProfilerActivity, profile

model.eval()

# Inputs already resident on the GPU:
# z: atomic numbers
# pos: atomic positions
# edge_index: graph connectivity
# batch: molecule assignments

# Warm up CUDA and any lazy initialization.
with torch.inference_mode():
    for _ in range(20):
        _ = model(z, pos, batch=batch)

torch.cuda.synchronize()

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with torch.inference_mode():
        _ = model(z, pos, batch=batch)

torch.cuda.synchronize()

print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=10,
    )
)
```

The profiler table could then be manually grouped into:

```text
element-wise
index_select / gather
index_add / scatter
GEMM
other
```

A more rigorous modern implementation would also export a Chrome trace:

```python
prof.export_chrome_trace("tensornet_profile.json")
```

That would allow the researchers to see the exact order and dependency structure of the kernels and determine which adjacent operators could safely be fused.

---

# Bottom line

The paper’s GPU profiling consisted of profiling a **4,096-atom, 131,072-edge TensorNet forward pass in PyTorch eager mode on an A100**, ranking raw CUDA kernels by accumulated execution time and call count, and grouping those kernels into higher-level TensorNet operations.

The key finding was that approximately **60.9% of execution time** came from fragmented, memory-oriented element-wise, gather, and scatter operations. That finding directly determined which operation sequences were fused into Triton kernels. The authors then validated the replacements using warm-up-controlled microbenchmarks, end-to-end inference timing, launch-count comparisons, and reported effective-bandwidth figures.

The profiling successfully identifies actionable bottlenecks, but it is primarily **operator-level timing analysis**, not a comprehensive GPU microarchitectural study. A stronger profiling section would additionally include Nsight timelines, hardware-counter measurements, exact profiler settings, synchronized timing code, and variance statistics.
