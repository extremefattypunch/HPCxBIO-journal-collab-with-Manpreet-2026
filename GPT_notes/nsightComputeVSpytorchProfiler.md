> [!fact]
> Profiling guides GPU optimization by revealing **where time is lost, what resource limits performance, and whether data crosses unnecessary kernel boundaries**. PyTorch Profiler identifies candidate operation sequences; Nsight Compute determines whether the remedy is fusion, tiling, memory-layout optimization, greater parallelism, tensor-core use, or a lower-level kernel implementation.

**Profiling does not directly say “use fusion”**

Instead, profiling produces a performance signature:

1. PyTorch Profiler shows **which operators and kernel sequences matter globally**.
2. The execution trace shows **where fusion boundaries could exist**.
3. Nsight Compute shows **why an individual kernel is slow**.
4. You choose an optimization whose mechanism addresses that measured limitation.

A useful approximation is:

$$
T_{\text{kernel}}
\approx
T_{\text{launch}}
+
\max\left(
\frac{\text{bytes transferred}}{\text{memory bandwidth}},
\frac{\text{operations}}{\text{compute throughput}}
\right)
+
T_{\text{stalls}}.
$$

Fusion is profitable when:

$$
T_{\text{saved launches}}
+
T_{\text{eliminated memory traffic}}

>

T_{\text{extra computation}}
+
T_{\text{register/shared-memory penalties}}.
$$

This last qualification matters: fusing more operations is not automatically faster.

---

## How PyTorch Profiler tells you **where** to fuse

PyTorch Profiler connects Python and ATen operations to CUDA kernel launches, records shapes and can track tensor allocation activity. It therefore helps identify consecutive operations operating on the same large tensors. ([PyTorch Documentation][1])

Look for the following trace pattern:

```text
aten::add       -> elementwise_kernel_1
aten::sigmoid   -> elementwise_kernel_2
aten::mul       -> elementwise_kernel_3
```

If all three operations have the same shape and each kernel:

* reads a full tensor from global memory,
* performs little computation,
* writes a full intermediate tensor,
* and launches immediately after the preceding kernel,

then the region is a strong **vertical-fusion candidate**.

Conceptually:

```text
Before fusion

x ── add ──> temporary_1 ── sigmoid ──> temporary_2 ── mul ──> output
     kernel 1                kernel 2                kernel 3
```

```text
After fusion

x ── fused add + sigmoid + multiply ──> output
                    kernel 1
```

PyTorch’s tuning guidance explicitly identifies pointwise operations as common memory-bound fusion targets: eager execution launches separate kernels with repeated reads and writes, while fusion can load and store the data only once. TorchInductor can automatically fuse eligible pointwise and reduction operations. ([PyTorch Documentation][2])

### A valid fusion boundary usually has these properties

The operations form a producer-consumer chain where:

* the intermediate is not needed elsewhere;
* operations have compatible iteration spaces;
* there are no synchronization or side-effect requirements between them;
* the fused working set fits reasonably in registers or shared memory;
* fusion does not make the kernel excessively large.

For example:

```python
def candidate(x, bias):
    a = x + bias
    b = torch.sigmoid(a)
    return b * x
```

In eager execution this may become several kernels. The first intervention should generally be:

```python
compiled_candidate = torch.compile(candidate)
```

TorchInductor’s most important GPU optimization is fusion, and it uses Triton as a key GPU code-generation component. ([PyTorch Documentation][3])

After compilation, profile again:

```text
Before torch.compile:
    add kernel          35 us
    sigmoid kernel      42 us
    multiply kernel     34 us
    Total              111 us

After torch.compile:
    fused kernel        48 us
```

At this point, you do **not** need to write a custom Triton kernel. The compiler has already performed the relevant transformation.

---

## Mapping profiler symptoms to optimizations

| Profiling result                                        | Likely explanation                                  | Appropriate intervention                               |
| ------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------ |
| Many consecutive, short pointwise kernels               | Launch overhead and repeated global-memory traffic  | Vertical fusion                                        |
| Several independent operations traverse the same tensor | Multiple redundant passes over identical data       | Horizontal fusion                                      |
| GEMM followed by bias, activation or scaling            | Output is written and reread for a cheap epilogue   | GEMM epilogue fusion                                   |
| Many very small matrix multiplications                  | Individual GEMMs cannot occupy the GPU efficiently  | Batched or grouped GEMM                                |
| GPU timeline has large CPU-side gaps                    | Python or launch overhead                           | `torch.compile`, CUDA graphs, batching                 |
| High DRAM utilization and low arithmetic intensity      | Memory-bandwidth limitation                         | Fusion, tiling, data reuse, lower precision            |
| Low DRAM utilization plus memory-dependency stalls      | Latency, uncoalesced access or poor locality        | Change layout, coalescing, caching or work assignment  |
| Low occupancy caused by registers                       | Kernel has too much live state                      | Smaller tiles, fewer fusion stages or split the kernel |
| Tensor-core utilization is low during GEMM-like work    | Wrong datatype, shape, layout or schedule           | Tensor-core-compatible shapes, precision and tiling    |
| Tensor cores and memory bandwidth are already near peak | Kernel is close to its hardware limit               | Optimize another region or change the algorithm        |
| Large host-to-device copies dominate                    | Data pipeline problem, not a compute-kernel problem | Pinned memory, overlap, prefetching, fewer transfers   |

Nsight Compute provides the corresponding hardware evidence through roofline analysis, Memory Workload Analysis, occupancy, scheduler statistics and warp-state metrics. ([NVIDIA Docs][4])

---

# Example 1: Pointwise fusion

Consider:

```python
def gated_activation(x, bias, gate):
    y = x + bias
    y = torch.tanh(y)
    y = y * gate
    return y
```

Suppose the profiler reports:

```text
Operation              CUDA time    Shape
aten::add                 90 us      [8192, 4096]
aten::tanh               115 us      [8192, 4096]
aten::mul                 88 us      [8192, 4096]
```

The three operations perform only a few arithmetic instructions per element, but each passes over roughly the entire tensor.

For FP32 data with $N$ elements, a simplified traffic estimate is:

### Unfused

* add: two reads and one write;
* tanh: one read and one write;
* multiply: two reads and one write.

Approximately:

$$
B_{\text{unfused}} \approx 8N \times 4\ \text{bytes},
$$

depending on broadcasting and cache reuse.

### Fused

A fused kernel can load `x`, `bias` and `gate`, keep intermediate values in registers, and write only the final result:

$$
B_{\text{fused}} \approx 4N \times 4\ \text{bytes}.
$$

The exact byte count depends on broadcasting and cache behavior, but the important change is that the full-size intermediates disappear.

### Implementation choice

Start with:

```python
gated_activation_compiled = torch.compile(gated_activation)
```

If the compiled profiler trace shows one generated kernel and adequate performance, stop.

Write a custom Triton kernel only when:

* TorchInductor introduces a graph break;
* it fails to fuse part of the chain;
* the generated indexing is inefficient;
* your shapes require a special schedule;
* or the operation is reused enough that additional optimization matters.

---

# Example 2: Fused softmax

A decomposed softmax resembles:

```python
def decomposed_softmax(x):
    maximum = x.max(dim=-1, keepdim=True).values
    numerator = torch.exp(x - maximum)
    denominator = numerator.sum(dim=-1, keepdim=True)
    return numerator / denominator
```

The profiler may expose several kernels:

```text
reduce_max
subtract
exp
reduce_sum
divide
```

This is a strong fusion opportunity because the intermediate tensors are much larger than the final row statistics.

A fused implementation can:

1. load one row or row tile;
2. calculate its maximum;
3. subtract the maximum;
4. calculate exponentials;
5. calculate the sum;
6. normalize;
7. write the result once.

The official Triton fused-softmax tutorial uses exactly this motivation: fusion is valuable for bandwidth-bound operations when a row can be retained in on-chip SRAM rather than repeatedly stored in global memory. ([Triton Language][5])

### What profiling tells you

Suppose PyTorch Profiler shows:

```text
Five kernels collectively: 420 us
Four large intermediate allocations
All shapes: [32768, 1024]
```

Nsight Compute then shows:

```text
DRAM throughput:              high
Arithmetic intensity:         low
Compute utilization:          low
Memory dependency stalls:     substantial
```

The interpretation is:

* this is not fundamentally a slow exponential problem;
* repeated global-memory movement dominates;
* fusion should increase arithmetic intensity by doing more work per byte loaded.

### Why Triton is appropriate

The operation has:

* a regular row-wise mapping;
* a reduction;
* a moderate amount of per-row state;
* straightforward masking for non-power-of-two widths;
* no need for extremely architecture-specific tensor-core scheduling.

That is an excellent Triton use case.

However, if the row is too large to retain in registers or shared memory, a single-program-per-row design can generate excessive register pressure or spilling. Nsight Compute would then report low occupancy, high register use or local-memory traffic. The solution may be tiled or multi-stage softmax—not simply “fuse everything.”

---

# Example 3: GEMM epilogue fusion

Consider:

```python
def linear_activation(x, weight, bias):
    y = x @ weight
    y = y + bias
    return torch.nn.functional.gelu(y)
```

An eager trace might show:

```text
cuBLAS GEMM
bias-add kernel
GELU kernel
```

The output of GEMM is written to global memory, loaded by the bias kernel, written again, loaded by GELU and finally written a third time.

The desired transformation is:

```text
GEMM accumulator
      ↓
add bias while accumulator is still local
      ↓
apply GELU
      ↓
one final global-memory write
```

This is **epilogue fusion**.

PyTorch currently exposes compiler options related to epilogue fusion and autotuned matrix multiplication, so `torch.compile(..., mode="max-autotune")` is a reasonable experiment before implementing a custom GEMM. ([PyTorch Documentation][6])

```python
compiled = torch.compile(
    linear_activation,
    mode="max-autotune",
)
```

### When not to replace the GEMM

Suppose Nsight Compute reports:

```text
GEMM duration:              900 us
Bias + GELU:                 25 us
Tensor-core utilization:     92%
Memory throughput:           85%
```

A custom replacement is unlikely to produce a large end-to-end improvement. Even eliminating the entire epilogue would save less than roughly $3%$ of this region.

By contrast:

```text
GEMM duration:              220 us
Bias + GELU:                170 us
Tensor-core utilization:     45%
```

This suggests substantial room for a fused or better-specialized implementation.

---

# Example 4: Many small GEMMs

Imagine a molecular or geometric model that performs 1,000 separate matrix multiplications of shape:

$$
32 \times 64 ;; \times ;; 64 \times 32.
$$

PyTorch Profiler might show:

```text
1,000 GEMM launches
Average kernel duration: 4 us
Large CPU launch contribution
Low overall GPU occupancy
```

Here, fusing arithmetic inside each GEMM is not the first problem. Each job is simply too small.

Better transformations include:

* combine them into a batched GEMM;
* use grouped GEMM for heterogeneous shapes;
* assign several independent problems to one persistent kernel;
* fuse the preceding data preparation and following epilogue into the grouped computation.

Torch compiler documentation describes horizontal fusion and grouped GEMM as forms of combining independent work to reduce execution overhead. ([PyTorch Documentation][7])

Triton provides an official Group GEMM tutorial and supports custom block-structured matmul implementations and autotuning. ([Triton Language][8])

---

# How Nsight Compute determines **how** to optimize

Once PyTorch Profiler identifies an important kernel, profile that kernel with Nsight Compute.

## Case A: Memory-bound and near peak bandwidth

Example metrics:

```text
Arithmetic intensity:       low
DRAM throughput:            88%
Compute throughput:         16%
```

The kernel is already moving data near the hardware limit. Making individual arithmetic instructions faster will barely matter.

Possible remedies:

* fuse its producer or consumer;
* reduce precision;
* avoid materializing intermediates;
* load data once and reuse it;
* reorganize the algorithm to perform more work per load.

The roofline model specifically identifies low-arithmetic-intensity kernels under the sloped memory-bandwidth roof as memory-bound. Moving performance upward generally requires increasing arithmetic intensity or reducing memory traffic. ([NVIDIA Docs][4])

## Case B: Low bandwidth but memory stalls

Example:

```text
DRAM throughput:             31%
Long-scoreboard stalls:      high
Global load efficiency:      poor
```

This does **not** mean there is unused bandwidth that fusion will automatically unlock.

Likely problems include:

* non-coalesced accesses;
* irregular gathers;
* poor spatial locality;
* cache misses;
* dependent pointer chasing;
* insufficient independent memory operations.

Better techniques may be:

* transpose or reorder data;
* change structure-of-arrays versus array-of-structures layout;
* tile into shared memory;
* sort or bucket irregular work;
* increase independent work per thread;
* prefetch future tiles.

Fusion helps only if it also improves locality or removes intermediate traffic.

## Case C: Low occupancy from registers

Example:

```text
Registers per thread:       192
Achieved occupancy:          18%
Local-memory loads:          present
```

This can be evidence of **overfusion**.

The fused kernel keeps too many values alive at once, causing:

* high register consumption;
* fewer resident warps;
* register spills into local memory;
* poor latency hiding.

Try:

* smaller tiles;
* fewer stages;
* recomputing cheap values instead of preserving them;
* splitting one oversized fused kernel into two;
* reducing the number of fused outputs;
* changing warp count.

Nsight Compute notes that low occupancy reduces the ability to hide latency, while register usage is one of the most important compiler-controlled occupancy constraints. ([NVIDIA Docs][9])

## Case D: Compute-bound but poor tensor-core use

Example:

```text
Compute-bound roofline position
Tensor-core active cycles:   low
FP32 pipelines:              high
```

Possible remedies:

* use FP16, BF16, TF32 or an appropriate lower-precision format;
* align matrix dimensions;
* change memory layout;
* use tensor-core-compatible tiles;
* fuse datatype conversion into the input-loading stage;
* use a specialized MMA schedule.

This is where CuTe DSL may become more attractive than a simple pointwise Triton kernel.

---

# Obtaining the Triton-versus-CuTe decision from profiling

The profiler does not literally output:

```text
Recommendation: use Triton
```

Instead, it reveals the **kernel class** and the amount of hardware control required.

## Use this escalation order

```text
PyTorch eager
    ↓
torch.compile / TorchInductor
    ↓
custom Triton
    ↓
CuTe DSL or CUDA/CUTLASS
```

Do not jump directly to the lowest level.

---

## Choose `torch.compile` first when

The candidate consists primarily of existing PyTorch operations and:

* graph capture works;
* shapes are reasonably stable;
* ordinary pointwise or reduction fusion is needed;
* compiler-generated kernels are already competitive.

TorchInductor automatically generates optimized GPU kernels and relies on Triton as a major code-generation building block for supported GPU backends. ([PyTorch Documentation][10])

You can inspect what Inductor decided using logging such as:

```bash
TORCH_LOGS="fusion,output_code,kernel_code,perf_hints" python benchmark.py
```

PyTorch documents these logging categories for viewing generated code, per-kernel code, fusion decisions and performance hints. ([PyTorch Documentation][11])

---

## Choose custom Triton when profiling shows

* several bandwidth-bound pointwise kernels should become one;
* producer-reduction-consumer fusion is needed;
* a row-wise or block-wise algorithm maps naturally to tiles;
* a custom matmul epilogue is needed;
* there are many regular small problems suitable for grouped scheduling;
* tile sizes or warp counts require autotuning;
* portability beyond NVIDIA is valuable;
* TorchInductor generated an inadequate schedule.

Triton’s official examples include fused softmax, layer normalization, attention, grouped GEMM, persistent matmul and block-scaled matrix multiplication. Its matmul tutorial demonstrates block tiling, L2-conscious program ordering and autotuning. ([Triton Language][8])

A practical Triton profiling signature is:

```text
Operation structure:       regular tensor tiles
Primary limitation:        launch or memory traffic
Required hardware control: moderate
Need to fuse custom math:   yes
```

---

## Choose CuTe DSL when profiling shows

* the expensive region is GEMM-, convolution- or attention-like;
* exact tensor-core instruction selection matters;
* shared-memory layout and swizzling matter;
* an asynchronous global-to-shared-memory pipeline must be controlled;
* producer and consumer warps need specialized roles;
* persistent or clustered execution is required;
* Triton cannot produce the desired architecture-specific schedule;
* the application is NVIDIA-specific and kernel performance justifies higher engineering complexity.

CuTe DSL is explicitly a lower-level model exposing layouts, tensors, hardware MMA/copy atoms and tiled operations, with control over the hardware thread and data hierarchy. Its pipeline API includes constructs for asynchronous threads, TMA loads and stores, and tensor-core MMA pipelines. ([NVIDIA Docs][12])

A practical CuTe profiling signature is:

```text
Operation structure:       tensor-core-intensive tiled computation
Primary limitation:        pipeline, layout or architecture utilization
Required hardware control: high
Target hardware:           NVIDIA-specific
```

### Concrete CuTe example

Suppose a custom block-scaled GEMM shows:

```text
Tensor-core utilization:      52%
DRAM throughput:              44%
Barrier stalls:               high
Shared-memory conflict rate:  high
Occupancy:                    acceptable
```

This suggests the kernel is neither saturating tensor cores nor global memory. The problem likely lies in the internal movement and scheduling of tiles.

A CuTe implementation lets you explicitly control:

```text
Global-memory layout
        ↓
TMA/copy atom
        ↓
Shared-memory layout and swizzle
        ↓
Thread/warp partitioning
        ↓
Tiled MMA instruction
        ↓
Accumulator and epilogue
```

CuTe’s core abstractions include precise tensor layouts, hardware atoms such as MMA and memory-copy operations, and tiled mappings across thread blocks and warps. ([NVIDIA Docs][12])

That is a much more direct response to shared-memory conflicts and tensor-core pipeline inefficiency than writing a generic fused pointwise kernel.

---

# Triton versus CuTe DSL decision table

| Profiling-derived requirement                |                   Triton |                     CuTe DSL |
| -------------------------------------------- | -----------------------: | ---------------------------: |
| Fuse pointwise operations                    |                Excellent |          Usually unnecessary |
| Fused reduction or normalization             |                Excellent | Possible but often excessive |
| Custom softmax                               |                Excellent |          Usually unnecessary |
| Regular custom matmul                        | Excellent starting point |                    Excellent |
| Custom matmul epilogue                       |                Excellent |                    Excellent |
| Cross-vendor GPU portability                 |                   Better |                  NVIDIA-only |
| Precise MMA instruction control              |         Limited/moderate |                    Excellent |
| Exact thread-data layout control             |                 Moderate |                    Excellent |
| TMA, cluster and specialized pipeline design |              Less direct |                    Excellent |
| Shared-memory swizzle control                |            More abstract |                     Explicit |
| Rapid experimentation                        |                   Easier |               More demanding |
| Maximum NVIDIA architecture specialization   |                Sometimes |              Designed for it |

CuTe DSL is designed as a low-level, NVIDIA-focused programming model with explicit layouts, tensors, hardware atoms and thread/data hierarchy control. ([NVIDIA Docs][12])

---

# A complete profiling-driven workflow

**1. Establish an unprofiled baseline**

Warm up the workload and measure end-to-end latency or throughput with CUDA synchronization at the measurement boundaries.

**2. Profile eager PyTorch**

Use PyTorch Profiler to find:

* expensive operators;
* repeated small kernels;
* intermediate allocations;
* operation shapes;
* CPU gaps;
* copies and synchronizations.

**3. Mark candidate subgraphs**

Identify consecutive operations whose intermediates do not need to leave the candidate region.

**4. Try compiler-level fusion**

Apply `torch.compile` to the smallest meaningful function or module and profile again.

**5. Quantify remaining importance**

Do not optimize a kernel that represents only $1%$ of total runtime unless it is called often enough to matter at scale.

**6. Inspect the dominant generated kernels with Nsight Compute**

Determine whether each is:

* launch-limited;
* memory-bandwidth-bound;
* latency-bound;
* occupancy-limited;
* compute-bound;
* tensor-core-limited.

**7. Select the implementation level**

* Existing library operation when one is already efficient.
* `torch.compile` for compiler-recognizable graphs.
* Triton for regular custom fusion and tiling.
* CuTe DSL for architecture-specific tensor-core and memory-pipeline control.

**8. Re-profile the fused kernel**

Verify that:

* total kernel count fell;
* global-memory traffic fell;
* execution time fell;
* register spills did not appear;
* occupancy did not collapse;
* numerical accuracy remains acceptable.

**9. Measure end-to-end performance again**

The final metric is not the speedup of one isolated kernel. It is the improvement in the real training step, inference request or simulation timestep.

[1]: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html?utm_source=chatgpt.com "PyTorch Profiler — PyTorch Tutorials 2.13.0+cu130 documentation"
[2]: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html?highlight=device&utm_source=chatgpt.com "Performance Tuning Guide — PyTorch Tutorials 2.13.0+cu130 documentation"
[3]: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_get_started.html?utm_source=chatgpt.com "Getting Started — PyTorch 2.12 documentation"
[4]: https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html?utm_source=chatgpt.com "2. Profiling Guide — NsightCompute 13.0 documentation"
[5]: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html?utm_source=chatgpt.com "Fused Softmax — Triton documentation"
[6]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html?utm_source=chatgpt.com "torch.compile — PyTorch 2.13 documentation"
[7]: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html?utm_source=chatgpt.com "Frequently Asked Questions — PyTorch 2.13 documentation"
[8]: https://triton-lang.org/main/getting-started/tutorials/?utm_source=chatgpt.com "Tutorials — Triton documentation"
[9]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?highlight=instance&utm_source=chatgpt.com "2. Profiling Guide — NsightCompute 13.3 documentation"
[10]: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html?utm_source=chatgpt.com "torch.compiler — PyTorch main documentation"
[11]: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html?utm_source=chatgpt.com "torch.compile Troubleshooting — PyTorch 2.12 documentation"
[12]: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html?utm_source=chatgpt.com "Overview — NVIDIA CUTLASS Documentation"
