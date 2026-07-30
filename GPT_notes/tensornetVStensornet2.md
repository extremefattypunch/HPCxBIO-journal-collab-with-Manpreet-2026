
> [!fact]
> **TensorNet2 is not a wholesale replacement for TensorNet’s equivariant tensor architecture.** It retains the same local rank-2 Cartesian-tensor message-passing backbone, but adds learned latent charge channels, molecule-wide neutral-charge equilibration, charge-conditioned messages, and an explicit long-range Coulomb energy. Consequently, TensorNet2 is better suited to charged and polar molecules, but introduces additional reductions, memory traffic, and—most importantly—an currently quadratic all-pairs Coulomb calculation. ([arXiv][1])

## 1. Naming clarification

There are three similarly named things:

* **TensorNet**: the NeurIPS 2023 equivariant neural-network architecture.
* **TorchMD-Net 2.0**: the 2024 software-framework release that included an optimized implementation of TensorNet.
* **TensorNet2**: the newer architecture introduced with AceFF-2 in 2026, extending TensorNet with neutral charge equilibration and Coulomb interactions. ([OpenReview][2])

Your question concerns the first and third.

---

# 2. What the original TensorNet does

## Rank-2 Cartesian atom features

TensorNet represents each atom $i$ using multiple learned $3\times3$ matrices:

$$
X_i\in\mathbb{R}^{3\times3\times F},
$$

where $F$ is the hidden-channel dimension.

Under a rotation $R$, each channel transforms as

$$
X_i\rightarrow RX_iR^\mathsf{T}.
$$

A general $3\times3$ matrix can be decomposed into three irreducible parts:

$$
X_i=I_i+A_i+S_i,
$$

where:

* $I_i$: scalar or isotropic component, with $1$ independent value;
* $A_i$: antisymmetric component, equivalent to a vector, with $3$ values;
* $S_i$: symmetric traceless rank-2 component, with $5$ values.

Thus the nine matrix components correspond to the $1+3+5$ irreducible representations of rotations. This lets TensorNet process scalar, vector and rank-2 geometric information without general Clebsch–Gordan tensor-product machinery. ([arXiv][3])

## Tensor embedding

For an edge $j\rightarrow i$, TensorNet computes:

1. distance $r_{ij}$;
2. normalized direction $\hat r_{ij}$;
3. radial basis features $e^{\mathrm{RBF}}(r_{ij})$;
4. atom-pair embeddings based on $Z_i$ and $Z_j$.

These produce edge contributions to the three representation types. Schematically,

$$
I_{ij}\sim f_I(r_{ij},Z_i,Z_j),
$$

$$
A_{ij}\sim f_A(r_{ij},Z_i,Z_j),[\hat r_{ij}]_\times,
$$

$$
S_{ij}\sim f_S(r_{ij},Z_i,Z_j)
\left(
\hat r_{ij}\hat r_{ij}^{\mathsf T}
-\frac13\mathbf I
\right).
$$

Messages are summed over neighbors to initialize the atomic tensor $X_i$.

## Interaction layers

Each interaction layer:

1. decomposes $X_i$ into $I_i,A_i,S_i$;
2. applies channel-mixing linear layers;
3. creates distance-dependent edge filters;
4. gathers features from neighboring atoms;
5. scales them with edge filters;
6. scatters and sums them into destination atoms;
7. composes the parts back into a full tensor;
8. mixes tensors using small $3\times3$ matrix products;
9. adds the result through a residual update.

The important point is that the expensive geometric mixing is reduced to fixed-size matrix operations instead of arbitrary spherical tensor products. This is one reason TensorNet can achieve strong accuracy with relatively few interaction layers. ([OpenReview][2])

## Energy and forces

At the output, TensorNet constructs invariant features such as

$$
|I_i|^2,\qquad |A_i|_F^2,\qquad |S_i|_F^2,
$$

passes them through an output network, and predicts an atomic energy $E_i$:

$$
E_{\mathrm{short}}=\sum_i E_i.
$$

Forces are obtained by differentiating the total energy:

$$
\mathbf F_i=-\frac{\partial E}{\partial\mathbf r_i}.
$$

With a fixed neighbor cutoff and bounded neighbor count, the message-passing portion scales approximately as

$$
O(N,k,F),
$$

where $N$ is the number of atoms and $k$ is the average number of neighbors.

---

# 3. What TensorNet2 adds

The underlying Cartesian tensor representation is essentially unchanged. TensorNet2 adds a **charge-processing pathway around the TensorNet backbone**.

| Component                 | TensorNet                                                | TensorNet2                                       |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------ |
| Atomic representation     | Rank-2 Cartesian tensors                                 | Same                                             |
| Local message passing     | Distance- and species-conditioned                        | Also conditioned on learned charge features      |
| Total molecular charge    | At most simple global conditioning in charged extensions | Explicitly enforced through charge equilibration |
| Per-atom latent charges   | No explicit learned charge system                        | Yes, several charge channels                     |
| Long-range electrostatics | Normally absent                                          | Explicit Coulomb energy                          |
| Main scaling              | Approximately linear with bounded neighbors              | Local part linear, present Coulomb part $O(N^2)$ |
| Best use case             | Neutral/local potential surfaces                         | Neutral and charged drug-like molecules          |
| Parameter/runtime cost    | Lower                                                    | Higher                                           |

## 3.1 Charge-prediction heads

After the initial tensor embedding and after each interaction layer, TensorNet2 decomposes the atomic tensor and forms invariant descriptors from $I_i,A_i,S_i$.

A small MLP predicts, for every atom, multiple latent charge hypotheses and associated redistribution weights:

$$
\mathbf q_i\in\mathbb R^{d_q},
\qquad
\mathbf w_i\in\mathbb R^{d_q}.
$$

The default documented charge dimension is

$$
d_q=16.
$$

These charge channels are latent features. They do not have to equal a particular physical charge partition such as Mulliken, Löwdin or MBIS unless one explicitly supervises a channel using such labels. AceFF-2 itself was trained on energy and force labels without partial-charge labels. ([arXiv][1])

## 3.2 Neutral charge equilibration

For each molecule and each charge channel, TensorNet2 corrects the raw atomic predictions so that they sum to the supplied molecular charge $Q$.

The conservation-enforcing form is

$$
q'_{ic}
=======

q_{ic}
+
\alpha_{ic}
\left(
Q-\sum_jq_{jc}
\right),
$$

with

$$
\alpha_{ic}
===========

\frac{w_{ic}}{\sum_j w_{jc}},
\qquad
\sum_i\alpha_{ic}=1.
$$

Therefore,

$$
\sum_iq'_{ic}=Q.
$$

The current TorchMD-Net implementation uses a positive correction proportional to $Q-\sum_iq_i$. The equation rendered in the AceFF manuscript appears to show a minus sign with the same residual definition, which would not enforce the stated constraint; the implementation contains the conservation-consistent positive update. ([GitHub][4])

This is much cheaper than traditional charge equilibration methods that construct and solve a global linear system.

## 3.3 Charges feed back into message passing

TensorNet edge filters originally depend mainly on radial information:

$$
f_{ij}=\operatorname{MLP}\left(e^{\mathrm{RBF}}(r_{ij})\right).
$$

TensorNet2 instead uses

$$
f_{ij}
======

\operatorname{MLP}
\left(
e^{\mathrm{RBF}}(r_{ij})
\oplus
\mathbf q_i
\oplus
\mathbf q_j
\right).
$$

Thus the local geometric representation influences the charges, and the charges influence the next geometric update:

$$
X^{(l)}
\rightarrow q^{(l)}
\rightarrow
X^{(l+1)}
\rightarrow q^{(l+1)}.
$$

This is why the authors describe it as a form of self-consistent processing, although it is not an iterative electronic-structure charge-equilibration solver. ([arXiv][1])

## 3.4 Explicit Coulomb energy

The learned charge channels are also used to calculate Coulomb-like energy terms. Conceptually,

$$
E_{\mathrm{Coulomb}}^{(c)}
\sim
\sum_{i<j}
\frac{q_{ic}q_{jc}}{r_{ij}},
$$

with smoothing or short-distance handling in the implementation. Contributions from different channels and interaction stages are combined, giving later layers greater importance.

The total prediction becomes

$$
E_{\mathrm{total}}
==================

E_{\mathrm{short}}
+
E_{\mathrm{Coulomb}}.
$$

This allows interactions beyond TensorNet’s ordinary local graph cutoff and substantially improves treatment of charged molecules and stretched configurations. ([arXiv][1])

## 3.5 Cost of the new architecture

For matched one-layer models with hidden dimension $128$ and $32$ radial basis functions, the authors report:

* TensorNet: $535{,}681$ parameters;
* TensorNet2 with $d_q=16$: $685{,}413$ parameters;
* parameter increase: approximately $28%$;
* 1,500-atom test: $100$ versus $75$ steps per second;
* TensorNet2 slowdown: approximately $25%$.

More importantly, the current explicit Coulomb implementation scales as $O(N^2)$, while the local TensorNet graph remains approximately linear for bounded neighbor count. ([arXiv][1])

---

# 4. What is and is not a TensorNet2 architectural change

Several properties sometimes associated with TensorNet2 actually belong to both architectures.

**Cartesian tensors**

Both use the $3\times3$ TensorNet representation.

**Chirality and reflection behavior**

The tensor–tensor interaction can mix even and odd parity components, reducing full reflection equivariance from $O(3)$ to rotational $SO(3)$ behavior in the relevant interaction pathway. This gives the architecture sensitivity to chirality. It is not primarily a TensorNet2 charge innovation. ([arXiv][1])

**Warp kernels**

The optimized NVIDIA Warp implementation applies to the TensorNet operations used by both TensorNet and TensorNet2. TensorNet2 merely adds more operations around the shared backbone.

**CUDA graphs**

Static edge-array shapes and fixed molecular composition can make either architecture compatible with CUDA graphs. This reduces CPU launch overhead, especially for small molecules.

---

# 5. Why TensorNet is particularly suitable for custom GPU kernels

TensorNet’s mathematics maps unusually well to specialized GPU operations.

## Bottleneck 1: tensor decomposition and composition

A naive PyTorch implementation may materialize three separate $3\times3$ tensors for $I,A,S$.

That representation stores

$$
9+9+9=27
$$

numbers per channel, even though the three objects contain only

$$
1+3+5=9
$$

independent values.

The optimized implementation keeps the compact layouts:

* $I$: `[N, 1, F]`;
* $A$: `[N, 3, F]`;
* $S$: `[N, 5, F]`.

A custom kernel can directly transform between the full nine-component tensor and the compact $1+3+5$ layout without creating zero-filled matrices.

The published Warp implementation reports approximately:

* $3\times$ faster training and inference;
* $3\times$ lower GPU memory usage.

The reduction comes from compact representation and avoiding large edge-sized autograd intermediates. ([arXiv][1])

## Bottleneck 2: gather–multiply–scatter message passing

A conventional framework often executes:

```text
gather source features
→ multiply by edge filter
→ materialize edge messages
→ scatter/index_add into destination atoms
```

For $E$ graph edges, the intermediate message tensors can have shapes such as

$$
[E,F],\quad [E,3,F],\quad [E,5,F].
$$

These consume substantial bandwidth and memory.

A fused kernel can instead:

1. load source atom features;
2. load the three edge-filter coefficients;
3. multiply in registers;
4. directly accumulate into destination atoms;
5. avoid writing full edge messages to global memory.

The current optimized code transforms the graph into row- and column-oriented sparse structures and dispatches custom message-passing kernels rather than relying only on generic `index_select` and `index_add`. ([GitHub][5])

## Bottleneck 3: fixed $3\times3$ products

TensorNet repeatedly performs very small matrix products such as

$$
Y_iM_i,
\qquad
M_iY_i,
\qquad
\Delta X_i\Delta X_i.
$$

Calling a general-purpose matrix-multiplication library for millions of independent $3\times3$ matrices can incur more dispatch and indexing overhead than arithmetic.

A specialized kernel can:

* completely unroll the $3\times3$ product;
* keep operands in registers;
* fuse the product with decomposition;
* fuse normalization and residual updates;
* avoid layout permutations before and after `matmul`.

The current source imports optimized Warp operations for tensor multiplication, decomposition, composition, norms and message passing. ([GitHub][5])

---

# 6. TensorNet2-specific GPU kernels

TensorNet2 introduces several new optimization targets.

## Charge prediction

The charge-head MLP is mostly standard dense linear algebra and will generally be handled efficiently by PyTorch’s GEMM kernels. The more specialized opportunity is to fuse:

$$
\text{tensor invariants}
\rightarrow
\text{normalization}
\rightarrow
\text{charge MLP input}.
$$

This removes intermediate arrays containing the concatenated $I$, $|A|^2$ and $|S|^2$ features.

## Molecule-wise charge equilibration

For every molecule and charge channel, TensorNet2 must calculate quantities such as

$$
Q_{\mathrm{pred},c}=\sum_iq_{ic},
$$

and

$$
W_c=\sum_iw_{ic}.
$$

A fused segmented-reduction kernel could:

1. compute both reductions in one pass;
2. calculate the molecular charge residual;
3. redistribute it to atoms;
4. write corrected charges only once.

For a batch of many small molecules, this can be noticeably better than several separate `index_add`, division and elementwise kernels.

## Charge-conditioned edge filters

The sequence

$$
e^{\mathrm{RBF}}_{ij}\oplus q_i\oplus q_j
\rightarrow \mathrm{MLP}
\rightarrow \text{three edge filters}
\rightarrow \text{message passing}
$$

can potentially be fused more deeply than in ordinary TensorNet.

A useful design would avoid storing the complete charge-augmented edge representation. The kernel could gather $q_i,q_j$, evaluate the final small filter transformation, and immediately apply it to the compact $I,A,S$ source features.

## Coulomb calculation: the largest new target

The current all-pairs Coulomb implementation is the most important TensorNet2-specific bottleneck because it scales as

$$
O(N^2d_q).
$$

A tiled GPU kernel could:

* process atom pairs in shared-memory tiles;
* calculate only $i<j$ and exploit symmetry;
* accumulate energy and force simultaneously;
* vectorize across charge channels;
* fuse distance, damping, reciprocal distance and charge products;
* avoid constructing an $N\times N\times d_q$ tensor;
* use custom backward or directly produce analytic coordinate gradients.

For larger systems, however, kernel tuning alone does not change the quadratic asymptotic cost. A more consequential architectural improvement would replace the all-pairs calculation with one of:

* cutoff electrostatics plus reaction-field correction;
* particle-mesh Ewald;
* treecode or fast multipole methods;
* hierarchical low-rank approximation;
* local electrostatic graph plus reciprocal-space correction.

That would change TensorNet2 from

$$
O(N^2)
$$

toward approximately

$$
O(N\log N)
$$

or $O(N)$, depending on the method.

---

# 7. CUDA graphs and small-molecule latency

Molecular dynamics repeatedly evaluates the same model with:

* fixed number of atoms;
* fixed atomic species;
* bounded or statically allocated edge arrays;
* identical tensor shapes each step.

That makes CUDA graph capture particularly useful. Instead of launching a large number of small kernels through Python and the CUDA driver on every timestep, the complete sequence can be replayed as a captured graph.

The AceFF study found that CUDA graph compatibility was crucial for surpassing $100$ steps per second in small systems, where kernel-launch latency otherwise dominates arithmetic. ([arXiv][1])

The optimization hierarchy is therefore:

$$
\text{CUDA graphs}
\quad\text{for low launch latency},
$$

$$
\text{fused Warp/Triton/CUDA kernels}
\quad\text{for memory bandwidth and intermediate elimination},
$$

$$
\text{better long-range algorithm}
\quad\text{for asymptotic scaling}.
$$

These solve different problems and are complementary.

---

# 8. Training kernels are harder than inference kernels

For inference-only molecular dynamics, one needs forces:

$$
\mathbf F=-\nabla_{\mathbf R}E.
$$

For force training, the loss depends on those forces:

$$
\mathcal L_F
============

|\mathbf F_{\mathrm{pred}}-\mathbf F_{\mathrm{ref}}|^2.
$$

Backpropagating $\mathcal L_F$ into model parameters therefore requires derivatives through the force calculation—effectively mixed second derivatives of energy.

A custom Triton, CUDA or Warp kernel must consequently provide:

* correct first-order backward for MD inference;
* differentiable backward or explicit double-backward support for force training;
* stable behavior near zero distances and graph cutoffs;
* identical equivariance and parity behavior;
* deterministic or sufficiently stable scatter accumulation.

This is a major reason why accelerating TensorNet training is a more substantial research contribution than writing an inference-only kernel.

---

# 9. Multi-GPU implications

## Training

Both models can use ordinary distributed data parallelism:

* each GPU processes different molecules;
* gradients are synchronized after the backward pass;
* scaling is usually straightforward when batches are large enough.

TensorNet2 has slightly more computation per sample, which can improve the computation-to-communication ratio.

## Single-trajectory inference

For the small ligand-sized systems targeted by AceFF, splitting one model evaluation across multiple GPUs is unlikely to help. The inter-GPU communication cost can exceed the local tensor computations.

For genuinely large systems, TensorNet’s local graph could be spatially decomposed across GPUs, but TensorNet2 introduces two global complications:

1. neutral charge equilibration needs molecule-wide reductions;
2. all-pairs Coulomb interactions require global positions and charges or a distributed long-range solver.

A scalable multi-GPU TensorNet2 implementation would therefore need:

$$
\text{domain-decomposed local message passing}
+
\text{global charge all-reduce}
+
\text{distributed PME/FMM electrostatics}.
$$

Simply distributing the existing $O(N^2)$ Coulomb matrix would accelerate a fixed size but would not resolve the fundamental scaling problem.

---

# 10. Bottom line for GPU-kernel research

**For TensorNet**, the highest-value kernel work is:

1. compact $1+3+5$ tensor storage;
2. fused decomposition–linear transform–composition;
3. fused gather–filter–scatter message passing;
4. unrolled $3\times3$ tensor products;
5. custom backward supporting force training;
6. CUDA graph-compatible static execution.

**For TensorNet2**, all of the above still applies, plus:

1. fused molecule-wise neutral-charge equilibration;
2. fused charge-conditioned edge filters;
3. tiled Coulomb energy-and-force kernels;
4. ultimately, replacement of the $O(N^2)$ Coulomb implementation with PME, FMM or another scalable long-range method.

The most defensible novel research direction is therefore **not merely “accelerate TensorNet2 with a fused kernel.”** The stronger contribution would be:

> A differentiable, CUDA-graph-compatible TensorNet2 backend combining compact equivariant message-passing kernels with a fused or asymptotically scalable long-range electrostatics implementation, evaluated for both energy/force training and molecular-dynamics inference.

That targets the real distinction between the architectures: TensorNet’s main bottleneck is local equivariant message passing, whereas TensorNet2 adds global charge reductions and long-range electrostatics that eventually dominate system-size scaling.

[1]: https://arxiv.org/html/2601.00581v2 "AceFF: A State-of-the-Art Machine Learning Potential for Small Molecules"
[2]: https://openreview.net/forum?id=BEHlPdBZ2e&utm_source=chatgpt.com "TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials | OpenReview"
[3]: https://arxiv.org/abs/2306.06482 "TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials"
[4]: https://raw.githubusercontent.com/torchmd/torchmd-net/main/torchmdnet/models/tensornet2.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/torchmd/torchmd-net/main/torchmdnet/models/tensornet.py "raw.githubusercontent.com"
