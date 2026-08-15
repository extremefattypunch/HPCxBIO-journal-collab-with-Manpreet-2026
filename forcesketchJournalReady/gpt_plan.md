# Journal-upgrade execution plan for the ForceSketch coding agent

The current manuscript already contains a publishable scientific core: small-$K$ sketches fail as direct replacements for extreme-tail force uncertainty, while a control-variate screening gate skips roughly $84%$–$87%$ of exact evaluations and retains about $96%$–$98%$ of high-uncertainty structures. It also reports honestly that the wall-clock benefit is strongly batch- and implementation-dependent.

For a strong journal version, the coding program should address the limitations that the manuscript itself identifies: one eight-head model family, one GPU, small $9$–$24$ atom rMD17 systems, no demonstrated relationship to reference-force error or downstream model quality, and uncertain behavior under distribution shift.

The most suitable primary target is **JCTC** if the project adds physical-error validation and end-to-end active learning. **JCP** becomes an equally credible target if the strongest additions are theoretical analysis, algorithmic generality, and robust computational scaling. JCTC explicitly publishes new theories, methodologies, and important applications in molecular dynamics and statistical mechanics; JCP welcomes advanced computational and ML-enhanced methods and software that reduces computational resources, while excluding straightforward implementations of established ideas. ([American Chemical Society Publications][1])

---

# 1. Final journal paper the agent should build toward

The journal article should support five claims.

**Claim 1 — Replacement limit**

A small, fixed number of randomized head-space directions cannot reliably preserve extreme-tail force acquisition, and the required $K$ grows with committee size, molecular dimension, and the severity of the acquisition tail.

**Claim 2 — Mechanism**

The failure is explained by estimator variance, extreme-value amplification across $D=3N$ force coordinates, and the residual spectrum of the centered head-space force Gram matrix.

**Claim 3 — Structured screening**

A learned leading subspace plus randomized residual directions can screen exact force uncertainty with a controlled false-negative risk, using exact fallback only for ambiguous structures.

**Claim 4 — Chemical relevance**

The screening procedure preserves the ability of exact multi-head disagreement to identify structures with high reference-force error and produces noninferior active-learning outcomes at lower uncertainty-evaluation cost.

**Claim 5 — Deployment boundary**

The method is useful in batched candidate-pool evaluation but may not be useful for single-structure MD; the break-even boundary can be predicted as a function of batch size $B$, atom count $N$, committee size $M$, sketch budget $K$, hardware, and fallback rate.

The present workshop manuscript already substantially supports Claims 1, 3, and part of 5. Its current contribution statement is therefore the correct foundation, not something to discard.

---

# 2. Definition of “journal ready”

The coding agent should not declare the project complete until the following evidence package exists.

## Mandatory package

1. All current results are reproduced from a frozen commit.
2. Design, calibration, and test sets are genuinely disjoint.
3. Reference-force-error evaluation is complete on every system.
4. At least one end-to-end active-learning experiment is complete with multiple seeds.
5. Committee-size scaling includes at least three values of $M$.
6. At least one system is substantially larger than the current rMD17 molecules.
7. At least two meaningfully different committee constructions are evaluated.
8. The control-variate and extreme-value behavior has a theoretical treatment verified numerically.
9. Timing is reproduced on at least two GPU classes.
10. Code, configurations, splits, raw records, and plotting scripts are publicly archived with a persistent DOI.

## Strong JCTC package

Add:

* two active-learning systems;
* at least one periodic or condensed-phase system;
* adaptive-$K$ screening;
* one second shared-trunk architecture or independently implemented multi-head model;
* explicit noninferiority testing of final model accuracy.

## Strong JCP package

Add:

* a rigorous concentration/sample-complexity argument;
* a closed-form control-variate variance analysis;
* a predictive model of the systems crossover;
* broader $M$, $N$, and hardware scaling;
* a polished open-source reference implementation.

These are internal project targets, not formal acceptance rules.

---

# 3. Agent operating rules

The coding agent must follow these rules throughout.

**R1 — Freeze the workshop result**

Create a permanent tag such as:

```text
workshop-v1.0
```

No journal experiment may silently alter the original data splits, checkpoint, score definitions, or timing procedure.

**R2 — Never tune on the test set**

All choices involving:

* $K$;
* $r_0$;
* $\alpha$;
* probe type;
* acquisition score;
* adaptive stopping thresholds;
* active-learning round size;
* committee size;
* basis-transfer strategy;

must be selected using design or validation data only.

**R3 — No manually transcribed numerical claims**

Every number in tables, figures, abstracts, and result paragraphs must be generated from machine-readable result records.

**R4 — Preserve negative results**

If a second architecture, larger system, or distribution shift causes ForceSketch to fail, record and report the failure. Do not remove failed systems or seeds.

**R5 — Compare with the fastest correct exact baseline**

At every $(B,N,M)$ setting, benchmark against whichever valid exact implementation is fastest:

* serial centered VJPs;
* batched VJPs;
* partially compiled serial;
* partially compiled batched, if it becomes valid;
* another backend if later supported.

**R6 — Keep statistical and systems claims separate**

A screening method can have strong recall and still be slower. Every result must distinguish:

$$
\text{decision quality}
\qquad\text{from}\qquad
\text{wall-clock benefit}.
$$

**R7 — Every phase ends with an audit**

No phase is complete until its audit script passes.

---

# 4. Repository architecture

Refactor the project before adding more experiments.

```text
forcesketch/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── src/
│   └── forcesketch/
│       ├── data/
│       │   ├── datasets.py
│       │   ├── splits.py
│       │   ├── manifests.py
│       │   └── structures.py
│       ├── models/
│       │   ├── base.py
│       │   ├── mace_adapter.py
│       │   ├── second_model_adapter.py
│       │   └── checkpoint_registry.py
│       ├── exact/
│       │   ├── member_forces.py
│       │   ├── centered_basis.py
│       │   ├── serial_vjp.py
│       │   └── batched_vjp.py
│       ├── sketches/
│       │   ├── gaussian.py
│       │   ├── haar.py
│       │   ├── rademacher.py
│       │   ├── pairwise.py
│       │   ├── head_subsampling.py
│       │   └── corrections.py
│       ├── control_variate/
│       │   ├── basis.py
│       │   ├── residual_frame.py
│       │   ├── estimator.py
│       │   └── transfer.py
│       ├── calibration/
│       │   ├── split_conformal.py
│       │   ├── risk_control.py
│       │   ├── adaptive_policy.py
│       │   └── diagnostics.py
│       ├── scores/
│       │   ├── uncertainty.py
│       │   ├── force_error.py
│       │   └── acquisition.py
│       ├── active_learning/
│       │   ├── pool.py
│       │   ├── acquisition.py
│       │   ├── retraining.py
│       │   └── loop.py
│       ├── evaluation/
│       │   ├── fidelity.py
│       │   ├── tail_metrics.py
│       │   ├── risk_coverage.py
│       │   ├── bootstrap.py
│       │   └── noninferiority.py
│       ├── benchmark/
│       │   ├── runner.py
│       │   ├── hardware.py
│       │   ├── profiler.py
│       │   └── cost_model.py
│       └── io/
│           ├── result_schema.py
│           ├── provenance.py
│           └── validation.py
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── experiments/
│   ├── active_learning/
│   └── benchmarks/
├── manifests/
│   ├── datasets/
│   ├── checkpoints/
│   └── splits/
├── tests/
├── scripts/
├── results/
│   ├── raw/
│   ├── validated/
│   └── summaries/
├── figures/
├── paper/
└── supplement/
```

---

# 5. Required result schema

Every experiment should write one immutable JSON or Parquet record per configuration and seed.

```text
experiment_id
experiment_version
git_commit
timestamp_utc

dataset_name
dataset_version
dataset_sha256
system_name
trajectory_id
split_manifest_sha256
split_role

model_architecture
model_config_sha256
checkpoint_sha256
model_training_seed
committee_construction
M

method
K
r0
probe_seed
probe_type
alpha
acquisition_score

dtype
batch_size
number_of_atoms
device_name
device_uuid
driver_version
cuda_version
pytorch_version
mace_version
e3nn_version

exact_score
approximate_score
reference_force_error
selected_exact
selected_approx
gate_cleared
gate_fallback

forward_time_ms
mean_force_time_ms
approx_uq_time_ms
fallback_time_ms
total_time_ms
peak_memory_bytes
kernel_launches

status
failure_reason
```

A separate aggregate script should produce every confidence interval and table.

---

# 6. Phase J0 — Reproduce and freeze the current paper

## Objective

Demonstrate that every current workshop result can be regenerated before changing the code.

## Tasks

### J0.1 Reproduce all current figures and tables

Rerun:

* 3BPA at $1200,\mathrm K$;
* ethanol;
* aspirin;
* azobenzene;
* all ten frozen probe seeds;
* exact, Gaussian, Haar, Rademacher, pairwise, head subsampling, and control-variate methods;
* all reported values of $K$.

### J0.2 Verify the current screening table

The present table shows that the control variate outperforms the free energy gate, exact-mean head subsampling, and plain Haar at matched lane budget.

The agent must regenerate this from raw records, including per-system rather than only range summaries.

### J0.3 Build numerical correctness tests

Required tests:

```text
explicit member forces == centered-basis exact forces
centered serial VJP == centered batched VJP
K = M-1 Haar == exact variance
full-rank control-variate completion == exact variance
mean-force seed == arithmetic mean of member forces
padding does not influence scores
float64 exact path agrees within specified tolerance
```

### J0.4 Archive the baseline

Produce:

```text
artifacts/workshop_v1/
    raw_results/
    summary_tables/
    figures/
    environment.yml
    git_commit.txt
    audit.json
```

## Acceptance gate

* All current main numerical claims regenerate within rounding tolerance.
* No seed, system, or failed configuration is missing.
* Every figure can be generated from one command.
* All correctness tests pass.

---

# 7. Phase J1 — Repair the statistical design

This is the first journal-critical phase.

The current manuscript says that conformal calibration is performed on a split disjoint from the design and test splits, but the leading basis is described as coming from a “calibration-set average.” That wording and potentially the implementation must be resolved.

## J1.1 Create three disjoint roles

For each system define:

$$
\mathcal D_{\mathrm{design}},
\qquad
\mathcal D_{\mathrm{cal}},
\qquad
\mathcal D_{\mathrm{test}}.
$$

Use:

* $\mathcal D_{\mathrm{design}}$ to learn $Q_{r_0}$ and choose $K$, $r_0$, score type, and method;
* $\mathcal D_{\mathrm{cal}}$ only to estimate the conformal multiplier and acquisition threshold;
* $\mathcal D_{\mathrm{test}}$ only for the final reported result.

A reasonable initial split is:

$$
20% / 20% / 60%,
$$

but the agent should calculate whether the calibration set is large enough for the desired $\alpha$.

## J1.2 Prevent trajectory leakage

Do not split adjacent frames independently.

Preferred order:

1. independent trajectories;
2. independent temperature or molecular-dynamics runs;
3. contiguous temporal blocks;
4. random frame split only as a last-resort diagnostic.

Store trajectory identifiers and frame ranges in the split manifest.

## J1.3 Freeze the basis correctly

Replace:

```text
calibration-set average A
```

with:

```text
design-set average A
```

both in code and manuscript.

For each model and system:

$$
\bar A_{\mathrm{design}}
========================

\frac{1}{|\mathcal D_{\mathrm{design}}|}
\sum_{x\in\mathcal D_{\mathrm{design}}}
P F(x)^\mathsf{T}F(x)P.
$$

The control-variate basis must be frozen before the calibration ratios are calculated.

## J1.4 Calibrate separately for each deployed estimator

For every fixed probe seed and fixed learned basis:

1. freeze estimator parameters;
2. calculate calibration ratios;
3. estimate the conformal order statistic;
4. apply that calibrated estimator to the test set.

Do not calibrate an average over ten sketches if deployment uses only one sketch.

## J1.5 Use generic score notation

Replace the method-level notation $S(x)$ with:

$$
q(x),
$$

where $q$ can be:

* maximum component uncertainty;
* maximum atom uncertainty;
* global trace uncertainty;
* reference-force error in later experiments.

## J1.6 Add coverage diagnostics

Report:

* empirical marginal coverage;
* high-UQ recall;
* false-negative probability;
* false-negative severity;
* coverage by score decile;
* coverage under temperature or molecule shift;
* confidence intervals.

## Acceptance gate

* Automated tests prove that no structure ID appears in more than one split.
* Learned basis hashes depend only on the design set.
* Calibration files depend only on the calibration set and frozen estimator.
* Test results are generated exactly once after method freezing.
* Coverage claims use correct finite-sample language.

---

# 8. Phase J2 — Reference-force-error validation

This is the highest-value new experiment because it requires no active retraining and directly connects the method to physical prediction error.

## J2.1 Define reference-error scores

For each structure calculate:

### Global force RMSE

$$
e_{\mathrm{RMSE}}(x)
====================

\sqrt{
\frac{1}{3N}
\sum_{a,\alpha}
\left(
\bar f_{a\alpha}(x)-f^{\mathrm{ref}}_{a\alpha}(x)
\right)^2
}.
$$

### Maximum atomwise force error

$$
e_{\max}(x)
===========

\max_a
\left|
\bar f_a(x)-f_a^{\mathrm{ref}}(x)
\right|_2.
$$

### Upper-tail component error

$$
e_{0.95}(x)
===========

\operatorname{quantile}_{0.95}
\left(
\left|
\bar f_d(x)-f_d^{\mathrm{ref}}(x)
\right|
\right).
$$

Make $e_{\max}$ primary because it most closely matches the current extreme-value acquisition setting.

## J2.2 Compare every uncertainty signal

Evaluate:

* exact MHC maximum-component disagreement;
* exact global disagreement;
* energy disagreement;
* exact-mean head subsampling;
* Gaussian;
* Haar;
* control variate;
* adaptive control variate, once available.

## J2.3 Required metrics

For each system and error score:

* Spearman and Kendall correlation;
* AUROC for top-$1%$, top-$5%$, and top-$10%$ error;
* AUPRC;
* high-error recall;
* enrichment factor;
* risk–coverage curve;
* area under the risk–coverage curve;
* mean and maximum error among cleared structures;
* distribution of errors in gate false negatives.

Define selective risk at coverage $c$ as:

$$
R(c)
====

\frac{1}{|\mathcal C_c|}
\sum_{x\in\mathcal C_c}
e(x),
$$

where $\mathcal C_c$ contains the structures accepted as sufficiently safe.

## J2.4 Critical comparison

Measure whether the ForceSketch gate preserves the error-detection quality of exact MHC.

The journal claim should be:

> ForceSketch is not a new universal uncertainty model; it preserves the physically relevant decisions made by an established committee uncertainty measure.

## J2.5 Failure analysis

For every gate miss, save:

* structure ID;
* geometry;
* reference error;
* exact uncertainty;
* approximate uncertainty;
* head-space spectrum;
* residual energy fraction;
* maximum-error atom;
* chemical environment around that atom.

Generate a gallery of the most severe misses.

## Acceptance gate

Proceed to the full journal application claim only if one of the following holds:

1. ForceSketch is statistically noninferior to exact MHC in high-reference-error recall; or
2. the screening gate preserves nearly all exact-MHC high-error selections; or
3. a clear mechanistic failure boundary is discovered and supported across systems.

If exact MHC itself performs poorly against force error, report that honestly and narrow the paper’s claim to efficient reproduction of committee decisions.

---

# 9. Phase J3 — Theoretical and mechanistic analysis

The coding agent can implement verification and symbolic checks, but final proofs require author review.

## J3.1 Uniform componentwise concentration

For each coordinate define:

$$
a_d^\mathsf{T}
==============

e_d^\mathsf{T}FQ.
$$

Derive a bound for:

$$
\Pr\left[
\max_{1\le d\le D}
\left|
\frac{\widehat v_d}{v_d}-1
\right|

> \epsilon
> \right].
> $$

The expected qualitative scaling is:

$$

K
=

\mathcal O\left(
\frac{\log(D/\delta)}{\epsilon^2}
\right),
$$

subject to assumptions and carefully verified constants.

The code should:

1. evaluate the theoretical bound;
2. draw synthetic $FQ$ matrices with controlled spectra;
3. run at least $10^5$ sketches per setting;
4. compare empirical failure rates against the bound;
5. validate dependence on $D$, $K$, and $\epsilon$.

## J3.2 Extreme-value bias

Quantify:

$$

\mathbb E
\left[
\max_d\widehat\sigma_d
\right]
-------

\max_d\sigma_d.
$$

Study its dependence on:

* $D=3N$;
* $K$;
* inter-coordinate correlation;
* score tail level;
* Gaussian versus Haar probes.

Fit empirical scaling models such as:

$$
\text{bias}
\sim
C
\sqrt{\frac{\log D}{K}},
$$

without claiming the form as a theorem unless proved.

## J3.3 Global-score variance

For:

$$

A(x)
====

Q^\mathsf{T}F(x)^\mathsf{T}F(x)Q,
$$

derive and numerically verify the estimator variance in terms of:

$$

\operatorname{tr}(A),
\qquad
\operatorname{tr}(A^2),
\qquad
r_{\mathrm{eff}}
================

\frac{\operatorname{tr}(A)^2}
{\operatorname{tr}(A^2)}.
$$

## J3.4 Control-variate residual variance

Define:

$$

R
=

P-Q_{r_0}Q_{r_0}^\mathsf{T},
$$

and the residual Gram matrix:

$$

A_R(x)
======

R F(x)^\mathsf{T}F(x)R.
$$

Show that only the residual contributes sketch variance and quantify the expected reduction using:

$$
\operatorname{tr}(A_R^2)
\quad\text{and}\quad
r_{\mathrm{eff}}(A_R).
$$

## J3.5 Basis-stability analysis

For every system calculate:

* fraction of disagreement energy captured by the design basis;
* principal angles between per-structure and design-set leading subspaces;
* basis drift across temperatures;
* basis drift across molecules;
* relationship between subspace drift and gate failures.

Useful quantities include:

$$

\eta_{r_0}(x)
=============

\frac{
\operatorname{tr}
\left(
Q_{r_0}^\mathsf{T}
A(x)
Q_{r_0}
\right)
}{
\operatorname{tr}(A(x))
}.
$$

## Acceptance gate

* Every analytical formula is validated by simulation.
* The control-variate advantage is predicted by residual spectral quantities.
* Extreme-value failure is explained quantitatively, not merely observed.
* Any unproved scaling is labeled empirical.

---

# 10. Phase J4 — Committee-size and construction scaling

The current $M=8$ setting limits both attainable speedup and scientific generality.

## J4.1 Committee-size grid

Evaluate:

$$
M\in{4,8,16}.
$$

Optional:

$$
M=32
$$

if the shared trunk and GPU memory make it practical.

For each $M$, sweep:

$$
K=1,\ldots,M-1.
$$

## J4.2 Committee constructions

At minimum include:

1. disjoint head-training data;
2. overlapping head-training data;
3. same-data heads with independent initialization or bootstrap;
4. one independently trained second shared-trunk committee.

The existing three head-distribution variants are useful, but they should be evaluated systematically rather than only treated as checkpoint variants.

## J4.3 Low-cost preliminary $M$ analysis

Before training new models:

* create multiple $M=4$ and $M=6$ subcommittees from the existing eight heads;
* sample several distinct head subsets;
* measure sensitivity to which heads are selected.

Label this as a subcommittee analysis, not equivalent to training genuine $M=4$ or $M=6$ committees.

## J4.4 Full training

For each final $M$ and construction:

* use at least three model-training seeds;
* keep architecture and training budget matched;
* freeze train/validation/test splits;
* report head diversity, energy accuracy, force accuracy, and exact disagreement magnitude.

## J4.5 Required analyses

Plot:

* required $K$ for a fixed recall target versus $M$;
* screening skip fraction versus $M$;
* total and incremental speedup versus $M$;
* effective and stable ranks versus $M$;
* head diversity versus sketchability;
* control-variate gain versus spectral concentration.

## Acceptance gate

The journal should contain at least:

* three values of $M$;
* two committee constructions;
* three model-training seeds for the principal comparison;
* no per-system retuning of $K$ or $r_0$ in the final generalization experiment.

---

# 11. Phase J5 — Larger and more diverse chemical systems

Use a staged system expansion so that the experiment grid remains manageable.

## Tier A — Mandatory

Retain:

* 3BPA;
* ethanol;
* aspirin;
* azobenzene.

Add one substantially larger molecule, preferably with at least $40$–$100$ atoms. A peptide-like or MD22-scale molecule is a good fit.

## Tier B — Strong journal version

Add one periodic or condensed-phase system, for example:

* liquid water;
* crystalline silicon;
* an organic crystal;
* a simple surface or material.

The exact choice should depend on where a compatible multi-head potential and public reference data can be obtained or trained reliably.

## Tier C — Optional biological extension

Add a small peptide or biomolecular system with heterogeneous local environments.

## Experimental policy

Do not run the full development grid on every new system.

### Anchor-system development

Use 3BPA to select:

* probe family;
* $K$;
* $r_0$;
* calibration level;
* adaptive policy.

### Frozen generalization

Apply the frozen configuration to every new system.

Only after reporting the frozen result may a separate system-specific optimum be shown.

## Acceptance gate

* At least one system exceeds the current molecular-size range.
* At least one evaluation tests a distribution or chemistry not represented in the design system.
* The basis-transfer and calibration requirements are clearly stated.

---

# 12. Phase J6 — End-to-end active learning

This is the strongest upgrade for a JCTC submission.

## J6.1 Primary experiment

Use a pool-based replay experiment in which all reference labels already exist.

A suitable structure is:

1. initial training set from the lower-temperature or in-distribution data;
2. candidate pool from a higher-temperature or extrapolative trajectory;
3. separate untouched final test set;
4. repeated acquisition, retraining, and evaluation rounds.

For 3BPA, the natural design is an initial model trained on lower-temperature data and a candidate pool drawn from higher-temperature structures.

## J6.2 Compared acquisition methods

Run:

1. exact multi-head force disagreement;
2. ForceSketch control-variate gate with exact fallback;
3. free energy-disagreement gate;
4. exact-mean head-subsampling gate;
5. plain Haar gate;
6. random acquisition.

Optional:

7. oracle reference-force-error acquisition, labeled as an unattainable upper bound.

## J6.3 Fairness rules

For each paired seed:

* identical initial labeled data;
* identical model initialization;
* identical candidate pool;
* identical query budget;
* identical retraining schedule;
* identical test set;
* identical total number of reference labels.

## J6.4 Gate-based acquisition algorithm

```text
for each active-learning round:
    compute ForceSketch score for every pool structure
    clear structures whose calibrated upper score is below tau
    compute exact committee uncertainty only for unresolved structures
    select the requested number of structures from the unresolved exact scores
    add reference labels
    retrain or fine-tune all compared methods under matched conditions
```

## J6.5 Required outcomes

At every round record:

* number of labeled structures;
* force MAE and RMSE;
* energy MAE;
* maximum force error;
* high-error recall;
* uncertainty-evaluation wall time;
* retraining wall time;
* candidate-scoring throughput;
* exact UQ evaluations avoided;
* overlap with exact-MHC acquisition;
* final model stability on held-out trajectories.

## J6.6 Statistical design

Run at least five paired active-learning seeds for the main system.

Define a noninferiority margin **before** final testing. For example:

* a practically meaningful absolute force-error margin; or
* a relative margin based on the variability of the exact-MHC baseline.

Report paired confidence intervals for:

$$

\Delta
======

## \mathrm{error}_{\mathrm{ForceSketch}}

\mathrm{error}_{\mathrm{exact,MHC}}.
$$

## J6.7 Secondary active-learning task

For a strong JCTC paper, repeat a smaller version on:

* one rMD17 system;
* the larger molecular system;
* or the periodic system.

## Acceptance gate

The strongest journal outcome is:

* final model quality is noninferior to exact-MHC acquisition;
* exact UQ evaluations are substantially reduced;
* candidate-scoring time is reduced in the batched regime;
* no severe high-error mode is systematically missed.

If the active-learning model degrades substantially, do not claim practical equivalence. Use the result to sharpen the failure boundary.

---

# 13. Phase J7 — Adaptive-$K$ screening

This is the highest-value optional algorithmic extension.

## J7.1 Sequential policy

Instead of evaluating fixed $K=4$ for every structure:

```text
compute mean and r0 exact leading directions
evaluate one residual direction
if confidently below threshold:
    clear
else:
    evaluate another direction
repeat
if still ambiguous:
    complete exact calculation
```

Use staged budgets such as:

$$
K_{\mathrm{res}}\in{1,2,4}.
$$

Staged doubling is likely more GPU-friendly than adding one direction at a time.

## J7.2 Batched active-set compaction

At each stage:

1. identify unresolved structures;
2. pack them into a smaller batch;
3. evaluate the next residual directions only for that batch;
4. scatter results back into the original order.

Measure whether compaction overhead erases the VJP savings.

## J7.3 Reusable orthogonal frame

Construct a complete residual orthogonal frame once.

The first columns are the screening sketch. If exact fallback is needed, evaluate only the missing columns.

Test:

$$

\widehat v_{\mathrm{partial}}
+
v_{\mathrm{missing}}
====================

v_{\mathrm{exact}}
$$

to float64 tolerance.

## J7.4 Sequential calibration

Repeated early-exit decisions require care.

Implement two options:

### Option A — Alpha spending

Assign:

$$
\alpha_1+\alpha_2+\cdots+\alpha_J\leq\alpha,
$$

and calibrate each stage separately.

### Option B — Policy-level risk control

Freeze the entire stopping policy using design data, then calibrate the final false-negative risk on a separate calibration split.

Option B more directly matches the scientific claim but requires a carefully implemented risk-control procedure.

## J7.5 Comparison

Compare adaptive and fixed $K$ at matched:

* high-UQ recall;
* false-negative risk;
* exact skipped fraction;
* total latency;
* average reverse lanes.

## Acceptance gate

Include adaptive-$K$ in the journal paper only if it provides at least one of:

* a meaningful reduction in average reverse lanes;
* a meaningful total-latency reduction;
* higher recall at matched cost;
* a clearer theoretical insight.

Do not include it merely because it is more elaborate.

---

# 14. Phase J8 — Systems and hardware study

## J8.1 Benchmark axes

Measure over:

$$
B\in{1,4,16,64},
$$

subject to memory, and over:

$$
M\in{4,8,16},
$$

with multiple $K$ values.

Use systems spanning atom count $N$.

The final performance model should therefore cover:

$$
T=T(B,N,M,K,\text{method},\text{hardware}).
$$

## J8.2 Hardware

Mandatory:

* current RTX 5070 Laptop GPU;
* one datacenter or workstation GPU with substantially different compute and memory characteristics.

Optional:

* third GPU generation;
* CPU reference;
* mixed-precision variant.

## J8.3 Exact baselines

Benchmark:

* serial explicit heads;
* serial centered basis;
* batched centered VJP;
* partially compiled serial;
* compiled batched if later supported;
* mean-force-only path;
* exact completion reusing screened directions.

## J8.4 Timing protocol

For every benchmark:

1. preload data;
2. preallocate tensors;
3. finish compilation and autotuning;
4. use CUDA events;
5. perform at least 100 warmups;
6. perform sufficient measured iterations for stable medians;
7. report median and IQR;
8. record peak memory;
9. record kernel count;
10. record hardware clocks and software versions;
11. separate compilation time.

## J8.5 Fallback accounting

The actual screening cost must be explicit.

If prior directions are reused:

$$

T_{\mathrm{screen}}
===================

T(B,K+1)
+
p_{\mathrm{fallback}}
T_{\mathrm{missing}}
(B_{\mathrm{fallback}},M-1-K).
$$

Do not model fallback as recomputing all directions unless that is what the code does.

## J8.6 Break-even surface

Generate a plot or table showing where:

$$
T_{\mathrm{screen}}
<
T_{\mathrm{exact}}.
$$

Axes should include at least:

* batch size;
* atom count;
* committee size;
* fallback rate.

The output should support statements such as:

> For $M=16$ and batched candidate pools above a measured size threshold, screening pays; for $M=8$ and $B=1$, it does not.

## Acceptance gate

* Two hardware classes.
* Fastest exact baseline selected independently at every setting.
* Exact and screened memory included.
* Total workflow speedup, not only incremental speedup, is primary.
* All crossover claims are based on measured data.

---

# 15. Phase J9 — Additional baselines

Add these before journal submission.

## J9.1 Exact leading-subspace only

Evaluate only the $r_0$ exact leading directions without a randomized residual.

This isolates the value contributed by the residual sketch.

## J9.2 Adaptive head subsampling

Sequentially evaluate heads until the decision becomes sufficiently certain.

Use the exact mean-force-assisted estimator, not ordinary sample variance.

## J9.3 Cross-system control-variate basis

Test:

* basis trained per system;
* basis trained on pooled systems;
* basis trained on one molecule and transferred;
* basis trained at one temperature and transferred.

## J9.4 Pooled calibration

Compare:

* system-specific calibration;
* molecule-family calibration;
* pooled calibration;
* no recalibration transfer.

## J9.5 Equal-wall-clock comparison

Because lane count is not equivalent to time under batched VJPs, compare methods under:

1. equal reverse-lane budget;
2. equal measured latency budget.

## J9.6 Standard randomized low-rank comparison

Implement the closest practical Hutch++-style or low-rank-plus-residual baseline applicable to the head-space statistic.

The journal paper should distinguish the ForceSketch control variate from established randomized trace-estimation constructions rather than only citing them.

---

# 16. Statistical analysis requirements

## Bootstrap policy

Use paired bootstrap when structures are independent.

For trajectory data:

* estimate temporal autocorrelation;
* use trajectory-level or block bootstrap;
* compare IID and block-bootstrap intervals.

## Tail uncertainty

For top-$5%$ recall, report:

* number of positive structures;
* point estimate;
* paired bootstrap interval;
* binomial-style interval as a robustness check.

## Model and sketch randomness

Separate:

$$
\text{model-training variability}
$$

from

$$
\text{probe-seed variability}.
$$

Use nested reporting:

* mean over model seeds;
* within-model variation over probe seeds;
* across-model variation.

## Multiple systems

Report each system separately before aggregates.

Do not average away system-specific failures.

## Pre-registration file

Before each final experiment, commit:

```text
protocols/<experiment_id>.yaml
```

containing:

* hypothesis;
* primary metric;
* secondary metrics;
* split hashes;
* methods;
* seeds;
* exclusion rules;
* success criterion.

---

# 17. Test suite the agent must implement

```text
test_centering_projector.py
test_centered_basis_exactness.py
test_member_force_equivalence.py
test_mean_force_seed.py
test_gaussian_unbiasedness.py
test_haar_unbiasedness.py
test_haar_full_rank_exactness.py
test_std_correction_gaussian.py
test_std_correction_haar.py
test_pairwise_unbiasedness.py
test_head_subsampling_exact_mean.py
test_control_variate_unbiasedness.py
test_control_variate_completion.py
test_basis_uses_design_only.py
test_split_disjointness.py
test_conformal_order_statistic.py
test_conformal_synthetic_coverage.py
test_adaptive_alpha_spending.py
test_batched_serial_equivalence.py
test_padding_and_masks.py
test_float32_float64_behavior.py
test_result_schema.py
test_no_missing_seeds.py
test_figure_numbers_match_results.py
```

CI should run all inexpensive tests on every commit. GPU integration tests can run nightly or before releases.

---

# 18. Reproducibility and journal compliance

JCTC and JCIM introduced a policy effective May 2026 asking original research articles to include a Data and Software Availability Statement and to provide the materials needed to reproduce key results whenever possible, including model details, splits, scripts, environments, and hardware metadata. ([American Chemical Society Publications][2])

AIP requires a Data Availability Statement for research articles, and its author guidance says the data needed to interpret and reproduce the conclusions should be available. ([AIP Publishing LLC][3])

## Required release package

Archive through Zenodo, Figshare, or an equivalent persistent repository:

```text
source code
environment lockfile
exact dependency versions
dataset acquisition scripts
dataset hashes
checkpoint hashes
split manifests
training configurations
probe seeds
raw per-structure predictions
raw timing records
profiler outputs
active-learning histories
all figure-generation scripts
all table-generation scripts
test suite
worked example
```

## One-command reproduction

Provide commands such as:

```bash
forcesketch reproduce --figure 2
forcesketch reproduce --table screening
forcesketch audit --all
forcesketch benchmark --config configs/benchmarks/journal.yaml
```

## AI disclosure

Because the workshop manuscript says an AI coding assistant contributed to implementation, experiments, diagnosis, and drafting, the journal submission should include a detailed disclosure.

AIP requires detailed Methods disclosure when AI use may affect findings, including tool name/version, task, and reason for use. ([AIP Publishing LLC][4])

The release should therefore record:

* tool name and model version;
* dates used;
* tasks assisted;
* which outputs were independently verified;
* tests used to validate generated code;
* confirmation that the authors checked references, calculations, and scientific claims.

---

# 19. Journal manuscript structure

## Proposed title

**Structured Jacobian Sketching for Safe Screening of Multi-Head Force Uncertainty**

Alternative:

**Limits and Opportunities of Randomized Head-Space VJPs for Molecular Force Uncertainty**

## Main paper outline

### 1. Introduction

* multi-head committee force-UQ bottleneck;
* why extreme-tail decisions differ from trace estimation;
* current gap;
* final contributions.

### 2. Theory

* centered Jacobian formulation;
* Gaussian and Haar estimators;
* finite-$K$ standard-deviation corrections;
* componentwise concentration;
* extreme-value effect;
* global variance and effective rank;
* control-variate residual variance.

### 3. Methods

* learned leading subspace;
* fixed-$K$ gate;
* adaptive-$K$ policy;
* conformal or risk-control calibration;
* exact fallback with direction reuse.

### 4. Experimental design

* models;
* committee sizes;
* systems;
* split protocol;
* reference-force metrics;
* active-learning protocol;
* hardware and timing.

### 5. Replacement limits

* fidelity against $K$, $M$, $N$;
* global versus maximum-component scores;
* theoretical and empirical scaling.

### 6. Screening performance

* free energy, head subsampling, Haar, leading-only, and control-variate baselines;
* fixed versus adaptive screening;
* coverage and fallback behavior.

### 7. Chemical relevance

* reference-force-error detection;
* risk–coverage;
* active-learning outcomes.

### 8. Computational performance

* two GPUs;
* batched versus serial exact paths;
* crossover map;
* memory and throughput;
* total workflow speedup.

### 9. Limitations

* distribution shift;
* model dependence;
* committee disagreement as proxy;
* hardware dependence;
* calibration data requirement.

### 10. Conclusion

Emphasize limits and valid deployment regime rather than universal acceleration.

---

# 20. Required main figures

## Figure 1 — Method and reusable fallback

Show:

```text
shared trunk
    |
M heads
    |
mean + leading exact directions + residual sketch
    |
calibrated gate
   / \
clear  unresolved
        |
evaluate only missing directions
```

## Figure 2 — Theory and extreme-value failure

Panels:

* uniform component error versus $K$ and $D$;
* maximum-score bias versus $\log D/K$;
* theoretical versus empirical failure probability.

## Figure 3 — Scaling across $M$

Panels:

* top-tail recall versus $K/(M-1)$;
* required $K$ for fixed recall;
* control-variate gain versus residual effective rank.

## Figure 4 — Physical-error validation

Panels:

* risk–coverage curves;
* high-reference-error recall;
* false-negative severity.

## Figure 5 — Screening Pareto

Axes:

$$
\text{exact evaluations skipped}
\quad\text{versus}\quad
\text{high-UQ recall}.
$$

Include all baselines and systems.

## Figure 6 — Active learning

Panels:

* force error versus reference-label count;
* force error versus acquisition wall time;
* exact UQ evaluations per round;
* final model noninferiority.

## Figure 7 — Systems crossover

Heatmap over:

$$
(B,N,M)
$$

showing:

$$
\frac{T_{\mathrm{exact}}}{T_{\mathrm{screen}}}.
$$

## Figure 8 — Adaptive policy

Show:

* distribution of per-structure $K(x)$;
* fraction terminating at each stage;
* recall versus average lane count.

---

# 21. Required tables

## Table 1 — Models and datasets

```text
system
atoms
periodic/nonperiodic
training structures
design structures
calibration structures
test structures
M
committee construction
model seeds
```

## Table 2 — Reference-force-error detection

```text
method
AUROC
AUPRC
top-5% error recall
risk-coverage area
false-negative maximum error
```

## Table 3 — Screening

```text
method
average lanes
exact skipped
high-UQ recall
coverage
total latency
speedup
```

## Table 4 — Active learning

```text
method
label budget
final force MAE
final max error
selection time
exact UQ evaluations
noninferiority interval
```

## Table 5 — Hardware scaling

```text
GPU
B
N
M
exact backend
screen backend
exact latency
screen latency
speedup
peak memory
```

---

# 22. Sixteen-week critical path

| Weeks | Work                                                           |
| ----- | -------------------------------------------------------------- |
| 1–2   | J0 reproduction, refactor, test suite, immutable result schema |
| 2–3   | J1 disjoint split implementation and conformal audit           |
| 3–5   | J2 reference-force-error evaluation                            |
| 3–6   | J3 theory scripts and Monte Carlo verification                 |
| 5–8   | J4 committee-size training and analysis                        |
| 6–9   | J5 larger molecular system; periodic-system feasibility        |
| 8–12  | J6 active-learning experiments                                 |
| 9–12  | J7 adaptive-$K$ implementation and calibration                 |
| 10–13 | J8 second-GPU benchmarks and crossover model                   |
| 12–14 | Freeze method and rerun final grids                            |
| 14–15 | Figures, tables, SI, statistical audits                        |
| 15–16 | Manuscript, repository release, archival DOI, cover letter     |

Parallelization:

* theory can run alongside reference-error experiments;
* model training can run while analysis code is developed;
* hardware scripts can be prepared before the second GPU becomes available;
* manuscript Methods can be drafted once J1 is frozen.

---

# 23. Go/no-go gates

## Gate A — Statistical correctness

Stop journal-scale experiments if split leakage, seed averaging, or fallback accounting cannot be resolved.

## Gate B — Physical relevance

If ForceSketch preserves exact MHC decisions but not reference-error detection, narrow the claim. Do not claim safer molecular simulation.

## Gate C — Active learning

If final model error is materially worse than exact-MHC acquisition, do not claim end-to-end equivalence. Analyze which structures were missed.

## Gate D — Generality

If performance depends entirely on the original $M=8$ checkpoint, reframe the article as an architecture-specific study rather than a general method.

## Gate E — Runtime

If the method is slower on both tested hardware classes in realistic candidate-pool regimes, remove acceleration from the title and lead with the statistical replacement limit.

## Gate F — Adaptive method

If adaptive-$K$ saves less than approximately $10%$–$15%$ total latency after compaction overhead, move it to the supplement or omit it.

---

# 24. Journal-selection decision after final experiments

**Submit to JCTC first when:**

* active learning is noninferior;
* physical-error screening is preserved;
* at least one larger or periodic application works;
* multiple committee sizes are included.

**Submit to JCP first when:**

* the theory of extreme-value sketch complexity is strong;
* residual-spectrum analysis predicts performance;
* scaling across $M$, $N$, and hardware is comprehensive;
* active learning is informative but not the strongest result.

**Consider JCIM when:**

* the main contribution becomes an open benchmark and software package;
* the comparison across UQ methods is broader than the chemical-physics theory.

---

# 25. Immediate first instructions for the coding agent

The agent’s next work queue should be exactly:

```text
1. Create workshop-v1.0 tag.
2. Reproduce every current number from raw records.
3. Introduce immutable design/calibration/test manifests.
4. Move control-variate basis learning to design data only.
5. Implement exact fallback by completing and reusing the residual frame.
6. Add reference-force-error scores and risk-coverage metrics.
7. Run reference-error evaluation on the current four systems.
8. Produce a journal-go/no-go report before training any new model.
```

The first new scientific result should be the reference-force-error analysis, not another probe distribution or another low-level kernel. That result will determine whether the paper can credibly become a chemical-methodology article, rather than remaining a technically strong study of how accurately one uncertainty proxy can approximate another.

[1]: https://pubs.acs.org/page/jctcce/about.html?utm_source=chatgpt.com "About Journal of Chemical Theory and Computation - ACS Publications"
[2]: https://pubs.acs.org/doi/abs/10.1021/acs.jctc.6c00733?utm_source=chatgpt.com "Advancing Reproducibility and Open Data in Theoretical and Computational Chemistry | Journal of Chemical Theory and Computation"
[3]: https://publishing.aip.org/resources/researchers/open-science/research-data-policy/?utm_source=chatgpt.com "Research Data Policy - AIP Publishing LLC"
[4]: https://publishing.aip.org/resources/researchers/policies-and-ethics/ai-policy/?utm_source=chatgpt.com "AI Policy - AIP Publishing LLC"
