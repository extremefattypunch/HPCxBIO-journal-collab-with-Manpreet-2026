# ForceSketch — §52 Paper Go/No-Go assessment

Execution stopped here per the agreed scope (plan Task 7.2). Every number below
traces to a file in `results/`; none was transcribed by hand.

**Systems:** 3BPA@1200K (2139 structures, disjoint/overlapping/same committees) and
rMD17 ethanol/aspirin/azobenzene (1000 structures each, joint 10-molecule committee).
M = 8 heads throughout. All estimator evaluation in float64; timing in float32.

---

## Verdict

**The §52 gate is NOT met as written.** Five of its six requirements pass; the
recall requirement fails, and it fails by a wide margin rather than marginally.

| §52 requirement | status | evidence |
|---|---|---|
| strongest exact batched baseline complete | **partial** | `is_grads_batched` unusable on MACE+e3nn 0.4.4; serial loop is the strongest *available* baseline, documented in `tests/test_real_model.py::test_batched_vjp_limitation_is_characterized` |
| mathematical tests passing | **pass** | 41/41, incl. §20, §21, §22, §23, §24 |
| one complete accuracy–latency Pareto curve | **pass** | `paper/figures/fig2_pareto_*.png` |
| top-5% recall ≥ 0.90 for at least one useful K | **FAIL** | best is 0.743 (control variate r0=2, K=4, 3BPA); 0.52–0.56 on rMD17 |
| incremental UQ speedup ≥ 1.5× | **pass** | **2.19–2.51×** at K=3; **3.11–4.03×** at K=2 (total workflow: 1.77–1.87× and 2.17–2.39×) |
| sketching competitive with head subsampling | **pass, conditional** | wins at equal total lanes *when an exact mean force is required* |

**§48 success criteria** (K≤4 **and** recall ≥0.90 **and** speedup ≥1.5× on ≥2 systems):
**not met** — the recall term fails on all four systems.

**§51 statistical kill gate:** does not trigger on 3BPA (Haar K=4 reaches ρ=0.859
against the 0.85 threshold — a narrow escape), but **triggers on all three rMD17
molecules**. Escalation rungs 3 (calibrated exact fallback) and 4 (low-rank control
variate) were both implemented in response, per the spec's ordering.

**§50 systems kill gate** (threshold 1.2× on incremental UQ): cleared comfortably and
uniformly — 2.19–2.51× at K=3 across every system and batch size tested.

---

## The result that does hold

Replacement fails; **screening succeeds, on every system tested**. Control variate
r0=2, K=4, α=0.05, evaluated on held-out 60% test splits with c_α and τ fitted on
the calibration split only:

| system | high-UQ recall | exact evaluations skipped | FNR | screening speedup |
|---|---|---|---|---|
| 3BPA@1200K | 0.982 | 86.0% | 0.018 | 1.25× |
| rMD17 ethanol | 0.973 | 79.3% | 0.027 | 1.16× |
| rMD17 aspirin | 0.988 | 76.5% | 0.013 | 1.12× |
| rMD17 azobenzene | 0.971 | 84.9% | 0.029 | 1.23× |

H5's targets (skip ≥50%, retain ≥95%) are met on all four with margin. This is
precisely the outcome §66 prescribes a framing for: **screening, not replacement.**

Screening speedups use the *measured* cost model T(L) = 10.38 + 11.78·L ms (least-squares
fit over the full lane scan), not lane counts. Lane count is not lane cost — the forward
pass is paid once regardless of L — and the lane-count model overstates the gate by
~0.06×. Both are recorded.

---

## §72 questions

**Q1 — best useful K?** K = 4 with an r0 = 2 control variate (5 total reverse lanes
including the mean-force lane). For plain sketching, Haar at K = 4.

**Q2 — top-5% recall at that K?** 0.743 on 3BPA; 0.52–0.56 on rMD17. **Below the
0.90 target on every system.** Reaching 0.90 requires K ≈ 6 of 7, a marginal saving.

**Q3 — beats head subsampling at equal budget?** Yes at equal *total* lanes when the
exact mean force is required: Haar K=3 (0.529) vs head-subsample K=3+mean (0.477) on
3BPA. No if the mean force is not required: head-subsample K=4 without a mean lane
scores 0.588. Both framings are in `results/raw/03_sketch_fidelity_*.jsonl`; the
distinction matters because MD needs the exact mean force.

**Q4 — incremental UQ speedup vs exact?** §45 defines this as
[T(L=8)−T(L=1)]/[T(L=1+K)−T(L=1)], i.e. uncertainty cost once the mean force is paid:
**2.19–2.51× at K=3** and **3.11–4.03× at K=2**, across four systems and
B ∈ {1,4,16,64}. Measured against the serial exact baseline.

**Q5 — total mean-force + UQ speedup?** T(L=8)/T(L=1+K), which includes the
mean-force lane and is necessarily smaller: **1.77–1.87× at K=3** and
**2.17–2.39× at K=2**. An earlier draft of this document quoted the total figure
under the incremental label; §45 forbids conflating them and they are now separate.

**Q6 — where does the benefit disappear?** It does not, over B = 1…64 — the ratio is
flat because real MACE is compute-bound even at B=1 (21 ms for a single lane). It
*would* shrink on a stack where batched VJP works; see the caveat below.

**Q7 — exact evaluations the gate can skip?** 76.5–86.0% at α = 0.05.

**Q8 — high-uncertainty recall retained?** 97.1–98.8%.

**Q9 — does Triton materially affect total runtime?** **No, measured.** On the real
committee at B=16, the full §41 postprocessing chain costs 0.16 ms against a 46.2 ms
ForceSketch(K=3) step — **0.3% of runtime**, thirty times below §43's 10% threshold.
The profile also shows why: the shared forward is 9.4 ms of a 100.6 ms exact step, so
the reverse pass dominates and postprocessing is negligible against it. §43's rule
therefore says do not implement the Triton kernel, and §42 explicitly permits
reporting that.

---

## Caveats that must survive into any write-up

1. **The exact baseline is serial, not batched.** `torch.autograd.grad(...,
   is_grads_batched=True)` is unusable on MACE + e3nn 0.4.4: it succeeds once or
   twice in a fresh process, then raises `Cannot access data pointer of Tensor that
   doesn't have storage` from the TorchScript interpreter. The specialization is
   cumulative and survives a fresh model instance, disabling
   `jit_script_fx`/`optimize_einsums`/`specialized_code`, disabling the profiling
   executor, and rebuilding the model unscripted. `torch.func.vjp`+`vmap` is
   separately blocked because MACE calls `requires_grad_()` inside forward. **On a
   stack where batched VJP works, the exact baseline would be faster and every
   speedup here correspondingly smaller.**

2. **cuEquivariance is unavailable** with the PR #800 readout, so all baselines are
   e3nn-only.

3. **§38's stated hypothesis is not supported.** The spectrum is fairly flat (stable
   rank 4.6–5.1 of 7 across systems), which is the regime that *favours* random
   sketching — so spectral concentration is not what limits recall here. The limit is
   simply estimator variance at small K.

4. **Timing is single-GPU, single-run.** §44's full protocol (≥500 iterations,
   counterbalanced ordering, subprocess isolation) is implemented for the lane scan
   only; medians and IQRs are recorded throughout. §46's phase profiling is done
   (Q9 above); §26's six-implementation comparison is moot for the batched and
   `torch.func` variants and unmeasured for the compiled-forward one; §31's batch
   sweep is covered by the lane scan at B in {1,4,16,64}.

6. **NVTX attribution is partial.** e3nn ships 18 compiled `ScriptModule`s and
   TorchScript rejects forward hooks on them, so per-module ranges stop at the
   tensor-product boundary. Their time is attributed to the enclosing range rather
   than lost, but kernel-level attribution *inside* the tensor products is not
   available through this route.

5. **rMD17 spans 9–24 atoms**, so molecule size is a weak axis; the size story rests
   on atoms-per-batch.

---

## Recommendation

The work supports a **§66-framed paper**: exact multi-head force uncertainty can be
screened, not replaced, by a small number of head-space VJPs — skipping ~80% of exact
evaluations while retaining ~98% of high-uncertainty structures, across four systems,
with a low-rank control variate as the enabling ingredient.

It does **not** support the headline the spec hoped for (K ≤ 4 replacing exact UQ at
≥0.90 recall). Presenting it as such would require dropping the rMD17 results, which
§47 and §70 both forbid.

The decision this needs from a human: whether a screening-framed result is worth
submitting, or whether to spend the remaining time on the §51 rungs not yet tried —
notably a *learned* (rather than eigen-) subspace, or per-atom rather than per-
structure gating.
