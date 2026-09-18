# Review — ForceSketch (`paper/main_sim2sci.pdf`), NeurIPS 2026 Workshop Sim2Sci

*Self-review written in the same format as the Sim2Sci Official Review form (Title / Summary /
Review / Strengths and Weaknesses / Limitations / numeric scores), for comparison against the
reviews of Submissions 193 and 274. Reviewed from the rendered PDF dated 30 Aug 2026.*

---

## TITLE

Unusually disciplined comparative work on a real bottleneck; the positive screening claim rests on a single untested operating point and a promised coverage number is missing

---

## SUMMARY

Multi-head committees make energy uncertainty nearly free for machine-learned interatomic
potentials: M readout heads share one message-passing trunk, so a single forward pass returns all
M energies. Forces do not amortize the same way. Forces are gradients, reverse mode returns the
gradient of one scalar per backward pass (which the paper calls a *lane*), and force disagreement —
the quantity acquisition rules and MD actually consume — therefore costs about M lanes. The paper
takes this limitation directly from the multi-head committee work it builds on and asks whether a
handful of cheap random probes can estimate the force-variance statistic instead of reconstructing
the member forces that define it.

The method (ForceSketch) replaces the exact (M−1)-direction centered head basis with K randomized
or orthogonal directions, each still one lane. Three estimator refinements are developed: Gaussian
probes centered into the disagreement subspace; Haar-orthogonal probes, which multiply estimator
variance by exactly (r−K)/(r+2) at no extra cost and are identified as the single largest
methodological lever; and a low-rank control variate that evaluates the r₀ leading eigendirections
of a design-split disagreement Gram matrix exactly and sketches only the residual. Finite-K
corrections for the concavity of the square root are given in closed form, including a Haar constant
rewritten to be finite at K = r where the conventional Beta expression is singular.

Experiments use the pretrained eight-head MACE committees released with Beck et al. (2025) — all
three head-distribution variants — on 3BPA at 1200 K (extrapolative relative to its 300 K training
set) and on rMD17 ethanol, aspirin and azobenzene, with ten frozen seeds and paired bootstraps.

The answer is deliberately two-sided. **Replacement fails**: under the primary max-component
acquisition rule the best sketch reaches only 0.484–0.570 top-5% recall at K=4 and 0.724–0.801 even
at K=6 of 7, well short of a 0.90 substitution line, and a maximum over 3N noisy estimates is
biased upward by 1.07–1.08 at K=4 in a way no marginal correction removes. **Screening
succeeds**: a split-conformal gate over the control-variate sketch (r₀=2, K=4, α=0.05) skips
0.836–0.872 of exact force-UQ evaluations while still routing 0.964–0.982 of high-uncertainty
structures to the exact path, on every system, and Pareto-dominates three matched-budget
alternatives — a free energy-disagreement gate, exact-mean head subsampling, and a plain Haar
sketch — in 11 of 12 gate-by-system comparisons. The wall-clock benefit is explicitly
regime-dependent: 1.85× total force-plus-UQ speedup at batch 16, collapsing to 1.11× at batch 1
where extra cotangent lanes are nearly free.

---

## REVIEW

**Significance.** The bottleneck is real, specific, and stated by the very work this paper extends —
the quoted admission that a single backward pass cannot yield forces for all heads is the entire
motivation, and it is a good one to build on. The Sim2Sci framing is also earned rather than
retrofitted: a learned potential is an emulator of an expensive mechanistic model that is
misspecified unevenly across configuration space, and the workflow needs not correctness everywhere
but a trustworthy account of where correctness fails, so the expensive model is called exactly
there. That is squarely the workshop's subject. The audience that runs committee UQ over MLIPs is
real and growing, and the screening result is something they could adopt.

What bounds the significance is that the payoff is narrow and its downstream value is asserted
rather than demonstrated. The deliverable is roughly 1.4× on the uncertainty portion of a *batched*
active-learning sweep, capped by M=8, with no active-learning loop run and no reference-force error
measured. The paper says this plainly — "we preserve a proxy, not a demonstrated outcome" — and
that honesty is worth more than an overclaim would be, but a reader deciding whether to adopt the
gate still has to take the downstream benefit on faith.

**Originality.** The components are drawn from established randomized linear algebra (Hutchinson
trace estimation, orthogonal random features, low-rank sketching) and from standard split conformal
prediction, and the paper says so explicitly rather than dressing them up. The originality is in
the transposition — recognising that *head space*, not parameter or input space, is the right thing
to compress at inference, and that the downstream object is a ranking or a threshold decision rather
than the member forces themselves — plus two smaller genuine contributions: the exact
(r−K)/(r+2) variance ratio that makes orthogonalization obviously worth doing, and the
non-singular form of the Haar finite-K constant. This is a modest but honest originality claim, and
the paper does not inflate it.

**Quality.** This is the paper's strongest dimension, and in the specific matter of comparative
discipline it is better than most submissions I see. Four things stand out.

First, the paper identifies and then closes the loophole that would otherwise void its positive
result: a calibrated gate is a generic wrapper, so any score correlated with exact disagreement
could be calibrated the same way, and the claim only stands if *this* gate beats the cheaper ones a
reader would actually reach for. It then builds three of them at matched lane budget with identical
splits, calibration and threshold — including a genuinely free one. Most papers assert the wrapper's
value; this one tests it.

Second, the free gate is reported as the informative failure it is: energy-disagreement recall
collapses to 0.636 on azobenzene, establishing that energy disagreement is a different signal rather
than a cheap proxy for force disagreement. That is a result, not a foil.

Third, the paper engineered the strongest baseline into existence rather than accepting a weak one.
Batched reverse mode failed on this stack from the third call onwards; the paper diagnoses it to the
TensorExpr fuser interacting with e3nn's module-level scripted spherical harmonics, fixes it, and
states outright that without the fix it "would have compared against a Python loop and overstated
every speedup." It then checks whether `torch.compile` composes with the batched path (it does not,
for the same storage-free BatchedTensor reason) and reports the eager batched path as primary
"because it is the fastest correct baseline available on this stack, not merely the most convenient
one." Very few papers do the work that makes their own numbers smaller.

Fourth, Appendix R reports a correction that cost the paper a favourable result: an earlier version
scored head subsampling with the sample variance while still charging it a mean-force lane and
reported Haar ahead; the corrected comparison, using the exact mean the baseline has already paid
for, reverses the ordering and head subsampling wins significantly on two of four systems. The paper
reports the corrected result and scopes its claims accordingly. Self-correction of this kind, in
print, against interest, is rare enough that it should be credited explicitly.

Supporting hygiene is consistent: ten seeds frozen before testing with none discarded; paired
bootstraps over structures with 10,000 resamples on the seed-averaged statistic, with seed-to-seed
deviations recorded separately; the explicit note that metrics are computed per seed then averaged
so the deployed budget is K lanes and not 10K; float64 for estimator evaluation with an actual
argument for it (κ≈36, so a float32 "exact" reference carries more relative error than the 10⁻⁵
tolerance at which full-rank exactness is verified); design/calibration/test splits whose separation
is tied to the conformal requirement that the score function be frozen before the calibration ratios
form, with disjointness asserted in code; and a mechanistic explanation of *why* sketching
underperforms (stable rank 2.91–3.08 of a maximum 7, leading direction carrying 0.325–0.344 against
0.143 isotropic) that correctly inverts the naive intuition — a concentrated spectrum is what breaks
a random sketch, not what saves it.

What holds quality at "good" rather than "excellent" is not sloppiness but coverage: the central
positive claim is measured at exactly one operating point, and a robustness check the paper promises
is absent. Both are detailed as W1 and W2 below, and both are cheap to fix given that everything
downstream of the cached head-force matrix is offline linear algebra.

**Clarity.** Dense but precise, and unusually well structured around its own negative-then-positive
argument; the notation is set up carefully and the hat convention for sketched quantities is
maintained. Four clarity problems are worth fixing: the title misdescribes the contribution; the
main text under-surfaces the Appendix R finding that the naive baseline beats the sketch as an
estimator; headline recall numbers are quoted without naming which acquisition rule and which
estimator produced them, in a paper that deliberately reports two rules whose values are numerically
close; and the appendices repeat their main-text run-in headings verbatim, with one appendix
containing no prose at all. Details in W6, W7 and W9.

**Recommendation.** Borderline accept, leaning positive. The comparative discipline, the reported
self-correction, and the baseline-engineering work are better than the norm, and a clean
negative-plus-screening result on a genuine bottleneck is a good thing for this workshop to host.
What keeps me from a clear accept is narrow and specific: the screening headline is one point where
it should be a curve, and the paper promises an observed-coverage number it never gives. Supplying
the α sweep and that coverage number would move this to a 5 for me — they are the difference between
"this configuration worked" and "this method works."

---

## STRENGTHS AND WEAKNESSES

### Strengths

**S1 — A real, precisely located bottleneck.** The asymmetry between energy and force uncertainty in
shared-trunk committees is quoted from the work being extended rather than manufactured, and the
lane abstraction (one reverse pass = one scalar's gradient) makes the cost model exact and auditable
instead of hand-waved.

**S2 — The wrapper loophole is identified and closed.** See the Review field. Building three
matched-budget gates including a free one, and reporting that the free one sometimes wins (perfect
recall on ethanol at only 0.322 skipped) and sometimes collapses (0.636 on azobenzene), is the
experiment the positive claim actually requires.

**S3 — Baseline fairness pursued past the point of convenience.** Diagnosing the batched
reverse-mode failure, fixing it, accepting a ~4% cost to the serial path from losing fusion, then
separately establishing that compilation does not compose with it — and saying explicitly that
without this the comparison would have been against a Python loop and every speedup overstated.
Appendix M is also a genuinely useful artifact in its own right: any e3nn-based UQ work will hit
that trap.

**S4 — A reported self-correction against interest (App. R).** Documented above. This should be kept
in the camera-ready exactly as written.

**S5 — The negative result is measured, explained, and not softened.** Replacement fails; the
extreme-value bias that causes part of the failure is measured directly (1.42–1.53 at K=1 falling to
1.07–1.08 at K=4 and to exactly 1.000 at full basis — which doubles as a pipeline check); and the
head-space spectrum is computed to explain the mechanism rather than left as speculation. The
recurring diagnosis — "a sketch that orders structures tolerably still fails to identify which ones
land in the extreme tail," with Spearman 0.648–0.781 against top-5% recall 0.484–0.570 at K=4 — is
the right way to characterise this kind of failure.

**S6 — Honest, regime-resolved treatment of speed.** Two speedup ratios are defined, kept separate,
and the more conservative one (total force-plus-UQ) is led with, on the stated ground that the
incremental ratio is unstable precisely in the regime the section is about. The B=1 collapse to
1.11× is in the abstract, and the conclusion states that the gate becomes a net slowdown there. The
abstract's closing line — "the decision-quality result is robust; the speed result is not" — is an
accurate summary of the paper's own evidence, which is not something one can say of every abstract.

**S7 — Cost accounting that refuses a free advantage.** Charging every gate the same fallback
structure (1+K lanes, then only r−K further passes on fallback rather than a fresh M), on the
explicit ground that crediting the reuse to the control variate alone "would manufacture much of its
advantage," and charging fallback a fresh forward rather than a retained graph as the conservative
choice.

**S8 — Numerical and statistical care with stated reasons.** The κ≈36 float64 argument; the finite-K
corrections verified by Monte Carlo at 4×10⁵ draws; the acknowledgement that the control variate
mixes exact and sketched directions and so has no exact constant, with the residual-frame Haar
correction applied as an approximation and the reason it does not matter for screening (cα absorbs
any constant scale error).

**S9 — Artifact quality.** Anonymous 4open.science link that is actually live, datasets and
checkpoints pinned by SHA-256 in a freeze manifest, ten frozen seeds and every raw result record
included, and the useful property that everything downstream of the cached head-force matrix
reproduces offline without a GPU.

### Weaknesses

**W1 (major, and the cheapest fix) — The screening result is one point where it needs to be a
curve.** Every headline screening number is reported at exactly r₀=2, K=4, α=0.05. There is no
sweep over α, no sweep over r₀, and no K-dependence for the *screening* result — Figure 1 gives
recall against K for the failed replacement task, and Appendix B gives the Pareto across *gates* at
fixed α, but nothing varies the knob that defines the operating point. This matters more here than
it would elsewhere, because dialing α is the entire value proposition of a conformal gate: the
practitioner's question is not "what does α=0.05 give" but "what does the (skip, recall) frontier
look like, and where do I want to sit on it." As written, a reader cannot tell whether 0.836–0.872
skipped at 0.964–0.982 recall is a favourable point on a gentle curve or a lucky point on a steep
one, nor whether r₀=2 was chosen because it is best. Recommend: the (exact-skipped, high-UQ-recall)
curve as α varies over roughly 0.01–0.20, for the control variate and at least the free gate, on all
four systems; and a small r₀ × K table. Given that this is offline linear algebra over cached
head-force matrices, the cost should be minutes.

**W2 (major, and a promise the paper makes) — The observed coverage number is never reported.**
Appendix F states that 3BPA at 1200 K calibrated against 300 K behaviour "is exactly such a shift,
so we report the observed coverage there rather than assume it." No observed coverage figure appears
anywhere in the paper. This is the one place where the paper's own standard of evidence is not met,
and it is conspicuous precisely because the rest of the paper is so careful: the conformal bound is
the formal backbone of the positive claim, and empirical coverage on the test split is a single
number. Report P̂[S(x) ≤ U(x)] per system against the nominal 1−α, ideally across the α sweep of W1,
so that the marginal guarantee is shown to hold rather than cited.

**W3 (moderate, framing) — The extrapolation story and the conformal guarantee are doing different
work, and the juxtaposition invites over-reading.** 3BPA at 1200 K is described as extrapolative
relative to the 300 K training set, which is true and is a real strength for stressing the
*estimator*. But the calibration and test splits are both drawn from the same 1200 K pool, so they
are exchangeable with each other and the conformal guarantee is an in-pool marginal guarantee; it
certifies nothing about a future shift, which is the deployment scenario the introduction motivates
(flagging extrapolation during MD, deciding when a structure must go back to DFT). Appendix F does
say the guarantee is neither conditional nor out-of-distribution, so the paper is not wrong — but
calling 1200 K-vs-300 K "exactly such a shift" in the same paragraph blurs a distinction the paper
elsewhere keeps sharp. Worth one sentence separating "the model is misspecified on this pool" from
"calibration and test are exchangeable within this pool," and, if feasible, one genuinely
shifted evaluation: calibrate on a lower-temperature or otherwise disjoint pool and measure coverage
degradation on 1200 K. That would be the most scientifically interesting addition after W1.

**W4 (moderate) — The positive sets are small enough that the headline range spans about one
structure.** Realised test prevalence is 0.028–0.050 on test splits of 1284 (3BPA) and 600 (each
rMD17 molecule), so the positive sets hold roughly 17–36 structures. The paper says this — "the
positive sets hold only tens of structures, so the recall intervals are correspondingly wide and we
report them rather than the point estimates alone," which is the right instinct — but the abstract
and conclusion still quote "96–98%" and "0.964–0.982" as if that two-point range were resolved, when
on ~20 positives it is roughly one structure wide, and the ethanol interval [0.940, 1.000] reaches
the ceiling. Either quote the intervals in the abstract or state the positive counts there. Pooling
across systems, or reporting the number of misses rather than a recall fraction, would also be more
honest about what was measured.

**W5 (moderate, scope) — M=8 caps the result, and the interesting regime is larger M.** The paper
notes that M=8 bounds the maximum saving, but the limitation runs deeper than the saving: "how many
of M−1 directions do you need" is a question whose answer is a function of M, and every number here
comes from r=7. At K=4 the method is already using 4 of 7 available directions, which is not the
asymptotic regime sketching is designed for. If the head-space stable rank grows sublinearly in M —
plausible, given it is 2.91–3.08 at r=7 — then replacement might well become viable at M=32 or 64,
and the paper's headline negative result would be specific to small committees rather than a
property of the approach. Training larger committees is out of scope for a workshop paper, but a
synthetic study is not: generate head-force matrices with controlled spectra matched to the measured
stable rank and leading-share, and show how top-5% recall at fixed K/r scales with r. That would
tell the reader whether "no for replacement" is a fact about the method or a fact about M=8, and it
is the single highest-value addition after W1 and W2.

**W6 (moderate, framing) — The finding that the naive baseline beats the sketch is too deeply
buried.** Appendix R establishes that at equal lanes, exact-mean head subsampling beats Haar
sketching significantly on 3BPA (−0.040 [−0.074, −0.008]) and aspirin (−0.070 [−0.126, −0.016]),
with no significant difference on the other two. This is important for a reader's mental model: the
randomization is not where the win comes from, and anyone who reads §5.1's negative result will
immediately wonder whether just computing three head forces exactly would have done better — the
answer is yes, as an estimator. The main text currently gives this three lines and a pointer ("It
depends on whether an exact mean force is required, and the two framings give opposite answers"),
which is honest but so compressed that it reads as a technicality rather than as the substantive
finding it is. Recommend promoting the conclusion (not the full analysis) into §5.1: state that the
sketch does not beat exact-mean head subsampling as an estimator, and that the paper's positive
claim rests on the control variate plus calibration, where head subsampling is included as a gate
baseline in its fair form and loses. This strengthens credibility rather than weakening the paper.

**W7 (moderate, clarity) — Headline numbers do not name their acquisition rule or estimator, and the
two rules produce confusably similar values.** The paper deliberately reports a primary
(max-component) rule and a secondary (global sum) statistic, which is good practice, but then quotes
figures without saying which. The abstract's "the best estimator recovers only 72–80%" names neither
the rule nor which estimator; §6 repeats 0.724–0.801 at K=6 under what appears to be the primary
rule; and Appendix G's estimator comparisons are explicitly on the *global* statistic, where the
control variate reaches 0.730 [0.692, 0.760] at K=4 — numerically almost identical to the primary
rule's 0.724 at K=6, and easy to conflate. Appendix G's numbers also appear to be 3BPA-specific
(0.619 + 0.111 = 0.730 matches §5.1's 3BPA Haar figure) without saying so, while the Haar-over-
Gaussian 0.225 is described as "significant on all four systems," leaving unclear whether 0.225 is
3BPA or pooled. Every reported figure should carry its rule, its estimator, its K and its system.

**W8 (minor, scope) — Single device and single architecture, where the conclusions are
device- and architecture-sensitive.** All timing is on one RTX 5070 Laptop GPU, and the speed story
*is* a crossover story: batched reverse mode is nearly flat in lane count at B=1, scales again at
B=16, and by B=64 is slower than serial at every lane count measured. Where that crossover sits
determines the paper's practical recommendation ("suits batched active-learning sweeps, not
single-structure MD"), and it is a function of memory bandwidth, occupancy and kernel-launch
overhead. One datacenter GPU (A100/H100) would materially strengthen a claim the abstract already
hedges. Separately, all results use one committee architecture (2 layers, 32 channels, ℓmax=1,
rcut=6 Å) — the three head-distribution variants on 3BPA are a real and welcome robustness check,
and should be credited as such, but the head-space spectrum that determines whether sketching can
work may differ at foundation-model scale.

**W9 (minor, presentation).**

- **The title misdescribes the contribution.** "Can we replace multi-head committees with screening
  methods for faster force uncertainty compute?" — nothing here replaces a committee with screening.
  The committee is retained throughout; what is screened is *when to compute its force uncertainty
  exactly*, and what is (unsuccessfully) replaced is the exact centered head basis. Consider
  something closer to "Randomized head-space probes cannot replace exact multi-head force
  uncertainty, but they can screen it."
- **Appendix H ("What ForceSketch replaces") contains no prose at all** — only Figure 3, whose
  caption carries the content. Either add a sentence or fold the figure into Appendix D.
- **Appendix K's heading is duplicated verbatim at two levels**: "K Does reducing directions reduce
  GPU time?" immediately followed by "K.1 Does reducing directions reduce GPU time?".
- **Most appendices repeat their main-text run-in heading verbatim** (D "Why energies are cheap and
  forces are not", E "Low-rank control variate", F "Screening rather than replacement", I "Finite-K
  corrections", J "Splits", L "What a screened evaluation costs"). This makes the appendices read as
  copy-paste extensions of the main text and costs a line each; drop the duplicated run-ins.
- **The venue line reads "Submitted to the Sim2Science"**, which is awkward and does not match the
  venue's name as it appears elsewhere.
- **Checklist item 6's justification misattributes parameters to §4**: it states that Section 4
  gives "the probe budgets K, the control-variate rank r₀, the miscoverage level α" — but r₀ and α
  appear only in §5.3.

**W10 (minor, but follows from the paper's own argument) — Are the reported decisions verified in
the precision they would be deployed in?** §4 draws a careful line: estimator evaluation in float64,
justified by κ≈36 making a float32 "exact" reference less accurate than the 10⁻⁵ verification
tolerance; deployment timing in float32. That argument is good, and it implies its own follow-up —
if float32 cancellation is bad enough to disqualify a float32 *reference*, what does it do to the
float32 *sketch scores* a deployed gate would actually threshold? The accuracy numbers come from
float64 and the timings from float32, so the paper never reports skip/recall for the configuration
one would ship. Recomputing the §5.3 table in float32 and confirming the gate decisions are
unchanged (or reporting how many flip) would close the loop.

### Questions

1. What is the observed conformal coverage per system against the nominal 0.95? (W2)
2. What does the (exact-skipped, high-UQ-recall) frontier look like as α varies, and is r₀=2 the
   best choice at K=4? (W1)
3. If calibration is performed on a pool disjoint in temperature from the test pool, how much does
   coverage degrade? (W3)
4. In a synthetic study with spectra matched to the measured stable rank, does top-5% recall at
   fixed K/r improve with r — i.e. is "no for replacement" a statement about the method or about
   M=8? (W5)
5. How many high-uncertainty structures are actually missed per system, in counts rather than
   fractions? (W4)
6. Do the §5.3 gate decisions change under float32? (W10)

---

## LIMITATIONS

Yes, and with notably little hedging — the limitations paragraph does real work rather than
discharging an obligation. It states that the time saving is regime-dependent and that at B=1 the
gate becomes a *net slowdown*; that the work preserves a proxy rather than a demonstrated outcome,
with no active-learning loop run and no reference-force error measured, so no claim is made about
model quality or oracle-query efficiency; that committee disagreement is only a proxy for error;
that calibration is marginal and under exchangeability; that M=8 bounds the maximum saving; and that
rMD17 spans only 9–24 atoms. Appendix F restates the conformal caveat formally, Appendix I flags
that the control variate's correction constant is an approximation, and Appendix R records a
corrected result that weakens one of the paper's own comparisons. Declining to claim a downstream
active-learning gain that would have been easy to imply is the right call and should be credited.

Four additions would complete it:

1. The single-operating-point limitation of W1 — that the screening headline is measured at one
   (r₀, K, α) and its sensitivity to α is untested — is not currently acknowledged anywhere, and it
   is the main caveat on the paper's central positive claim.
2. The distinction in W3 between "the model is misspecified on this pool" and "calibration and test
   are exchangeable within this pool." The paper has the right formal statement but places it
   beside a sentence that blurs it.
3. That M=8 limits not only the achievable saving but the *generality of the negative result* (W5) —
   these are different limitations and only the first is stated.
4. Single timing device and single committee architecture (W8), given that the speed conclusion is
   a crossover claim.

No societal-impact concerns; the work is methodological, uses public datasets and public
checkpoints, and its failure mode (an over-permissive gate skipping a high-uncertainty structure) is
exactly what the paper measures and calibrates against.

---

## SCORES

| Field | Score |
|---|---|
| Quality | 3 (good) — held here by W1 and W2, not by execution; fixing both would make it 4 |
| Clarity | 3 (good) |
| Significance | 3 (good) |
| Originality | 3 (good) |
| Relevance | 5 (highly relevant) |
| **Rating** | **4 (Borderline accept, leaning positive)** |
| Confidence | 4 |
| Ethical concerns | No ethical concerns |

### Note on calibration against the other two reviews

For consistency, the same standard was applied to all three papers. The one structural criticism
levelled at Submission 274 — that the positive claim is measured at a single untested operating
point of the parameter governing the paper's own central tradeoff — applies to this paper too, with
α and r₀ in place of the sentinel share, which is why both land at 4 rather than 5. On every other
axis this paper is the stronger of the two: real pretrained models and real molecular systems rather
than 1D toys, a live and pinned artifact, a matched-budget baseline suite that includes a free
gate, a method with no specification gap of the kind found in 274's weight clipping, and a reported
self-correction against interest. Submission 193 sits at 5 because its contribution — the curated
dataset — is largely independent of the one framing claim its evidence underdetermines, whereas
here the untested operating point sits directly under the headline result.
