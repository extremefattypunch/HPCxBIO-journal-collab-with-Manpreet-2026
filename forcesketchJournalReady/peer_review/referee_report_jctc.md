# Referee Report — *Journal of Chemical Theory and Computation*

**Manuscript:** "When is committee force uncertainty worth computing in MD? Decision quality, cheap gates, and the baselines that must be beaten"

**Reviewed artifact:** `paper/main_jctc.pdf`, 18 pages, 5 figures, 2 tables, 11 references. No Supporting Information document accompanied the submission.

**Recommendation: Major Revision.**

---

## 1. Summary of the submission

The manuscript asks whether query-by-committee force uncertainty — the dominant structure-selection heuristic for machine-learned interatomic potentials (MLIPs) — actually earns the backward passes it costs. The author separates two questions that the literature normally conflates:

- **Factor A (oracle quality):** does exact committee force disagreement rank structures by their *true error against the reference method*?
- **Factor B (preservation):** given whatever oracle you trust, how cheaply can a surrogate reproduce its decisions?

The literature, the author argues, validates cheap estimators against the committee as if the committee were ground truth, and therefore never asks Factor A at all. Three results follow.

1. **Factor A is moderate and the choice of statistic matters.** On six small-molecule systems (3BPA under three committee constructions, three rMD17 molecules), exact global disagreement reaches AUROC 0.737–0.848 for detecting the top-5% highest-error structures. The global (trace) statistic beats the conventional `max-component` rule in 24 of 24 cells.
2. **Randomisation does not pay.** A deterministic leading-subspace estimator beats a randomised control variate at matched lane budget on 5 of 6 systems. A finite-sample Beta law (`v̂_d/v_d ~ (r/K)·Beta(K/2,(r−K)/2)`) reproduces measured top-5% recall to MAE 0.015 with no free parameters, and a Chernoff argument shows the required sketch width is 10³–10⁴ against an available budget of `K ≤ r = 7`.
3. **On a realistic materials pool the committee loses to a free signal.** On 136,923 MPtraj crystals under a frozen MACE-MP trunk with eight heads, the max mean-force norm ‖f̄‖ — already computed in the forward pass — reaches AUROC 0.943 against the committee's 0.798, and adding the committee to a nested model moves held-out AUROC by 0.0002 (not significant). The global statistic is shown to be *extensive* (it sums `v_d` over 3N coordinates), so on a variable-size pool it acquires large structures rather than difficult ones; the size normalisation `S_α = (Σ_d v_d)/(3N)^α` restores parity with `max-component`, not superiority.

## 2. Assessment of significance

The reframing in Section 1 is the paper's real contribution and it is a good one. It is genuinely common in this literature to report that a cheap uncertainty proxy "agrees with the ensemble to within X%" and treat that as validation, without ever establishing that the ensemble ranking is worth agreeing with. Stating Factor A as a separately answerable, offline, sketch-free question is a clean methodological correction, and it is the kind of correction that changes how a subfield reports results.

The extensivity diagnosis in Section 8.2 is the single most useful technical observation in the paper. `Σ_d v_d` grows with system size whether or not the model is uncertain; on a fixed-`N` benchmark this is one harmless constant, and on a composition- and size-spanning pool it silently converts an uncertainty ranking into a size ranking. The author's demonstration that the acquired set has 2.50× the median atom count of the `max-component` selection, and that the Section 4 ordering reverses on all four error scores, is convincing, and the post-hoc stratification by atom-count quartile is the right control.

The reproducibility engineering exceeds what JCTC normally receives: content-hashed splits with guard bands sized from a measured integrated autocorrelation time, hash-bound pre-registration of protocols, offline re-derivation of every headline number in ~4 minutes on CPU, and a declared policy that no number in the manuscript is transcribed by hand. Pre-registered predictions are reported *including the failures* (2 of 5 confirmed in Section 8; 3 of 5 in Section 7.2). Section "Use of Large Language Models" is unusually candid and, more importantly, explains why the verification is mechanical rather than a matter of trust. I want to record explicitly that I regard this as exemplary and that none of my objections below are about the integrity of the reported numbers.

I spot-checked internal consistency and found none of the arithmetic wanting: the abstract's 82–94% matches §5's 0.822–0.943; the 0.067–0.182 AUROC margin matches §4.2; §8.1's 0.798 matches Table 1; and §7.1's "247 subcommittees" is exactly 2⁸ − 1 − 8, i.e. every subset of at least two heads from eight.

That said, the headline claim is currently over-reached relative to what the experiments establish, for the reasons in Section 3.

---

## 3. Major points

### M1. The MPtraj committee is the weakest possible committee, and the paper's own data say so

This is my principal objection and it must be addressed before publication.

The headline result — ‖f̄‖ at 0.943 versus the committee at 0.798 — is obtained on a **frozen foundation-model trunk carrying eight fine-tuned heads** (Section 8). Heads on a shared frozen trunk share every learned representation; their disagreement is close to degenerate by construction. The manuscript itself supplies the evidence: Section 7.2 reports that independently trained committees are **2.28× overconfident against 25.77× for the matched shared-trunk committee** — an order-of-magnitude calibration difference — and Section 7.3 reports that under distribution shift a matched independently-trained committee is 2.7× more accurate and its Factor A collapses for a different reason.

So the paper compares a free signal against a committee construction it has independently shown to be badly miscalibrated, on the one pool where it draws its strongest conclusion, and then generalises in the abstract: *"the ensemble adds nothing significant on top of it."*

The independently-trained control exists in the paper — it is just never run on MPtraj. **Please run it there.** Even a reduced version (eight independently initialised heads, or independently fine-tuned trunks, on a subsample of the pool sufficient for the AUROC confidence intervals you already compute) would settle whether the finding is about *committees* or about *frozen-trunk multi-head committees*. These are very different papers. If the free signal still wins against a properly constructed committee, the result becomes considerably stronger than it is now. If it does not, the claim needs to be restated as a limitation of the multi-head amortisation of ref. 2 — which is still a publishable and useful result, and arguably a more precise one.

Until this is done, the title, abstract, and Recommendation 1 should be scoped explicitly to shared-trunk multi-head committees.

### M2. No closed-loop active-learning experiment supports the practical recommendation

The entire paper is about which structures get labelled, yet no retraining is ever performed. This is acknowledged in Limitations ("No retraining was performed, so no claim is made about downstream model quality — only about which structures are selected"), and the honesty is appreciated, but the acknowledgement does not fully discharge the problem, because Section 8.4 also reports that the selection-overlap route to bounding the downstream difference **is unavailable**: Jaccard overlap is 0.794, short of the pre-registered threshold.

The paper therefore has neither a downstream measurement nor a bound that would substitute for one. What remains is AUROC-for-detecting-top-5%-error as a proxy for acquisition value — itself an unvalidated proxy, and the very move ("validate against a proxy and call it done") the paper's own framing indicts. I do not think the manuscript can carry a recommendation as strong as "ensemble uncertainty is unnecessary" on that basis.

The remedy need not be large. One closed-loop experiment on one system — a fixed label budget, `n` acquisition rounds, ‖f̄‖-selected versus committee-selected versus random, force RMSE on a held-out set — would convert the central claim from a ranking correlation into a demonstration. With the infrastructure described in Section 10 this appears very achievable.

### M3. AUROC is the wrong figure of merit for an acquisition decision, and the paper argues this itself

Acquisition takes the top `k`. AUROC at a 5% positive rate integrates over the whole ranking and is dominated by the bulk, not by the head where the decision actually happens. Section 8.4 makes precisely this argument, correctly and forcefully, about Jaccard overlap: *"Overlap statistics must always be reported with their acquisition rate, since any two rankings agree when a large fraction of the pool is taken."* The same reasoning applies to the paper's own headline metric and is not applied there.

Please report **precision@k / enrichment at the realistic acquisition rates** you actually use (the budgets of 1000 and 8000 in Figure 4a would be natural), for every signal in Table 1 and Figure 1. It is entirely possible that the committee and the free signal separate differently at the head of the ranking than in the AUROC integral, in either direction, and this is the number a practitioner needs.

### M4. The error targets are force-magnitude-scaled, which partly builds in the free signal's advantage

All four error scores in Table 1 (`e_max`, `e_rmse`, `e_max c`, `e_q95`) are absolute force-error magnitudes. Absolute force error correlates mechanically with force magnitude: structures with large forces have large errors under almost any model. A predictor that *is* the max force magnitude is therefore advantaged by construction on these targets.

The author clearly anticipated this — the "by force" column, stratifying within force-magnitude quintiles, is the right control, and ‖f̄‖ retains 0.710 against the committee's 0.626–0.656 there. That is a real and non-trivial defence, and it should be promoted from a post-hoc column into the main narrative rather than left in a table caption.

But it is not the whole answer. Please add **at least one scale-free error target** — relative force error `‖Δf‖/‖f‖`, or cosine error on force directions, or an energy-per-atom error — and report the full signal comparison on it. If ‖f̄‖ still wins on a scale-free target, M4 is closed decisively. If it does not, that is an important qualification to the headline.

Relatedly, Table 1's `by size` cell for `exact max-atom` is a dash with no explanation in the caption. Please explain the omission.

### M5. No non-committee UQ baseline other than force magnitude and energy spread

The paper's closing recommendation is that future methods be measured against the free signal. That is a good norm. But the manuscript's own comparison set is narrow: exact committee statistics, energy-head standard deviation, and ‖f̄‖. The obvious and widely used cheap competitor — **distance to the training set in the model's latent/descriptor space** — is absent, as are GP/GAP-style predictive variances and any distance-based or density-based novelty score. Several of these are approximately as cheap as ‖f̄‖ on a frozen trunk, since the descriptors are already computed.

A paper whose thesis is "here is the baseline that must be beaten" is materially weakened by not having surveyed the cheap-baseline space it is legislating over. Please add at least a latent-space-distance baseline to Table 1 and Figure 1.

### M6. Multiplicity is not addressed

The manuscript reports "19 of them significantly at 95%" (§4.1, 24 cells), "significant in 9 of 10 comparisons" (§4.2), "significant gains on 8 of 10 systems" (§8.3), and "2 significant wins, one tie, and 1 significant loss" (§8.2). The paired moving-block bootstrap is the right tool for autocorrelated MD data and I have no objection to it. But with 24 simultaneous comparisons at nominal 95% one expects on the order of one false positive, and no correction or family-wise statement is made anywhere. Please state the multiplicity handling explicitly, or apply a correction (Holm or Benjamini–Hochberg) and report the corrected counts. The 24/24 direction-of-effect result is robust to this; the "19 significant" count is not.

### M7. Scope: Section 9 and the GPU benchmarking belong in Supporting Information

Section 9, Figure 5, and Table 2 are a careful three-architecture study of when lane reduction becomes compute-bound rather than launch-bound. The work is sound and the observation that the crossover batch size *increases* with GPU speed is a nice one. But the manuscript states plainly that "acceleration is deliberately not the headline," and for a JCTC readership this material dilutes the chemistry argument considerably — roughly a page and a half of text plus a figure and a table are hardware characterisation. I recommend moving Section 9, Figure 5, and Table 2 to the SI and retaining two or three sentences in the main text with a pointer.

### M8. Supporting Information is required, and the artifact should have a DOI

No SI accompanied this submission, and the manuscript is not self-contained without one. Missing, at minimum:

- Architecture, training and fine-tuning protocol for the eight heads; the same for the eight independently trained models of §7.2.
- Definition of the three 3BPA committee constructions — **"disjoint," "overlapping," and "same" are used in Figures 1 and 2 and in Section 4 and are never defined anywhere in the manuscript.** A reader cannot interpret the primary figure.
- Precise definitions of `e_max`, `e_rmse`, `e_max c`, `e_q95`.
- The conformal level α actually shipped (it is referred to as "the α we ship" but never given), and the calibration/test block sizes for all 28 splits.
- MD and PIMD settings for the water system of §7.3 (thermostat, timestep, beads, sampling interval) and the measured autocorrelation times used to size the guard bands.
- The definition of the 20 systems in §7.1 — the main text has introduced six.

On availability: pointing at `https://github.com/extremefattypunch/forcesketch` is good practice but a live repository is mutable and is not archival. The paper already archives *someone else's* inputs at a Zenodo DOI; please mint a versioned DOI for your own artifact and cite that alongside the repository, per ACS data-availability guidance.

---

## 4. Minor points

1. **Cross-reference error.** Section 1 states "Section 10 describes the reproduction path." Section 10 is *Recommendations and limitations*; the reproduction path is the unnumbered Reproducibility Statement that follows it. Please fix the reference.
2. **"Lane" is load-bearing and under-introduced.** It is defined in §3.3 ("A projection (lane) costs one reverse pass") but used as a cost unit in the framing well before that, and it is not standard vocabulary for this readership. Define it at first use and consider a plainer term.
3. **Undefined or under-defined at first use:** "head space," "stable rank," "guard band," "Helmert basis," "content-hashed," "pre-registered." A computational chemistry audience will not have all of these. One clarifying clause each would help substantially.
4. **Prose register.** The writing is compressed and aphoristic — "the recommendation was never wrong where it was measured, it was measured only where it could not fail"; "the committee loses to a signal that costs nothing." It is good writing, but several passages require re-reading to extract the claim, and JCTC prose norms are plainer. I would ask for a pass that unpacks the densest sentences, particularly in §§1, 5.1 and 8.2.
5. **Section 7.3 is too thin to carry its conclusion.** The condensed-phase and distribution-shift result — arguably the most chemically interesting setting in the paper — is one paragraph with a single number (2.7×), no figure, and no table. Either develop it with the supporting numbers and a figure, or move it to SI and soften the claim.
6. **"Correcting earlier statements of our own."** Limitations refers to results that "correct earlier statements of our own" with the record "retained in the repository." In a journal submission this is confusing unless the earlier statements are citable. Either cite the prior artifact or drop the sentence; a repository is not a citable prior claim.
7. **Single author, first person plural.** "We" and "our" throughout with one author is acceptable but inconsistent with "Ian Poon" as sole author and "The author directed the work" in the LLM statement. Choose one convention.
8. **Reference list is thin (11 items)** for a paper whose contribution is a critique of prevailing practice in MLIP active learning. The UQ-for-MLIP and active-learning literature the paper is arguing with is largely uncited — ensemble/deep-ensemble UQ for interatomic potentials, calibration studies for MLIP uncertainty, latent-distance and GP-variance selection, and on-the-fly learning. This matters beyond bibliographic courtesy: it is how a reader judges whether the "baseline that must be beaten" claim is novel.
9. **Reference formatting.** Ref. 9 uses "others" where ACS style requires the full author list or "et al."; ref. 10 (Künsch, *Ann. Stat.* **1989**, *17*) is missing page numbers; ref. 2 appends an arXiv identifier to a published *J. Chem. Phys.* citation. ACS copyediting will flag these.
10. **Figure 1 legibility.** With six signals on six systems, markers overlap (the `exact max-atom` and `energy std` markers nearly coincide on 3BPA disjoint). Consider faceting, jitter, or splitting into two panels. The distinct-marker choice for grayscale is good and should be kept.
11. **Figure 4b needs the design-split curve.** α* = 0.75 is design-selected and the held-out curve is shown; without the design curve the reader cannot judge how sharply α was fitted. Please overlay both.
12. **Title.** Three clauses and an interrogative lead is long for an ACS title. Consider a declarative form naming the finding.
13. **Abstract.** "Ensemble uncertainty is unnecessary where that free signal suffices" is close to tautological as written. The substantive claim — that the free signal *does* suffice on a realistic materials pool, and that this must henceforth be the control — deserves the sentence instead.

---

## 5. Questions I would like answered in the response

1. What is the AUROC of the free signal versus an **independently trained** committee on the MPtraj pool (M1)?
2. What are precision@1000 and precision@8000 for every signal in Table 1 (M3)?
3. Does ‖f̄‖ still outrank the committee on a **relative** or **scale-free** force-error target (M4)?
4. How does latent-space distance to the fine-tuning set compare on the same pool (M5)?
5. Why is the `by size` entry for `exact max-atom` absent from Table 1?
6. What are the three 3BPA committee constructions, and which of them (if any) corresponds to the standard practice the paper is critiquing?
7. What conformal α is shipped, and how were the 28 splits chosen?
8. In §7.2 the acceleration claim is stated as "a derivation rather than a measurement." Please make the derivation explicit — the argument that `Fw` requires all `M` columns of `F` without a shared trunk is correct but should be written out rather than asserted.

---

## 6. Recommendation and rationale

**Major Revision.**

I am recommending against rejection because the core methodological argument is correct and worth publishing, the extensivity result is a genuine and immediately actionable finding, the negative results are honestly reported with their failed pre-registrations intact, and the reproducibility standard is well above what this journal typically sees. I am recommending against acceptance because the paper's most quotable claim — that ensemble uncertainty adds nothing over a free signal — currently rests on a single pool, a single trunk, a committee construction the paper's own data show to be an order of magnitude worse calibrated than the alternative, an error target that partly favours the winning signal by construction, and a ranking metric that is not the one the decision uses.

The distinguishing feature of this submission is that every one of those gaps is closable with the infrastructure the author already describes, and several of the required controls already exist elsewhere in the manuscript and simply need to be run in the setting that carries the headline. If M1–M4 are addressed and the claim is scoped to what they support, I would expect to recommend acceptance.

---

### Confidential comments to the editor

The reproducibility and pre-registration apparatus here is the most thorough I have reviewed for this journal, and the LLM-use disclosure is a model of how such a statement should be written — it states the exposure plainly and then explains the mechanical controls that make the exposure verifiable rather than asking for trust. I would encourage the editors to note it regardless of the outcome.

My concerns are scientific scope, not integrity. The author has, if anything, been more forthcoming about the weaknesses (the unavailable overlap bound, the failed pre-registered predictions, the "this model family failing badly there" caveat in §7.3) than the norm. The revision I am asking for is additional controls and a narrower claim, not a reinvestigation.

One process note: the manuscript is not self-contained without Supporting Information, and at least one primary figure (Figure 1) cannot be interpreted as submitted because its system labels are undefined. I would treat SI as a condition of any further round rather than a copyediting matter.
