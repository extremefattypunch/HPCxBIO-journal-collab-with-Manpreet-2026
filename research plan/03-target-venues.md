# Target Workshops — Submission Shortlist (workshop held on/before December 2026)

**Compiled:** 2026-05-30 (via live web search).
**Scope (per your instruction):** **workshops only** — no main-track submissions — within the **ICML / NeurIPS / ICLR** families.
**Hard constraint:** the **workshop must be *held* on or before December 2026** (not just the deadline).
**Paper:** *Portable Fused GPU Kernels for Sensitive Protein Homology Search* (see `01-research-paper-plan.md`).

---

## Bottom line

After applying both filters (workshops only **and** held ≤ Dec 2026, with deadlines still open as of 30 May 2026), the field collapses to **one viable family:**

> ### ➜ NeurIPS 2026 workshops (held **11–13 December 2026**; paper deadlines ~**late Aug 2026**).

Everything else is eliminated:

- **ICML 2026 workshops** are *held* in time (10–11 Jul 2026) but their **paper deadlines have already passed** (the flagship bio workshop, GenBio, closed **8 May 2026**; ICML workshops notify authors by ~mid/late May).
- **ICLR 2026 workshops** were held **April 2026** — already in the past.
- **ICLR 2027 workshops** are held in **2027** — after your December 2026 cutoff (deadlines ~Feb 2027 anyway).

So: aim at the **NeurIPS 2026 workshops**, and within them, **MLSB** is your best topical home.

---

## ✅ Viable: NeurIPS 2026 workshops

**Umbrella facts** (from the official Call for Workshops):

| Item | Date |
|---|---|
| Workshop line-up announced to organizers | **11 Jul 2026** (individual CFPs appear shortly after) |
| NeurIPS-suggested paper submission date | **~29 Aug 2026 (AoE)** — each workshop sets its own, typically late Aug–mid Sep |
| Mandatory accept/reject notification | **29 Sep 2026 (AoE)** |
| Workshop days | **11–13 Dec 2026** (Sydney 11–12; Paris & Atlanta 12–13) |

Official: [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops) · [NeurIPS 2026 Dates](https://neurips.cc/Conferences/2026/Dates)

> ⚠️ The **2026 editions of the specific workshops below are not announced yet** (they can't be, until the 11 Jul 2026 line-up). The rows give the recurring series, why it fits, and an **estimated** deadline from the 2025 edition + the NeurIPS-suggested 29 Aug date. **Re-check each site in mid-to-late July 2026** for the confirmed CFP.

### Recommended workshops (in priority order for this paper)

| # | Workshop (series) | Why it fits your paper | Est. 2026 deadline | Fit | Official site |
|---|---|---|---|---|---|
| 1 | **MLSB — Machine Learning in Structural Biology** | Your kernels accelerate the MSA-generation substrate behind AlphaFold/ColabFold and the Foldseek structural search you benchmark — directly in MLSB's scope (structure prediction, protein search, biomolecular methods). 6th edition expected at NeurIPS 2026. | ~late Aug / early Sep 2026 | ✅ **Strongest** | [mlsb.io](https://www.mlsb.io/) |
| 2 | **AI for Science** | Broad "compute/AI enabling scientific discovery"; your speed/cost/energy gains that unlock larger sensitive searches fit its tools-for-science framing. Runs at NeurIPS most years. | ~late Aug / early Sep 2026 | ✅ Good | [ai4sciencecommunity.github.io](https://ai4sciencecommunity.github.io/) |
| 3 | **ML for Systems** | Systems/GPU-kernel angle (fusion, roofline, cross-vendor portability). **Caveat:** its theme is *using ML to improve computer systems* (learned heuristics), whereas you hand-write Triton kernels for a bio workload — a thematic stretch despite the systems flavour. 2025 deadline was 22 Aug 2025. | ~late Aug 2026 | ⚠️ Borderline | [mlforsystems.org](http://mlforsystems.org/) · [CFP](http://mlforsystems.org/call_for_papers.html) |

> When the 11 Jul 2026 line-up drops, also scan it for any new **"efficient ML / systems"** or **"AI for (accelerated) science"** workshop — these recur under varying names and any of them would suit the systems half of the paper.

---

## ❌ Ruled out (and exactly why)

| Venue | Held | Deadline | Why excluded |
|---|---|---|---|
| **ICML 2026 workshops** (GenBio, AI for Science, life-sciences FM, efficiency workshops, …) | 10–11 Jul 2026 ✅ in time | **Passed** — GenBio closed **8 May 2026**; ICML workshops notify by ~15–25 May | **Deadlines already passed** as of 30 May 2026 |
| **ICLR 2026 workshops** | Apr 2026 (Rio) | Feb 2026 | Workshop already **happened** |
| **ICLR 2027 workshops** | ~2027 (TBA) | ~Feb 2027 (est.) | **Held after** Dec 2026 cutoff; deadline also after cutoff |
| ICML 2027 / NeurIPS 2027 workshops | 2027 | 2027 | Held after Dec 2026 |

**Evidence for the ICML exclusion:** GenBio ICML 2026 (the flagship Generative/Agentic-AI-for-Biology workshop) lists submission **8 May 2026**, notification **25 May 2026**, workshop **10 Jul 2026** — i.e., the most relevant ICML bio workshop closed three weeks before today. The full ICML 2026 line-up (44 workshops, incl. "AI for Science: AI Scientists", "future of AI for biology", "Multi-modal Foundation Models … for Life Sciences", several efficiency/adaptive-inference workshops) is announced [here](https://blog.icml.cc/2026/04/06/announcing-the-icml-2026-workshops-and-affinity-workshops/), but all operate on the same ~early-May deadline cycle.

---

## Why a workshop is the right move here (and the journal still works)

- **Non-archival.** NeurIPS workshops have no formal proceedings, so a workshop paper **does not block the full *Bioinformatics* journal submission** in `01-research-paper-plan.md`. You get early visibility *and* the journal.
- **Right length.** Workshops want ~4–9 pages (NeurIPS style files) — a natural condensation of the plan: lead with **H1 (NVIDIA bit-exact parity)** + **H2 (first sensitive GPU filter on AMD MI300X)**; keep the full benchmark suite for the journal.
- **Binding milestone.** To make the ~**29 Aug 2026** deadline you need Phases 0–1 (NVIDIA filter at bit-exact parity) **and** Phase 4a (the AMD MI300X port — your headline) **measured by ~mid-August 2026**. Everything else (fusion ablations, SWG, end-to-end ColabFold/Foldseek) can follow for the journal version.

---

## Action calendar

| When | Action |
|---|---|
| Now → mid-Aug 2026 | Run Phases 0–1 + 4a on the cluster → H1 (parity) + H2 (AMD port). |
| **11 Jul 2026** | NeurIPS 2026 workshop line-up announced. Confirm MLSB / AI4Science / ML-for-Systems 2026 CFPs + exact deadlines + page limits. |
| **~29 Aug 2026** | Submit 4–9p workshop paper to **MLSB @ NeurIPS 2026** (primary); optionally also AI for Science. |
| **29 Sep 2026** | Workshop accept/reject notifications (NeurIPS mandatory date). |
| **11–13 Dec 2026** | Present at NeurIPS 2026 workshop (Sydney / Paris / Atlanta). |
| Parallel | Submit the full paper to *Bioinformatics* (unaffected by non-archival workshop acceptance). |

---

## Sources (official / primary)

- NeurIPS 2026 Call for Workshops — https://neurips.cc/Conferences/2026/CallForWorkshops
- NeurIPS 2026 Dates & Deadlines — https://neurips.cc/Conferences/2026/Dates
- ICML 2026 Workshops announcement (44 workshops) — https://blog.icml.cc/2026/04/06/announcing-the-icml-2026-workshops-and-affinity-workshops/
- ICML 2026 workshop list (virtual) — https://icml.cc/virtual/2026/events/workshop
- GenBio @ ICML 2026 (evidence ICML bio workshops are closed: deadline 8 May 2026) — https://genbio-workshop.github.io/2026/
- MLSB (Machine Learning in Structural Biology) — https://www.mlsb.io/
- AI for Science workshop series — https://ai4sciencecommunity.github.io/
- ML for Systems workshop — http://mlforsystems.org/ · CFP — http://mlforsystems.org/call_for_papers.html
- ICLR Future Meetings (confirms no official ICLR 2027 listing yet) — https://iclr.cc/Conferences/FutureMeetings

*Outside the three families you named, the most topically ideal workshop-style options held within 2026 would be domain venues like **MLCB (Machine Learning in Computational Biology, ~Nov–Dec 2026)** or the **MLSys** workshop track — say the word and I'll search those too.*
