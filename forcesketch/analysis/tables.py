"""Tables 1-3 (spec SS54) and LaTeX macros, generated from results/raw only.

Spec SS68: "Do not manually transcribe results into tables." So the manuscript
\\input{macros.tex} and contains no literal number; re-running this script after a
re-measurement updates every figure in the paper at once. `check_cherry_picking`
enforces spec SS70's rule that no configuration may be silently dropped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path("results/raw")
PROC = Path("results/processed")

# Measured lane-cost model, refit by `fit_cost_model` from lane_scaling_*.jsonl.
DEFAULT_COST = (9.50, 11.93)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fit_cost_model(system: str = "disjoint", batch_size: int = 1) -> tuple[float, float]:
    """Least-squares fit of T(L) = a + b*L to the measured lane scan."""
    path = RAW / f"lane_scaling_{system}.jsonl"
    if not path.exists():
        return DEFAULT_COST
    recs = [r for r in load_jsonl(path) if r["batch_size"] == batch_size]
    if len(recs) < 2:
        return DEFAULT_COST
    L = np.array([r["lanes"] for r in recs], dtype=float)
    T = np.array([r["median_ms"] for r in recs], dtype=float)
    b, a = np.polyfit(L, T, 1)
    return float(a), float(b)


def mean_by(records: list[dict], keys: tuple[str, ...], fields: tuple[str, ...]) -> dict:
    """Group by `keys`, average `fields`, and keep the seed count for provenance."""
    acc: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        acc[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in acc.items():
        out[k] = {f: float(np.mean([r[f] for r in rs])) for f in fields if f in rs[0]}
        out[k]["n_seeds"] = len(rs)
    return out


def check_cherry_picking(fidelity: list[dict], expected_seeds: int) -> list[str]:
    """Spec SS70: the analysis must not discard failed configurations."""
    problems = []
    grouped = mean_by(fidelity, ("method", "K", "with_mean_lane"), ("top5_recall",))
    for key, v in sorted(grouped.items()):
        if v["n_seeds"] != expected_seeds:
            problems.append(f"{key}: {v['n_seeds']} seeds, expected {expected_seeds}")
    methods = {k[0] for k in grouped}
    for m in methods:
        ks = sorted({k[1] for k in grouped if k[0] == m})
        if m in ("gaussian", "rademacher", "pairwise") and ks != [1, 2, 3, 4, 7]:
            problems.append(f"{m}: K grid is {ks}, expected [1, 2, 3, 4, 7]")
    return problems


def table1_primary(system_tag: str, cost=None) -> str:
    """Spec SS54 Table 1. Rows are NEVER filtered by outcome."""
    a, b = cost or fit_cost_model()
    fid = load_jsonl(RAW / f"03_sketch_fidelity_{system_tag}.jsonl")
    g = mean_by(fid, ("method", "K", "with_mean_lane", "total_lanes"),
                ("spearman", "top5_recall", "top5_jaccard", "p90_rel_err_S",
                 "same_max_atom_frac"))
    t_exact = a + b * 8
    lines = ["| method | K | lanes | Spearman | top-5% recall | top-5% Jaccard "
             "| incr. UQ ms | incr. speedup | total ms | total speedup |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (m, K, wm, L), v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        incr = (a + b * L) - (a + b * 1)
        incr_ex = t_exact - (a + b * 1)
        name = m.replace("_", " ") + (" +mean" if wm else "")
        lines.append(
            f"| {name} | {K} | {L} | {v['spearman']:.3f} | {v['top5_recall']:.3f} "
            f"| {v['top5_jaccard']:.3f} | {incr:.1f} | {incr_ex/incr:.2f}x "
            f"| {a+b*L:.1f} | {t_exact/(a+b*L):.2f}x |")
    return "\n".join(lines)


def table2_scaling() -> str:
    """Spec SS54 Table 2: system, atoms, batch size, exact vs ForceSketch latency.

    Answers what the saving actually tracks. Columns are measured medians, not a
    fitted model, so the fit in `fit_cost_model` can be checked against them.
    """
    import glob
    lines = ["| system | atoms/struct | B | total atoms | exact UQ ms (L=8) "
             "| FS K=3 ms (L=4) | incr. speedup | K=2 speedup |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for f in sorted(glob.glob(str(RAW / "lane_scaling_*.jsonl"))):
        recs = load_jsonl(Path(f))
        by = {(r["batch_size"], r["lanes"]): r for r in recs}
        a_per = recs[0]["num_atoms"] // recs[0]["batch_size"]
        for B in sorted({r["batch_size"] for r in recs}):
            if (B, 8) not in by:
                continue
            t1, t3, t4, t8 = (by[(B, l)]["median_ms"] for l in (1, 3, 4, 8))
            lines.append(
                f"| {recs[0]['system']} | {a_per} | {B} | {by[(B,8)]['num_atoms']} "
                f"| {t8:.2f} | {t4:.2f} | {(t8-t1)/(t4-t1):.2f}x | {(t8-t1)/(t3-t1):.2f}x |")
    return "\n".join(lines)


def table3_screening(system_tags: list[str]) -> str:
    """Spec SS54 Table 3, across systems."""
    lines = ["| system | method | K | alpha | exact skipped | high-UQ recall "
             "| FNR | screening speedup |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for tag in system_tags:
        p = RAW / f"06_screening_{tag}.jsonl"
        if not p.exists():
            continue
        g = mean_by(load_jsonl(p), ("method", "K", "r0", "alpha"),
                    ("frac_exact_skipped", "high_uq_recall", "false_negative_rate",
                     "screening_speedup"))
        for (m, K, r0, alpha), v in sorted(g.items()):
            name = f"{m.replace('_',' ')}" + (f" r0={r0}" if r0 else "")
            lines.append(
                f"| {tag} | {name} | {K} | {alpha:.2f} | {v['frac_exact_skipped']:.3f} "
                f"| {v['high_uq_recall']:.3f} | {v['false_negative_rate']:.3f} "
                f"| {v['screening_speedup']:.2f}x |")
    return "\n".join(lines)


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def emit_macros(out: Path) -> dict:
    """One \\newcommand per frozen number. The manuscript cites these, never literals."""
    a, b = fit_cost_model()
    macros, vals = [], {}

    def put(name, value, fmt="{:.3f}"):
        s = fmt.format(value) if not isinstance(value, str) else value
        macros.append(_macro(name, s))
        vals[name] = value

    put("fsCostIntercept", a, "{:.2f}")
    put("fsCostSlope", b, "{:.2f}")

    # LaTeX control sequences must be letters only -- no digits in macro names.
    WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 7: "Seven", 64: "SixtyFour"}

    fid = load_jsonl(RAW / "03_sketch_fidelity_disjoint_test_1200K.jsonl")
    g = mean_by(fid, ("method", "K"), ("top5_recall", "spearman"))
    for K in (2, 3, 4):
        put(f"fsHaarRecallK{WORD[K]}", g[("haar", K)]["top5_recall"])
        put(f"fsHaarSpearmanK{WORD[K]}", g[("haar", K)]["spearman"])
    put("fsHeadSubRecallKThree", g[("head_subsample", 3)]["top5_recall"])
    put("fsGaussRecallKThree", g[("gaussian", 3)]["top5_recall"])

    # lane-scaling ceilings
    ls = RAW / "lane_scaling_disjoint.jsonl"
    if ls.exists():
        recs = {(r["batch_size"], r["lanes"]): r["median_ms"] for r in load_jsonl(ls)}
        # Spec SS45 defines these differently and forbids conflating them:
        #   incremental UQ speedup = [T(L=8) - T(L=1)] / [T(L=1+K) - T(L=1)]
        #   total workflow speedup =  T(L=8) / T(L=1+K)
        # The raw ratio INCLUDES the mean-force lane, so it is the total, not the
        # incremental, figure.
        for B in (1, 64):
            if (B, 8) in recs:
                t1, t3, t4, t8 = (recs[(B, l)] for l in (1, 3, 4, 8))
                put(f"fsTotalKThreeB{WORD[B]}", t8 / t4, "{:.2f}")
                put(f"fsTotalKTwoB{WORD[B]}", t8 / t3, "{:.2f}")
                put(f"fsIncrKThreeB{WORD[B]}", (t8 - t1) / (t4 - t1), "{:.2f}")
                put(f"fsIncrKTwoB{WORD[B]}", (t8 - t1) / (t3 - t1), "{:.2f}")

    # screening headline: control variate r0=2, K=4, alpha=0.05, per system
    for tag, short in [("disjoint_test_1200K", "ThreeBPA"), ("rmd17-disjoint_ethanol", "Ethanol"),
                       ("rmd17-disjoint_aspirin", "Aspirin"),
                       ("rmd17-disjoint_azobenzene", "Azobenzene")]:
        p = RAW / f"06_screening_{tag}.jsonl"
        if not p.exists():
            continue
        gg = mean_by(load_jsonl(p), ("method", "K", "r0", "alpha"),
                     ("frac_exact_skipped", "high_uq_recall", "false_negative_rate",
                      "screening_speedup"))
        key = ("control_variate", 4, 2, 0.05)
        if key in gg:
            put(f"fsGateSkip{short}", gg[key]["frac_exact_skipped"])
            put(f"fsGateRecall{short}", gg[key]["high_uq_recall"])
            put(f"fsGateSpeedup{short}", gg[key]["screening_speedup"], "{:.2f}")

    # --- bootstrap CIs (spec SS47). The manuscript must never quote a bare mean.
    bs = RAW / "04_bootstrap_disjoint_test_1200K.jsonl"
    if bs.exists():
        recs = load_jsonl(bs)
        nb = {int(r["top5_recall_n_boot"]) for r in recs
              if r["experiment_id"] == "04_bootstrap_ci"}
        put("fsNBoot", f"{sorted(nb)[0]:,}" if len(nb) == 1 else "MIXED")
        by = {(r["method"], r["K"], r["r0"], r.get("with_mean_lane", False)): r
              for r in recs if r["experiment_id"] == "04_bootstrap_ci"}
        for K in (2, 3, 4):
            r = by.get(("haar", K, 0, False))
            if r:
                put(f"fsHaarRecallK{WORD[K]}CI",
                    f"[{r['top5_recall_ci_lo']:.3f}, {r['top5_recall_ci_hi']:.3f}]")
                put(f"fsHaarSpearmanK{WORD[K]}CI",
                    f"[{r['spearman_ci_lo']:.3f}, {r['spearman_ci_hi']:.3f}]")
        r = by.get(("control_variate", 4, 2, False))
        if r:
            put("fsCVRecallKFour", r["top5_recall"])
            put("fsCVRecallKFourCI",
                f"[{r['top5_recall_ci_lo']:.3f}, {r['top5_recall_ci_hi']:.3f}]")
            put("fsCVSpearmanKFour", r["spearman"])
        # paired differences -- the statistic that actually decides SS49
        for rec in recs:
            if rec["experiment_id"] != "04_bootstrap_diff":
                continue
            if "head_subsample K=3 +mean" in rec["comparison"]:
                nm = "fsDeltaVsHeadSub"
            elif "gaussian" in rec["comparison"]:
                nm = "fsDeltaOrtho"
            elif "control_variate" in rec["comparison"]:
                nm = "fsDeltaCV"
            else:
                continue
            put(nm, rec["delta_top5_recall"])
            put(nm + "CI", f"[{rec['ci_lo']:.3f}, {rec['ci_hi']:.3f}]")

    sp = PROC / "spectrum_disjoint_test_1200K.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        put("fsStableRank", s["stable_rank"], "{:.2f}")
        put("fsTopEigFraction", s["top1_fraction"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("% Generated by analysis/tables.py -- do not edit by hand.\n"
                   + "\n".join(macros) + "\n")
    return vals
