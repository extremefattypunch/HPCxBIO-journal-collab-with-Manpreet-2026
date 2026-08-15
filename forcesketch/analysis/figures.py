"""Figures 2-5 (spec SS29, SS30, SS53). Generated from results/raw only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Measured lane-cost model: T(L) = 9.50 + 11.93 L ms (3BPA, RTX 5070 Laptop, fp32).
INTERCEPT, SLOPE = 9.50, 11.93
STYLE = {
    "haar": ("o", "#1f77b4"), "gaussian": ("s", "#d62728"),
    "rademacher": ("^", "#2ca02c"), "pairwise": ("D", "#9467bd"),
    "head_subsample": ("v", "#ff7f0e"), "control_variate": ("*", "#17becf"),
}


def lane_ms(lanes: int) -> float:
    return INTERCEPT + SLOPE * lanes


def incremental_uq_ms(lanes_total: int) -> float:
    """T_UQ = T(L=1+K) - T(L=1): the mean-force lane is not uncertainty cost (spec SS45)."""
    return lane_ms(lanes_total) - lane_ms(1)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def pareto(fidelity_jsonl: Path, out: Path, *, title: str) -> None:
    """Figure 2 (spec SS29): incremental force-UQ latency vs top-5% recall."""
    recs = load_jsonl(fidelity_jsonl)
    agg: dict[tuple, list] = defaultdict(list)
    for r in recs:
        if r["method"] == "head_subsample" and not r["with_mean_lane"]:
            continue  # the +mean variant is the like-for-like comparison
        agg[(r["method"], r["K"], r["total_lanes"])].append(r["top5_recall"])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for method in STYLE:
        pts = sorted((incremental_uq_ms(L), float(np.mean(v)), K)
                     for (m, K, L), v in agg.items() if m == method)
        if not pts:
            continue
        marker, colour = STYLE[method]
        x, y, ks = zip(*pts)
        ax.plot(x, y, marker=marker, color=colour, label=method.replace("_", " "),
                lw=1.4, ms=7, alpha=0.9)
        for xi, yi, k in pts:
            ax.annotate(f"{k}", (xi, yi), textcoords="offset points",
                        xytext=(5, -9), fontsize=7, color=colour)

    ax.axhline(0.90, ls="--", c="0.4", lw=1)
    ax.text(ax.get_xlim()[1], 0.905, "H2 target 0.90", ha="right", fontsize=8, color="0.35")
    ax.set_xlabel("incremental force-UQ latency  $T(L{=}1{+}K) - T(L{=}1)$   [ms]")
    ax.set_ylabel("recall of exact top-5% uncertainty set")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def screening_curve(screening_jsonl: Path, out: Path, *, title: str) -> None:
    """Figure 5 (spec SS34): fraction of exact evaluations skipped vs high-UQ recall."""
    recs = load_jsonl(screening_jsonl)
    agg: dict[tuple, list] = defaultdict(list)
    for r in recs:
        key = (r["method"], r["K"], r["r0"], r["alpha"])
        agg[key].append((r["frac_exact_skipped"], r["high_uq_recall"]))

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    series: dict[tuple, list] = defaultdict(list)
    for (m, K, r0, alpha), v in agg.items():
        a = np.mean(v, axis=0)
        series[(m, K, r0)].append((alpha, a[0], a[1]))
    for (m, K, r0), pts in sorted(series.items()):
        pts.sort()
        _, x, y = zip(*pts)
        marker, colour = STYLE["control_variate" if r0 else m]
        label = f"{m}{f' r0={r0}' if r0 else ''} K={K}"
        ax.plot(x, y, marker=marker, ms=6, lw=1.3, alpha=0.85, label=label, color=colour)

    ax.axhline(0.95, ls="--", c="0.4", lw=1)
    ax.axvline(0.50, ls=":", c="0.6", lw=1)
    ax.text(0.51, 0.9505, "H5 targets", fontsize=8, color="0.35")
    ax.set_xlabel("fraction of exact UQ evaluations skipped")
    ax.set_ylabel("recall of exact high-uncertainty configurations")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
