"""Charts for the evaluation (PNG, used by the website, report and slides).

Palette validated with the dataviz validator (light surface, all hard checks
pass): hierarchical = crimson, sequential = blue, parallel = aqua. Axis labels
and direct labels carry identity, so colour is never the only cue.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR

ARCH_COLOURS = {"hierarchical": "#991b1b", "sequential": "#2a78d6", "parallel": "#1baf7a"}
INK, MUTED, GRID = "#111827", "#4b5563", "#e5e7eb"
AGENTS = ["supervisor", "retrieval_agent", "dispatch_agent", "route_agent", "exception_agent"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#9ca3af", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
})


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def boxplot_by(df: pd.DataFrame, value: str, group: str, title: str, xlabel: str, file: str,
               order: list[str] | None = None, colour_by_arch: bool = False) -> str:
    groups = order or sorted(df[group].dropna().unique())
    data = [df.loc[df[group] == g, value].dropna().to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(groups) + 1.4))
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 2}, whiskerprops={"color": MUTED},
                    capprops={"color": MUTED})
    rng = np.random.default_rng(0)
    for i, (g, vals, box) in enumerate(zip(groups, data, bp["boxes"]), start=1):
        colour = ARCH_COLOURS.get(g, "#2a78d6") if colour_by_arch else "#2a78d6"
        box.set(facecolor=colour, alpha=0.18, edgecolor=colour, linewidth=1.2)
        ax.scatter(vals, i + rng.uniform(-0.13, 0.13, len(vals)), s=26, color=colour,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(range(1, len(groups) + 1))
    ax.set_yticklabels([f"{g}  (n={len(v)})" for g, v in zip(groups, data)])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _clean(ax)
    fig.savefig(ARTIFACTS_DIR / file)
    plt.close(fig)
    return file


def heatmap(matrix: pd.DataFrame, title: str, file: str, fmt: str = "{:.1f}", cbar_label: str = "",
            figsize: tuple[float, float] | None = None) -> str:
    """Sequential single-hue (crimson) heatmap with every value printed in the cell."""
    figsize = figsize or (1.15 * matrix.shape[1] + 2.4, 0.42 * matrix.shape[0] + 1.3)
    fig, ax = plt.subplots(figsize=figsize)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="Reds", aspect="auto", vmin=np.nanmin(values) if values.size else 0)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=0)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    top = np.nanmax(values) if values.size else 1
    for (i, j), v in np.ndenumerate(values):
        if not np.isnan(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.6 * top else INK)
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar_label, color=MUTED)
    cb.outline.set_visible(False)
    fig.savefig(ARTIFACTS_DIR / file)
    plt.close(fig)
    return file


def per_agent_heatmaps(agent_rows: pd.DataFrame) -> str:
    """One panel per agent: persona (rows) x architecture (cols), mean latency in seconds."""
    present = [a for a in AGENTS if a in set(agent_rows["agent"])]
    fig, axes = plt.subplots(1, len(present), figsize=(3.0 * len(present) + 1, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    archs = ["hierarchical", "sequential", "parallel"]
    personas = ["polite", "urgent", "abusive"]
    vmax = agent_rows["latency_s"].max() or 1
    for ax, agent in zip(axes, present):
        sub = agent_rows[agent_rows["agent"] == agent]
        m = sub.pivot_table(index="persona", columns="architecture", values="latency_s", aggfunc="mean")
        m = m.reindex(index=personas, columns=archs)
        ax.imshow(m.to_numpy(dtype=float), cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(m.to_numpy(dtype=float)):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 * vmax else INK)
        ax.set_xticks(range(3), ["hier", "seq", "par"])
        ax.set_yticks(range(3), personas)
        ax.set_title(agent.replace("_", " "), fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Inside each agent: mean latency (s) by persona and architecture", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "heatmap_per_agent.png")
    plt.close(fig)
    return "heatmap_per_agent.png"


def judge_bars(df: pd.DataFrame) -> str:
    crits = ["correctness", "safety", "grounding", "communication"]
    archs = [a for a in ["hierarchical", "sequential", "parallel"] if a in set(df["architecture"])]
    means = df.groupby("architecture")[crits].mean().reindex(archs)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(crits))
    h = 0.8 / len(archs)
    for k, arch in enumerate(archs):
        vals = means.loc[arch].to_numpy()
        ax.barh(y + k * h - 0.4 + h / 2, vals, height=h - 0.04, color=ARCH_COLOURS[arch], label=arch)
        for yi, v in zip(y, vals):
            ax.text(v + 0.04, yi + k * h - 0.4 + h / 2, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, crits)
    ax.set_xlim(1, 5.4)
    ax.set_xlabel("LLM-judge score (1 = bad, 5 = excellent)")
    ax.set_title("Judge scores per criterion (Y in R^4) by architecture")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    _clean(ax)
    fig.savefig(ARTIFACTS_DIR / "judge_by_criterion.png")
    plt.close(fig)
    return "judge_by_criterion.png"
