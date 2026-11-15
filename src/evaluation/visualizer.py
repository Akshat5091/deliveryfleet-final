"""Charts for the evaluation (PNG, used by the website, report and slides).

Colours: hierarchical = crimson, sequential = blue, parallel = aqua. Every series
is also labelled, so colour is never the only cue.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR

ARCH_COLOURS = {"hierarchical": "#991b1b", "sequential": "#2a78d6", "parallel": "#1baf7a"}
# fixed widths so text is the same size in every chart
HALF_W, FULL_W = 7.2, 14.4
INK, MUTED, GRID = "#111827", "#4b5563", "#e5e7eb"
AGENTS = ["supervisor", "retrieval_agent", "dispatch_agent", "route_agent", "exception_agent"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#9ca3af", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 150, "savefig.dpi": 400,
    "savefig.bbox": "standard", "figure.constrained_layout.use": True,
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
    fig, ax = plt.subplots(figsize=(HALF_W, 0.62 * len(groups) + 1.4))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
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


def per_agent_heatmaps(agent_rows: pd.DataFrame, stacked: bool = False) -> str:
    """One panel per agent: persona (rows) x architecture (cols), mean latency in seconds.
    Wide (one row of panels) for the website; stacked (two rows, HALF_W) for the A4 report."""
    present = [a for a in AGENTS if a in set(agent_rows["agent"])]
    if stacked:
        fig, grid = plt.subplots(2, 3, figsize=(HALF_W, 4.6))
        axes = grid.ravel()
        for spare in axes[len(present):]:
            spare.set_visible(False)
    else:
        fig, axes = plt.subplots(1, len(present), figsize=(FULL_W, 3.2), sharey=True)
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
    file = "heatmap_per_agent_stacked.png" if stacked else "heatmap_per_agent.png"
    fig.savefig(ARTIFACTS_DIR / file)
    plt.close(fig)
    return file


AGENT_METRICS = [("latency_s", "time (s)", "{:.1f}"), ("llm_calls", "LLM calls", "{:.1f}"),
                 ("tool_calls", "tool calls", "{:.1f}"), ("tokens", "tokens", "{:,.0f}"),
                 ("cost_usd", "cost (USD)", "{:.4f}")]


def agent_metric_heatmap(agent_rows: pd.DataFrame) -> str:
    """Agents x metrics, mean per run. Units differ, so each column is shaded against its own
    maximum and the real value is printed in every cell."""
    present = [a for a in AGENTS if a in set(agent_rows["agent"])]
    means = agent_rows.groupby("agent")[[m for m, _, _ in AGENT_METRICS]].mean().reindex(present)
    shade = means / means.max().replace(0, 1)
    fig, ax = plt.subplots(figsize=(HALF_W, 0.5 * len(present) + 1.3))
    ax.imshow(shade.to_numpy(dtype=float), cmap="Reds", vmin=0, vmax=1.15, aspect="auto")
    for (i, j), v in np.ndenumerate(means.to_numpy(dtype=float)):
        ax.text(j, i, AGENT_METRICS[j][2].format(v), ha="center", va="center", fontsize=8.5,
                color="white" if shade.iat[i, j] > 0.7 else INK)
    ax.set_xticks(range(len(AGENT_METRICS)), [label for _, label, _ in AGENT_METRICS])
    ax.set_yticks(range(len(present)), [a.replace("_", " ") for a in present])
    ax.xaxis.tick_top()
    ax.set_title("Mean work per run (darker = largest in column)", pad=26)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(ARTIFACTS_DIR / "agent_metric_heatmap.png")
    plt.close(fig)
    return "agent_metric_heatmap.png"


OUTCOME_COLOURS = {"DISPATCHED": "#166534", "RECOVERED_AFTER_DISRUPTION": "#2a78d6",
                   "ESCALATED": "#d97706", "BLOCKED": "#991b1b"}
OUTCOME_LABELS = {"DISPATCHED": "dispatched", "RECOVERED_AFTER_DISRUPTION": "recovered after disruption",
                  "ESCALATED": "escalated to a human", "BLOCKED": "blocked by guard"}


def outcome_bars(df: pd.DataFrame) -> str:
    """Stacked count of run outcomes for each architecture (same 24 inputs per architecture)."""
    archs = [a for a in ["hierarchical", "sequential", "parallel"] if a in set(df["architecture"])]
    counts = df.groupby(["architecture", "outcome"]).size().unstack(fill_value=0).reindex(archs).fillna(0)
    outcomes = [o for o in OUTCOME_COLOURS if o in counts.columns]
    fig, ax = plt.subplots(figsize=(HALF_W, 0.62 * len(archs) + 1.6))
    left = np.zeros(len(archs))
    for o in outcomes:
        vals = counts[o].to_numpy(dtype=float)
        ax.barh(archs, vals, left=left, height=0.55, color=OUTCOME_COLOURS[o], edgecolor="white",
                linewidth=2, label=OUTCOME_LABELS[o])
        for y, (v, start) in enumerate(zip(vals, left)):
            if v:
                ax.text(start + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white")
        left += vals
    ax.invert_yaxis()
    ax.set_xlabel("runs")
    ax.set_title("What happened in each run, by architecture")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8.5)
    _clean(ax)
    fig.savefig(ARTIFACTS_DIR / "outcomes_by_architecture.png")
    plt.close(fig)
    return "outcomes_by_architecture.png"


def prompt_style_cost(df: pd.DataFrame, stacked: bool = False) -> str:
    """ReAct vs Direct on the same inputs: time and tokens per run (side by side, or stacked for the report)."""
    styles = [s for s in ["ReAct", "Direct"] if s in set(df["prompt_style"])]
    fig, axes = (plt.subplots(2, 1, figsize=(HALF_W, 3.8)) if stacked
                 else plt.subplots(1, 2, figsize=(FULL_W, 2.8), sharey=True))
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, [("latency_s", "seconds per run"), ("tokens", "tokens per run")]):
        data = [df.loc[df["prompt_style"] == s, col].dropna().to_numpy() for s in styles]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 2}, whiskerprops={"color": MUTED},
                        capprops={"color": MUTED})
        for i, (vals, box) in enumerate(zip(data, bp["boxes"]), start=1):
            box.set(facecolor="#2a78d6", alpha=0.18, edgecolor="#2a78d6", linewidth=1.2)
            ax.scatter(vals, i + rng.uniform(-0.13, 0.13, len(vals)), s=22, color="#2a78d6",
                       edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_yticks(range(1, len(styles) + 1), [f"{s}  (n={len(v)})" for s, v in zip(styles, data)])
        ax.set_xlabel(label)
        _clean(ax)
    fig.suptitle("What each prompt style costs: time and tokens per run", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    file = "cost_by_prompt_style_stacked.png" if stacked else "cost_by_prompt_style.png"
    fig.savefig(ARTIFACTS_DIR / file)
    plt.close(fig)
    return file


def judge_check_bars(check: dict) -> str:
    """Judge score on a clean run vs the same run with one planted fault (scripts/judge_sanity_check.py)."""
    cases = check["cases"]
    fig, ax = plt.subplots(figsize=(HALF_W, 0.62 * len(cases) + 1.5))
    y = np.arange(len(cases))
    clean = [c["original"] for c in cases]
    broken = [c["broken"] for c in cases]
    ax.barh(y - 0.19, clean, height=0.36, color="#166534", label="clean run")
    ax.barh(y + 0.19, broken, height=0.36, color="#991b1b", label="same run with the fault planted")
    for yi, (a, b) in enumerate(zip(clean, broken)):
        ax.text(a + 0.06, yi - 0.19, f"{a}", va="center", fontsize=8.5, color=INK)
        ax.text(b + 0.06, yi + 0.19, f"{b}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [f"{c['case']}\n({c['target']})" for c in cases], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 5.6)
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("judge score for the criterion the fault breaks (1 = bad, 5 = excellent)")
    ax.set_title("Clean run vs planted fault")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    _clean(ax)
    fig.savefig(ARTIFACTS_DIR / "judge_check.png")
    plt.close(fig)
    return "judge_check.png"


def judge_bars(df: pd.DataFrame) -> str:
    crits = ["correctness", "safety", "grounding", "communication"]
    archs = [a for a in ["hierarchical", "sequential", "parallel"] if a in set(df["architecture"])]
    means = df.groupby("architecture")[crits].mean().reindex(archs)
    fig, ax = plt.subplots(figsize=(HALF_W, 3.2))
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
