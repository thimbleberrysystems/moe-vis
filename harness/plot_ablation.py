#!/usr/bin/env python3
"""Render the ablation figure as a grouped bar chart (no heatmap) from
ablation_validation.csv, so it can be regenerated without re-running the model.

For each prompt type (x groups) we draw one bar per ablation condition. The
grey 'random' bar is the control; if specialization is causal, ablating a task's
experts (the matching coloured bar) should exceed random on that task. Matching
(task == prompt type) bars are outlined in black.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MOE_MODEL", "qwen3:30b-a3b")


def load():
    conds, rows = [], []
    with open(os.path.join(HERE, "ablation_validation.csv")) as f:
        r = csv.reader(f)
        sets = next(r)[1:]
        for line in r:
            conds.append(line[0]); rows.append([float(x) for x in line[1:]])
    return conds, sets, np.array(rows)


def plot(conds, sets, M):
    # M[cond, set]; transpose to group by prompt type
    colors = {"ablate math": "#d62728", "ablate knowledge": "#ff7f0e",
              "ablate language": "#2ca02c", "ablate code": "#1f77b4",
              "ablate random": "#9aa0aa"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(conds)
    width = 0.8 / n
    x = np.arange(len(sets))
    for ci, cond in enumerate(conds):
        vals = M[ci]
        offs = x + (ci - (n - 1) / 2) * width
        col = colors.get(cond, "#888888")
        # outline the bar where the ablated task matches the prompt type
        edges = ["k" if cond == f"ablate {s}" else "none" for s in sets]
        lws = [2.0 if cond == f"ablate {s}" else 0 for s in sets]
        ax.bar(offs, vals, width, label=cond, color=col,
               edgecolor=edges, linewidth=lws, zorder=3)

    ax.set_xticks(x); ax.set_xticklabels([f"{s}\nprompts" for s in sets])
    ax.set_ylabel("output divergence from un-ablated baseline")
    ax.set_title(f"{MODEL}: causal effect of ablating task-specialized experts\n"
                 "black-outlined = ablating that task's own experts; grey = random control")
    ax.legend(ncol=len(conds), fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ablation_validation.png"), dpi=130,
                bbox_inches="tight")
    print("wrote ablation_validation.png")


if __name__ == "__main__":
    plot(*load())
