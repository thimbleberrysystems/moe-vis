#!/usr/bin/env python3
"""Render the ablation result as small multiples (no heatmap) from
ablation_validation.csv, so it can be regenerated without re-running the model.

One panel per ablation condition. Each panel bars the output divergence across
the evaluated prompt sets; the set whose own experts were ablated is highlighted
in red, and the grey random-ablation control is drawn as a black tick on each bar.
If specialization is causal, the red bar clears its own tick.
"""
import csv
import math
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
    rand_i = conds.index("ablate random") if "ablate random" in conds else None
    rand = M[rand_i] if rand_i is not None else np.zeros(len(sets))
    panels = [i for i in range(len(conds)) if i != rand_i]

    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows),
                             sharey=True, squeeze=False)
    x = np.arange(len(sets))
    short = [s.replace("_", "\n") for s in sets]
    ymax = M.max() * 1.15

    for k, ci in enumerate(panels):
        ax = axes.flat[k]
        cond = conds[ci]
        target = cond.replace("ablate ", "")
        vals = M[ci]
        colors = ["#e23b3b" if s == target else "#4a6fa5" for s in sets]
        ax.bar(x, vals, color=colors, zorder=3, width=0.7)
        # random-ablation control = black tick across each bar
        ax.hlines(rand, x - 0.35, x + 0.35, color="k", linewidth=1.6, zorder=5)
        ax.set_title(cond, fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.5)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.25, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k % ncols == 0:
            ax.set_ylabel("divergence vs baseline")
    for k in range(len(panels), nrows * ncols):
        axes.flat[k].axis("off")

    fig.suptitle(f"{MODEL}: causal ablation — each panel ablates one task family's "
                 f"experts\nred = that family's own prompts · black tick = random-"
                 f"ablation control (red above tick ⇒ causal & specific)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(HERE, "ablation_validation.png"), dpi=130)
    print("wrote ablation_validation.png")


if __name__ == "__main__":
    plot(*load())
