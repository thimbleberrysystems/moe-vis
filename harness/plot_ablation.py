#!/usr/bin/env python3
"""Render the ablation figure from ablation_validation.csv (so it can be
regenerated without re-running the model).

Left panel  : raw output divergence, every condition incl. the random control.
Right panel : divergence *above the random-ablation baseline* -- the clean
              specificity view; a hot diagonal means each task's experts perturb
              that task more than random experts do.
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
    rows, conds = [], []
    with open(os.path.join(HERE, "ablation_validation.csv")) as f:
        r = csv.reader(f)
        sets = next(r)[1:]
        for line in r:
            conds.append(line[0]); rows.append([float(x) for x in line[1:]])
    return conds, sets, np.array(rows)


def plot(conds, sets, M):
    rand_i = conds.index("ablate random") if "ablate random" in conds else -1
    task_rows = [i for i in range(len(conds)) if i != rand_i]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 5))

    im = axl.imshow(M, cmap="magma", aspect="auto")
    axl.set_xticks(range(len(sets))); axl.set_xticklabels(sets)
    axl.set_yticks(range(len(conds))); axl.set_yticklabels(conds)
    axl.set_title("raw output divergence vs baseline\n(random row = control)")
    axl.set_xlabel("prompt type")
    for i in range(len(conds)):
        for j in range(len(sets)):
            axl.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                     color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=9)
    fig.colorbar(im, ax=axl, fraction=0.046, label="divergence")

    if rand_i >= 0:
        C = M[task_rows] - M[rand_i][None, :]
        clabels = [conds[i] for i in task_rows]
        v = float(np.abs(C).max()) or 1.0
        im2 = axr.imshow(C, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        axr.set_xticks(range(len(sets))); axr.set_xticklabels(sets)
        axr.set_yticks(range(len(clabels))); axr.set_yticklabels(clabels)
        axr.set_title("divergence ABOVE random-ablation control\n"
                      "(hot diagonal = task-specific causal effect)")
        axr.set_xlabel("prompt type")
        for i in range(len(clabels)):
            for j in range(len(sets)):
                axr.text(j, i, f"{C[i,j]:+.2f}", ha="center", va="center",
                         color="k", fontsize=9)
        fig.colorbar(im2, ax=axr, fraction=0.046, label="divergence − random")

    fig.suptitle(f"{MODEL}: causal ablation of task-specialized experts")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ablation_validation.png"), dpi=120)
    print("wrote ablation_validation.png")


if __name__ == "__main__":
    plot(*load())
