#!/usr/bin/env python3
"""Expert Atlas — a map of the model's experts.

Every used (layer, expert) expert is one point. Two experts sit close together
if they are recruited on the *same prompts* (co-activation similarity), found by
embedding each expert's 110-prompt usage profile to 2D with t-SNE. Each point is
colored by the task it most specializes for and sized by how strongly. Task
"modules" emerge as colored continents — no grid, no decoding.

Reads activations.npz (generation phase, gate-weighted). Output: expert_atlas.png
"""
import os

import numpy as np
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_T = float(os.environ.get("MOE_SHARED_T", "0.30"))   # log2 below this = "shared"


def main():
    d = np.load(os.path.join(HERE, "activations.npz"), allow_pickle=True)
    model = str(d["model"])
    w = d["wsum_gen"]
    cats_all = [str(c) for c in d["categories"]]
    cats = sorted(set(cats_all))
    R, L, E = w.shape

    A = 0.5
    lt = w.sum(axis=2, keepdims=True)
    frac = (w + A) / (lt + A * E)                                  # (R,L,E)
    idx = {c: [i for i, x in enumerate(cats_all) if x == c] for c in cats}
    cat_frac = np.stack([frac[idx[c]].mean(0) for c in cats])      # (C,L,E)
    overall = frac.mean(0)
    spec = np.log2(cat_frac / overall[None])                       # (C,L,E)

    # one feature vector per expert = its usage across the R prompts
    used = (w.sum(0) > 0)                                          # (L,E) drop dead
    cells = np.argwhere(used)                                      # (n,2)
    X = np.stack([frac[:, l, e] for l, e in cells])               # (n, R)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)  # usage pattern
    print(f"{model}: embedding {len(cells)} experts ...")

    xy = TSNE(n_components=2, perplexity=40, metric="cosine", init="pca",
              random_state=0).fit_transform(X)

    sp = np.stack([spec[:, l, e] for l, e in cells])              # (n, C)
    dom = sp.argmax(1)
    strength = sp.max(1)
    is_shared = strength < SHARED_T

    # ---- draw the map -------------------------------------------------------
    palette = plt.cm.tab10(np.linspace(0, 1, 10))[:len(cats)]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#0f1117"); fig.set_facecolor("#0f1117")

    # shared/core experts as faint grey backdrop
    ax.scatter(xy[is_shared, 0], xy[is_shared, 1], s=6, c="#3a3f4b",
               alpha=0.5, linewidths=0, label=None)
    # task-specialized experts as colored continents
    for ci, c in enumerate(cats):
        m = (~is_shared) & (dom == ci)
        if not m.any():
            continue
        s = 12 + 90 * np.clip((strength[m] - SHARED_T) / 2.0, 0, 1)
        ax.scatter(xy[m, 0], xy[m, 1], s=s, color=palette[ci], alpha=0.85,
                   linewidths=0, label=c)
        # label the continent at its core
        core = (~is_shared) & (dom == ci) & (strength > strength[m].mean())
        if core.sum() >= 3:
            cx, cy = np.median(xy[core, 0]), np.median(xy[core, 1])
            ax.text(cx, cy, c, color=palette[ci], fontsize=18, fontweight="bold",
                    ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=3, foreground="#0f1117")])

    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(f"Expert Atlas — {model}\n"
                 f"{len(cells)} (layer,expert) experts · position = co-activation "
                 f"similarity · color = specialized task · grey = shared/core",
                 color="w", fontsize=13, pad=14)
    leg = ax.legend(loc="upper right", framealpha=0.2, labelcolor="w",
                    facecolor="#0f1117", edgecolor="#444", title="dominant task")
    leg.get_title().set_color("w")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "expert_atlas.png"), dpi=140,
                facecolor=fig.get_facecolor())
    print("wrote expert_atlas.png")
    # quick stats
    for ci, c in enumerate(cats):
        print(f"  {c:10s}: {int(((~is_shared)&(dom==ci)).sum())} specialized experts")
    print(f"  shared/core: {int(is_shared.sum())}")


if __name__ == "__main__":
    main()
