#!/usr/bin/env python3
"""Turn activations.npz into heatmaps showing which experts each task type
activates.

Produces (in harness/):
  heatmap_category_expert.png   expert-usage fraction per task category
  heatmap_specialization.png    log2(usage / overall) -- task specialization
  heatmap_layers.png            per-category layer x expert routing (grid)
  category_expert_fraction.csv  raw numbers
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    npz = np.load(os.path.join(HERE, "activations.npz"), allow_pickle=True)
    return (npz["counts"], list(npz["categories"]), list(npz["ids"]),
            int(npz["n_used"]))


def main():
    counts, categories, ids, n_used = load()
    R, n_layers, n_experts = counts.shape
    cats = sorted(set(categories))
    print(f"{R} requests, {n_layers} layers, {n_experts} experts, top-k={n_used}")

    # (n_cat, n_experts): activations per category summed over layers + requests
    cat_expert = np.zeros((len(cats), n_experts), dtype=np.float64)
    for i, c in enumerate(categories):
        cat_expert[cats.index(c)] += counts[i].sum(axis=0)

    # normalize each category row to a probability distribution over experts
    frac = cat_expert / cat_expert.sum(axis=1, keepdims=True).clip(min=1)

    # overall expert usage distribution (baseline)
    overall = cat_expert.sum(axis=0)
    overall = overall / max(overall.sum(), 1)

    # ---- heatmap 1: raw usage fraction --------------------------------------
    fig, ax = plt.subplots(figsize=(16, 3 + 0.5 * len(cats)))
    im = ax.imshow(frac, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel("expert id")
    ax.set_title(f"Expert usage fraction per task category "
                 f"({n_experts} experts, top-{n_used} routing)")
    fig.colorbar(im, ax=ax, label="fraction of activations")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "heatmap_category_expert.png"), dpi=130)
    plt.close(fig)

    # ---- heatmap 2: specialization (log2 ratio vs overall) ------------------
    eps = 1e-9
    spec = np.log2((frac + eps) / (overall[None, :] + eps))
    vmax = float(np.percentile(np.abs(spec), 99)) or 1.0
    fig, ax = plt.subplots(figsize=(16, 3 + 0.5 * len(cats)))
    im = ax.imshow(spec, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel("expert id")
    ax.set_title("Expert specialization by task  "
                 "(log2 of category usage / overall usage; red = task-preferred)")
    fig.colorbar(im, ax=ax, label="log2 ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "heatmap_specialization.png"), dpi=130)
    plt.close(fig)

    # ---- heatmap 3: per-category layer x expert routing ---------------------
    ncol = 2
    nrow = (len(cats) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(9 * ncol, 3.2 * nrow),
                             squeeze=False)
    for k, cat in enumerate(cats):
        mask = [j for j, c in enumerate(categories) if c == cat]
        m = counts[mask].sum(axis=0).astype(np.float64)  # (n_layers, n_experts)
        m = m / m.sum(axis=1, keepdims=True).clip(min=1)  # normalize per layer
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(m, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(f"{cat}: layer x expert routing")
        ax.set_xlabel("expert id")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.025)
    for k in range(len(cats), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "heatmap_layers.png"), dpi=120)
    plt.close(fig)

    # ---- CSV + console summary ---------------------------------------------
    csv = os.path.join(HERE, "category_expert_fraction.csv")
    with open(csv, "w") as f:
        f.write("category," + ",".join(f"e{e}" for e in range(n_experts)) + "\n")
        for i, c in enumerate(cats):
            f.write(c + "," + ",".join(f"{x:.6f}" for x in frac[i]) + "\n")

    print("\nMost task-specialized experts (log2 usage vs overall):")
    for i, c in enumerate(cats):
        top = np.argsort(spec[i])[::-1][:8]
        pretty = ", ".join(f"e{e}(+{spec[i][e]:.2f})" for e in top)
        print(f"  {c:10s}: {pretty}")

    print("\nwrote: heatmap_category_expert.png, heatmap_specialization.png, "
          "heatmap_layers.png, category_expert_fraction.csv")


if __name__ == "__main__":
    main()
