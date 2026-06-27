#!/usr/bin/env python3
"""Clustered expert heatmap.

Instead of showing experts in arbitrary index order, this groups the
significantly-specialized (layer, expert) experts by their *across-task*
routing profile using hierarchical clustering. Experts that specialize for the
same task fall into the same block, so the task structure is visible directly.

Rows  = individual (layer, expert) experts (true per-layer identity, not summed)
Cols  = task categories
Color = log2(task routing / overall routing); red = task-preferred
Left strip = each expert's dominant task; dendrogram shows the clustering.

Output: heatmap_clustered.png
Reads activations.npz (generation phase, gate-weighted) -- run run_trace first.
"""
import math
import os

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = float(os.environ.get("MOE_ALPHA", "0.05"))
MAX_ROWS = int(os.environ.get("MOE_MAX_ROWS", "500"))


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def bh_crit(pvals, alpha):
    flat = np.sort(pvals.ravel())
    m = flat.size
    passed = flat <= alpha * (np.arange(1, m + 1) / m)
    return flat[np.where(passed)[0].max()] if passed.any() else -1.0


def main():
    d = np.load(os.path.join(HERE, "activations.npz"), allow_pickle=True)
    model = str(d["model"])
    w = d["wsum_gen"]
    cats_all = [str(c) for c in d["categories"]]
    cats = sorted(set(cats_all))
    R, L, E = w.shape

    A = float(os.environ.get("MOE_PSEUDOCOUNT", "0.5"))           # Dirichlet smoothing
    lt = w.sum(axis=2, keepdims=True)
    frac = (w + A) / (lt + A * E)
    idx = {c: [i for i, x in enumerate(cats_all) if x == c] for c in cats}
    cat_frac = np.stack([frac[idx[c]].mean(0) for c in cats])     # (C,L,E)
    overall = frac.mean(0)
    eps = 1e-12
    spec = np.log2(cat_frac / overall[None])                      # (C,L,E)

    # significance per (task, layer, expert): Welch z vs other tasks + BH
    pvals = np.ones((len(cats), L, E))
    floor = overall < (0.2 / E)
    for ci, c in enumerate(cats):
        ins, out = np.array(idx[c]), np.array([i for i in range(R) if cats_all[i] != c])
        a, b = frac[ins], frac[out]
        se = np.sqrt(a.var(0, ddof=1) / len(ins) + b.var(0, ddof=1) / len(out)) + eps
        z = (a.mean(0) - b.mean(0)) / se
        p = np.vectorize(norm_p)(z)
        p[floor] = 1.0
        pvals[ci] = p
    sig = pvals <= bh_crit(pvals, ALPHA)

    # rows = experts significant & up-routed for at least one task
    up = sig & (spec > 0)
    keep = np.argwhere(up.any(axis=0))                            # (n,2) -> (layer,expert)
    if len(keep) == 0:
        print("no significant experts; run a full trace first")
        return
    M = np.stack([spec[:, l, e] for l, e in keep])               # (n_rows, C)
    if len(M) > MAX_ROWS:                                        # cap for legibility
        top = np.argsort(M.max(1))[::-1][:MAX_ROWS]
        keep, M = keep[top], M[top]
    print(f"{model}: clustering {len(M)} significant experts x {len(cats)} tasks")

    Z = linkage(M, method="ward")
    clusters = fcluster(Z, t=len(cats), criterion="maxclust")
    dominant = M.argmax(1)

    # cluster -> dominant task purity summary
    print("clusters (dominant task : purity, size):")
    for cl in sorted(set(clusters)):
        members = dominant[clusters == cl]
        vals, cnts = np.unique(members, return_counts=True)
        top = vals[cnts.argmax()]
        print(f"  cluster {cl}: {cats[top]:10s}  purity={cnts.max()/len(members):.2f}  n={len(members)}")

    # ---- figure: dendrogram | heatmap | dominant-task strip -----------------
    fig = plt.figure(figsize=(9, 11))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 3.2, 0.35], wspace=0.05)
    ax_d = fig.add_subplot(gs[0]); ax_h = fig.add_subplot(gs[1]); ax_s = fig.add_subplot(gs[2])

    dn = dendrogram(Z, orientation="left", no_labels=True, ax=ax_d,
                    color_threshold=0, link_color_func=lambda _: "#888888")
    ax_d.set_xticks([]); ax_d.set_title("clustering", fontsize=10)
    order = dn["leaves"][::-1]   # top-to-bottom to match imshow origin upper

    vmax = min(float(np.percentile(np.abs(M), 99)) or 1.0, 4.0)
    im = ax_h.imshow(M[order], aspect="auto", cmap="coolwarm",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_h.set_xticks(range(len(cats))); ax_h.set_xticklabels(cats, rotation=30, ha="right")
    ax_h.set_yticks([]); ax_h.set_ylabel(f"{len(M)} specialized (layer,expert) experts")
    ax_h.set_title(f"{model} — experts clustered by task-routing profile\n"
                   "(gen phase, gate-weighted; red = task-preferred)", fontsize=10)
    fig.colorbar(im, ax=ax_s, fraction=0.6, label="log2 routing ratio")

    # dominant-task color strip
    palette = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    strip = palette[dominant[order]]
    ax_s.imshow(strip.reshape(-1, 1, 4), aspect="auto")
    ax_s.set_xticks([]); ax_s.set_yticks([]); ax_s.set_title("task", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(len(cats))]
    ax_h.legend(handles, cats, title="dominant task", loc="lower left",
                fontsize=8, framealpha=0.9)

    fig.savefig(os.path.join(HERE, "heatmap_clustered.png"), dpi=130, bbox_inches="tight")
    print("\nwrote: heatmap_clustered.png")


if __name__ == "__main__":
    main()
