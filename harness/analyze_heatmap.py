#!/usr/bin/env python3
"""Analyze MoE expert activations with statistically-grounded, layer-aware
methods.

Fixes over the naive version:
  * Operates on (layer, expert) pairs -- expert id is only meaningful within a
    layer, so we never sum a given index across layers.
  * Uses the **generation** phase by default (the model's own task output),
    which removes most of the shared-instruction-wrapper bias that pollutes the
    prefill tokens. Set MOE_PHASE=pre to inspect prefill.
  * Weights each activation by its **gating probability**, not a binary count.
  * Tests **significance** per (category, layer, expert) across prompts with a
    Welch z-test + Benjamini-Hochberg FDR control, and masks the heatmaps to
    significant cells only.
  * Adds coverage metrics: per-task routing entropy, task-overlap (Jaccard of
    significantly up-routed experts), and dead / always-on expert counts.

Outputs (harness/):
  heatmap_specialization_LE.png   per-task (layer x expert) specialization, FDR-masked
  task_overlap.png                Jaccard overlap of task-preferred experts
  routing_entropy.png             per-task per-layer routing entropy
  specialization_significant.csv  significant (layer,expert) cells per task
"""
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.environ.get("MOE_PHASE", "gen")   # 'gen' or 'pre'
ALPHA = float(os.environ.get("MOE_ALPHA", "0.05"))
USE_WEIGHTED = os.environ.get("MOE_WEIGHTED", "1") != "0"


def norm_p(z):
    """Two-sided p-value from a z-score (normal approx, no scipy)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def bh_mask(pvals, alpha):
    """Benjamini-Hochberg FDR: return boolean mask of rejected (significant)."""
    flat = pvals.ravel()
    m = flat.size
    order = np.argsort(flat)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = flat[order] <= thresh
    kmax = np.where(passed)[0].max() + 1 if passed.any() else 0
    crit = flat[order][kmax - 1] if kmax else -1.0
    return (pvals <= crit)


def main():
    d = np.load(os.path.join(HERE, "activations.npz"), allow_pickle=True)
    model = str(d["model"]); k = int(d["n_used"])
    counts = d[f"counts_{PHASE}"]
    wsum = d[f"wsum_{PHASE}"]
    cats_all = [str(c) for c in d["categories"]]
    signal = wsum if USE_WEIGHTED else counts.astype(np.float64)
    R, L, E = signal.shape
    cats = sorted(set(cats_all))
    cat_idx = {c: [i for i, x in enumerate(cats_all) if x == c] for c in cats}
    print(f"model={model}  phase={PHASE}  weighted={USE_WEIGHTED}  "
          f"{R} requests, {L} layers x {E} experts, k={k}")

    # per-prompt within-layer routing fraction with a Dirichlet pseudocount, so
    # rarely-used experts can't produce explosive (near-0 baseline) ratios.
    A = float(os.environ.get("MOE_PSEUDOCOUNT", "0.5"))            # in weight units
    layer_tot = signal.sum(axis=2, keepdims=True)
    frac = (signal + A) / (layer_tot + A * E)

    cat_frac = np.stack([frac[cat_idx[c]].mean(0) for c in cats])   # (C,L,E)
    overall = frac.mean(0)                                          # (L,E)
    eps = 1e-12
    spec = np.log2(cat_frac / overall[None])                       # (C,L,E)

    # significance: Welch z of category-c prompts vs the rest, per (c,L,E)
    pvals = np.ones((len(cats), L, E))
    for ci, c in enumerate(cats):
        inside = np.array(cat_idx[c])
        outside = np.array([i for i in range(R) if cats_all[i] != c])
        a, b = frac[inside], frac[outside]
        ma, mb = a.mean(0), b.mean(0)
        va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
        se = np.sqrt(va / len(inside) + vb / len(outside)) + eps
        z = (ma - mb) / se
        # ignore essentially-unused cells (avoid spurious significance)
        floor = overall < (0.2 / E)
        for li in range(L):
            for ei in range(E):
                pvals[ci, li, ei] = 1.0 if floor[li, ei] else norm_p(z[li, ei])
    sig = bh_mask(pvals, ALPHA)
    print(f"significant (FDR<{ALPHA}) cells per task: "
          + ", ".join(f"{c}={int(sig[i].sum())}" for i, c in enumerate(cats)))

    # ---- heatmap 1: per-task (layer x expert) specialization, masked --------
    spec_masked = np.where(sig, spec, np.nan)
    vmax = float(np.nanpercentile(np.abs(spec_masked), 99)) if sig.any() else 1.0
    vmax = min(max(vmax, 0.5), 4.0)
    ncol = 2
    nrow = (len(cats) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(10 * ncol, 3.4 * nrow), squeeze=False)
    cmap = plt.cm.coolwarm.copy(); cmap.set_bad("#dddddd")
    for ki, c in enumerate(cats):
        ax = axes[ki // ncol][ki % ncol]
        im = ax.imshow(spec_masked[ki].T, aspect="auto", cmap=cmap,
                       vmin=-vmax, vmax=vmax, interpolation="nearest", origin="lower")
        ax.set_title(f"{c}: expert specialization (FDR<{ALPHA}); red=preferred")
        ax.set_xlabel("layer"); ax.set_ylabel("expert id")
        fig.colorbar(im, ax=ax, fraction=0.025, label="log2 ratio")
    for ki in range(len(cats), nrow * ncol):
        axes[ki // ncol][ki % ncol].axis("off")
    fig.suptitle(f"{model} — per-task (layer x expert) routing specialization "
                 f"[{PHASE} phase, {'gate-weighted' if USE_WEIGHTED else 'counts'}]")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "heatmap_specialization_LE.png"), dpi=120)
    plt.close(fig)

    # ---- heatmap 2: task overlap (Jaccard of up-routed significant cells) ----
    up = sig & (spec > 0)
    J = np.zeros((len(cats), len(cats)))
    for i in range(len(cats)):
        for j in range(len(cats)):
            inter = np.logical_and(up[i], up[j]).sum()
            union = np.logical_or(up[i], up[j]).sum()
            J[i, j] = inter / union if union else 0.0
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(J, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center",
                    color="w" if J[i, j] < 0.6 else "k", fontsize=9)
    ax.set_title("Task overlap: Jaccard of\ntask-preferred experts")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "task_overlap.png"), dpi=120)
    plt.close(fig)

    # ---- heatmap 3: routing entropy per task per layer ----------------------
    # H = -sum_E p log p, normalized by log E (1=uniform routing, 0=one expert)
    p = cat_frac / cat_frac.sum(axis=2, keepdims=True).clip(min=eps)
    ent = -(p * np.log(p + eps)).sum(axis=2) / math.log(E)   # (C,L)
    fig, ax = plt.subplots(figsize=(14, 3 + 0.4 * len(cats)))
    im = ax.imshow(ent, aspect="auto", cmap="magma", vmin=ent.min(), vmax=1.0)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_xlabel("layer")
    ax.set_title("Routing entropy per task per layer "
                 "(1 = experts used evenly, low = concentrated)")
    fig.colorbar(im, ax=ax, label="normalized entropy")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "routing_entropy.png"), dpi=120)
    plt.close(fig)

    # ---- coverage metrics + CSV --------------------------------------------
    total_use = signal.sum(0)                       # (L,E)
    dead = int((total_use == 0).sum())
    # always-on: selected for ~every token in its layer (counts share high)
    tok_per_layer = counts.sum(0).sum(1, keepdims=True).clip(min=1)  # gen tokens * ... per layer
    share = counts.sum(0) / tok_per_layer
    always_on = int((share > 0.95).sum())
    print(f"coverage: {dead}/{L*E} (layer,expert) cells never used ({PHASE}); "
          f"{always_on} near-always-on")

    csv = os.path.join(HERE, "specialization_significant.csv")
    with open(csv, "w") as f:
        f.write("category,layer,expert,log2_ratio,cat_frac,overall_frac\n")
        for ci, c in enumerate(cats):
            cells = np.argwhere(sig[ci] & (spec[ci] > 0))
            cells = sorted(cells, key=lambda le: -spec[ci, le[0], le[1]])[:30]
            for li, ei in cells:
                f.write(f"{c},{li},{ei},{spec[ci,li,ei]:.3f},"
                        f"{cat_frac[ci,li,ei]:.5f},{overall[li,ei]:.5f}\n")

    print("\nTop task-preferred (layer,expert) experts [significant]:")
    for ci, c in enumerate(cats):
        cells = np.argwhere(sig[ci] & (spec[ci] > 0))
        cells = sorted(cells, key=lambda le: -spec[ci, le[0], le[1]])[:6]
        s = ", ".join(f"L{li}/e{ei}(+{spec[ci,li,ei]:.2f})" for li, ei in cells)
        print(f"  {c:10s}: {s}")

    print("\nwrote: heatmap_specialization_LE.png, task_overlap.png, "
          "routing_entropy.png, specialization_significant.csv")


if __name__ == "__main__":
    main()
