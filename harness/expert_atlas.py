#!/usr/bin/env python3
"""Expert Atlas — a map of the model's experts.

Every used (layer, expert) expert is one point. Two experts sit close together
if they are recruited on the same prompts (co-activation similarity), via t-SNE
of each expert's per-prompt usage profile. Color encodes the task the expert
specializes for, using a *family* hue with a per-member shade — so related tasks
(e.g. all sciences) share a colour region. Grey = shared/core.

Reads activations.npz (generation phase, gate-weighted). Output: expert_atlas.png
"""
import os

import numpy as np
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.colors import hsv_to_rgb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_T = float(os.environ.get("MOE_SHARED_T", "0.30"))

# family -> base hue (0..1). Members of a family get distinct shades of this hue.
FAMILY_HUE = {"math": 0.00, "soc": 0.07, "code": 0.58, "sci": 0.33,
              "hum": 0.80, "lang": 0.50}   # neutral handled separately (grey)


def family_of(cat):
    return cat.split("_")[0]


def build_palette(cats):
    """category -> rgb, grouped by family hue with a shade per member."""
    fams = {}
    for c in cats:
        fams.setdefault(family_of(c), []).append(c)
    color = {}
    for fam, members in fams.items():
        members = sorted(members)
        if fam == "neutral":
            for c in members:
                color[c] = (0.55, 0.57, 0.60)
            continue
        h = FAMILY_HUE.get(fam, 0.0)
        n = len(members)
        for i, c in enumerate(members):
            t = 0 if n == 1 else i / (n - 1)
            s = 0.55 + 0.40 * t          # shade by saturation ...
            v = 0.95 - 0.30 * t          # ... and value
            color[c] = tuple(hsv_to_rgb([h, s, v]))
    return color


def main():
    d = np.load(os.path.join(HERE, "activations.npz"), allow_pickle=True)
    model = str(d["model"])
    w = d["wsum_gen"]
    cats_all = [str(c) for c in d["categories"]]
    cats = sorted(set(cats_all))
    R, L, E = w.shape
    palette = build_palette(cats)

    A = 0.5
    lt = w.sum(axis=2, keepdims=True)
    frac = (w + A) / (lt + A * E)
    idx = {c: [i for i, x in enumerate(cats_all) if x == c] for c in cats}
    cat_frac = np.stack([frac[idx[c]].mean(0) for c in cats])
    overall = frac.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.log2(cat_frac / np.maximum(overall[None], 1e-12))  # (C,L,E)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

    used = (w.sum(0) > 0)
    cells = np.argwhere(used)
    X = np.stack([frac[:, l, e] for l, e in cells])
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    print(f"{model}: embedding {len(cells)} experts, {len(cats)} categories ...")

    xy = TSNE(n_components=2, perplexity=40, metric="cosine", init="pca",
              random_state=0).fit_transform(X)

    sp = np.stack([spec[:, l, e] for l, e in cells])              # (n, C)
    dom = sp.argmax(1)
    strength = sp.max(1)
    is_shared = strength < SHARED_T

    fig, ax = plt.subplots(figsize=(13, 10.5))
    ax.set_facecolor("#0f1117"); fig.set_facecolor("#0f1117")
    ax.scatter(xy[is_shared, 0], xy[is_shared, 1], s=5, c="#343842",
               alpha=0.45, linewidths=0)
    for ci, c in enumerate(cats):
        m = (~is_shared) & (dom == ci)
        if not m.any():
            continue
        s = 10 + 80 * np.clip((strength[m] - SHARED_T) / 2.0, 0, 1)
        ax.scatter(xy[m, 0], xy[m, 1], s=s, color=palette[c], alpha=0.85,
                   linewidths=0, label=c)

    # label by FAMILY at the family's centroid (keeps the map uncluttered)
    fam_of_cat = {c: family_of(c) for c in cats}
    fams = sorted({f for f in fam_of_cat.values() if f != "neutral"} | {"neutral"})
    for fam in fams:
        cis = [ci for ci, c in enumerate(cats) if fam_of_cat[c] == fam]
        m = (~is_shared) & np.isin(dom, cis)
        if m.sum() < 5:
            continue
        cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
        col = palette[cats[cis[len(cis) // 2]]]
        ax.text(cx, cy, fam, color=col, fontsize=17, fontweight="bold",
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="#0f1117")])

    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(f"Expert Atlas — {model}\n{len(cells)} (layer,expert) experts · "
                 f"position = co-activation similarity · colour = task family "
                 f"(hue) + sub-task (shade) · grey = shared/core",
                 color="w", fontsize=13, pad=12)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), framealpha=0.15,
                    labelcolor="w", facecolor="#0f1117", edgecolor="#444",
                    fontsize=8, title="category", ncol=1)
    leg.get_title().set_color("w")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "expert_atlas.png"), dpi=140,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("wrote expert_atlas.png")
    for ci, c in enumerate(cats):
        print(f"  {c:16s}: {int(((~is_shared)&(dom==ci)).sum())} specialized")
    print(f"  shared/core: {int(is_shared.sum())}")


if __name__ == "__main__":
    main()
