#!/usr/bin/env python3
"""Physically prune a MoE GGUF down to chosen task(s)' most-used experts.

Produces a NEW, smaller .gguf: for every layer it keeps the top-K experts (by the
selected tasks' gate-weighted usage from activations.npz) and drops the rest,
slicing them out of the stacked expert tensors and the router, and lowering
`<arch>.expert_count` to K. Whole experts are contiguous in the file, so the
quantized blocks are sliced byte-exact with no requantization.

Because `expert_count` is global, every layer keeps the same K (different actual
experts per layer, same count). K is the knob:
  K=126  ~lossless for coding (~1.6% smaller)   K=96  ~25% smaller, ~99% routing kept
  K=64   ~50% smaller, ~94% kept (quality dips)  smaller K needs healing/finetune.

The first argument SELECTS which task(s) to optimize for (matched by category-name
prefix, so families or specific categories both work):
  code                keep coding's experts
  code,math           keep a UNION of coding + math experts
  sci_physics         keep one specific category
  drop:lang           keep everything EXCEPT language (prune that family)
  drop:lang,neutral   prune several

Usage:  python prune_export.py <spec> <K> [out.gguf]
        python prune_export.py code 96
        python prune_export.py drop:lang 110
"""
import glob
import os
import re
import sys

import numpy as np
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))


def source_gguf():
    blobs = glob.glob(os.path.expanduser("~/.ollama/models/blobs/sha256-*"))
    return max(blobs, key=os.path.getsize)            # the model weights = largest blob


def select_requests(spec, cats):
    """Resolve a spec ('code', 'code,math', 'drop:lang') to the request indices
    whose usage the kept experts are optimized for."""
    drop = spec.startswith("drop:")
    body = spec[len("drop:"):] if drop else spec
    prefixes = [p for p in body.split(",") if p]
    def matches(x):
        return any(x.startswith(p) for p in prefixes)
    return [i for i, x in enumerate(cats) if (matches(x) != drop)]


def keep_sets(spec, K):
    """Per-layer sorted indices of the K experts to KEEP for `spec`."""
    d = np.load(os.path.join(HERE, "activations.npz"), allow_pickle=True)
    c = d["counts_gen"]; cats = [str(x) for x in d["categories"]]
    reqs = select_requests(spec, cats)
    if not reqs:
        sys.exit(f"spec '{spec}' selected no categories; have {sorted(set(cats))}")
    use = c[reqs].sum(0)                              # (L, E)
    L, E = use.shape
    if not (0 < K <= E):
        sys.exit(f"K must be in 1..{E}")
    keep = {l: sorted(np.argsort(use[l])[::-1][:K].tolist()) for l in range(L)}
    return keep, L, E, len(reqs)


MOE_RE = re.compile(r"blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight$|"
                    r"blk\.(\d+)\.ffn_gate_inp\.weight$")


def moe_layer(name):
    m = MOE_RE.match(name)
    if not m:
        return None
    return int(m.group(1) if m.group(1) is not None else m.group(2))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    spec = sys.argv[1]; K = int(sys.argv[2])
    tag = re.sub(r"[^a-z0-9]+", "-", spec.lower()).strip("-")   # safe for file/model names
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, f"pruned-{tag}-k{K}.gguf")

    keep, L, E, nreq = keep_sets(spec, K)
    src = source_gguf()
    print(f"source: {src} ({os.path.getsize(src)/1e9:.1f} GB)")
    print(f"spec '{spec}' ({nreq} requests): keep top {K}/{E} experts per layer "
          f"-> {L} layers, expert_count {E}->{K}")

    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(out, arch)
    w.data_alignment = r.alignment                    # match the source alignment

    # --- metadata: copy everything, override expert_count/used_count ----------
    for key, f in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        val, vtype = f.contents(), f.types[0]
        if key.endswith(".expert_count"):
            val = K
        elif key.endswith(".expert_used_count"):
            val = min(int(val), K)
        if vtype == gguf.GGUFValueType.ARRAY:
            w.add_key_value(key, val, vtype, sub_type=f.types[1])
        else:
            w.add_key_value(key, val, vtype)

    # n_expert is axis 0 of data.shape for the 4 MoE tensors; slice it to `keep`.
    # pass 1: tensor infos (shape/nbytes only, no materializing) --------------
    sliced = 0
    for t in r.tensors:
        d = t.data
        l = moe_layer(t.name)
        if l is not None:
            assert d.shape[0] == E, f"{t.name} axis0 {d.shape[0]} != {E}"
            shape = (K,) + d.shape[1:]
            nbytes = int(np.prod(shape)) * d.dtype.itemsize
            sliced += 1
        else:
            shape, nbytes = d.shape, d.nbytes
        w.add_tensor_info(t.name, shape, d.dtype, nbytes, raw_dtype=t.tensor_type)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    # pass 2: tensor data, slicing one tensor at a time ----------------------
    for t in r.tensors:
        d = t.data
        l = moe_layer(t.name)
        if l is not None:
            d = np.ascontiguousarray(d[keep[l]])
        w.write_tensor_data(d)
    w.close()
    print(f"sliced {sliced} expert/router tensors (expect {L*4})")
    print(f"wrote {out} ({os.path.getsize(out)/1e9:.1f} GB, "
          f"{100*(1-os.path.getsize(out)/os.path.getsize(src)):.1f}% smaller)")
    name = f"{arch}-{tag}-k{K}"
    print("\nimport + run:")
    print(f"  printf 'FROM {out}\\n' > /tmp/Modelfile.pruned")
    print(f"  ollama create {name} -f /tmp/Modelfile.pruned")
    print(f"  ollama run {name} 'write a python function to reverse a string'")


if __name__ == "__main__":
    main()
