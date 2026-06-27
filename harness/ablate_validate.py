#!/usr/bin/env python3
"""Causal check via output divergence (cheap, reasoning-model friendly).

Generating full correct answers from a 30B reasoning model on CPU is too slow
for a multi-condition sweep, so instead of accuracy we measure how much ablating
a task's experts *changes the model's output* on different prompt types.

For each condition we restart the patched server with OLLAMA_ABLATE_EXPERTS set
(those experts' routing scores -> -inf, never selected), greedily (temp 0)
generate a short continuation for every prompt, and compare it token-for-token to
the un-ablated baseline via difflib. Divergence = 1 - similarity in [0,1].

The causal claim is **specificity**: ablating a task's top experts should
diverge that task's prompts much more than neutral prompts, and more than random
ablation does. A heatmap of (ablated set) x (prompt type) makes this visible.

Outputs: ablation_validation.png, ablation_validation.csv
"""
import difflib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLLAMA_BIN = os.environ.get("OLLAMA_BIN", os.path.join(ROOT, "ollama-src", "ollama"))
MODEL = os.environ.get("MOE_MODEL", "qwen3:30b-a3b")
HOST = f"127.0.0.1:{os.environ.get('MOE_PORT', '11436')}"
N_ABLATE = int(os.environ.get("MOE_ABLATE_N", "48"))
N_PROMPTS = int(os.environ.get("MOE_EVAL_N", "8"))      # per task-set
N_TOK = int(os.environ.get("MOE_EVAL_TOK", "48"))
SETS = ["math", "knowledge", "language", "neutral"]


def wait_ready(timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://{HOST}/api/tags", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def gen(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "think": True,
                       "options": {"temperature": 0, "num_predict": N_TOK, "seed": 0,
                                   "num_gpu": 0}}).encode()
    req = urllib.request.Request(f"http://{HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return (d.get("thinking", "") or "") + (d.get("response", "") or "")


def top_experts(task, npz, n):
    cats = [str(c) for c in npz["categories"]]
    w = npz["wsum_gen"]
    A = 0.5
    lt = w.sum(axis=2, keepdims=True)
    frac = (w + A) / (lt + A * w.shape[2])
    overall = frac.mean(0)
    idx = [i for i, c in enumerate(cats) if c == task]
    spec = np.log2(frac[idx].mean(0) / overall)
    order = np.argsort(spec.ravel())[::-1][:n]
    return [divmod(int(f), spec.shape[1]) for f in order]


def fmt(cells):
    return ",".join(f"{l}:{e}" for l, e in cells)


def run_condition(name, ablate_cells, prompts):
    env = dict(os.environ)
    env.update(OLLAMA_HOST=HOST, OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1",
               OLLAMA_KEEP_ALIVE="30m")
    env.pop("OLLAMA_EXPERT_TRACE", None)
    if ablate_cells:
        env["OLLAMA_ABLATE_EXPERTS"] = fmt(ablate_cells)
    else:
        env.pop("OLLAMA_ABLATE_EXPERTS", None)
    log = open(os.path.join(HERE, "ablate_serve.log"), "w")
    srv = subprocess.Popen([OLLAMA_BIN, "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready():
            sys.exit("server not ready; see harness/ablate_serve.log")
        gen("hi")
        out = {}
        t0 = time.time()
        for s in SETS:
            out[s] = [gen(p) for p in prompts[s]]
        print(f"  [{name}] generated {sum(len(v) for v in out.values())} continuations "
              f"({time.time()-t0:.0f}s)")
        return out
    finally:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(timeout=15)
        except subprocess.TimeoutExpired:
            srv.kill()


def divergence(a, b):
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def main():
    npz = np.load(os.path.join(HERE, "activations.npz"), allow_pickle=True)
    bench = json.load(open(os.path.join(HERE, "benchmarks.json")))
    bench = bench.get("tasks", bench)
    prompts = {s: [it["prompt"] for it in bench[s][:N_PROMPTS]] for s in SETS}

    ablations = {
        "ablate math": top_experts("math", npz, N_ABLATE),
        "ablate knowledge": top_experts("knowledge", npz, N_ABLATE),
        "ablate language": top_experts("language", npz, N_ABLATE),
    }
    rng = np.random.default_rng(0)
    L, E = npz["wsum_gen"].shape[1:]
    ablations["ablate random"] = [(int(rng.integers(L)), int(rng.integers(E)))
                                  for _ in range(N_ABLATE)]

    print(f"N_ablate={N_ABLATE}, {N_PROMPTS} prompts/set, {N_TOK} tokens each")
    base = run_condition("baseline", None, prompts)
    div = {}  # condition -> {set -> mean divergence vs baseline}
    for name, cells in ablations.items():
        out = run_condition(name, cells, prompts)
        div[name] = {s: float(np.mean([divergence(out[s][i], base[s][i])
                                       for i in range(len(prompts[s]))])) for s in SETS}
        print("    " + "  ".join(f"{s}={div[name][s]:.2f}" for s in SETS))

    # ---- write CSV + render figure (raw + contrast-vs-random) ---------------
    conds = list(ablations)
    M = np.array([[div[c][s] for s in SETS] for c in conds])
    with open(os.path.join(HERE, "ablation_validation.csv"), "w") as f:
        f.write("ablated," + ",".join(SETS) + "\n")
        for i, c in enumerate(conds):
            f.write(c + "," + ",".join(f"{M[i,j]:.4f}" for j in range(len(SETS))) + "\n")

    import plot_ablation
    plot_ablation.plot(conds, SETS, M)
    print("wrote: ablation_validation.png, ablation_validation.csv")


if __name__ == "__main__":
    main()
