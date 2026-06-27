#!/usr/bin/env python3
"""Drive the patched Ollama build over benchmark prompts and collect MoE
expert activations -- per (layer, expert), weighted by gating probability, and
split into prefill vs generation phases.

The patched runtime emits two record kinds to $OLLAMA_EXPERT_TRACE:
  {"layer":L,"name":"ffn_moe_down-L","n_used":k,"n_tokens":T,"experts":[[...]]}
  {"wlayer":L,"n_used":k,"n_tokens":T,"weights":[[...]]}
We keep the 'down' experts record (one per layer/forward) and pair it, per layer
in occurrence order, with the matching weights record so each selected expert
gets its gating weight. n_tokens>1 marks the prefill batch; n_tokens==1 a decode
step.

Output: activations.npz with, for phase in {gen, pre}:
  counts_<phase>  int32   (R, n_layers, n_experts)  selection counts
  wsum_<phase>    float32 (R, n_layers, n_experts)  summed normalized gate weight
plus categories, ids, model, n_used, block_count.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

OLLAMA_BIN = os.environ.get("OLLAMA_BIN", os.path.join(ROOT, "ollama-src", "ollama"))
MODEL = os.environ.get("MOE_MODEL", "qwen3:30b-a3b")
PORT = os.environ.get("MOE_PORT", "11435")
HOST = f"127.0.0.1:{PORT}"
TRACE_FILE = os.path.join(HERE, "expert_trace.jsonl")
NUM_PREDICT = int(os.environ.get("MOE_NUM_PREDICT", "96"))


def wait_ready(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}/api/tags", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def api(path, payload):
    req = urllib.request.Request(f"http://{HOST}/api/{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def model_dims():
    """Return (block_count, expert_used_count) from /api/show, or (None, None)."""
    try:
        mi = api("show", {"model": MODEL}).get("model_info", {})
    except Exception:  # noqa: BLE001
        return None, None
    bc = next((v for k, v in mi.items() if k.endswith(".block_count")), None)
    ec = next((v for k, v in mi.items() if k.endswith(".expert_used_count")), None)
    return bc, ec


def generate(prompt):
    return api("generate", {
        "model": MODEL, "prompt": prompt, "stream": False, "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": NUM_PREDICT, "seed": 0},
    })


def parse_request(text):
    """Pair down-experts records with weights records per layer/forward.

    Yields (phase, layer, expert_id, norm_weight) for every selected expert.
    phase is 'pre' (n_tokens>1) or 'gen' (n_tokens==1).
    """
    downs = defaultdict(list)   # layer -> [ (n_tokens, [[experts]...]) ]
    wts = defaultdict(list)     # layer -> [ (n_tokens, [[weights]...]) ]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "experts" in r:
            if "down" in r.get("name", ""):
                downs[r["layer"]].append(r["experts"])
        elif "weights" in r:
            wts[r["wlayer"]].append(r["weights"])

    for layer, dlist in downs.items():
        wlist = wts.get(layer, [])
        for fi, d_forward in enumerate(dlist):
            w_forward = wlist[fi] if fi < len(wlist) else None
            phase = "pre" if len(d_forward) > 1 else "gen"  # n_tokens>1 -> prefill batch
            for ti, experts in enumerate(d_forward):
                wrow = (w_forward[ti] if w_forward and ti < len(w_forward) else None)
                if wrow and len(wrow) == len(experts):
                    s = sum(wrow) or 1.0
                    norm = [w / s for w in wrow]
                else:
                    norm = [1.0 / len(experts)] * len(experts)  # fallback: uniform
                for e, w in zip(experts, norm):
                    yield phase, layer, e, w


def main():
    if not os.path.exists(OLLAMA_BIN):
        sys.exit(f"patched ollama binary not found at {OLLAMA_BIN}; set OLLAMA_BIN")
    with open(os.path.join(HERE, "benchmarks.json")) as f:
        bench = json.load(f)
    bench = bench.get("tasks", bench)  # tolerate {tasks:..., neutral:...} or flat
    limit = int(os.environ.get("MOE_LIMIT", "0"))
    if limit:
        bench = {c: items[:limit] for c, items in bench.items()}

    open(TRACE_FILE, "w").close()
    env = dict(os.environ)
    env.update(OLLAMA_HOST=HOST, OLLAMA_EXPERT_TRACE=TRACE_FILE,
               OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1",
               OLLAMA_KEEP_ALIVE="30m")
    env.pop("OLLAMA_ABLATE_EXPERTS", None)  # tracing run: no ablation

    serve_log = open(os.path.join(HERE, "serve.log"), "w")
    print(f"starting patched server: {OLLAMA_BIN} serve ({HOST})")
    server = subprocess.Popen([OLLAMA_BIN, "serve"], env=env,
                              stdout=serve_log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready():
            sys.exit("server not ready; see harness/serve.log")
        block_count, k_meta = model_dims()
        print(f"model dims: block_count={block_count} expert_used_count={k_meta}")
        print("warming up (first load reads the model into RAM)...")
        generate("hi")

        records = []  # (category, id, dict[(phase,layer,expert)] -> [count, wsum])
        max_layer = max_expert = 0

        for cat, items in bench.items():
            for it in items:
                off0 = os.path.getsize(TRACE_FILE)
                t0 = time.time()
                try:
                    generate(it["prompt"])
                except urllib.error.URLError as e:
                    print(f"  [{cat} {it['id']}] FAILED: {e}")
                    continue
                with open(TRACE_FILE) as f:
                    f.seek(off0)
                    chunk = f.read(os.path.getsize(TRACE_FILE) - off0)

                acc = defaultdict(lambda: [0, 0.0])
                ntok = 0
                for phase, layer, e, w in parse_request(chunk):
                    ntok += 1
                    max_layer = max(max_layer, layer)
                    max_expert = max(max_expert, e)
                    cell = acc[(phase, layer, e)]
                    cell[0] += 1
                    cell[1] += w
                if not acc:
                    print(f"  [{cat} {it['id']}] WARNING: empty trace slice")
                    continue
                records.append((cat, it["id"], dict(acc)))
                print(f"  [{cat} {it['id']}] {ntok} activations, {time.time()-t0:.1f}s")

        n_layers = max_layer + 1
        n_experts = max_expert + 1
        R = len(records)
        out = {p: {"counts": np.zeros((R, n_layers, n_experts), np.int32),
                   "wsum": np.zeros((R, n_layers, n_experts), np.float32)}
               for p in ("pre", "gen")}
        cats, ids = [], []
        for i, (cat, rid, acc) in enumerate(records):
            cats.append(cat)
            ids.append(rid)
            for (phase, layer, e), (c, w) in acc.items():
                out[phase]["counts"][i, layer, e] = c
                out[phase]["wsum"][i, layer, e] = w

        # --- validation (gap #5) ----------------------------------------------
        gen_layers = int((out["gen"]["counts"].sum(axis=(0, 2)) > 0).sum())
        if block_count is not None and gen_layers != block_count:
            print(f"  WARNING: captured {gen_layers} gen layers != block_count "
                  f"{block_count} -- trace may be missing layers")
        else:
            print(f"  OK: captured {gen_layers} layers (block_count={block_count})")
        print(f"  distinct experts seen: {n_experts} (model expert_used_count k={k_meta})")

        np.savez_compressed(
            os.path.join(HERE, "activations.npz"),
            counts_gen=out["gen"]["counts"], wsum_gen=out["gen"]["wsum"],
            counts_pre=out["pre"]["counts"], wsum_pre=out["pre"]["wsum"],
            categories=np.array(cats), ids=np.array(ids),
            model=MODEL, n_used=(k_meta or 0), block_count=(block_count or n_layers))
        print(f"\nwrote activations.npz: {R} requests, {n_layers}x{n_experts} "
              f"(gen+prefill, counts+weighted)")
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
