#!/usr/bin/env python3
"""Drive the patched Ollama build over the benchmark prompts and collect
per-request MoE expert activations.

For each prompt we record the trace file size before/after the request and
parse only the bytes written in between (requests are serialized via
OLLAMA_NUM_PARALLEL=1), so each slice belongs to exactly one prompt.

Output: activations.npz
  counts      int32 (n_requests, n_layers, n_experts)  expert-activation counts
  categories  str   (n_requests,)
  ids         str   (n_requests,)
  n_used      int   experts selected per token (top-k)
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

OLLAMA_BIN = os.environ.get("OLLAMA_BIN", os.path.join(ROOT, "ollama-src", "ollama"))
MODEL = os.environ.get("MOE_MODEL", "qwen3:30b-a3b")
PORT = os.environ.get("MOE_PORT", "11435")
HOST = f"127.0.0.1:{PORT}"
TRACE_FILE = os.path.join(HERE, "expert_trace.jsonl")
NUM_PREDICT = int(os.environ.get("MOE_NUM_PREDICT", "96"))


def wait_ready(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}/api/tags", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def generate(prompt):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": NUM_PREDICT, "seed": 0},
    }).encode()
    req = urllib.request.Request(f"http://{HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def parse_slice(text):
    """Yield (layer, [expert ids...]) per token across the slice.

    Each MoE layer emits gate/up/down mul_mat_id ops that share the same
    selected_experts; we keep only the 'down' op so every (layer, token) is
    counted exactly once.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "down" not in rec["name"]:
            continue
        layer = rec["layer"]
        for tok in rec["experts"]:
            yield layer, tok


def main():
    if not os.path.exists(OLLAMA_BIN):
        sys.exit(f"patched ollama binary not found at {OLLAMA_BIN}; set OLLAMA_BIN")
    with open(os.path.join(HERE, "benchmarks.json")) as f:
        benchmarks = json.load(f)
    limit = int(os.environ.get("MOE_LIMIT", "0"))  # 0 = all
    if limit:
        benchmarks = {c: items[:limit] for c, items in benchmarks.items()}

    open(TRACE_FILE, "w").close()  # truncate

    env = dict(os.environ)
    env["OLLAMA_HOST"] = HOST
    env["OLLAMA_EXPERT_TRACE"] = TRACE_FILE
    env["OLLAMA_NUM_PARALLEL"] = "1"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["OLLAMA_KEEP_ALIVE"] = "30m"

    serve_log = open(os.path.join(HERE, "serve.log"), "w")
    print(f"starting patched server: {OLLAMA_BIN} serve  (host {HOST})")
    server = subprocess.Popen([OLLAMA_BIN, "serve"], env=env,
                              stdout=serve_log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready():
            sys.exit("server did not become ready; see harness/serve.log")
        print("server ready; warming up model (first load reads ~18GB into RAM)...")
        generate("hi")  # load model; ignore its trace (we slice per request below)

        per_request = []   # list of (category, id, Counter{(layer,expert):n})
        max_layer = max_expert = 0
        n_used_seen = 0

        for cat, items in benchmarks.items():
            for it in items:
                off0 = os.path.getsize(TRACE_FILE)
                t0 = time.time()
                try:
                    generate(it["prompt"])
                except urllib.error.URLError as e:
                    print(f"  [{cat} {it['id']}] request failed: {e}")
                    continue
                off1 = os.path.getsize(TRACE_FILE)
                with open(TRACE_FILE) as f:
                    f.seek(off0)
                    chunk = f.read(off1 - off0)

                ctr = Counter()
                ntok = 0
                for layer, experts in parse_slice(chunk):
                    ntok += 1
                    n_used_seen = max(n_used_seen, len(experts))
                    if layer > max_layer:
                        max_layer = layer
                    for e in experts:
                        if e > max_expert:
                            max_expert = e
                        ctr[(layer, e)] += 1
                per_request.append((cat, it["id"], ctr))
                print(f"  [{cat} {it['id']}] {ntok} token-activations, "
                      f"{time.time()-t0:.1f}s")

        n_layers = max_layer + 1
        n_experts = max_expert + 1
        R = len(per_request)
        counts = np.zeros((R, n_layers, n_experts), dtype=np.int32)
        categories, ids = [], []
        for i, (cat, rid, ctr) in enumerate(per_request):
            categories.append(cat)
            ids.append(rid)
            for (layer, e), n in ctr.items():
                counts[i, layer, e] = n

        out = os.path.join(HERE, "activations.npz")
        np.savez_compressed(out, counts=counts,
                            categories=np.array(categories),
                            ids=np.array(ids), n_used=n_used_seen)
        print(f"\nwrote {out}: {R} requests, "
              f"{n_layers} layers x {n_experts} experts, top-k={n_used_seen}")
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
