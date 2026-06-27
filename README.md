# moe-vis — mapping & causally validating MoE expert specialization

See **which experts ("sub-models") a Mixture-of-Experts LLM uses for which task**,
using a custom-patched [Ollama](https://ollama.com) build to trace expert routing
— then map it as an **Expert Atlas** and **causally validate** it by ablation.

Reference model: **`qwen3:30b-a3b`** (qwen3moe: 48 layers, 128 experts, top-8
routing) on **CPU**, across 15 fine-grained prompt categories grouped into 7 task
families — math, code, science, humanities, social, language, and a content-free
*neutral* control (e.g. `sci_physics`, `hum_law`, `lang_translate`).

## Headline: the Expert Atlas

![atlas](results/expert_atlas.png)

### How to read it

- **Each dot is one expert** — a single expert FFN in a single layer, i.e. one
  `(layer, expert)` pair. qwen3-30b has 48 layers × 128 experts; 6,026 of those
  are actually used and plotted. (Expert index is *per-layer*: expert 40 in layer
  3 is a different network from expert 40 in layer 20, so each is its own dot.)
- **Position = co-activation similarity.** Every expert gets a 110-dimensional
  fingerprint — its gate-weighted usage on each of the 110 benchmark prompts
  (generation phase). t-SNE (cosine metric) projects those fingerprints to 2D, so
  **experts that fire on the same prompts land near each other.** Distances and
  clusters are meaningful; the absolute x/y axes are not (typical of t-SNE).
- **Colour = the task the expert specializes for** (highest `log2(task usage /
  overall usage)`), encoded hierarchically: each **family** gets a hue (math reds,
  science greens, humanities purples, …) and each sub-task a **shade** of it — so
  related tasks share a colour region. Family labels sit on each region; the
  legend lists the 15 sub-tasks.
- **Grey = shared / core** — experts whose top specialization is weak: the
  general-purpose backbone every task routes through.
- **Size = specialization strength** (how far above baseline its preferred task
  routes to it).

### What it shows

The model **self-organizes into task "continents"** — each family occupies a
distinct coloured region, with related sub-tasks (same hue, different shade)
clustering together and a shared grey core in the middle. In other words
qwen3-30b doesn't spread every task across all experts — it routes each task to a
largely distinct, spatially-coherent sub-network on top of a common core, and
related tasks sit near each other. The clean separation is the finding; the
[ablation](#causal-validation-ablation) shows it's *causal*.

---

## How it works

### 1. The trace patch (`patches/expert-trace.patch`)

Ollama runs MoE models through bundled **llama.cpp / ggml**. The patch adds an
opt-in hook (active only when `$OLLAMA_EXPERT_TRACE` is set) that records, per
token:

| what | where it's hooked | record |
|------|-------------------|--------|
| **selected experts** | `ggml_mul_mat_id` (both `ggml-cpu.c` *and* `repack.cpp`) | `{"layer":L,"experts":[[...]]}` |
| **gating weights** | dispatcher post-op on `ffn_moe_weights` | `{"wlayer":L,"weights":[[...]]}` |
| **ablation** | dispatcher pre-argsort on `ffn_moe_probs` | masks `$OLLAMA_ABLATE_EXPERTS` to −inf |

> **Two gotchas, both load-bearing:**
> 1. ggml *repacks* quantized expert weights into a blocked layout with its
>    **own** `mul_mat_id` kernel in `repack.cpp`. Hooking only `ggml-cpu.c`
>    silently captures ~half the layers. The patch hooks both.
> 2. A given expert index is **per-layer** — analysis keeps `(layer, expert)` as
>    the unit and never sums an index across layers.

### 2. The harness (`harness/`)

| script | role |
|--------|------|
| `fetch_benchmarks.py` | ~12 prompts each for 15 fine-grained categories in 7 families (GSM8K, HumanEval, MBPP, MMLU subjects, opus-100, XSum) + a neutral control. Edit the `CATEGORIES` dict to change them; `family_member` naming drives the atlas colours. |
| `run_trace.py` | Drive the patched server (serialized, `think:false`), slice the trace by byte offset per request, pair experts with gating weights, **split prefill vs generation**, validate captured layers == `block_count`. → `activations.npz` |
| `expert_atlas.py` | Embed experts by co-activation similarity (t-SNE) → the Atlas. |
| `ablate_validate.py` + `plot_ablation.py` | **Causal test**: ablate each task's top experts, measure task-specific output divergence. |

### 3. Methodology (what makes it trustworthy)

- **`(layer, expert)` units** — never pools an expert index across layers.
- **Generation phase, not prefill** (`MOE_PHASE=gen`) — the model's own output
  tokens, which avoids most shared-instruction-wrapper bias.
- **Gate-weighted** — each activation is weighted by its routing probability, not
  a binary top-k membership.
- **Pseudocount-smoothed** specialization ratios (no near-zero-baseline blow-ups).
- **Neutral control** category as a routing baseline.
- **Causally validated** by ablation — correlation alone isn't claimed.

---

## Reproduce

### 0. Prerequisites

- **Runs on CPU** — and the trace *requires* CPU execution. It hooks ggml's CPU
  kernels, so the harness forces `num_gpu: 0`; on a CUDA/Metal machine the experts
  would otherwise run on the GPU and bypass the trace entirely. (`run_trace.py`
  warns if captured layers ≠ the model's layer count, which catches this.)
- **Go ≥ 1.26**, **CMake ≥ 3.24**, a C/C++ compiler, **git**, **Python 3.10+**.
- **~30 GB disk** and **~24 GB free RAM** for `qwen3:30b-a3b` (CPU inference).
  Low on RAM? Use a smaller MoE — `ollama pull granite3.1-moe:3b` (~2 GB) and set
  `MOE_MODEL=granite3.1-moe:3b`.
- **Internet** — the build fetches llama.cpp, and `fetch_benchmarks.py` pulls
  prompts live from the HuggingFace datasets-server.

Reproducibility notes: generation is deterministic (`temperature 0`, `seed 0`,
`num_gpu 0`) and the Atlas embedding is seeded (`random_state 0`), so a given
build + model reproduces the same `activations.npz` and figures. The patch is
pinned to one llama.cpp revision (below); a different ollama version needs the
patch regenerated.

```bash
python3 -m venv venv && ./venv/bin/pip install cmake ninja numpy scipy scikit-learn matplotlib
cp env.sh.example env.sh   # edit paths, then:  source env.sh
```

### 1. Build the patched Ollama

The patch is generated against an exact llama.cpp revision, so the build must use
the matching pins. Ollama auto-applies any `*.patch` under `llama/compat/`.

| component | pin |
|-----------|-----|
| ollama | tag `v0.30.5` = commit `3370ff8b1cda259b1b4cf947422a2faff7aaa58b` |
| llama.cpp (fetched by the build) | `b9509` (ollama's `LLAMA_CPP_VERSION`) |

```bash
git clone --depth 1 --branch v0.30.5 https://github.com/ollama/ollama.git ollama-src

# verify the exact revisions the patch targets, else fail before the long build
[ "$(git -C ollama-src rev-parse HEAD)" = 3370ff8b1cda259b1b4cf947422a2faff7aaa58b ] \
  && grep -qx b9509 ollama-src/LLAMA_CPP_VERSION \
  || { echo "version mismatch -- regenerate the patch for this ollama/llama.cpp"; exit 1; }

cp patches/expert-trace.patch ollama-src/llama/compat/
cd ollama-src && cmake -B build . && cmake --build build --parallel && cd ..
```

The patch targets two functions in `ggml/src/ggml-cpu/ggml-cpu.c` and one in
`ggml/src/ggml-cpu/repack.cpp`. On a different llama.cpp revision `git apply` will
reject it at the build's patch step (it won't misapply silently); regenerate the
patch against that revision and retry.

### 2. Pull a model

```bash
ollama pull qwen3:30b-a3b      # ~18 GB; any MoE model works
```

### 3. Trace, map, validate

```bash
source env.sh && cd harness
python fetch_benchmarks.py     # benchmarks.json (+ gold answers, neutral set)
python run_trace.py            # activations.npz  (~13 min for 110 prompts)
python expert_atlas.py         # expert_atlas.png — the headline map
python ablate_validate.py      # causal ablation (~9 min; restarts server x4)
```

`run_trace.py` starts its own patched server (port 11435) on your existing
`~/.ollama` store, so it won't clash with a running Ollama.

---

## Causal validation (ablation)

This is **not** run by the trace pipeline — it's a separate, optional step
(`ablate_validate.py`, ~9 min, restarts the model server four times).

It forces a task's top-N experts out of routing (their scores → −inf, so the
router can never pick them) and measures the **causal effect** as how much the
model's greedy output changes. (Generating full correct answers from a 30B
reasoning model on CPU is too slow for a 4-condition sweep, so we measure **output
divergence** — short deterministic continuations, with vs. without ablation —
rather than accuracy.)

It ablates one representative category per task family (plus a random control) and
shows the result as **small multiples** — one panel per ablation. In each panel the
**red bar** is that family's own prompts and the **black tick** is the random-
ablation control on each bar. **Red above its tick ⇒ that family's experts causally
and specifically matter** for that family's prompts.

![ablation](results/ablation_validation.png)

In the reference run the red bar clears its random tick in **all six families** —
dramatically for `sci_biology` (~0.68 vs ~0.10 random) and `math_algebra`
(~0.20 vs ~0.03), more modestly for humanities/social/language — so the
specialization is causal and task-specific across the board, not an artifact of
routing statistics.

---

## Outputs

| file | meaning |
|------|---------|
| `expert_atlas.png` | the Expert Atlas (headline). |
| `ablation_validation.png` / `.csv` | causal output-divergence per ablation condition. |
| `activations.npz` | per-request gen/prefill, count/weighted `(layer,expert)` tensors for custom analysis. |

## Customizing

`run_trace.py`: `MOE_MODEL`, `MOE_NUM_PREDICT`, `MOE_LIMIT`, `MOE_PORT`, `OLLAMA_BIN`.
`expert_atlas.py`: `MOE_SHARED_T` (specialization threshold for "shared").
`ablate_validate.py`: `MOE_ABLATE_N`, `MOE_EVAL_N`, `MOE_EVAL_TOK`.

## Caveats

- Routing reflects the *generated token mix*; with a reasoning model some
  "thinking" style remains even at `think:false`.
- t-SNE positions are relative — read clusters/neighbourhoods, not absolute axes.
- Specialization is measured against the in-set baseline (the 15 categories).
- Causal validation uses output divergence, not task accuracy: it shows the
  experts are *causally influential and task-specific*, not the exact accuracy
  cost of removing them.

## Repo layout

```
patches/expert-trace.patch   the ggml trace + weights + ablation hooks
harness/                     fetch / run / expert_atlas / ablate (+ plot_ablation)
results/                     example Atlas + ablation figure from the reference run
env.sh.example               toolchain PATH template
```

The Ollama tree, llama.cpp clone, venv, and generated artifacts are not
committed (`.gitignore`); the steps above recreate them.
