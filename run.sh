#!/usr/bin/env bash
# moe-vis — one-shot pipeline. Hit and run:
#
#   ./run.sh                      # full pipeline, default model (qwen3:30b-a3b)
#   ./run.sh gpt-oss:20b          # any MoE model that runs on the llama.cpp engine
#   SKIP_ABLATE=1 ./run.sh        # skip the (~20 min) causal ablation
#
# The model is an optional first argument (defaults to qwen3:30b-a3b). The run is
# idempotent: each stage is skipped if already done. Stages: toolchain check ->
# venv -> build patched ollama -> pull model -> fetch -> trace -> Atlas -> ablate.
# It opens the output graphs at the end.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

MODEL="${1:-${MODEL:-qwen3:30b-a3b}}"
OLLAMA_TAG="v0.30.5"
OLLAMA_SHA="3370ff8b1cda259b1b4cf947422a2faff7aaa58b"
LLAMA_PIN="b9509"
PORT_PULL="127.0.0.1:11439"

say() { printf '\n\033[1;36m[%s]\033[0m %s\n' "$1" "$2"; }

# --- toolchain ---------------------------------------------------------------
if ! command -v go >/dev/null; then
  for d in "$HOME"/sdk/go*/bin /usr/local/go/bin; do [ -d "$d" ] && PATH="$d:$PATH"; done
fi
command -v go >/dev/null || { echo "ERROR: need Go >= 1.26 on PATH (https://go.dev/dl)"; exit 1; }
command -v git >/dev/null || { echo "ERROR: need git"; exit 1; }
command -v gcc >/dev/null || command -v cc >/dev/null || { echo "ERROR: need a C/C++ compiler"; exit 1; }
export CC="${CC:-gcc}" CXX="${CXX:-g++}"

# --- python venv (also provides cmake + ninja) -------------------------------
if [ ! -x venv/bin/python ]; then
  say setup "creating venv + installing python deps"
  python3 -m venv venv
  venv/bin/pip -q install --upgrade pip
  venv/bin/pip -q install cmake ninja numpy scipy scikit-learn matplotlib
fi
export PATH="$ROOT/venv/bin:$PATH"            # cmake/ninja/python all from the venv

# --- build the patched ollama ------------------------------------------------
if [ ! -x ollama-src/ollama ]; then
  say build "cloning ollama $OLLAMA_TAG and applying the trace patch"
  [ -d ollama-src ] || git clone --depth 1 --branch "$OLLAMA_TAG" \
    https://github.com/ollama/ollama.git ollama-src
  if [ "$(git -C ollama-src rev-parse HEAD)" != "$OLLAMA_SHA" ] || \
     ! grep -qx "$LLAMA_PIN" ollama-src/LLAMA_CPP_VERSION; then
    echo "ERROR: ollama/llama.cpp revision != pinned ($OLLAMA_SHA / $LLAMA_PIN)."
    echo "       Regenerate patches/expert-trace.patch against this revision."
    exit 1
  fi
  cp patches/expert-trace.patch ollama-src/llama/compat/
  ( cd ollama-src && cmake -B build . && cmake --build build --parallel )
fi
export OLLAMA_BIN="$ROOT/ollama-src/ollama"

# --- ensure the model is present (pull via the patched binary's own server) --
name="${MODEL%%:*}"; tag="${MODEL##*:}"; [ "$tag" = "$MODEL" ] && tag="latest"
manifest="$HOME/.ollama/models/manifests/registry.ollama.ai/library/$name/$tag"
if [ ! -f "$manifest" ]; then
  say pull "$MODEL (large download)"
  OLLAMA_HOST="$PORT_PULL" "$OLLAMA_BIN" serve >/tmp/moe-vis-pull.log 2>&1 &
  pull_pid=$!
  for _ in $(seq 1 90); do
    curl -sf "http://$PORT_PULL/api/tags" >/dev/null 2>&1 && break; sleep 1
  done
  OLLAMA_HOST="$PORT_PULL" "$OLLAMA_BIN" pull "$MODEL"
  kill "$pull_pid" 2>/dev/null || true
  wait "$pull_pid" 2>/dev/null || true
fi
export MOE_MODEL="$MODEL"

# --- pipeline ----------------------------------------------------------------
cd harness
[ -f benchmarks.json ] || { say fetch "downloading benchmark prompts"; python fetch_benchmarks.py; }
say trace "tracing expert routing for $MODEL (CPU; this is the slow part)"
python run_trace.py
say atlas "rendering the Expert Atlas"
python expert_atlas.py
if [ "${SKIP_ABLATE:-0}" != "1" ]; then
  say ablate "causal ablation (restarts the server per condition)"
  python ablate_validate.py
fi

# --- show the output graphs --------------------------------------------------
open_graph() {  # open in the platform image viewer, and always print the path
  local f="$1"; [ -f "$f" ] || return 0
  if   command -v xdg-open >/dev/null; then (xdg-open "$f" >/dev/null 2>&1 &)
  elif command -v open     >/dev/null; then open "$f"
  fi
  echo "  $f"
}

say done "outputs:"
open_graph "$ROOT/harness/expert_atlas.png"
[ "${SKIP_ABLATE:-0}" != "1" ] && open_graph "$ROOT/harness/ablation_validation.png"
