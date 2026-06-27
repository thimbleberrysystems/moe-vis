#!/usr/bin/env bash
# One-command pipeline: fetch prompts -> trace experts -> render the Expert Atlas.
# Causal ablation is a separate, optional step: python ablate_validate.py
set -euo pipefail
cd "$(dirname "$0")"
source ../env.sh
[ -f benchmarks.json ] || python fetch_benchmarks.py
python run_trace.py
python expert_atlas.py
echo "done -> harness/expert_atlas.png"
