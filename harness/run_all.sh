#!/usr/bin/env bash
# One-command pipeline: fetch prompts -> trace experts -> render heatmaps.
set -euo pipefail
cd "$(dirname "$0")"
source ../env.sh
[ -f benchmarks.json ] || python fetch_benchmarks.py
python run_trace.py
python analyze_heatmap.py
echo "done -> harness/*.png, category_expert_fraction.csv"
