#!/usr/bin/env bash
# Reproduce every number and figure in the README.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/download_data.py

for seed in 2021 2022 2023; do
  python scripts/train.py --dataset ETTh1 --seed "$seed"
done
python scripts/train.py --dataset weather --seed 2021

python scripts/evaluate_pruning.py --dataset ETTh1 --seeds 2021 2022 2023
python scripts/evaluate_pruning.py --dataset weather --seeds 2021 --val_step 4
python scripts/benchmark_efficiency.py --dataset ETTh1 --seed 2021
python scripts/benchmark_efficiency.py --dataset weather --seed 2021
python scripts/make_figures.py
