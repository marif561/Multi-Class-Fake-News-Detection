#!/usr/bin/env bash
# Run all three models for Task 3a (EN -> EN) and Task 3b (EN -> DE).
# Usage: bash scripts/run_all.sh <data_dir>   (expects <data_dir>/en/{train,dev,test}.csv and <data_dir>/de/test.csv)
set -euo pipefail
D=${1:?"data directory required"}
EN="--train $D/en/train.csv --dev $D/en/dev.csv"

for M in train_pac train_bilstm train_transformer; do
  echo ">> $M · Task 3a"
  python -m fakenews.$M $EN --test "$D/en/test.csv"
  if [[ -f "$D/de/test.csv" ]]; then
    echo ">> $M · Task 3b"
    python -m fakenews.$M $EN --test "$D/de/test.csv" --test-lang de
  fi
done
echo "Done. Results in outputs/"
