#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

python scripts/run_baseline.py \
  --adapter_path models/qlora-adapter-epoch1 \
  --mode zero \
  --output outputs/qlora_epoch1.json

python scripts/run_baseline.py \
  --adapter_path models/qlora-adapter-epoch2 \
  --mode zero \
  --output outputs/qlora_epoch2.json

python scripts/run_baseline.py \
  --adapter_path models/qlora-adapter-epoch3 \
  --mode zero \
  --output outputs/qlora_epoch3.json

echo "All epoch evaluations complete."
