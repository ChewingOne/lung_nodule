#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python -m monai.bundle run \
  --config_file "['configs/train.json','configs/train_lndbv4_text_mlp_fold0_agr2_stage1.json']"
