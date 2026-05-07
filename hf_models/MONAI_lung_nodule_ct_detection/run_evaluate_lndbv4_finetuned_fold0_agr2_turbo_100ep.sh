#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

EVAL_CKPT_FILENAME="${EVAL_CKPT_FILENAME:-best.pt}"

python -m monai.bundle run \
  --config_file "['configs/train.json','configs/evaluate.json','configs/evaluate_lndbv4_finetuned_fold0_agr2_turbo_100ep.json']" \
  --eval_ckpt_filename "$EVAL_CKPT_FILENAME"
