#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PRETRAINED_CKPT_FILENAME="${PRETRAINED_CKPT_FILENAME:-model.pt}"
BEST_CKPT_FILENAME="${BEST_CKPT_FILENAME:-best.pt}"
LAST_CKPT_FILENAME="${LAST_CKPT_FILENAME:-last.pt}"

python -m monai.bundle run \
  --config_file "['configs/train.json','configs/train_lndbv4_finetune_fold0_agr2_turbo_100ep.json']" \
  --pretrained_ckpt_filename "$PRETRAINED_CKPT_FILENAME" \
  --best_ckpt_filename "$BEST_CKPT_FILENAME" \
  --last_ckpt_filename "$LAST_CKPT_FILENAME"
