# LNDbv4 MONAI LUNA16 Fine-tune Experiment

## 1. Data Preparation

- Source dataset: `/root/autodl-tmp/LNDbv4`
- Aligned dataset: `/root/autodl-tmp/LNDbv4_monai_luna16_aligned`
- Conversion script: `/root/autodl-tmp/prepare_lndbv4_for_monai_luna16.py`
- Image spacing: `0.703125 x 0.703125 x 1.25 mm`
- Intensity: saved as HU-like `int16`; MONAI applies `[-1024, 300] -> [0, 1]`
- Box mode: `cccwhd` world coordinates
- Label mapping: `nodule -> 0`
- Filtering:
  - keep `Nodule = 1`
  - remove `Nodule = 0`
  - use `AgrLevel >= 2`

The final split file is:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0.json
```

Split:

```text
training   = Fold1 + Fold2 + Fold3, 177 CTs, 167 boxes, 93 empty-label CTs
validation = Fold0, 59 CTs, 54 boxes, 32 empty-label CTs
test       = Fold0, 59 CTs, 54 boxes, 32 empty-label CTs
```

The official LNDbv4 58-case test set is not used for metric computation because it has no public GT labels.

## 2. Baseline Evaluation

Model:

```text
/root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection/models/model.pt
```

This is the MONAI LUNA16 pre-trained RetinaNet model evaluated directly on LNDbv4 Fold0.

Command:

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
python -m monai.bundle run \
  --config_file "['configs/train.json','configs/evaluate.json','configs/evaluate_lndbv4_fold0.json']"
```

Result:

```text
AP_IoU_0.10_MaxDet_100        = 0.4908313303
AR_IoU_0.10_MaxDet_100        = 0.9814814925
mAP_IoU_0.10_0.50_MaxDet_100  = 0.4891738062
mAR_IoU_0.10_0.50_MaxDet_100  = 0.9773662620
```

Metric file:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_eval_fold0_agr2/metrics.csv
```

## 3. Fine-tune 10 Epoch Test

Training setup:

```text
pretrained checkpoint = MONAI LUNA16 model.pt
epochs                = 10
learning_rate         = 0.001
batch_size            = 8 patch samples per CT
validation interval   = 2 epochs
training data         = Fold1 + Fold2 + Fold3
validation data       = Fold0
```

Training command:

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
./run_finetune_lndbv4_fold0_agr2_turbo_10ep.sh
```

Evaluation command:

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
./run_evaluate_lndbv4_finetuned_fold0_agr2_turbo_10ep.sh
```

Result:

```text
AP_IoU_0.10_MaxDet_100        = 0.5294895494
AR_IoU_0.10_MaxDet_100        = 0.9814814925
mAP_IoU_0.10_0.50_MaxDet_100  = 0.5287746161
mAR_IoU_0.10_0.50_MaxDet_100  = 0.9794238806
```

Metric file:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_10ep/eval_best/metrics.csv
```

Change from baseline:

```text
AP_IoU_0.10  +0.0386582191
mAP          +0.0396008099
AR_IoU_0.10  unchanged
mAR          +0.0020576186
```

Interpretation:

```text
Fine-tuning improves AP/mAP, mainly by improving prediction confidence/ranking or reducing false positives.
Recall was already high before fine-tuning, so AR changes little.
```

## 4. Prepared 100 Epoch Run

Training command:

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
./run_finetune_lndbv4_fold0_agr2_turbo_100ep.sh
```

Evaluation command after training:

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
./run_evaluate_lndbv4_finetuned_fold0_agr2_turbo_100ep.sh
```

100 epoch output:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep
```

Key 100 epoch settings:

```text
epochs              = 100
val_interval        = 5
learning_rate       = 0.001
LR schedule         = warmup 3 epochs, StepLR step_size=30, gamma=0.1
batch_size          = 8 patch samples per CT
num_train_workers   = 8
num_val_workers     = 4
pin_memory          = true
persistent_workers  = true
```

`val_interval=5` is used for the 100 epoch run to reduce validation overhead. Each Fold0 validation pass takes about 4 minutes on the current setup.

## 5. Text-Conditioned Query-Level Experiments

### 5.1 Text datalist generation

Text-conditioned datalists were generated from:

```text
/root/autodl-tmp/LNDbv4/allNods.csv
/root/autodl-tmp/LNDbv4/report.csv
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0.json
```

Generation script:

```text
/root/autodl-tmp/create_lndbv4_text_datalist.py
```

Generated files:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_with_text.json
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_text.json
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_image_only.json
```

Text policy:

```text
reliable text: Where == TextReport+RadAnnotation
missing/unreliable text: "A pulmonary nodule."
coordinate leakage guard: x/y/z are not included in text prompts
```

Coverage:

```text
training queries   = 167, reliable text = 120, generic = 47
validation queries = 54,  reliable text = 26,  generic = 28
```

### 5.2 MLP text encoder implementation

Implemented in:

```text
/root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection/scripts/text_conditioning.py
```

Model:

```text
text_fields + optional patch spatial encoding
  -> structured feature vector
  -> MLP text encoder
  -> text embedding
  -> FiLM gamma/beta
  -> RetinaNet classification branch

RetinaNet regression branch remains image-only.
```

Training uses the current image-only 100 epoch best checkpoint as initialization:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/models/best.pt
```

Stage 1 freezes the image feature extractor and regression branch, and trains:

```text
classification_head
text_encoder
text_to_film
```

### 5.3 Query-level image-only baseline

This baseline uses the 100 epoch image-only model but evaluates on the query-level validation file with text fields removed.

Evaluation command:

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
OMP_NUM_THREADS=1 python -m monai.bundle run \
  --config_file "['configs/train.json','configs/train_lndbv4_finetune_fold0_agr2_turbo_100ep.json','configs/evaluate.json']" \
  --data_list_file_path /root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_image_only.json \
  --eval_ckpt_filename best.pt \
  --output_dir /root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/eval_query_image_only_best
```

Result:

```text
AP_IoU_0.10_MaxDet_100        = 0.3634630439
mAP_IoU_0.10_0.50_MaxDet_100  = 0.3622955941
AR_IoU_0.10_MaxDet_100        = 0.9814814925
mAR_IoU_0.10_0.50_MaxDet_100  = 0.9794238806
FROC CPM                      = 0.5185185185
num_predictions               = 1206
```

Metric file:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/eval_query_image_only_best/metrics.csv
```

### 5.4 Text MLP without patch spatial encoding

This experiment used the first MLP version, where the text feature vector had 49 dimensions and did not include patch spatial features.

Best checkpoint result:

```text
AP_IoU_0.10_MaxDet_100        = 0.3669232512
mAP_IoU_0.10_0.50_MaxDet_100  = 0.3656532630
AR_IoU_0.10_MaxDet_100        = 0.9814814925
mAR_IoU_0.10_0.50_MaxDet_100  = 0.9794238806
FROC CPM                      = 0.5264550265
num_predictions               = 1404
```

Last checkpoint result:

```text
AP_IoU_0.10_MaxDet_100        = 0.3477858098
mAP_IoU_0.10_0.50_MaxDet_100  = 0.3464795420
AR_IoU_0.10_MaxDet_100        = 0.9814814925
mAR_IoU_0.10_0.50_MaxDet_100  = 0.9794238806
FROC CPM                      = 0.5357142857
num_predictions               = 1502
```

Metric files:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_fold0_agr2_stage1/eval_best/metrics.csv
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_fold0_agr2_stage1/eval_last/metrics.csv
```

### 5.5 Text MLP with patch spatial encoding, 10 epoch

Patch spatial encoding was added to give the MLP access to the current crop position:

```text
patch_center_x, patch_center_y, patch_center_z,
patch_extent_x, patch_extent_y, patch_extent_z
```

The full feature vector is now:

```text
text_fields + patch spatial encoding -> 55 dimensions
```

Training configuration:

```text
/root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection/configs/train_lndbv4_text_mlp_fold0_agr2_stage1.json
```

Training command:

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
./run_train_lndbv4_text_mlp_fold0_agr2_stage1.sh
```

Output:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_fold0_agr2_stage1
```

Before training, text and spatial features were checked with:

```bash
python /root/autodl-tmp/check_lndbv4_text_spatial_features.py
```

Sanity check summary:

```text
TEXT_FEATURE_DIM = 55
training records = 167, reliable text = 120, generic = 47
validation records = 54, reliable text = 26, generic = 28
train crop spatial features vary by crop
validation full-volume spatial features are approximately [0, 0, 0, 1, 1, 1]
```

Evaluation command:

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
OMP_NUM_THREADS=1 python -m monai.bundle run \
  --config_file "['configs/train.json','configs/train_lndbv4_text_mlp_fold0_agr2_stage1.json','configs/evaluate.json']" \
  --eval_ckpt_filename best.pt \
  --output_dir /root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_fold0_agr2_stage1/eval_best
```

Best checkpoint result:

```text
AP_IoU_0.10_MaxDet_100        = 0.3555558536
mAP_IoU_0.10_0.50_MaxDet_100  = 0.3540233656
AR_IoU_0.10_MaxDet_100        = 0.9814814925
mAR_IoU_0.10_0.50_MaxDet_100  = 0.9794238806
FROC CPM                      = 0.5363756614
num_predictions               = 1321
```

Metric file:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_fold0_agr2_stage1/eval_best/metrics.csv
```

### 5.6 Query-level comparison

All values below use the same 54-query validation protocol.

| Model | AP@0.10 | mAP@0.10:0.50 | AR@0.10 | mAR@0.10:0.50 | FROC CPM | Predictions |
|---|---:|---:|---:|---:|---:|---:|
| Image-only query baseline | 0.3635 | 0.3623 | 0.9815 | 0.9794 | 0.5185 | 1206 |
| MLP text, no patch spatial | 0.3669 | 0.3657 | 0.9815 | 0.9794 | 0.5265 | 1404 |
| MLP text + patch spatial, 10 epoch | 0.3556 | 0.3540 | 0.9815 | 0.9794 | 0.5364 | 1321 |

Interpretation:

```text
The no-spatial MLP gives a small AP/mAP and FROC CPM gain over the query-level image-only baseline.
Adding patch spatial encoding further improves FROC CPM, especially low-FP sensitivity, but lowers AP/mAP.
The current result suggests spatial features may help low-FP ranking, but the overall precision ranking is not yet stable.
```

Important caveat:

```text
CT-level metrics and query-level metrics should not be compared directly.
CT-level evaluation inputs one CT and evaluates all nodules in that CT.
Query-level evaluation inputs one CT plus one query and evaluates only the query target.
```

### 5.7 Real / generic / shuffled text ablation

Validation-only ablation datalists were generated with:

```bash
python /root/autodl-tmp/create_lndbv4_text_ablation_datalists.py
```

Generated files:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_text_generic_val.json
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_text_shuffled_val.json
```

Ablation protocol:

```text
real text     = original query_text validation file
generic text  = all 54 validation queries use "A pulmonary nodule." and empty structured text fields
shuffled text = reliable validation text fields are cyclically shifted across the 26 reliable-text queries;
                generic validation queries remain generic
```

All runs use the same spatial text-MLP best checkpoint:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_fold0_agr2_stage1/models/best.pt
```

Results:

| Validation text | AP@0.10 | mAP@0.10:0.50 | AR@0.10 | mAR@0.10:0.50 | FROC CPM | Predictions |
|---|---:|---:|---:|---:|---:|---:|
| Real text | 0.3556 | 0.3540 | 0.9815 | 0.9794 | 0.5364 | 1321 |
| Generic text | 0.3640 | 0.3624 | 0.9815 | 0.9794 | 0.5344 | 1333 |
| Shuffled text | 0.3641 | 0.3623 | 0.9815 | 0.9774 | 0.5397 | 1330 |

Metric files:

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_fold0_agr2_stage1/eval_ablation_real_best/metrics.csv
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_fold0_agr2_stage1/eval_ablation_generic_best/metrics.csv
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_fold0_agr2_stage1/eval_ablation_shuffled_best/metrics.csv
```

Interpretation:

```text
The current spatial text-MLP checkpoint does not show evidence that it uses real text semantics.
Generic and shuffled validation text both outperform real text on AP/mAP, and shuffled text has the best FROC CPM.
This suggests the observed FROC gain over image-only query baseline is likely from the text-conditioned architecture,
training changes, spatial features, or score calibration rather than correct text-query grounding.
```
