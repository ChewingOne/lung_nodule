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
