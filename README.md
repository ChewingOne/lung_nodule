# LNDbv4 肺结节检测：MONAI LUNA16 迁移微调实验

本项目基于 **MONAI Lung Nodule CT Detection Bundle**，将 LNDbv4 肺结节 CT 数据整理为接近 LUNA16 bundle 的输入格式，并在 LNDbv4 Fold0 验证集上进行迁移微调与评估。

仓库只建议保存代码、配置、说明文档和轻量级实验记录；**不建议提交 LNDbv4、LUNA16 原始数据、重采样影像、模型权重和训练日志大文件**。

## 项目内容

- 整理 LNDbv4 数据格式，并说明其与 LUNA16 的差异。
- 将 LNDbv4 的 `.mhd/.raw` CT 和 `trainNodules_gt.csv` 标注转换为 MONAI detection datalist。
- 按 MONAI LUNA16 bundle 的目标 spacing 对 CT 进行重采样。
- 使用 MONAI LUNA16 预训练 RetinaNet 模型在 LNDbv4 上微调。
- 保留当前主要实验配置：`Fold1 + Fold2 + Fold3` 训练，`Fold0` 验证，`AgrLevel >= 2`，训练 `100 epoch`。

## 推荐仓库结构

```text
.
├── README.md
├── data.md
├── prepare_lndbv4_for_monai_luna16.py
├── LNDbv4_monai_luna16_aligned/
│   ├── README_preparation.md
│   ├── EXPERIMENT_LOG.md
│   ├── summary.json
│   ├── lndbv4_train_folds1-3_test_fold0.json
│   ├── lndbv4_labeled_train_as_validation.json
│   └── lndbv4_test_as_validation.json
└── hf_models/
    └── MONAI_lung_nodule_ct_detection/
        ├── configs/
        │   ├── train.json
        │   ├── evaluate.json
        │   ├── train_lndbv4_finetune_fold0_agr2_turbo_100ep.json
        │   └── evaluate_lndbv4_finetuned_fold0_agr2_turbo_100ep.json
        ├── scripts/
        ├── run_finetune_lndbv4_fold0_agr2_turbo_100ep.sh
        └── run_evaluate_lndbv4_finetuned_fold0_agr2_turbo_100ep.sh
```

本地工作目录中可能还有以下大文件目录，但它们不应进入 GitHub：

```text
LNDbv4/
LNDbv4_monai_luna16_aligned/images/
LNDbv4_monai_luna16_aligned/monai_finetune_*/
datasets/
hf_models/Bio_ClinicalBERT/
hf_models/MONAI_lung_nodule_ct_detection/models/
```

## 数据说明

LNDbv4 主 CT 使用 MetaImage 格式，每个病例由一对文件组成：

```text
LNDb-0001.mhd
LNDb-0001.raw
```

主要标注文件：

- `trainCTs.csv`：训练 CT ID。
- `testCTs.csv`：测试 CT ID。
- `trainFolds.csv`：训练集 fold 划分。
- `trainNodules_gt.csv`：多医生合并后的结节标注。
- `report.csv`：结构化报告信息。

更完整的数据格式说明见 `data.md`。

## 数据准备

默认路径假设：

```text
原始 LNDbv4: /root/autodl-tmp/LNDbv4
输出目录:   /root/autodl-tmp/LNDbv4_monai_luna16_aligned
```

执行数据转换：

```bash
python prepare_lndbv4_for_monai_luna16.py \
  --lndb-dir /root/autodl-tmp/LNDbv4 \
  --out-dir /root/autodl-tmp/LNDbv4_monai_luna16_aligned \
  --min-agr-level 2
```

转换脚本会生成：

- 重采样后的 CT 图像。
- MONAI datalist JSON。
- `summary.json`。
- 数据准备说明文件。

当前目标 spacing：

```text
0.703125 x 0.703125 x 1.25 mm
```

检测框格式：

```text
cccwhd = [center_x, center_y, center_z, width, height, depth]
```

类别映射：

```text
nodule -> 0
```

## 训练配置

当前保留的主要训练配置：

```text
hf_models/MONAI_lung_nodule_ct_detection/configs/train_lndbv4_finetune_fold0_agr2_turbo_100ep.json
```

核心参数：

| 参数 | 值 |
|---|---:|
| epochs | 100 |
| val_interval | 5 |
| learning_rate | 0.001 |
| batch_size | 8 |
| num_train_workers | 8 |
| num_val_workers | 4 |
| LR schedule | warmup 3 epochs + StepLR |
| StepLR step_size | 30 |
| StepLR gamma | 0.1 |

训练集和验证集：

```text
training   = Fold1 + Fold2 + Fold3
validation = Fold0
```

标注过滤：

```text
Nodule = 1
AgrLevel >= 2
```

## 运行训练

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
./run_finetune_lndbv4_fold0_agr2_turbo_100ep.sh
```

训练输出默认保存到：

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/
```

权重命名规则：

```text
原始预训练权重: hf_models/MONAI_lung_nodule_ct_detection/models/model.pt
训练最优权重:   LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/models/best.pt
训练最终权重:   LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/models/last.pt
```

训练脚本支持通过环境变量覆盖权重文件名：

```bash
PRETRAINED_CKPT_FILENAME=model.pt \
BEST_CKPT_FILENAME=best.pt \
LAST_CKPT_FILENAME=last.pt \
./run_finetune_lndbv4_fold0_agr2_turbo_100ep.sh
```

## 运行评估

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
./run_evaluate_lndbv4_finetuned_fold0_agr2_turbo_100ep.sh
```

默认评估 `best.pt`。如果要评估最终权重：

```bash
EVAL_CKPT_FILENAME=last.pt ./run_evaluate_lndbv4_finetuned_fold0_agr2_turbo_100ep.sh
```

评估结果默认保存到：

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/eval_best/
```

## 当前实验结果

Fold0 验证集，`AgrLevel >= 2`：

| 模型 | AP@IoU0.10 | mAP@IoU0.10:0.50 | AR@IoU0.10 | mAR@IoU0.10:0.50 |
|---|---:|---:|---:|---:|
| MONAI LUNA16 预训练模型 | 0.4908 | 0.4892 | 0.9815 | 0.9774 |
| LNDbv4 微调 10 epoch | 0.5295 | 0.5288 | 0.9815 | 0.9794 |
| LNDbv4 微调 100 epoch | 0.5530 | 0.5511 | 0.9815 | 0.9794 |

100 epoch 微调相比直接使用 LUNA16 预训练模型，AP/mAP 有稳定提升；AR 变化较小，说明预训练模型召回已经较高，微调主要改善了预测排序或误报控制。

## GitHub 上传建议

建议通过 `.gitignore` 排除以下内容：

```gitignore
# datasets
LNDbv4/
datasets/
LNDbv4_monai_luna16_aligned/images/

# checkpoints and outputs
*.pt
*.pth
*.ckpt
*.ts
*.onnx
*.npy
*.raw
*.mhd
*.rar
*.zip
*.db
*.log
events.out.tfevents*
LNDbv4_monai_luna16_aligned/monai_finetune_*/
LNDbv4_monai_luna16_aligned/monai_eval_*/

# local/cache files
__pycache__/
.ipynb_checkpoints/
.autodl/
.Trash-0/
```

如果希望公开可复现实验，建议保留：

- 数据转换脚本。
- datalist JSON。
- 训练和评估配置。
- 轻量级 `metrics.csv` 或整理后的实验结果表。
- 数据下载与放置路径说明。

## 注意事项

- 当前配置中的路径是本地绝对路径，例如 `/root/autodl-tmp/...`。在其他机器运行前，需要同步修改 JSON 配置和脚本中的路径。
- LNDbv4 原始数据和 LUNA16 数据各自有独立的数据使用协议，上传仓库时不要包含未获授权的数据文件。
- MONAI bundle 和预训练权重可能也有单独许可证要求，公开仓库时建议在文档中说明来源和许可证。
