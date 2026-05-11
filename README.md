# LNDbv4 肺结节检测：MONAI LUNA16 迁移微调实验

本项目基于 **MONAI Lung Nodule CT Detection Bundle**，将 LNDbv4 肺结节 CT 数据整理为接近 LUNA16 bundle 的输入格式，并在 LNDbv4 Fold0 验证集上进行迁移微调与评估。

仓库只建议保存代码、配置、说明文档和轻量级实验记录；**不建议提交 LNDbv4、LUNA16 原始数据、重采样影像、模型权重和训练日志大文件**。

## 项目内容

- 整理 LNDbv4 数据格式，并说明其与 LUNA16 的差异。
- 将 LNDbv4 的 `.mhd/.raw` CT 和 `trainNodules_gt.csv` 标注转换为 MONAI detection datalist。
- 按 MONAI LUNA16 bundle 的目标 spacing 对 CT 进行重采样。
- 使用 MONAI LUNA16 预训练 RetinaNet 模型在 LNDbv4 上微调。
- 保留当前主要实验配置：`Fold1 + Fold2 + Fold3` 训练，`Fold0` 验证，`AgrLevel >= 2`，训练 `100 epoch`。
- 在评估流程中加入 FROC 指标，统计 CPM 和不同 FP/scan 下的 sensitivity。
- 从 LNDbv4 `allNods.csv` 构造文本条件检测 datalist，用于后续 report/query-conditioned nodule detection。

## 推荐仓库结构

```text
.
├── README.md
├── data.md
├── prepare_lndbv4_for_monai_luna16.py
├── create_lndbv4_text_datalist.py
├── create_lndbv4_text_ablation_datalists.py
├── check_lndbv4_text_spatial_features.py
├── TEXT_CONDITIONED_DETECTION_PLAN.md
├── LNDbv4_monai_luna16_aligned/
│   ├── README_preparation.md
│   ├── EXPERIMENT_LOG.md
│   ├── summary.json
│   ├── lndbv4_train_folds1-3_test_fold0.json
│   ├── lndbv4_train_folds1-3_test_fold0_with_text.json
│   ├── lndbv4_train_folds1-3_test_fold0_query_text.json
│   ├── lndbv4_train_folds1-3_test_fold0_query_image_only.json
│   ├── lndbv4_train_folds1-3_test_fold0_query_text_generic_val.json
│   ├── lndbv4_train_folds1-3_test_fold0_query_text_shuffled_val.json
│   ├── lndbv4_labeled_train_as_validation.json
│   └── lndbv4_test_as_validation.json
└── hf_models/
    └── MONAI_lung_nodule_ct_detection/
        ├── configs/
        │   ├── train.json
        │   ├── evaluate.json
        │   ├── train_lndbv4_finetune_fold0_agr2_turbo_100ep.json
        │   ├── train_lndbv4_text_mlp_fold0_agr2_stage1.json
        │   └── evaluate_lndbv4_finetuned_fold0_agr2_turbo_100ep.json
        ├── scripts/
        ├── run_finetune_lndbv4_fold0_agr2_turbo_100ep.sh
        ├── run_train_lndbv4_text_mlp_fold0_agr2_stage1.sh
        └── run_evaluate_lndbv4_finetuned_fold0_agr2_turbo_100ep.sh
```

本地工作目录中可能还有以下大文件目录，但它们不应进入 GitHub：

```text
LNDbv4/
LNDbv4_monai_luna16_aligned/images/
LNDbv4_monai_luna16_aligned/monai_finetune_*/
LNDbv4_monai_luna16_aligned/monai_text_mlp_*/
LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_*/
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
- `allNods.csv`：影像标注、报告文本实体和二者匹配后的综合表。
- `chars_trainNodules.csv`：医生级结节特征评分。

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

## 文本条件数据生成

LNDbv4 提供结构化报告抽取和影像-文本匹配信息，可用于文本辅助检测。当前使用 `allNods.csv` 生成两份文本 datalist：

```bash
python create_lndbv4_text_datalist.py
```

输出文件：

```text
LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_with_text.json
LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_text.json
```

文本构造策略：

- 只把 `Where == TextReport+RadAnnotation` 作为可靠图文配对，标记为 `text_valid = 1`。
- 无可靠文本的结节使用通用 prompt：`A pulmonary nodule.`，标记为 `text_valid = 0`。
- 使用字段包括 `TextInstanceID`, `TextQuestion`, `Pos_Text`, `Diam_Text`, `NodType`, `Caract_Text` 和报告肺叶位置。
- 不把 `x/y/z` 坐标写入文本，避免直接泄漏定位答案。
- 原始 `image/box/label` 与 `lndbv4_train_folds1-3_test_fold0.json` 保持一致。

两种 datalist 格式：

| 文件 | 用途 |
|---|---|
| `*_with_text.json` | 保留当前一张 CT 多个框的检测格式，并为每个 box 增加 `text`, `text_valid`, `text_source`, `finding_id` |
| `*_query_text.json` | Query-conditioned 格式，每条样本是一条 `query_text` 对应一个 GT box |

当前文本覆盖情况：

| split | CT 数 | box 数 | 可靠文本 | generic 文本 |
|---|---:|---:|---:|---:|
| training | 177 | 167 | 120 | 47 |
| validation/Fold0 | 59 | 54 | 26 | 28 |
| test/Fold0 | 59 | 54 | 26 | 28 |

示例 prompt：

```text
A 8.5 mm nodule in the apical region of the right upper lobe with margin score 1.
A possible nodule in the left upper lobe with calcification score 5.
A 6 mm micronodule in the superior justapleural region of the left lower lobe.
```

文本条件检测的整体技术路线见 `TEXT_CONDITIONED_DETECTION_PLAN.md`。首版建议优先使用 `*_query_text.json` 做 query-conditioned detection，并与 image-only、generic text、shuffled text 对照。

消融验证 datalist 可用以下脚本生成：

```bash
python create_lndbv4_text_ablation_datalists.py
```

输出：

```text
LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_text_generic_val.json
LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_text_shuffled_val.json
```

## MLP 文本编码器

当前已实现首版轻量结构化 MLP 文本编码器：

```text
text_fields + patch spatial encoding -> 55 维结构化特征 -> MLP -> 128 维 text embedding
```

代码入口：

```text
hf_models/MONAI_lung_nodule_ct_detection/scripts/text_conditioning.py
```

核心组件：

| 组件 | 作用 |
|---|---|
| `encode_text_fields` | 将 `loc`, `position`, `diameter_mm`, `nodule_type`, `characteristics`, `uncertainty` 编码为固定长度向量 |
| `encode_patch_spatial_features` | 从 `MetaTensor.affine/original_affine/spatial_shape` 编码 patch 在原 CT 中的归一化中心和覆盖范围 |
| `StructuredTextMLPEncoder` | 结构化文本特征的 MLP 编码器 |
| `TextConditionedRetinaNet` | 在 RetinaNet classification branch 上使用 FiLM 文本调制 |
| `TextConditionedRetinaNetDetector` | 支持 `text_features` 参数传入 detector |
| `freeze_for_text_stage1` | Stage 1 冻结 image backbone/FPN/regression branch，只训练文本编码器、FiLM 和分类头 |

训练配置：

```text
hf_models/MONAI_lung_nodule_ct_detection/configs/train_lndbv4_text_mlp_fold0_agr2_stage1.json
```

该配置默认：

- 使用 `lndbv4_train_folds1-3_test_fold0_query_text.json`。
- 从 `monai_finetune_fold0_agr2_turbo_100ep/models/best.pt` 初始化图像检测权重。
- 非 strict 加载 checkpoint，允许新增文本模块参数随机初始化。
- 输出到 `monai_text_mlp_spatial_fold0_agr2_stage1/`，避免和未加入空间编码的旧文本实验混淆。
- 训练 10 epoch，`val_interval = 1`，用于快速验证 MLP + 空间编码是否有效。
- 只调制 RetinaNet classification branch，box regression branch 保持 image-only。

空间编码包含 6 个值：

```text
patch_center_x, patch_center_y, patch_center_z,
patch_extent_x, patch_extent_y, patch_extent_z
```

其中 center 按原 CT index space 归一化到约 `[-1, 1]`，extent 表示 patch 覆盖原 CT 尺寸的比例。这样文本里的 `right upper lobe`, `apical`, `peripheral` 等位置描述可以和当前 patch 的空间位置一起进入 MLP。

训练前可运行以下脚本抽样检查文本字段和 patch 空间编码：

```bash
python check_lndbv4_text_spatial_features.py
```

运行：

```bash
cd /root/autodl-tmp/hf_models/MONAI_lung_nodule_ct_detection
./run_train_lndbv4_text_mlp_fold0_agr2_stage1.sh
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

重新评估 100 epoch 微调模型 `best.pt` 后，FROC 结果如下：

| 指标 | 值 |
|---|---:|
| froc_cpm | 0.6931 |
| froc_iou_threshold | 0.1 |
| froc_num_scans | 59 |
| froc_num_gt | 54 |
| froc_num_predictions | 1177 |
| sensitivity @ 0.125 FP/scan | 0.3657 |
| sensitivity @ 0.25 FP/scan | 0.4398 |
| sensitivity @ 0.5 FP/scan | 0.5278 |
| sensitivity @ 1 FP/scan | 0.6852 |
| sensitivity @ 2 FP/scan | 0.8889 |
| sensitivity @ 4 FP/scan | 0.9630 |
| sensitivity @ 8 FP/scan | 0.9815 |

### Query-Level 文本条件实验

文本条件实验使用 `lndbv4_train_folds1-3_test_fold0_query_text.json`，每条样本是一条 `query_text` 对应一个 GT box。为了公平比较，image-only baseline 也使用同一 query-level 验证协议，文本字段被移除后保存为：

```text
LNDbv4_monai_luna16_aligned/lndbv4_train_folds1-3_test_fold0_query_image_only.json
```

Query-level 验证集：

```text
validation queries = 54
reliable text      = 26
generic text       = 28
```

结果：

| 模型 | AP@IoU0.10 | mAP@IoU0.10:0.50 | AR@IoU0.10 | mAR@IoU0.10:0.50 | FROC CPM | predictions |
|---|---:|---:|---:|---:|---:|---:|
| Image-only query baseline | 0.3635 | 0.3623 | 0.9815 | 0.9794 | 0.5185 | 1206 |
| MLP text, no patch spatial | 0.3669 | 0.3657 | 0.9815 | 0.9794 | 0.5265 | 1404 |
| MLP text + patch spatial, 10 epoch | 0.3556 | 0.3540 | 0.9815 | 0.9794 | 0.5364 | 1321 |

对应 metrics：

```text
LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/eval_query_image_only_best/metrics.csv
LNDbv4_monai_luna16_aligned/monai_text_mlp_fold0_agr2_stage1/eval_best/metrics.csv
LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_fold0_agr2_stage1/eval_best/metrics.csv
```

阶段性结论：

- no-spatial MLP 相比 image-only query baseline，AP/mAP 和 FROC CPM 都有小幅提升。
- 加入 patch spatial 后，FROC CPM 进一步提升，但 AP/mAP 下降。
- 当前结果说明空间编码可能改善低 FP 区间排序，但整体检测精度尚未稳定提升。
- CT-level 指标和 query-level 指标不可直接横向比较；前者是一张 CT 找全部结节，后者是一条文本 query 对应一个目标结节。

### Real / Generic / Shuffled Text 消融

消融目的：验证模型是否真正利用真实文本语义，而不是只受文本分支、空间编码或 score calibration 影响。

消融设置：

| 验证文本 | 定义 |
|---|---|
| Real text | 使用原始 `query_text` 和 `text_fields` |
| Generic text | 54 条验证 query 全部替换为 `A pulmonary nodule.`，结构化文本字段置空 |
| Shuffled text | 26 条 reliable-text 验证 query 的文本字段循环打乱，28 条 generic query 保持 generic |

同一个 spatial text-MLP `best.pt` 在三种验证文本上的结果：

| Validation text | AP@IoU0.10 | mAP@IoU0.10:0.50 | AR@IoU0.10 | mAR@IoU0.10:0.50 | FROC CPM | predictions |
|---|---:|---:|---:|---:|---:|---:|
| Real text | 0.3556 | 0.3540 | 0.9815 | 0.9794 | 0.5364 | 1321 |
| Generic text | 0.3640 | 0.3624 | 0.9815 | 0.9794 | 0.5344 | 1333 |
| Shuffled text | 0.3641 | 0.3623 | 0.9815 | 0.9774 | 0.5397 | 1330 |

消融结论：

- 当前 spatial text-MLP `best.pt` 尚未证明模型利用了真实文本语义。
- Generic/Shuffled 的 AP/mAP 不低于 Real text，Shuffled 的 FROC CPM 最高。
- 当前收益更可能来自文本条件结构、训练方式、空间特征或分数校准，而不是可靠的文本-query grounding。
- 下一步若继续推进文本辅助检测，应优先做更严格的 real/generic/shuffled 分别训练，或增强 query-level 负样本设计。

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
LNDbv4_monai_luna16_aligned/monai_text_mlp_*/
LNDbv4_monai_luna16_aligned/monai_text_mlp_spatial_*/
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
