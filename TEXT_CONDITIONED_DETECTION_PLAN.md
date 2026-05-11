# LNDbv4 文本条件肺结节检测技术路线

本文档记录后续在当前 MONAI LUNA16 RetinaNet 检测实验基础上，引入 LNDbv4 结节描述文本模块的技术路线。目标方案是 **方案 2：文本作为模型输入**，即进行 report-conditioned / query-conditioned nodule detection。

## 1. 目标定义

当前基线模型：

```text
CT volume -> RetinaNet -> nodule boxes
```

计划改造为：

```text
CT volume + nodule/report text -> text-conditioned RetinaNet -> nodule boxes
```

文本用于直接调制检测模型，使模型根据报告中的结节描述增强定位和分类能力。

需要明确的是，该方案推理阶段也需要文本输入，因此它回答的问题是：

```text
当给定 CT 和报告/结节描述时，文本是否能辅助模型更准确地检测对应结节？
```

它不同于纯 CT 检测模型，也需要和 image-only baseline 分开评估。

## 2. 可用文本数据

LNDbv4 中与文本相关的主要文件：

```text
/root/autodl-tmp/LNDbv4/report.csv
/root/autodl-tmp/LNDbv4/allNods.csv
/root/autodl-tmp/LNDbv4/trainNodules_gt.csv
/root/autodl-tmp/LNDbv4/chars_trainNodules.csv
/root/autodl-tmp/LNDbv4/text2Fleischner.csv
```

### 2.1 report.csv

`report.csv` 是从医学报告中抽取出的结构化结果。每个 CT 最多有 6 个报告结节实例：

```text
num_report,
loc_0, unc_0, rem_0,
loc_1, unc_1, rem_1,
...
loc_5, unc_5, rem_5
```

字段含义：

| 字段 | 含义 | 示例 |
|---|---|---|
| `loc_i` | 肺叶或肺区位置 | `RUL`, `LUL`, `RLL`, `LLL`, `ML` |
| `unc_i` | 不确定性问题 | `how many?`, `is it?` |
| `rem_i` | 半结构化结节描述 | `apical,8.5,nod,margin: 1` |

`rem_i` 通常可拆成：

```text
position, diameter, nodule_type, characteristics
```

### 2.2 allNods.csv

`allNods.csv` 是首选数据源，因为它整合了影像人工标注、报告文本和二者的匹配关系。

关键字段：

| 字段 | 含义 |
|---|---|
| `LNDbID` | CT ID |
| `FindingID` | 合并后的结节 ID |
| `x, y, z` | 结节中心世界坐标 |
| `DiamEq_Rad` | 影像标注等效直径 |
| `Texture` | 纹理评分 |
| `Lobe` | 肺叶编号 |
| `TextInstanceID` | 报告中的结节实例编号 |
| `TextQuestion` | 文本抽取时的不确定性问题 |
| `Pos_Text` | 报告描述的位置 |
| `Diam_Text` | 报告描述的直径 |
| `NodType` | 报告描述的结节类型 |
| `Caract_Text` | 报告描述的结节特征 |
| `Where` | 该结节来自人工标注、文本报告，或两者匹配 |

常见 `Where` 取值：

```text
RadAnnotation
TextReport+RadAnnotation
TextReport+RadAnnotation (created: texture)
TextReport
TextReport (created: size)
TextReport (created: texture)
TextReport (created: size, texture)
```

首版训练建议只把 `TextReport+RadAnnotation` 作为可靠图文配对。带 `created:*` 的样本可作为扩展实验使用。

## 3. 当前文本覆盖情况

基于当前 Fold1+2+3 训练、Fold0 验证、`AgrLevel >= 2`、`Nodule = 1` 的检测设置：

| 项目 | 数量 |
|---|---:|
| 当前检测 GT 框总数 | 221 |
| 训练集 GT 框 | 167 |
| 验证集 GT 框 | 54 |
| 训练集中有可靠文本匹配的框 | 120 |
| 验证集中有可靠文本匹配的框 | 26 |

因此文本覆盖率可用，但不完整。模型和 datalist 需要支持部分结节没有文本描述的情况。

## 4. 文本构造方式

推荐从 `allNods.csv` 为每个匹配结节构造一条 prompt。

### 4.1 结构化 prompt

示例：

```text
lobe 3; position apical; diameter 8.5 mm; type nodule; margin score 1
```

优点是字段稳定，解析简单，适合小数据量。

### 4.2 自然语言 prompt

示例：

```text
A 8.5 mm nodule in the apical region of lobe 3 with margin score 1.
```

或使用更医学化表达：

```text
Pulmonary nodule, lobe 3, apical position, 8.5 mm, margin score 1.
```

首版建议使用英文结构化自然句，方便接入 Bio_ClinicalBERT / ClinicalBERT。

### 4.3 缺失文本处理

对于没有可靠文本匹配的 GT 结节，可选策略：

| 策略 | 做法 | 建议 |
|---|---|---|
| 通用 prompt | 使用 `a pulmonary nodule` | 首版推荐 |
| 跳过文本融合 | 只参与 image-only detection loss | 可做对照 |
| 从影像属性生成 prompt | 使用 `Text`, `Volume`, `chars_trainNodules.csv` 生成弱文本 | 后续扩展 |

首版建议：

```text
有可靠文本匹配 -> 使用 allNods.csv 构造具体 prompt
无可靠文本匹配 -> 使用 generic prompt: "a pulmonary nodule"
```

## 5. Datalist 扩展设计

当前 datalist 格式：

```json
{
  "image": "images/LNDb-0001.mhd",
  "box": [[...]],
  "label": [0]
}
```

建议扩展为：

```json
{
  "image": "images/LNDb-0001.mhd",
  "box": [[...]],
  "label": [0],
  "text": [
    "A 8.5 mm nodule in the apical region of lobe 3 with margin score 1."
  ],
  "text_source": [
    "TextReport+RadAnnotation"
  ],
  "text_valid": [
    1
  ],
  "finding_id": [
    1
  ]
}
```

对于一个 CT 有多个结节的情况：

```json
{
  "image": "images/LNDb-0002.mhd",
  "box": [[...], [...], [...]],
  "label": [0, 0, 0],
  "text": [
    "A pulmonary nodule in lobe 1 with calcification score 5.",
    "A pulmonary nodule in lobe 1 with calcification score 5.",
    "A pulmonary nodule."
  ],
  "text_source": [
    "TextReport+RadAnnotation",
    "TextReport+RadAnnotation",
    "generic"
  ],
  "text_valid": [
    1,
    1,
    0
  ],
  "finding_id": [
    1,
    2,
    3
  ]
}
```

注意：如果训练设计为 query-conditioned detection，则每条样本也可以改造成“一条文本 query 对应一个 CT 和目标框”的形式：

```json
{
  "image": "images/LNDb-0001.mhd",
  "query_text": "A 8.5 mm nodule in the apical region of lobe 3 with margin score 1.",
  "box": [[...]],
  "label": [0],
  "query_finding_id": 1
}
```

首版建议采用 query-conditioned 格式，监督关系更清楚。

## 6. 模型结构路线

当前 MONAI bundle 模型：

```text
3D ResNet50 backbone -> FPN -> RetinaNet classification/regression heads
```

文本条件模型建议：

```text
text prompt -> text encoder -> text embedding
CT volume -> 3D ResNet50 -> FPN features
FPN features + text embedding -> conditioned classification head
FPN features -> box regression head
```

### 6.1 文本编码器

首版可选两种实现。

#### 轻量结构化编码器

将结构化字段编码为 embedding / 数值特征：

```text
Lobe embedding
NodType embedding
Diam_Text normalized value
Caract_Text parsed attribute vector
Pos_Text embedding
```

再通过 MLP 得到 text embedding。

优点：

```text
训练稳定
参数少
适合小数据
容易解释
```

#### BERT 文本编码器

使用本地已有模型：

```text
/root/autodl-tmp/hf_models/Bio_ClinicalBERT
```

流程：

```text
prompt -> tokenizer -> Bio_ClinicalBERT -> [CLS] embedding -> projection MLP
```

首版建议冻结 BERT，只训练 projection MLP 和融合模块。数据量较小，不建议一开始端到端微调 BERT。

### 6.2 融合模块

首版推荐使用 FiLM 调制分类 head 特征：

```text
gamma, beta = MLP(text_embedding)
conditioned_feature = gamma * image_feature + beta
```

然后将 `conditioned_feature` 输入 RetinaNet classification head。

推荐只调制分类 head，不调制 box regression head：

```text
classification branch: text-conditioned
box regression branch: image-only
```

原因：

```text
文本描述对“是否是该类/该 query 对应结节”更直接
定位回归对图像几何稳定性要求高，首版避免文本干扰 box regression
```

后续可扩展：

```text
FPN-level FiLM
cross-attention between text token embeddings and image features
text-conditioned proposal / anchor scoring
```

## 7. 训练策略

### 7.1 初始化

建议从当前微调好的检测模型初始化：

```text
/root/autodl-tmp/LNDbv4_monai_luna16_aligned/monai_finetune_fold0_agr2_turbo_100ep/models/best.pt
```

这样文本模块是在已有强检测模型基础上学习条件调制，而不是从头训练。

### 7.2 分阶段训练

建议分 3 个阶段：

```text
Stage 1:
  冻结 backbone、FPN、box head
  训练 text encoder projection、fusion module、classification head

Stage 2:
  解冻 FPN 和 classification head
  小学习率微调

Stage 3:
  端到端小学习率微调
  BERT 仍建议冻结，除非验证集显示明显欠拟合
```

### 7.3 Loss

首版保持检测 loss：

```text
L = L_retinanet_cls + L_retinanet_box
```

后续可增加图文匹配辅助 loss：

```text
L_total = L_det + lambda * L_image_text_match
```

图文匹配 loss 可以基于结节 crop embedding 和 text embedding 做 InfoNCE，但这属于第二阶段扩展，不建议首版就加入。

## 8. 推理方式

有两种实验设定。

### 8.1 Query-conditioned detection

每次输入一条文本 query：

```text
CT volume + "A 8.5 mm nodule in the apical region of lobe 3 with margin score 1."
```

输出与该 query 对应的结节框。

优点：

```text
一条文本对应一个或少数目标
监督关系明确
适合证明模型是否利用文本
```

首版推荐这个设定。

### 8.2 CT-level report-conditioned detection

一次输入整份 CT 的多个结节描述：

```text
CT volume + [query_1, query_2, ..., query_k]
```

输出该报告中所有结节框。

这种形式更接近真实临床报告辅助检测，但模型和后处理更复杂，建议在 query-conditioned 实验稳定后再做。

## 9. 评估设计

必须和 image-only baseline 及文本消融实验对比。

### 9.1 对照组

| 实验 | 输入 | 目的 |
|---|---|---|
| Image-only baseline | CT | 当前 RetinaNet 基线 |
| Generic text | CT + `a pulmonary nodule` | 判断文本模块本身是否带来结构性收益 |
| Real text | CT + 真实匹配文本 | 主实验 |
| Shuffled text | CT + 打乱到其他结节的文本 | 检查模型是否真正利用文本 |
| Missing-text subset | CT + generic text | 评估文本缺失情况下性能 |

### 9.2 指标

沿用当前检测指标：

```text
AP_IoU_0.10_MaxDet_100
mAP_IoU_0.10_0.50_MaxDet_100
AR_IoU_0.10_MaxDet_100
mAR_IoU_0.10_0.50_MaxDet_100
```

建议额外统计：

```text
matched-text subset AP
generic-text subset AP
false positives per scan
query hit rate
```

其中 `query hit rate` 可定义为：

```text
给定一个 query，top-k 预测框中是否命中该 query 对应 GT 框。
```

### 9.3 关键判据

如果出现以下结果，说明文本模块有效：

```text
Real text > Generic text
Real text > Shuffled text
Real text 在 matched-text subset 上提升明显
Shuffled text 不应稳定提升
```

如果 `Real text` 和 `Shuffled text` 都提升，说明模型可能只是受额外参数或训练策略影响，并未真正使用文本语义。

## 10. 首版最小可行实现

建议第一版按以下范围实现：

```text
1. 从 allNods.csv 构造 query-conditioned datalist
2. 只使用 TextReport+RadAnnotation 样本作为可靠 query
3. 无可靠文本的结节使用 generic prompt
4. 使用 Bio_ClinicalBERT 冻结编码文本，或先用结构化 MLP 编码器
5. 使用 FiLM 调制 RetinaNet classification head
6. box regression head 保持 image-only
7. 从现有 best.pt 初始化
8. 先冻结 backbone/FPN，只训练文本投影和分类融合模块
9. 对比 image-only、generic text、real text、shuffled text
```

首版不建议加入：

```text
cross-attention
端到端微调 BERT
文本生成 pseudo label
CT-level 多 query 联合推理
```

这些可以作为第二阶段扩展。

## 11. 可能风险

### 11.1 文本泄漏

报告文本可能包含位置和大小信息，例如：

```text
RUL apical 8.5 mm nodule
```

如果评估目标是纯 CT 自动检测，则这属于额外信息，不能和 image-only 模型直接公平比较。应明确该实验是 report-assisted detection。

### 11.2 文本覆盖不完整

当前 Fold0 验证集中只有部分 GT 框有可靠文本匹配。需要同时报告：

```text
全验证集指标
有可靠文本匹配子集指标
无文本或 generic prompt 子集指标
```

### 11.3 数据量较小

可靠图文配对样本数量有限，复杂模型容易过拟合。首版应控制参数量，优先冻结大模型。

### 11.4 多结节 CT 的 query 歧义

同一 CT 可能多个结节共享相同描述，例如同一肺叶多个 `nod` 或 `micro`。Query-conditioned 训练时需要允许一个 query 对应多个可能目标，或在 datalist 中保留一对一匹配的 `FindingID` 用于分析。

## 12. 后续任务清单

建议按以下顺序推进：

```text
1. 编写 datalist 扩展脚本，生成 text-conditioned JSON
2. 检查每个 box 与 prompt 的匹配关系
3. 先实现结构化 MLP 文本编码器版本
4. 改造 RetinaNet classification head 的 FiLM 融合
5. 复用当前 best.pt 初始化训练
6. 跑 generic text / real text / shuffled text 三组实验
7. 若 real text 明显有效，再接入 Bio_ClinicalBERT
8. 补充 matched-text subset 评估脚本
9. 视结果扩展到 CT-level 多 query 检测
```

