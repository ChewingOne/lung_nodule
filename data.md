# LNDbv4 数据格式说明与 LUNA16 对比

本文档基于 `/root/autodl-tmp/LNDbv4` 目录下的实际文件结构整理，说明 LNDbv4 的数据格式、与肺结节领域常用数据集 LUNA16 的差异，以及这类 3D CT 肺结节数据的常见处理办法。

## 1. LNDbv4 数据集概况

本地目录名为 `LNDbv4`，不是 `LNDb4`。该目录总大小约 149 GB，主体由 3D 胸部 CT、医生标注、医学报告结构化信息和若干任务相关 CSV 组成。

主要文件数量如下：

| 类型 | 数量 | 说明 |
|---|---:|---|
| `.mhd` 主 CT | 294 | MetaImage 头文件 |
| `.raw` 主 CT | 294 | 与 `.mhd` 配对的原始体素数据 |
| `.csv` | 16 | CT 划分、结节标注、报告抽取、Fleischner 分类等 |
| `masks/*.mhd` | 481 | 其中 480 个有对应 `.raw`，另有 1 个 checkpoint 文件 |
| `masks/*.raw` | 480 | 医生分割 mask |
| `predictedNodulesB/*.npy` | 625 | 80x80x80 的布尔体块 |
| `.rar/.zip` | 若干 | 原始压缩包、脚本、提交样例等 |

CT 划分：

| 集合 | CT 数量 | 文件 |
|---|---:|---|
| 训练集 | 236 | `trainCTs.csv` |
| 测试集 | 58 | `testCTs.csv` |
| 总计 | 294 | `trainCTs.csv + testCTs.csv` 无重叠，正好覆盖全部 CT |

CT 编号范围为 1 到 312，但编号不连续，中间缺少部分 ID。文件名 `LNDb-0001.mhd` 对应 CSV 中的 `LNDbID = 1`。

## 2. CT 影像格式

LNDbv4 的主 CT 使用 MetaImage 格式，每个病例由一对文件组成：

```text
LNDb-0001.mhd
LNDb-0001.raw
```

`.mhd` 是文本头文件，记录维度、spacing、origin、方向矩阵、数据类型和 `.raw` 文件名。`.raw` 是真实体素数据。

样例：

```text
ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = -158.1962890625 -309.1962890625 -297.5
AnatomicalOrientation = RAI
ElementSpacing = 0.607421875 0.607421875 1
DimSize = 512 512 328
ElementType = MET_SHORT
ElementDataFile = LNDb-0001.raw
```

关键含义：

| 字段 | 含义 |
|---|---|
| `NDims = 3` | 三维 CT 体数据 |
| `DimSize` | 体素维度，顺序通常是 x, y, z |
| `ElementSpacing` | 体素间距，单位 mm，顺序通常是 x, y, z |
| `Offset` | 图像原点的世界坐标 |
| `TransformMatrix` | 图像坐标到世界坐标的方向矩阵 |
| `ElementType = MET_SHORT` | 主 CT 为 16-bit signed short，通常可按 HU 值处理 |
| `ElementDataFile` | 对应的 `.raw` 文件 |

本地检查结果：

| 属性 | 范围 |
|---|---|
| x 维度 | 512 到 804 |
| y 维度 | 512 到 574 |
| z 维度 | 251 到 631 |
| x/y spacing | 约 0.43 到 0.89 mm |
| z spacing | 0.5 到 1.4 mm |
| 主 CT 数据类型 | `MET_SHORT` |

使用 SimpleITK 读取时要注意，`.mhd` 中 `DimSize` 是 x, y, z，但 `sitk.GetArrayFromImage` 得到的 numpy 数组通常是 z, y, x。

示例：

```python
import SimpleITK as sitk

image = sitk.ReadImage("LNDbv4/LNDb-0001.mhd")
array = sitk.GetArrayFromImage(image)  # shape: z, y, x
spacing = image.GetSpacing()           # x, y, z
origin = image.GetOrigin()             # x, y, z
direction = image.GetDirection()
```

## 3. 分割 Mask 格式

医生分割 mask 位于 `LNDbv4/masks`：

```text
masks/LNDb-0001_rad1.mhd
masks/LNDb-0001_rad1.raw
masks/LNDb-0001_rad2.mhd
masks/LNDb-0001_rad2.raw
```

命名规则：

```text
LNDb-XXXX_radR.mhd
```

含义：

| 部分 | 含义 |
|---|---|
| `XXXX` | CT 的 LNDbID |
| `radR` | 第 R 位放射科医生 |

mask 的 `.mhd` 结构与主 CT 类似，但数据类型为：

```text
ElementType = MET_UCHAR
```

即 8-bit unsigned char。mask 不是单纯二值图，而是一个标签图。每个非零体素值表示该医生标注的 finding ID，对应 `trainNodules.csv` 中该医生的 `FindingID`。

例如：

```text
mask voxel value = 0  表示背景
mask voxel value = 1  表示该医生的 finding 1
mask voxel value = 2  表示该医生的 finding 2
```

本地检查结果：

| 项目 | 数量 |
|---|---:|
| 有 mask 的训练 CT | 236 |
| rad1 mask | 236 |
| rad2 mask | 177 |
| rad3 mask | 67 |

测试集没有对应的人工分割 mask。

## 4. 主要 CSV 文件

### 4.1 CT 级文件

| 文件 | 行数 | 说明 |
|---|---:|---|
| `trainCTs.csv` | 236 | 训练 CT ID，包含 `RadN` 字段，表示该 CT 有多少位医生标注 |
| `testCTs.csv` | 58 | 测试 CT ID |
| `LNDbAcqParams.csv` | 294 | 每个 CT 的扫描采集参数 |
| `trainFolds.csv` | 59 | 训练/验证折划分，每列一个 fold |
| `trainFleischner.csv` | 236 | 每个训练 CT 的 Fleischner 分类 |

`LNDbAcqParams.csv` 字段包括：

```text
LNDbID, Modality, Manufacturer, ManufacturerModelName,
SliceThickness, KVP, DataCollectionDiameter,
ReconstructionDiameter, DistanceSourceToDetector,
DistanceSourceToPatient, GantryDetectorTilt, TableHeight,
RotationDirection, ExposureTime, XRayTubeCurrent, Exposure,
FilterType, ConvolutionKernel, SamplesPerPixel,
PhotometricInterpretation, PixelSpacing
```

### 4.2 结节标注文件

`trainNodules.csv` 是医生级标注，每一行对应某个医生标注的一个 finding：

```text
LNDbID, RadID, FindingID, x, y, z, Nodule, Volume, Text
```

字段含义：

| 字段 | 含义 |
|---|---|
| `LNDbID` | CT ID |
| `RadID` | 放射科医生 ID |
| `FindingID` | 该医生在该 CT 中的 finding 编号 |
| `x, y, z` | 世界坐标，单位 mm |
| `Nodule` | 1 表示结节，0 表示非结节 |
| `Volume` | 分割体积，单位通常按 mm3 理解 |
| `Text` | 纹理评分，后续可转为 GGO / part-solid / solid |

`trainNodules_gt.csv` 是合并多医生标注后的 ground truth：

```text
LNDbID, RadID, RadFindingID, FindingID, x, y, z,
AgrLevel, Nodule, Volume, Text
```

其中：

| 字段 | 含义 |
|---|---|
| `RadID` | 参与该合并 finding 的医生列表，如 `1,2,3` |
| `RadFindingID` | 对应医生原始 finding ID 列表 |
| `FindingID` | 合并后的 finding ID |
| `AgrLevel` | 有多少位医生认为这是结节 |
| `Nodule` | 合并后的结节/非结节标签 |
| `Volume` | 多医生分割体积的平均值 |
| `Text` | 多医生纹理评分的平均值 |

本地统计：

| 文件 | 总行数 | 结节 | 非结节 |
|---|---:|---:|---:|
| `trainNodules.csv` | 1527 | 1033 | 494 |
| `trainNodules_gt.csv` | 1219 | 768 | 451 |

### 4.3 结节特征文件

`chars_trainNodules.csv` 包含医生级结节特征评分：

```text
LNDbID, RadID, FindingID,
calcification, internalStructure, lobulation, malignancy,
margin, sphericity, spiculation, subtlety
```

这些特征常用于结节性质分析、恶性度预测或与报告文本描述对齐。

### 4.4 医学报告相关文件

`report.csv` 是从医学报告中抽取出的结构化结果。一个 CT 最多有 6 个报告中提到的结节实例：

```text
num_report,
loc_0, unc_0, rem_0,
loc_1, unc_1, rem_1,
...
loc_5, unc_5, rem_5
```

字段含义：

| 字段 | 含义 |
|---|---|
| `num_report` | CT / 报告 ID |
| `loc_i` | 结节位置 |
| `unc_i` | 不确定性参数，例如 `how many?` |
| `rem_i` | 剩余属性，包含二级位置、大小、类型、特征等 |

`allNods.csv` 是影像标注与报告文本匹配后的综合表，包含所有被识别出的结节：

```text
LNDbID, RadID, RadFinding, FindingID, Nodule,
x, y, z, DiamEq_Rad,
Texture, Calcification, InternalStructure, Lobulation,
Malignancy, Margin, Sphericity, Spiculation, Subtlety,
Lobe, TextInstanceID, TextQuestion, Pos_Text, Diam_Text,
NodType, Caract_Text, Where
```

其中 `Where` 表示该结节来自哪里：

| `Where` 示例 | 含义 |
|---|---|
| `RadAnnotation` | 只在影像人工标注中出现 |
| `TextReport` | 只在医学报告中出现 |
| `TextReport+RadAnnotation` | 报告和影像标注均匹配到 |
| `TextReport (created: size)` | 报告中缺部分信息，由规则补全大小 |
| `TextReport (created: texture)` | 报告中缺部分信息，由规则补全纹理 |

本地 `allNods.csv` 中 `Where` 分布：

| 来源 | 数量 |
|---|---:|
| `RadAnnotation` | 732 |
| `TextReport+RadAnnotation` | 377 |
| `TextReport+RadAnnotation (created: texture)` | 43 |
| `TextReport (created: texture)` | 47 |
| `TextReport (created: size, texture)` | 17 |
| `TextReport` | 12 |
| `TextReport (created: size)` | 6 |

### 4.5 Fleischner 相关文件

`rad2Fleischner.csv` 和 `text2Fleischner.csv` 用于根据影像标注或报告文本计算 Fleischner 随访建议：

```text
LNDbID, FindingID, Nodule, Volume, Text, Where
```

`trainFleischner.csv` 给出训练 CT 的最终 Fleischner 类别：

```text
LNDbID, Fleischner
```

类别为 0 到 3。代码 `calcFleischner.py` 中的规则主要考虑：

| 因素 | 类别 |
|---|---|
| 结节数量 | 无结节、单发、多发 |
| 体积 | `<100 mm3`, `100-250 mm3`, `>=250 mm3` |
| 纹理 | GGO、part-solid、solid |

本地 `trainFleischner.csv` 分布：

| Fleischner 类别 | 数量 |
|---|---:|
| 0 | 112 |
| 1 | 4 |
| 2 | 60 |
| 3 | 60 |

## 5. 坐标系统

LNDbv4 的 CSV 中 `x, y, z` 为世界坐标，单位 mm，不是 numpy 数组下标。

世界坐标转图像体素坐标的一般流程：

```python
import numpy as np
import SimpleITK as sitk

image = sitk.ReadImage("LNDbv4/LNDb-0001.mhd")
origin = np.array(image.GetOrigin())
spacing = np.array(image.GetSpacing())
direction = np.array(image.GetDirection()).reshape(3, 3)

world = np.array([x, y, z])
voxel_xyz = np.linalg.inv(direction @ np.diag(spacing)) @ (world - origin)
voxel_xyz = np.round(voxel_xyz).astype(int)

array = sitk.GetArrayFromImage(image)
z, y, x = voxel_xyz[2], voxel_xyz[1], voxel_xyz[0]
value = array[z, y, x]
```

如果方向矩阵是单位矩阵，也可以近似写成：

```python
voxel_xyz = np.round((world - origin) / spacing).astype(int)
```

但为了兼容所有 CT，建议始终使用 direction 矩阵。

## 6. 与 LUNA16 的对比

LUNA16 是肺结节检测领域最常用的公开 benchmark 之一，来源于 LIDC-IDRI 的筛选子集。它和 LNDbv4 都是 3D 胸部 CT 数据集，都常用于肺结节检测、候选生成和假阳性抑制，但数据组织和任务侧重点不同。

| 对比项 | LNDbv4 | LUNA16 |
|---|---|---|
| 数据规模 | 294 个 CT | 888 个 CT |
| 数据来源 | LNDb 数据集及其 v4 扩展 | LIDC-IDRI 筛选子集 |
| 影像格式 | `.mhd + .raw` | `.mhd + .raw` |
| CT ID | `LNDbID`，文件名如 `LNDb-0001.mhd` | `seriesuid`，文件名通常是一串 DICOM series UID |
| 官方划分 | 236 train / 58 test，另有 train folds | 10 个 subset：`subset0` 到 `subset9` |
| 结节坐标 | CSV 中 `x, y, z` 世界坐标，单位 mm | `annotations.csv` 中 `coordX, coordY, coordZ` 世界坐标，单位 mm |
| 结节大小 | `Volume`、等效直径、报告直径等 | `diameter_mm` |
| 分割 mask | 提供医生级 3D mask，mask 值对应 finding ID | LUNA16 官方主要提供中心点和直径，不提供完整分割 mask |
| 医生标注 | 1 到 3 位医生，保留医生级 mask 和合并 GT | 来源于 LIDC 多医生标注，LUNA16 常用的是共识后的结节列表 |
| 非结节/候选 | `Nodule=0` 表示非结节 finding | 提供 `candidates.csv` / `candidates_V2.csv`，含候选点与 label |
| 文本报告 | 有 `report.csv`、`allNods.csv`、`text2Fleischner.csv` 等 | 通常没有医学报告文本结构化信息 |
| 临床随访 | 有 Fleischner 分类文件和计算脚本 | 主要面向结节检测，不直接提供 Fleischner 随访标签 |
| 典型任务 | 结节检测、分割、医生一致性、报告-影像匹配、Fleischner 分类 | 结节检测、候选生成、假阳性抑制 |

### 6.1 LUNA16 常见文件

LUNA16 典型目录结构如下：

```text
subset0/
subset1/
...
subset9/
annotations.csv
candidates.csv
candidates_V2.csv
sampleSubmission.csv
```

`annotations.csv` 常见字段：

```text
seriesuid, coordX, coordY, coordZ, diameter_mm
```

`candidates.csv` 常见字段：

```text
seriesuid, coordX, coordY, coordZ, class
```

其中 `class = 1` 表示真结节候选，`class = 0` 表示假阳性候选。

### 6.2 任务差异

LUNA16 更像是标准检测 benchmark：

1. 给定 CT。
2. 检测结节中心点。
3. 根据距离阈值和 FROC 指标评估。
4. 常见流程是候选生成加假阳性抑制。

LNDbv4 信息更丰富：

1. 有 CT。
2. 有医生级结节与非结节 finding。
3. 有 3D 分割 mask。
4. 有合并后的 ground truth。
5. 有医学报告抽取信息。
6. 有报告结节与影像结节的匹配结果。
7. 有 Fleischner 随访分类。

因此 LNDbv4 更适合做多任务研究，例如：

| 任务 | 可用文件 |
|---|---|
| 结节检测 | `trainNodules.csv`, `trainNodules_gt.csv`, CT |
| 结节分割 | `masks/*.mhd`, CT |
| 结节分类 | `chars_trainNodules.csv`, `trainNodules_gt.csv` |
| 影像-报告匹配 | `report.csv`, `allNods.csv` |
| 随访建议分类 | `trainFleischner.csv`, `rad2Fleischner.csv`, `text2Fleischner.csv` |

## 7. 常见处理办法

### 7.1 读取 CT 和 mask

推荐使用 SimpleITK：

```python
import SimpleITK as sitk

def read_mhd(path):
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)  # z, y, x
    spacing = image.GetSpacing()           # x, y, z
    origin = image.GetOrigin()
    direction = image.GetDirection()
    return array, spacing, origin, direction
```

对于 CT：

```python
ct, spacing, origin, direction = read_mhd("LNDbv4/LNDb-0001.mhd")
```

对于 mask：

```python
mask, _, _, _ = read_mhd("LNDbv4/masks/LNDb-0001_rad1.mhd")
```

如果只想取某个 finding 的二值 mask：

```python
finding_id = 2
binary_mask = (mask == finding_id).astype("uint8")
```

### 7.2 HU 截断和归一化

胸部 CT 常用 HU window：

```python
import numpy as np

def normalize_lung_window(ct, low=-1000, high=400):
    ct = np.clip(ct, low, high)
    ct = (ct - low) / (high - low)
    return ct.astype("float32")
```

常见窗口：

| 任务 | HU 范围 |
|---|---|
| 肺实质/结节检测 | `[-1000, 400]` |
| 更宽 CT 归一化 | `[-1200, 600]` |
| 纵隔相关分析 | 可使用更高上限，如 `[-160, 240]`，但肺结节检测较少单独使用 |

### 7.3 重采样到统一 spacing

由于不同 CT 的 spacing 不一致，训练 3D 模型前通常会重采样到统一 spacing。

常见设置：

| 目标 spacing | 说明 |
|---|---|
| `1.0 x 1.0 x 1.0 mm` | 最常见，适合 3D CNN |
| `0.7 x 0.7 x 1.0 mm` | 尽量保持层内分辨率 |
| 原 spacing | 适合某些候选点裁块任务，避免插值影响 |

图像重采样用线性插值，mask 重采样必须用最近邻插值：

```python
import SimpleITK as sitk

def resample_image(image, new_spacing=(1.0, 1.0, 1.0), is_mask=False):
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()

    new_size = [
        int(round(old_size[i] * old_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )

    return resampler.Execute(image)
```

### 7.4 世界坐标与 voxel 坐标转换

LNDbv4 和 LUNA16 的结节坐标都是世界坐标，单位 mm。训练时通常需要转为数组坐标，用于裁剪 patch 或生成标签图。

推荐使用 SimpleITK 内置转换：

```python
import SimpleITK as sitk

image = sitk.ReadImage("LNDbv4/LNDb-0001.mhd")
world = (x, y, z)
voxel_xyz = image.TransformPhysicalPointToIndex(world)
```

然后访问 numpy 数组时要换顺序：

```python
array = sitk.GetArrayFromImage(image)  # z, y, x
x, y, z = voxel_xyz
value = array[z, y, x]
```

### 7.5 提取结节 3D patch

检测或分类常用以结节中心为中心裁剪 3D patch。

常见 patch 大小：

| patch size | 用途 |
|---|---|
| `32^3` | 小结节分类、快速实验 |
| `48^3` | 候选点分类 |
| `64^3` | 结节上下文更充分 |
| `80^3` | LNDbv4 脚本中使用的默认 cube 大小 |
| `96^3` 或更大 | 检测网络或需要更多肺部上下文的任务 |

LNDbv4 自带 `utils.py` 中有 `extractCube` 函数，默认从中心点提取 `80x80x80` 的 cube，并将物理尺寸统一到约 `51 mm`。

### 7.6 生成检测标签

如果训练 3D 检测模型，常见标签形式有三类：

1. 中心点热力图：以结节中心生成 3D Gaussian heatmap。
2. Anchor/box 标签：用中心点和直径/等效直径生成 3D bounding box。
3. 分割监督：直接使用医生 mask 或合并 mask。

LNDbv4 有 mask，因此可以比 LUNA16 更自然地做分割监督。LUNA16 通常只有中心点和直径，所以常见做法是用 `diameter_mm` 近似生成球形 mask 或 bounding box。

### 7.7 肺野分割

肺结节检测前常做肺野分割，目的是减少假阳性和计算量。

常见办法：

| 方法 | 说明 |
|---|---|
| HU 阈值 + 连通域 | 快速、无需训练，常用阈值约 `-600 HU` |
| 形态学处理 | 填洞、开闭运算、去除气管和体外空气 |
| 预训练肺分割模型 | 如 lungmask，效果更稳，适合批量处理 |
| 不显式分割 | 直接让 3D detector 学习，但需要更多数据和算力 |

LNDbv4 的 `help.txt` 中也提到了可使用 lungmask 相关代码。

### 7.8 正负样本构造

肺结节检测数据通常严重类别不平衡。常见策略：

| 样本类型 | 来源 |
|---|---|
| 正样本 | `Nodule=1` 的结节中心 |
| 困难负样本 | `Nodule=0` 的非结节 finding，或模型高分假阳性 |
| 随机负样本 | 肺野内随机采样，避开真实结节 |
| 边界负样本 | 接近血管、胸膜、气道的位置 |

LNDbv4 的优势是 `trainNodules.csv` 和 `trainNodules_gt.csv` 中明确包含 `Nodule=0` 的非结节 finding，可直接作为困难负样本来源。LUNA16 中常用 `candidates.csv` 的 `class=0` 作为负样本。

### 7.9 多医生标注处理

LNDbv4 保留医生级标注和合并后 GT，因此可以有多种标签策略：

| 策略 | 说明 |
|---|---|
| 使用 `trainNodules_gt.csv` | 最简单，直接用合并后的 finding |
| 使用 `AgrLevel >= 2` | 只保留至少两位医生同意的高置信结节 |
| 使用所有 `Nodule=1` | 提高召回，可能引入更多争议标注 |
| 使用 soft label | 用 `AgrLevel / RadN` 表示结节置信度 |
| 医生级训练 | 使用 `trainNodules.csv` 和 `masks/LNDb-XXXX_radR.mhd` 分别建模 |

如果目标是检测，建议从 `trainNodules_gt.csv` 开始；如果目标是分析医生差异或不确定性，可以使用医生级 `trainNodules.csv` 和 `masks`。

### 7.10 数据增强

3D 肺结节任务常用增强：

| 增强 | 注意点 |
|---|---|
| 随机翻转 | 左右翻转通常可用；若使用肺叶位置标签，要谨慎 |
| 随机旋转 | 小角度旋转更安全 |
| 随机缩放 | 注意保持结节大小标签一致 |
| 随机平移 | patch 训练常用 |
| Gaussian noise | 模拟扫描噪声 |
| HU window jitter | 改变窗宽窗位，提高鲁棒性 |
| Cutout / Mixup | 需要谨慎，医学任务中可能破坏结构含义 |

mask 增强必须和 CT 使用相同空间变换，且 mask 插值使用最近邻。

### 7.11 常见训练任务设计

#### 结节检测

输入：

```text
CT volume
```

标签：

```text
结节中心点 + 直径/体积/box
```

LNDbv4 可用：

```text
trainNodules_gt.csv
```

LUNA16 可用：

```text
annotations.csv
```

#### 候选点假阳性抑制

输入：

```text
候选点周围 3D patch
```

标签：

```text
结节 / 非结节
```

LNDbv4 可用：

```text
trainNodules.csv 中的 Nodule 字段
```

LUNA16 可用：

```text
candidates.csv 或 candidates_V2.csv 中的 class 字段
```

#### 结节分割

输入：

```text
CT patch
```

标签：

```text
binary mask 或 instance mask
```

LNDbv4 可直接使用：

```text
masks/LNDb-XXXX_radR.mhd
```

LUNA16 官方通常没有完整分割 mask，若需要分割，常回到 LIDC-IDRI 原始 XML 标注或使用球形伪 mask。

#### Fleischner 分类

输入可以是：

```text
CT 级结节列表
每个结节的体积、纹理、数量
```

标签：

```text
trainFleischner.csv 中的 Fleischner
```

LNDbv4 更适合这个任务，因为它直接提供相关 CSV 和计算脚本。LUNA16 通常不直接提供 Fleischner 标签。

## 8. 推荐的 LNDbv4 基础处理流程

一个稳妥的 baseline 流程如下：

1. 读取 `trainCTs.csv` 和 `testCTs.csv`，建立病例划分。
2. 用 SimpleITK 读取 `.mhd`，得到 CT、spacing、origin、direction。
3. 将 CT HU 截断到 `[-1000, 400]` 并归一化到 `[0, 1]`。
4. 根据任务决定是否重采样到 `1 mm isotropic`。
5. 对 `trainNodules_gt.csv` 中的 `x, y, z` 做世界坐标到 voxel 坐标转换。
6. 检测任务：生成中心点 heatmap、anchor 标签或裁剪候选 patch。
7. 分割任务：读取 `masks/LNDb-XXXX_radR.mhd`，按 finding ID 提取二值 mask。
8. 分类任务：以结节中心裁剪 `48^3`、`64^3` 或 `80^3` patch。
9. 训练时进行 3D 空间增强和 HU 增强。
10. 评估时注意世界坐标与 voxel 坐标转换，输出结果应回到世界坐标。

## 9. 实践注意事项

1. 不要把 CSV 中的 `x, y, z` 直接当作 numpy 下标，它们是世界坐标。
2. SimpleITK 数组是 z, y, x，CSV 和 spacing 通常是 x, y, z。
3. CT 重采样用线性插值，mask 重采样用最近邻插值。
4. mask 中的非零值是 finding ID，不是统一的 1。
5. `trainNodules.csv` 是医生级标注，`trainNodules_gt.csv` 是合并后标注，二者不要混用。
6. LNDbv4 的测试集没有公开人工 mask，训练分割模型时应只用训练集 mask。
7. 如果和 LUNA16 共同训练，建议统一到相同 spacing、HU window、坐标转换和标签定义。
8. LUNA16 通常用直径 `diameter_mm` 表示大小，LNDbv4 常用 `Volume` 或等效直径，合并时需要统一。
9. LNDbv4 有报告文本相关标签，LUNA16 没有；做多模态或报告匹配任务时不能直接把 LUNA16 当作同构数据。
10. 对 Fleischner 分类任务，应以 CT 为单位聚合所有结节，而不是对单个结节直接分类。

