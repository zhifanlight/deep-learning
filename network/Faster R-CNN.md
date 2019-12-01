# $\mathrm{Faster \ R-CNN}$

## 思想

- 用 $\mathrm{RPN}$ 替换 $\mathrm{Selective \ Search}$，并将 $\mathrm{RPN}$ 与 $\mathrm{Fast \ R-CNN}$ 统一到同一个 $\mathrm{CNN}$ 中，实现端到端检测

- $\mathrm{RPN}$ 与 $\mathrm{Fast \ R-CNN}$ 共享卷积层，减少计算量

- 在 $\mathrm{GPU}$ 上的运行速度可以达到 $5 \ \mathrm{fps}$

- 检测范围：

  - $\mathrm{PASCAL \ VOC}$：背景类、$20$ 个前景类

  - $\mathrm{MS \ COCO}$：背景类、$80$ 个前景类

## 网络结构

<center>
<img src="images/faster_rcnn.png"/>
</center>

### $\mathrm{Fast \ R-CNN}$

- 结构与 $\mathrm{Fast \ R-CNN}$ 相同：

  - 将 $\mathrm{VGG}$ 最后一个 $\mathrm{max \ pooling}$ 层替换为 $\mathrm{ROI \ pooling}$ 层

  - 通过全连接层同时对 $\mathrm{Region \ Proposal}$ 进行分类、$\mathrm{Bounding \ Box}$ 回归

  - 关于 $\mathrm{Fast \ R-CNN}$，参考 [$\mathrm{Fast\ R-CNN.md}$](Fast R-CNN.md)

### $\mathrm{RPN}$

- $\mathrm{Region \ Proposal \ Network}$

- 首先对 $\mathrm{VGG}$ 最后一个卷积层进行 $3 \times 3$ 卷积，以进一步提取特征

- 分为两部分：

  - $\mathrm{Softmax}$ 计算每个 $\mathrm{Region \ Proposal}$ 分别属于前景类、背景类的概率

  - $\mathrm{Bounding \ Box}$ 输出每个 $\mathrm{Region \ Proposal}$ 的偏移量、缩放尺度

#### $\mathrm{Anchor}$

- $\mathrm{Anchor}$ 是特征图上的矩形，每个 $\mathrm{Anchor}$ 对应输入图像中一个区域

- $\mathrm{Anchor}$ 有 $3$ 种形状、$3$ 种尺寸，共 $9$ 种组合：

  - 长宽比：$1:1 \quad 1:2 \quad 2:1$

  - 短边长：$128 \quad 256 \quad 512$

- 通过多尺度 $\mathrm{Anchor}$ 实现多尺度上的目标检测，节省计算

#### $\mathrm{Softmax}$ 模块

- 通过 $1 \times 1$ 卷积，将特征图中不同通道的特征联系在一起

- 判断特征图上每个点所属的类别，每个点对应 $2k$ 维向量：

  - 对于每个点，共有 $k = 9$ 个 $\mathrm{Anchor}$

  - 使用二分类 $\mathrm{Softmax}$ 判断每个 $\mathrm{Anchor}$ 分别属于前景类、背景类的概率

- 来自同一张图片的 $256$ 个样本组成一个 $\mathrm{batch}$，正负样本定义如下：

  - 正样本：$\mathrm{Anchor}$ 与任一 $\mathrm{Ground \ Truth}$ 的 $\mathrm{IoU} \geq 0.7$

    - 检测不到正样本时，将最大 $\mathrm{IoU}$ 对应的 $\mathrm{Anchor}$ 作为正样本

  - 负样本：$\mathrm{Anchor}$ 与所有 $\mathrm{Ground \ Truth}$ 的 $\mathrm{IoU} < 0.3$

  - 无效样本：除了正、负样本之外的样本，不参与 $\mathrm{RPN}$ 训练

  - 如果某个 $\mathrm{Anchor}$ 既满足正样本定义，又满足负样本定义，按负样本处理

  - 为防止训练结果偏向负样本，每个 $\mathrm{batch}$ 内正、负样本数量相同

    - 正样本数量不足时，用负样本填充，组成一个完整的 $\mathrm{batch}$

- $\mathrm{Softmax \ Loss}$ 计算如下：

  $$
  L_{\mathrm{cls}} = \frac{1}{N_{\mathrm{cls}}} \sum_{a} L_{\mathrm{cls}} \left( p_{a}, \ p_{a}^{*} \right)
  $$

  - $N_{\mathrm{cls}}$ 是参与训练的 $\mathrm{Anchor}$ 数量

  - $p_{a}$ 是第 $a$ 个 $\mathrm{Anchor}$ 被预测为前景类的概率

  - $p_{a}^{*}$ 是第 $a$ 个 $\mathrm{Anchor}$ 的标签：

    - 前景类：$1$

    - 背景类：$0$

- $\mathrm{Softmax}$ 模块输出 $18$ 通道的特征图，分别表示每种 $\mathrm{Anchor}$ 属于前景、背景概率

#### $\mathrm{Bounding \ Box}$ 模块

- 通过 $1 \times 1$ 卷积，将特征图中不同通道的特征联系在一起

- 计算特征图上每个点对应的 $\mathrm{Bounding \ Box}$ 参数，每个点对应 $4k$ 维向量：

  - 对于每个点，共有 $k = 9$ 个 $\mathrm{Anchor}$

  - 为每个 $\mathrm{Anchor}$ 训练一个 $\mathrm{Bounding \ Box}$ 回归器

- $\mathrm{BBox \ Loss}$ 计算如下：

  $$
  L_{\mathrm{reg}} = \frac{1}{N_{\mathrm{reg}}} \sum_{a} p_{a}^{*} \cdot \mathrm{{Smooth}_{L_{1}}} \left( t_{a} - t_{a}^{*} \right)
  $$

  $$
  \mathrm{Smooth_{L_{1}}} \left( x \right) = \left\{ \begin{matrix} 0.5 x^{2} \quad \ \quad |x| < 1 \\ |x| - 0.5 \quad |x| > 1 \end{matrix} \right.
  $$

  - $N_{\mathrm{reg}}$ 是 $\mathrm{RPN}$ 特征图上 $\mathrm{Anchor}$ 的不同位置数

- $t_{a}$ 第 $a$ 个 $\mathrm{Anchor}$ 的预测 $\mathrm{Bounding \ Box}$ 参数

  - $t_{a}^{*}$ 是第 $a$ 个 $\mathrm{Anchor}$ 的实际 $\mathrm{Bounding \ Box}$ 参数

- 系数 $p_{a}^{*}$ 表示只计算前景类的 $\mathrm{Smooth \ L1}$ 损失

- 计算 $\mathrm{Bounding \ Box}$ 参数时所需的特征，来自 $\mathrm{RPN}$ 的 $3 \times 3$ 卷积层

- 计算的是从 $\mathrm{Anchor}$ 到最近邻 $\mathrm{Ground \ Truth}$ 的 $\mathrm{Bounding \ Box}$ 参数

- $\mathrm{Bounding \ Box}$ 模块输出 $36$ 通道的特征图，分别表示每种 $\mathrm{Anchor}$ 的 $4$ 个偏移量

#### 损失函数

- 整个 $\mathrm{RPN}$ 的损失函数计算如下：

  $$
  L_{\mathrm{RPN}} = \frac{1}{N_{\mathrm{cls}}} \sum_{a} L_{\mathrm{cls}} \left( p_{a}, \ p_{a}^{*} \right) + \lambda \cdot \frac{1}{N_{\mathrm{reg}}} \sum_{a} p_{a}^{*} \cdot \mathrm{Smooth_{L_{1}}} \left( t_{a} - t_{a}^{*} \right)
  $$

- 超参数 $\lambda = 10$:

  - 对于 $\mathrm{VGG}$ 而言，$1000 \times 600$ 的输入图像在 $\mathrm{RPN}$ 特征图上的尺寸约为 $60 \times 40$

  - 实际训练时，选择 $256$ 个 $\mathrm{Anchor}$ 组成 $\mathrm{mini-batch}$

### $\mathrm{Faster \ R-CNN \ with \ ResNet-101}$

- 使用 $\mathrm{ResNet-101}$ 作为主干网络时，结构略有不同

  - 去掉 $\mathrm{Fast \ R-CNN}$ 的两个全连接层，构成 $\mathrm{FCN}$

  - 在 $\mathrm{conv4}$ 特征图上进行 $\mathrm{RPN}$ 和 $\mathrm{ROIPooling}$

  - $\mathrm{ROIPooling}$ 后的特征图维度变为 $14 \times 14$

  - 将 $\mathrm{ROIPooling}$ 结果作为 $\mathrm{conv5}$ 的输入

  - 对 $\mathrm{conv5}$ 的 $\mathrm{global \ average \ pooling}$ 结果送入两个子网络进行分类和 $\mathrm{BBox}$ 回归

## 训练过程

- 只进行单尺度检测，将图像缩放至短边长度 $s = 600$

- 从 $\mathrm{VGG}$ 的 $\mathrm{conv3\_1}$ 层开始进行 $\mathrm{fine-tuning}$

### 四步交替训练

#### $\mathrm{RPN \ fine-tuning}$

- 对预训练的 $\mathrm{CNN}$ 模型进行 $\mathrm{fine-tuning}$，得到 $\mathrm{Region \ Proposal}$，用于后续 $\mathrm{Fast \ R-CNN}$

#### $\mathrm{Fast \ R-CNN \ fine-tuning}$

- 利用上一步中得到的 $\mathrm{Region \ Proposal}$，对预训练的 $\mathrm{CNN}$ 模型进行 $\mathrm{fine-tuning}$

#### $\mathrm{RPN \ fine-tuning}$

- 基于上一步得到的模型，仅对 $\mathrm{RPN}$ 进行 $\mathrm{fine-tuning}$，实现卷积层共享

#### $\mathrm{Fast \ R-CNN \ fine-tuning}$

- 固定卷积层和 $\mathrm{RPN}$，仅对 $\mathrm{Fast-RCNN}$ 进行 $\mathrm{fine-tuning}$，完成对 $\mathrm{Faster \ R-CNN}$ 的训练

### $\mathrm{Anchor}$ 数量

- 对于 $\mathrm{VGG}$ 而言，$1000 \times 600$ 的输入图像在 $\mathrm{RPN}$ 特征图上的尺寸约为 $60 \times 40$，共有 $2400 \times 9 \approx 2,0000$ 个 $\mathrm{Anchor}$

- 忽略超出边界、高或宽小于 $16$ 的 $\mathrm{Region \ Proposal}$，大约还剩 $6000$ 个

  - 在训练时，直接过滤掉超出边界的 $\mathrm{Region \ Proposal}$，否则容易导致不收敛；在测试时，对超出边界的 $\mathrm{Region \ Proposal}$ 进行裁剪

  - $\mathrm{VGG}$ 在最后一个卷积层上的累积 $\mathrm{stride} = 16$，当高或宽小于 $16$ 时，$\mathrm{Region \ Proposal}$ 无法对应到特征图上的点

- 通过 $\mathrm{NMS}$（阈值 $0.7$）过滤重叠区域，大约还剩 $2000$ 个

- $\mathrm{Anchor}$ 选择：

  - 训练时从这 $2000$ 个 $\mathrm{Anchor}$ 中随机选取 $256$ 个 $\mathrm{Anchor}$

  - 测试时只选取得分最高的 $300$ 个 $\mathrm{Anchor}$ 作为 $\mathrm{Fast \ R-CNN}$ 输入