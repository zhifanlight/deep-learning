# $\mathrm{YOLO}$

## $\mathrm{YOLO \ v1}$

### 思想

- 不选择 $\mathrm{Region \ Proposal}$，直接在输出层回归目标的位置和对应分类

- 由于省略了 $\mathrm{Region \ Proposal}$ 过程，速度可以达到 $\mathrm{45 \ FPS}$；代价是精度降低（预测位置不准）

### 检测流程

#### $\mathrm{Grid}$

- 输入图像划分成 $S \times S$ 的 $\mathrm{Grid}$，每个 $\mathrm{Grid}$ 对应 $B$ 个 $\mathrm{BBox}$ 向量 $\left( \mathrm{conf}, \ x, \ y, \ w, \ h \right)$ 和一个 $C$ 分类的向量，最终输出为 $S \times S \times \left( 5B + C \right)$ 维向量

  - $\left( x, \ y \right)$ 是 $\mathrm{BBox}$ 中心相对于当前 $\mathrm{Grid}$ 左上角的位移，归一化到 $\left[ 0, \ 1 \right)$

  - $\left( w, \ h \right)$ 是 $\mathrm{BBox}$ 相对于输入图像的宽和高，归一化到 $\left[ 0, \ 1 \right)$

    - 由损失函数部分可知，$\left( w, \ h \right)$ 实际是归一化后的平方根

  - $\mathrm{conf}$ 代表了所预测的 $\mathrm{BBox}$ 中有目标的置信度和 $\mathrm{BBox}$ 回归精度的两重信息：

    $$
    \mathrm{conf} = P \left( \mathrm{Object} \right) \cdot \mathrm{IoU}_{\mathrm{Pred}}^{\mathrm{Truth}}
    $$

    - 如果有目标的中心落在当前 $\mathrm{Grid}$ 中，$P \left( \mathrm{Object} \right) = 1$；否则 $P \left( \mathrm{Object} \right) = 0$

- 在 $\mathrm{PASCAL \ VOC}$ 上，$S = 7, \ B = 2, \ C = 20$，输出为 $7 \times 7 \times \left( 5 \times 2 + 20 \right) = 1470$ 维，最多可以检测 $7 \times 7 \times 2 = 98$ 个目标

#### 训练

- 对于有目标中心落入的每个 $\mathrm{Grid}$，从其预测的 $B$ 个 $\mathrm{BBox}$ 中选择与 $\mathrm{Ground \ Truth}$ 的 $\mathrm{IoU}$ 最大的一个，负责对 $\mathrm{Ground \ Truth}$ 的预测，用于计算 $\mathrm{loss}$

- 对各种 $\mathrm{loss}$ 加权求和：

  - $\mathrm{BBox \ loss}$ 和 $\mathrm{Softmax \ loss}$ 权重不应该相同

    - 更重视对 $\mathrm{BBox}$ 的预测，$\lambda_{\mathrm{coord}} = 5$

  - 如果一个 $\mathrm{Grid}$ 中没有目标，其 $\mathrm{Confidence}$ 会被置零；由于大部分 $\mathrm{Grid}$ 中没有目标，使得无目标部分的梯度在训练时占比过高，容易造成网络训练不稳定

    - 降低无目标 $\mathrm{Grid}$ 的分类 $\mathrm{loss}$ 权重，即 $\lambda_{\mathrm{noobj}} = 0.5$

- $\mathrm{Loss}$ 由以下 $5$ 项组成：

  - $\mathrm{BBox}$ 位置 $\mathrm{Loss}$

    $$
    \lambda_{\mathrm{coord}} \sum_{i = 0}^{S^{2}} \sum_{j = 0}^{B} \mathrm{1}_{ij}^{\mathrm{obj}} \left[ \left( x_{i} - \hat{x_{i}} \right)^{2} + \left( y_{i} - \hat{y_{i}} \right)^{2} \right]
    $$

    - $\mathrm{1}_{ij}^{\mathrm{obj}}$ 判断第 $i$ 个 $\mathrm{Grid}$ 中的第 $j$ 个 $\mathrm{BBox}$ 是否负责预测 $\mathrm{Ground \ Truth}$

  - $\mathrm{BBox}$ 尺度 $\mathrm{Loss}$

    $$
    \lambda_{\mathrm{coord}} \sum_{i = 0}^{S^{2}}\sum_{j = 0}^{B} \mathbb{1}_{ij}^{\mathrm{obj}} \left[ \left( \sqrt{w_{i}} - \sqrt{\hat{w_{i}}} \right)^{2} + \left( \sqrt{h_{i}} - \sqrt{\hat{h_{i}}} \right)^{2} \right]
    $$

    - $w$ 或 $h$ 差距相同时，小物体的 $\mathrm{loss}$ 应该更大

  - 目标的 $\mathrm{Confidence \ Loss}$

    $$
    \sum_{i = 0}^{S^{2}} \sum_{j = 0}^{B} \mathbb{1}_{ij}^{\mathrm{obj}} \left( C_{i} - \hat{C_{i}} \right)^{2}
    $$

  - 背景的 $\mathrm{Confidence \ Loss}$

    $$
    \lambda_{\mathrm{noobj}} \sum_{i = 0}^{S^{2}} \sum_{j = 0}^{B} \mathbb{1}_{ij}^{\mathrm{noobj}} \left( C_{i} - \hat{C_{i}} \right)^{2}
    $$

  - 分类 $\mathrm{Loss}$

    $$
    \sum_{i = 0}^{S^{2}} \mathbb{1}_{i}^{obj} \sum_{c \in \mathrm{classes}} \left( p_{i} \left( c \right) - \hat{p_{i}} \left( c \right) \right)^{2}
    $$

    - $\mathbb{1}_{i}^{\mathrm{obj}}$ 判断是否有目标中心落在第 $i$ 个 $\mathrm{Grid}$ 中

- 由 $\mathrm{loss}$ 定义可知：

  - 只有某个 $\mathrm{Grid}$ 中有目标的时候，才惩罚分类 $\mathrm{loss}$

  - 只有某个 $\mathrm{BBox}$ 负责预测某个 $\mathrm{Ground \ Truth}$ 时，才惩罚 $\mathrm{BBox}$ 和 $\mathrm{Confidence \ loss}$

    - 具体惩罚哪个 $\mathrm{BBox}$，根据 $\mathrm{IoU}$ 决定

#### 测试

- 根据 $\mathrm{Confidence}$ 和分类信息，计算每个 $\mathrm{Grid}$ 的每个 $\mathrm{BBox}$ 对每个类别的置信度：

  $$
  P \left( \mathrm{Object} \right) \cdot \mathrm{IoU}_{\mathrm{Pred}}^{\mathrm{Truth}} \cdot P \left( \mathrm{Class_{i}} | \mathrm{Object} \right) = P \left( \mathrm{Class_{i}} \right) \cdot \mathrm{IoU}_{\mathrm{Pred}}^{\mathrm{Truth}}
  $$

- 过滤掉置信度较低的 $\mathrm{BBox}$，通过 $\mathrm{NMS}$ 得到最终的检测结果

### 网络结构

<center>
<img src="images/yolo_v1.png"/>
</center>

- 输入图像缩放到 $448 \times 448$

- 最后的两个全连接层输出 $1470$ 维向量

- 采用 $\mathrm{Leaky \ ReLU}$ 作为激活函数

### 缺点

- 对靠太近的物体、小群体的检测效果不好

  - 多个目标中心落在同一 $\mathrm{Grid}$ 上，但每个 $\mathrm{Grid}$ 只能预测两个 $\mathrm{BBox}$，而且只属于一类

- 对小目标的定位准确性较差

  - 训练时小目标和大目标的 $\mathrm{BBox \ loss}$ 对损失函数影响相同（平方根只是缓解，不能彻底解决问题）

## $\mathrm{YOLO \ v2}$

### 思想

- 采用一系列方法进行优化，保证 $\mathrm{YOLO \ v1}$ 速度的情况下，提高检测精度

### $\mathrm{Better}$

#### $\mathrm{Batch \ Normalization}$

- 在每一个卷积层之后添加 $\mathrm{Batch \ Normalization}$ 层，解决 $\mathrm{Covariate \ Shift}$ 问题

- 同时在一定程度上对模型进行正则化约束

#### 高分辨率分类器

- 检测算法大多使用 $\mathrm{ImageNet}$ 上的预训练模型提取特征，但这些分类模型的输入一般是 $224 \times 224$，分辨率较低，会给检测带来困难

- 在 $\mathrm{YOLO \ v2}$ 中，首先在 $\mathrm{ImageNet}$ 上对 $448 \times 448$ 的分类模型进行 $\mathrm{fine-tuning}$，再在检测任务上进行 $\mathrm{fine-tuning}$

#### $\mathrm{Anchor}$ 思想

- $\mathrm{YOLO \ v1}$ 使用最后的全连接层预测 $\mathrm{BBox}$，导致较多空间信息的丢失，定位不准

- 借鉴 $\mathrm{Faster \ R-CNN}$ 的 $\mathrm{Anchor}$ 思想预测 $\mathrm{BBox}$；去掉全连接层和最后的 $\mathrm{pooling}$ 层，以确保输出的特征图具有更高的分辨率

- 修改输入尺寸为 $416 \times 416$，使最后一层特征图尺寸为奇数 $13 \times 13$

  - 经过 $5$ 次 $\mathrm{pooling}$，特征图缩小 $32$ 倍

  - 大目标通常占据图像的中间位置

    - 如果特征图是奇数，只需使用中间的 $1$ 个 $\mathrm{Cell}$ 进行预测

    - 如果特征图是偶数，需要使用中间的 $4$ 个 $\mathrm{Cell}$ 进行预测

#### $\mathrm{K-means}$ 聚类

- 在 $\mathrm{Faster \ R-CNN}$ 中，需要人工选择 $\mathrm{Anchor}$；如果可以选择更好的 $\mathrm{Anchor}$，模型的学习会更容易

- 通过 $\mathrm{K-means}$ 对 $\mathrm{Ground \ Truth}$ 进行聚类，根据 $k$ 个聚类中心的尺寸，设置 $\mathrm{Anchor}$

  - 如果使用欧式距离，大目标会比小目标产生更多的误差，导致聚类中心偏向大目标

  - 使用 $\mathrm{IoU}$ 作为距离衡量的标准，消除尺度对聚类误差的影响：

    $$
    D \left( \mathrm{box}, \ \mathrm{center} \right) = 1 - \mathrm{IoU} \left( \mathrm{box}, \ \mathrm{center} \right)
    $$

  - $k = 5$ 时，模型复杂度和召回率达到较好的平衡点

#### 直接预测位置

- 直接使用 $\mathrm{Anchor}$ 时，模型训练不稳定，主要问题是 $\mathrm{BBox}$ 对 $\left( x, \ y \right)$ 偏移量的预测

  - 由于没有数值限定，可能导致 $\mathrm{Anchor}$ 预测很远的 $\mathrm{BBox}$ 目标，效率较低

  - 正确的做法是，每一个 $\mathrm{Anchor}$ 只负责周围的 $\mathrm{BBox}$

- 将 $\mathrm{BBox}$ 的位置预测值 $\left( t_{x}, \ t_{y} \right)$ 改为相对于 $\mathrm{Cell}$ 左上角的偏移量

  - $\mathrm{Cell}$ 左上角坐标用其位置表示，$\left( c_{x}, \ c_{y} \right)$ 表示第 $\left( c_{x}, \ c_{y} \right)$ 个 $\mathrm{Cell}$

  - $\mathrm{BBox}$ 预测值 $\left( t_{x}, \ t_{y} \right)$ 通过 $\mathrm{Sigmoid}$ 函数归一化到 $\left( 0, \ 1 \right)$ 范围

  - 对于形状为 $\left( p_{w}, \ p_{h} \right)$ 的 $\mathrm{Anchor}$，最终的 $\mathrm{BBox}$ 回归过程计算如下：

    $$
    b_{x} = \sigma \left( t_{x} \right) + c_{x}
    $$

    $$
    b_{y} = \sigma \left( t_{y} \right) + c_{y}
    $$

    $$
    b_{w} = p_{w} \cdot \mathrm{e}^{t_{w}}
    $$

    $$
    b_{h} = p_{h} \cdot \mathrm{e}^{t_{h}}
    $$

#### 细粒度特征

- $13 \times 13$ 的特征图足以进行大目标的检测，但对小目标的检测效果较差

- 把浅层的 $26 \times 26$ 进行特征重排后，与 $13 \times 13$ 的特征图进行拼接，方便小目标检测

  - 通过隔行隔列采样，$26 \times 26$ 的特征图可以得到 $4$ 个 $13 \times 13$ 特征图

  - 原论文 $\mathrm{Reorg}$ 层实现方式很诡异，但精度略高

#### 多尺度训练

- 为了保证 $\mathrm{YOLO \ v2}$ 在不同尺度输入图像上的性能，进行多尺度训练

  - 每隔 $10$ 个 $\mathrm{batch}$，随机选择一种新尺寸的输入图像进行训练

- 由于特征图维度变为原图的 $1 / 32$，训练时使用 $32$ 的整数倍作为输入图像尺寸：

  $$
  \left( 320, \ 352, \cdots, \ 608 \right)
  $$

#### $\mathrm{Anchor}$ 输出

- 每个 $\mathrm{Anchor}$ 的输出是 $5 + C$ 维的向量

  - $5$ 维分别是 $\left( t_{x}, \ t_{y}, \ t_{w}, \ t_{h}, \ \mathrm{conf} \right)$

    - $t_{x}, \ t_{y}$ 是 $\mathrm{BBox}$ 中心相对于当前 $\mathrm{Cell}$ 左上角的偏移百分比

    - $t_{w}, \ t_{h}$ 是 $\mathrm{BBox}$ 的尺度变换量

    - $\mathrm{conf}$ 代表所预测的 $\mathrm{BBox}$ 中有目标的置信度和 $\mathrm{BBox}$ 回归精度的两重信息：

      $$
      \mathrm{conf} = P \left( \mathrm{Object} \right) \cdot \mathrm{IoU}_{\mathrm{Pred}}^{\mathrm{Truth}}
      $$

      - 如果有目标的中心落在当前 $\mathrm{Grid}$ 中，$P \left( \mathrm{Object} \right) = 1$；否则 $P \left( \mathrm{Object} \right) = 0$

  - $C$ 维分别是当前目标属于 $C$ 个前景类的概率 $P \left( \mathrm{Class_{i}} | \mathrm{Object} \right)$

  - 每个 $\mathrm{BBox}$ 对每个类别的置信度计算如下：

    $$
    P \left( \mathrm{Class_{i}} \right) \cdot \mathrm{IoU}_{\mathrm{Pred}}^{\mathrm{Truth}} = P \left( \mathrm{Object} \right) \cdot \mathrm{IoU}_{\mathrm{Pred}}^{\mathrm{Truth}} \cdot P \left( \mathrm{Class_{i}} | \mathrm{Object} \right)
    $$

### $\mathrm{Faster}$

#### $\mathrm{Darknet-19}$

- 使用 $\mathrm{Darknet-19}$ 作为基础网络，相比 $\mathrm{VGG}$ 和 $\mathrm{GoogLeNet}$，运算速度更快，精度更高

- $\mathrm{Darknet-19}$ 共有 $7$ 组卷积（$19$ 层），$5$ 个 $\mathrm{pooling}$ 层

  - 前 $6$ 组卷积之间通过 $\mathrm{stride} = 2$ 的 $2 \times 2 \ \mathrm{pooling}$ 实现下采样

  - 最后的卷积层经过 $\mathrm{global \ average \ pooling}$ 后进行 $\mathrm{Softmax}$ 分类