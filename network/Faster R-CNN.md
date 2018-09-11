<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Faster R-CNN

## 思想

- 用 RPN 替换 Selective Search，并将 RPN 与 Fast R-CNN 统一到同一个 CNN 中，实现端到端检测

- RPN 与 Fast R-CNN 共享卷积层，减少计算量

- 在 GPU 上的运行速度可以达到 5 fps

- 检测范围：

	- PASCAL VOC：背景类、20 个前景类

	- MS COCO：背景类、80 个前景类

## 网络结构

![img](images/faster_rcnn.png)

### Fast R-CNN

- 结构与 Fast R-CNN 相同：

	- 将 VGG 最后一个 max pooling 层替换为 ROI pooling 层

	- 通过全连接层同时对 Region Proposal 进行分类、Bounding Box 回归

	- 关于 Fast R-CNN，参考 [Fast R-CNN.md](Fast\ R-CNN.md)

### RPN

- Region Proposal Network

- 首先对 VGG 最后一个卷积层进行 \\(3 \times 3\\) 卷积，以进一步提取特征

- 分为两部分：

	- Softmax 计算每个 Region Proposal 分别属于前景类、背景类的概率

	- Bounding Box 输出每个 Region Proposal 的偏移量、缩放尺度

#### Anchor

- Anchor 是特征图上的矩形，每个 Anchor 对应输入图像中一个区域

- Anchor 有 3 种形状、3 种尺寸，共 9 种组合：

	- 长宽比：\\(1:1 \quad 1:2 \quad 2:1\\)

	- 短边长：\\(128 \quad 256 \quad 512\\)

- 通过多尺度 Anchor 实现多尺度上的目标检测，节省计算

#### Softmax 模块

- 通过 \\(1 \times 1\\) 卷积，将特征图中不同通道的特征联系在一起

- 判断特征图上每个点所属的类别，每个点对应 \\(2k\\) 维向量：

	- 对于每个点，共有 \\(k=9\\) 个 Anchor

	- 使用二分类 Softmax 判断每个 Anchor 分别属于前景类、背景类的概率

- 来自同一张图片的 256 个样本组成一个 batch，正负样本定义如下：

	- 正样本：Anchor 与任一 Ground Truth 的 \\(IoU \geq 0.7\\)

		- 检测不到正样本时，将最大 \\(IoU\\) 对应的 Anchor 作为正样本

	- 负样本：Anchor 与所有 Ground Truth 的 \\(IoU < 0.3\\)

	- 无效样本：除了正、负样本之外的样本，不参与 RPN 训练

	- 如果某个 Anchor 既满足正样本定义，又满足负样本定义，按负样本处理

	- 为防止训练结果偏向负样本，每个 batch 内正、负样本数量相同

		- 正样本数量不足时，用负样本填充，组成一个完整的 batch

- Softmax Loss 计算如下：

	$$ L\_{cls} = \frac{1}{N\_{cls}} \sum\_{a} L\_{cls}(p\_{a}, p\_{a}^{\*}) $$

	- \\(N\_{cls}\\) 是参与训练的 Anchor 数量

	- \\(p\_{a}\\) 是第 \\(a\\) 个 Anchor 被预测为前景类的概率

	- \\(p\_{a}^{\*}\\) 是第 \\(a\\) 个 Anchor 的标签：

		- 前景类：1

		- 背景类：0

- Softmax 模块输出 18 通道的特征图，分别表示每种 Anchor 属于前景、背景概率

#### Bounding Box 模块

- 通过 \\(1 \times 1\\) 卷积，将特征图中不同通道的特征联系在一起

- 计算特征图上每个点对应的 Bounding Box 参数，每个点对应 \\(4k\\) 维向量：

	- 对于每个点，共有 \\(k=9\\) 个 Anchor

	- 为每个 Anchor 训练一个 Bounding Box 回归器

- BBox Loss 计算如下：

	$$ L\_{reg} = \frac{1}{N\_{reg}} \sum\_{a} p\_{a}^{\*} \cdot Smooth\_{L\_{1}}(t\_{a} - t\_{a}^{\*}) $$

	$$ Smooth\_{L\_{1}}(x) = \left\\{ \begin{matrix} 0.5 x^{2} \quad |x| < 1 \\\\ |x| - 0.5 \quad |x| > 1 \end{matrix} \right. $$

	- \\(N\_{reg}\\) 是 RPN 特征图上 Anchor 的不同位置数

	- \\(t\_{a}\\) 第 \\(a\\) 个 Anchor 的预测 Bounding Box 参数

	- \\(t\_{a}^{\*}\\) 是第 \\(a\\) 个 Anchor 的实际 Bounding Box 参数

	- 系数 \\(p\_{a}^{\*}\\) 表示只计算前景类的 Smooth L1 损失

- 计算 Bounding Box 参数时所需的特征，来自 RPN 的 \\(3 \times 3\\) 卷积层

- 计算的是从 Anchor 到最近邻 Ground Truth 的 Bounding Box 参数

- Bounding Box 模块输出 36 通道的特征图，分别表示每种 Anchor 的 4 个偏移量

#### 损失函数

- 整个 RPN 的损失函数计算如下：

	$$ L\_{RPN} = \frac{1}{N\_{cls}} \sum\_{a} L\_{cls}(p\_{a}, p\_{a}^{\*}) + \lambda \cdot \frac{1}{N\_{reg}} \sum\_{a} p\_{a}^{\*} \cdot Smooth\_{L\_{1}}(t\_{a} - t\_{a}^{\*}) $$

- 超参数 \\(\lambda = 10\\):

	- 对于 VGG 而言，\\(1000 \times 600\\) 的输入图像在 RPN 特征图上的尺寸约为 \\(60 \times 40\\)

	- 实际训练时，选择 256 个 Anchor 组成 mini-batch

### Faster R-CNN with ResNet-101

- 使用 ResNet-101 作为主干网络时，结构略有不同

	- 去掉 Fast R-CNN 的两个全连接层，构成 FCN

	- 在 conv4 特征图上进行 RPN 和 ROIPooling

	- ROIPooling 后的特征图维度变为 \\(14 \times 14\\)

	- 将 ROIPooling 结果作为 conv5 的输入

	- 对 conv5 的 global average pooling 结果送入两个子网络进行分类和 BBox 回归

## 训练过程

- 只进行单尺度检测，将图像缩放至短边长度 \\(s = 600\\)

- 从 VGG 的 conv3_1 层开始进行 fine-tuning

### 四步交替训练

#### RPN fine-tuning

- 对预训练的 CNN 模型进行 fine-tuning，得到 Region Proposal，用于后续 Fast R-CNN

#### Fast R-CNN fine-tuning

- 利用上一步中得到的 Region Proposal，对预训练的 CNN 模型进行 fine-tuning

#### RPN fine-tuning

- 基于上一步得到的模型，仅对 RPN 进行 fine-tuning，实现卷积层共享

#### Fast R-CNN fine-tuning

- 固定卷积层和 RPN，仅对 Fast-RCNN 进行 fine-tuning，完成对 Faster R-CNN 的训练

### Anchor 数量

- 对于 VGG 而言，\\(1000 \times 600\\) 的输入图像在 RPN 特征图上的尺寸约为 \\(60 \times 40\\)，共有 \\(2400 \times 9 \approx 20000\\) 个 Anchor

- 忽略超出边界、高或宽小于 16 的 Region Proposal，大约还剩 6000 个

	- 在训练时，直接过滤掉超出边界的 Region Proposal，否则容易导致不收敛；在测试时，对超出边界的 Region Proposal 进行裁剪

	- VGG 在最后一个卷积层上的累积 \\(stride = 16\\)，当高或宽小于 16 时， Region Proposal 无法对应到特征图上的点

- 通过 NMS（阈值 0.7）过滤重叠区域，大约还剩 2000 个

- Anchor 选择：

	- 训练时从这 2000 个 Anchor 中随机选取 256 个 Anchor

	- 测试时只选取得分最高的 300 个 Anchor 作为 Fast R-CNN 输入