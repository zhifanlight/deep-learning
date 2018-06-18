<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Faster R-CNN

## 思想

- 用 RPN 替换 Selective Search 过程，并将 RPN 与 Fast R-CNN 统一到同一个 CNN 中，实现端到端检测

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

	- Softmax 计算每个区域属于前景类、背景类的概率

	- Bounding Box 输出每个 Region Proposal 的最终位置

#### Anchor

- Anchor 是特征图上的矩形，每个 Anchor 对应输入图像中一个区域

- Anchor 有 3 种形状、3 种尺寸，共 9 种 Anchor：

	- 形状（长宽比）：

		- 1:1

		- 1:2

		- 2:1

	- 尺寸（短边）：

		- 128

		- 256

		- 512

#### Softmax 模块

- 通过 \\(1 \times 1\\) 卷积，将特征图中不同通道的特征联系在一起

- 判断特征图上每个点所属的类别，每个点对应 \\(2k\\) 维向量：

	- 对于每个点，共有 \\(k=9\\) 种 Anchor

	- 使用二分类 Softmax 判断每个 Anchor 属于前景类的概率、属于背景类的概率

- 为平衡样本数量差距，正负样本定义如下：

	- 正样本：Anchor 与任一 Ground Truth 的 \\(IoU \geq 0.7\\)

	- 负样本：Anchor 与所有 Ground Truth 的 \\(IoU < 0.3\\)

	- 无效样本：除了正、负样本之外的样本，不参与 RPN 训练

- 交叉熵损失函数计算如下：

	$$ L\_{cls} = \frac{1}{N\_{cls}} \sum\_{a} L\_{cls}(p\_{a}, p\_{a}^{\*}) $$

	- \\(N\_{cls}\\) 是参与训练的 Anchor 数量

	- \\(p\_{a}\\) 是第 \\(a\\) 个 Anchor 被预测为前景类的概率

	- \\(p\_{a}^{\*}\\) 是第 \\(a\\) 个 Anchor 的标签：

		- 前景类：1

		- 背景类：0

#### Bounding Box 模块

- 通过 \\(1 \times 1\\) 卷积，将特征图中不同通道的特征联系在一起

- 计算特征图上每个点对应的 Bounding Box 参数，每个点对应 \\(4k\\) 维向量：

	- 对于每个点，共有 \\(k=9\\) 种 Anchor

	- 每个 Anchor 对应 4 个 Bounding Box 参数

- Smooth L1 损失函数计算如下：

	$$ L\_{reg} = \frac{1}{N\_{reg}} \sum\_{a} p\_{a}^{\*} \cdot Smooth\_{L\_{1}}(t\_{a} - t\_{a}^{\*}) $$

	- \\(N\_{reg}\\) 是 RPN 特征图上 Anchor 的不同位置数

	- \\(t\_{a}\\) 第 \\(a\\) 个 Anchor 的预测 Bounding Box 参数

	- \\(t\_{a}^{\*}\\) 是第 \\(a\\) 个 Anchor 的实际 Bounding Box 参数

	- 系数 \\(p\_{a}^{\*}\\) 表示只计算前景类的 Smooth L1 损失

- 计算 Bounding Box 参数时所需的特征，来自 RPN 的 \\(3 \times 3\\) 卷积层

- 计算的是从 Anchor 到最近邻 Ground Truth 的 Bounding Box 参数

#### 损失函数

- RPN 的总损失函数计算如下：

	$$ L\_{RPN} = \frac{1}{N\_{cls}} \sum\_{a} L\_{cls}(p\_{a}, p\_{a}^{\*}) + \lambda \cdot \frac{1}{N\_{reg}} \sum\_{a} p\_{a}^{\*} \cdot Smooth\_{L\_{1}}(t\_{a} - t\_{a}^{\*}) $$

- 超参数 \\(\lambda = 10\\):

	- 对于 VGG 而言，\\(1000 \times 600\\) 的输入图像在 RPN 特征图上的尺寸约为 \\(60 \times 40\\)

	- 实际训练时，选择 256 个 Anchor 组成 mini-batch

## 训练过程

- 只进行单尺度检测，将图像缩放至短边长度 \\(s = 600\\)

- 从 VGG 的 conv3_1 层开始进行 fine-tuning

### 四步交替训练

#### RPN fine-tuning

- 对 ImageNet 上预训练的 CNN 模型进行 fine-tuning，得到经过 Bounding Box 的 Region Proposal，用于后续 Fast R-CNN

#### Fast R-CNN fine-tuning

- 利用上一步中得到的 Region Proposal，对 ImageNet 上预训练的 CNN 模型进行 fine-tuning

#### RPN fine-tuning

- 将 Fast R-CNN 的卷积核复制到 RPN 并固定，仅对 RPN 进行 fine-tuning，实现卷积层共享

#### Fast R-CNN fine-tuning

- 继续固定卷积层，仅对 Fast-RCNN 进行 fine-tuning，完成对 Faster R-CNN 的训练

### Anchor 数量

- 对于 VGG 而言，\\(1000 \times 600\\) 的输入图像在 RPN 特征图上的尺寸约为 \\(60 \times 40\\)，共有 \\(2400 * 9 \approx 20000\\) 个 Anchor

- 忽略超出边界、高或宽小于 16 的 Region Proposal，大约还剩 6000 个

	- 在训练时，直接过滤掉超出边界的 Region Proposal；在测试时，对超出边界的 Region Proposal 进行裁剪

	- VGG 在最后一个卷积层上的累积 \\(stride = 16\\)，当高或宽小于 16 时， Region Proposal 无法对应到特征图上的点

- 通过 NMS（阈值 0.7）过滤重叠区域，大约还剩 2000 个

- Anchor 选择：

	- 训练时从这 2000 个 Anchor 中随机选取 256 个 Anchor

	- 测试时只选取得分最高的 300 个 Anchor 用于 Fast R-CNN 检测