<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# YOLO v1

## 思想

- 不选择 Region Proposal，直接在输出层回归目标的位置和对应分类

- 由于省略了 Region Proposal 过程，速度可以达到 45FPS；代价是精度降低

## 检测流程

### Grid

- 输入图像划分成 \\(S \times S\\) 的 Grid，每个 Grid 对应 \\(B\\) 个 BBox 向量 \\((conf, x, y, w, h)\\) 和一个 \\(C\\) 分类的向量，最终输出为 \\(S \times S \times (5B + C)\\) 维向量

	- \\((x, y)\\) 是 BBox 中心相对于当前 Grid 左上角的位移，归一化到 \\([0, 1)\\)

	- \\((w, h)\\) 是 BBox 相对于输入图像的宽和长，归一化到 \\([0, 1)\\)

		- 由损失函数部分可知，\\((w, h)\\) 实际是归一化后的平方根

	- \\(conf\\) 代表了所预测的 BBox 中有目标的置信度和 BBox 回归精度的两重信息：

		$$ conf = Pr(Object) \cdot IoU^{Truth}\_{Pred} $$

		- 如果有目标的中心落在当前 Grid 中，\\(Pr(Object) = 1\\)；否则 \\(Pr(Object) = 0\\)

- 在 PASCAL VOC 上，\\(S = 7, B = 2, C = 20\\)，输出为 \\(7 \times 7 \times (5 \times 2 + 20) = 1470\\) 维，最多可以检测 \\(7 \times 7 \times 2 = 98\\) 个目标

### 训练

- 对于有目标中心落入的每个 Grid，从其预测的 \\(B\\) 个 BBox 中选择与 Ground Truth 的 \\(IoU\\) 最大的一个，负责对 Ground Truth 的预测，用于计算 loss

- 对各种 loss 加权求和：

	- BBox loss 和 Softmax loss 权重不应该相同

		- 更重视对 BBox 的预测，\\(\lambda\_{coord} = 5\\)

	- 如果一个 BBox 中没有目标，其 Confidence 会被置零；由于大部分 BBox 中没有目标，会导致网络训练不稳定

		- 对于没有目标的 BBox，其 loss 影响较小，\\(\lambda\_{noobj} = 0.5\\)

- Loss 由以下 5 项组成：

	- BBox 位置 Loss

		$$ \lambda\_{coord} \sum\_{i=0}^{S^{2}}\sum\_{j=0}^{B} \mathbb{1}\_{ij}^{obj} \left[ \left(x\_{i} - \hat{x\_{i}}\right)^{2} + \left(y\_{i} - \hat{y\_{i}}\right)^{2}\right] $$

		- \\(\mathbb{1}^{obj}\_{ij}\\) 判断第 \\(i\\) 个 Grid 中的第 \\(j\\) 个 BBox 是否负责预测 Ground Truth

	- BBox 尺度 Loss

		$$ \lambda\_{coord} \sum\_{i=0}^{S^{2}}\sum\_{j=0}^{B} \mathbb{1}\_{ij}^{obj} \left[ \left(\sqrt{w\_{i}} - \sqrt\{\hat{w\_{i}}}\right)^{2} + \left(\sqrt{h\_{i}} - \sqrt{\hat{h\_{i}}}\right)^{2}\right] $$

		- \\(w\\) 或 \\(h\\) 差距相同时，小物体的 loss 应该更大

	- 目标的 Confidence Loss

		$$ \sum\_{i=0}^{S^{2}} \sum\_{j=0}^{B} \mathbb{1}^{obj}\_{ij} \left(C\_{i} - \hat{C\_{i}}\right)^{2} $$

	- 背景的 Confidence Loss

		$$ \lambda\_{noobj} \sum\_{i=0}^{S^{2}} \sum\_{j=0}^{B} \mathbb{1}^{noobj}\_{ij} \left(C\_{i} - \hat{C\_{i}}\right)^{2} $$

	- 分类 Loss

		$$ \sum\_{i=0}^{S^{2}} \mathbb{1}^{obj}\_{i} \sum\_{c \in classes} \left(p\_{i} \ (c) - \hat{p\_{i}} \ (c)\right)^{2} $$

		- \\(\mathbb{1}^{obj}\_{i}\\) 判断是否有目标中心落在第 \\(i\\) 个 Grid 中

- 由 loss 定义可知：

	- 只有某个 Grid 中有目标的时候，才惩罚分类 loss

	- 只有某个 BBox 负责预测某个 Ground Truth 时，才惩罚 BBox 和 Confidence loss

		- 具体惩罚哪个 BBox，根据 \\(IoU\\) 决定

### 测试

- 根据 Confidence 和分类信息，计算每个 Grid 的每个 BBox 对每个类别的置信度：

	$$ Pr(Object) \cdot IoU^{Truth}\_{Pred} \cdot Pr(Class\_{i} | Object) = Pr(Class\_{i}) \cdot IoU^{Truth}\_{Pred} $$

- 过滤掉置信度较低的 BBox，通过 NMS 得到最终的检测结果

## 网络结构

![img](images/yolo_v1.png)

- 输入图像缩放到 \\(448 \times 448\\)

- 最后的两个全连接层输出 \\(1470\\) 维向量

- 采用 Leaky ReLU 作为激活函数

## 缺点

- 对靠太近的物体、小群体的检测效果不好

	- 多个目标中心落在同一 Grid 上，但每个 Grid 只能预测两个 BBox，而且只属于一类

- 对小目标的定位准确性较差

	- 训练时小目标和大目标的 BBox loss 对损失函数影响相同（平方根只是缓解，不能彻底解决问题）