<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# HOG 特征

## 基本思想

- 在边缘具体位置未知的情况下，边缘方向的分布也可以表示目标的外形轮廓

- 由于 HOG 特征在图像的局部方格上操作，所以对几何形变、光学形变都能保持很好的不变性，这两种形变只会出现在更大的空间领域上

- 特别适合做计算机视觉中的人体检测

## 计算过程

### 图像预处理

- 由于颜色信息作用不大，通常先将 RGB 图像转换为灰度图像

- 通过 Gamma 变换，调节图像对比度，同时抑制噪声干扰

### 梯度计算

- 计算每个像素的梯度，包括大小和方向，以捕获轮廓信息

	- 大小：

		$$ G(x,y) = \sqrt{G\_{x}(x,y)^{2} + G\_{y}(x,y)^{2}} $$

	- 方向：

		$$ \alpha(x,y) = arc \ tan \left( \frac{G\_{y}(x,y)}{G\_{x}(x,y)}\right) $$

### cell 处理

- 将图像划分为若干个不重叠的 cell，比如每个 cell 分辨率为 \\(6 \times 6\\)

- 采用 \\(9\\) 个 bin 统计每个 cell 内的梯度，梯度方向决定统计区间，梯度大小作为统计权重

### block 处理

- 由于光照强度变化、前景-背景对比，梯度强度变化范围可能特别大，需要做归一化

- 把相邻（比如 \\(2 \times 2\\) 个）cell 组成更大的 block，将一个 block 内的所有 cell 向量串联起来组成该 block 的向量

- 对每个 block 向量分别归一化，串联所有归一化后的 block 向量作为最终 HOG 特征