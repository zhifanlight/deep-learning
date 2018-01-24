<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 训练技巧

## 数据增强

- 由于直接采集新样本代价较高，通常对已有样本进行预处理，以扩充数据集大小

- 对于图像数据，一般采用如下方法：

	- 随机裁剪

		- 从原始图像上随机截取固定大小的图像作为新样本

	- 旋转与翻转

		- 旋转一定角度或水平翻转，改变图像内容朝向

	- 缩放

		- 按一定比例对图像进行放大或缩小

	- 添加噪声

		- 对整张图像加高斯噪声或椒盐噪声

	- 调整光照

		- 将 RGB 图像变换到 HSV 空间，固定 H，调整 SV

		- 关于颜色空间，参考 [ColorSpace.md](../vision/ColorSpace.md)

## 正则化

- 通过 \\(L\_{1}\\) 或 \\(L\_{2}\\) 正则项，使参数大部分为 0 或尽量接近 0，一定程度上简化模型

- 关于正则化，参考 [Regularization.md](../basic/Regularization.md)

## Dropout

- 仅在训练时使用，以概率 \\(p\\) 让某些神经元不工作，并将其他神经元输出扩大 \\(\frac{1}{1-p}\\) 倍

- Dropout 使神经元随机失效，可以减少神经元之间的耦合，一定程度上简化模型

- 每次 Dropout 得到的模型都不相同，不同模型之间的过拟合可能相互抵消，从而整体上减少过拟合