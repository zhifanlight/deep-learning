<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# MobileNet v1

## 设计思想

- 在保证准确率的前提下，通过卷积分解提高响应速度，同时也可以减少参数量

- 模型较小，不易过拟合，可以减少正则化的使用

### 卷积分解

- 将标准卷积层分解成两个卷积层：depthwise 卷积和 pointwise 卷积

	- 第一个卷积层的每个 filter 只处理一个输入调整图

	- 第二个卷积层通过 \\(1 \times 1\\) 卷积将上述结果进行组合，提取新的特征

- 网络中大部分是 \\(1 \times 1\\) 卷积，可以最大程度的加速计算：

	- \\(1 \times 1\\) 卷积无需经过 im2col，可以直接按通道线性加权

## 网络结构

- 不使用 pooling 层，在 depthwise 卷积层中使用步长为 \\(2\\) 的 \\(3 \times 3\\) 卷积

- 网络的第一层是标准卷积层，最后一层通过 global average pooling 后进行 softmax 分类，中间层结构如下：

![img](images/mobilenet_v1.png)

## 性能分析

- 假设输入特征图维度为 \\(D\_{F} \times D\_{F} \times M\\)，输出特征图维度为 \\(D\_{G} \times D\_{G} \times N\\)，卷积核尺度为 \\(D\_{K} \times D\_{K}\\)，步长 \\(stride = 1\\)

- 对于标准卷积，计算量如下：

	$$ D\_{F} \cdot D\_{F} \cdot M \cdot N \cdot D\_{K} \cdot D\_{K} $$

- 经过卷积分解，计算量如下：

	$$ D\_{F} \cdot D\_{F} \cdot M \cdot D\_{K} \cdot D\_{K} + D\_{F} \cdot D\_{F} \cdot M \cdot N $$

	- depthwise 卷积计算量：

		$$ D\_{F} \cdot D\_{F} \cdot M \cdot D\_{K} \cdot D\_{K} $$

	- pointwise 卷积计算量：

		$$ D\_{F} \cdot D\_{F} \cdot M \cdot N $$

- 相比标准卷积，运行时间变为：

	$$ T = \frac{1}{N} + \frac{1}{D\_{K}^{2}} $$

	- 对于 \\(3 \times 3\\) 卷积，理论上加速 \\(8-9\\) 倍

	- 使用 Caffe 训练时，实际速度较慢：

		- Caffe 的 group 卷积按 group 依次运行，无法充分发挥 GPU 优势

		- 内核函数多次启动导致的额外时间，进一步降低了运行速度

- 对于分类任务，运行速度是 VGG 的 27 倍，准确率接近

- 对于检测任务，相比 Faster R-CNN 和 SSD，准确率有所下降，但速度提升较大

## 超参数

- 除了上述基准模型，还可以实现更小的 MobileNet

- MobileNet 系列由以下两个超参数控制

### 宽度乘子 \\(\alpha\\)

- \\(\alpha \in (0, 1]\\)，通常取值为 \\(0.25, \ 0.5, \ 0.75, \ 1.0\\)

- 对于给定层和宽度乘子 \\(\alpha\\)，输入通道数由 \\(M\\) 变为 \\(\alpha M\\)，输出通道数由 \\(N\\) 变为 \\(\alpha N\\)

- 总计算量变为：

	$$ D\_{F} \cdot D\_{F} \cdot \alpha M \cdot D\_{K} \cdot D\_{K} + D\_{F} \cdot D\_{F} \cdot \alpha M \cdot \alpha N $$

	- 即 \\(\alpha\\) 对计算量的影响是平方关系

### 分辨率乘子 \\(\rho\\)

- 对于给定层和分辨率乘子 \\(\rho\\)，输入图像尺度由 \\(D\_{F}\\) 变为 \\(\rho D\_{F}\\)

- 实际实现时，通过控制输入尺寸实现衰减，输入尺寸分别为 \\(\\{224, \ 192, \ 160, \ 128\\}\\)

- 总计算量变为：

	$$ \rho D\_{F} \cdot \rho D\_{F} \cdot M \cdot D\_{K} \cdot D\_{K} + \rho D\_{F} \cdot \rho D\_{F} \cdot M \cdot N $$

	- 即 \\(\rho\\) 对计算量的影响也是平方关系