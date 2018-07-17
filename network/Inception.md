<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Inception

## Inception v1

### Network In Network

- 为了提升网络性能，需要增加网络的深度或宽度，意味着大量参数：

	- 一方面会导致过拟合

	- 另一方面会导致计算量增加

- 在保证计算量不变的前提下，通过设计 Inception 结构，增加网络深度和宽度

	- 对于同一输入，通过组合不同尺度的卷积核，可以进行更好的提取特征

### Inception 结构

- 整个网络由多个 Inception 节点组成，而 Inception 节点的结构如下：

	![img](images/inception_v1.png)

#### 特征提取

- 采用 \\(1 \times 1, \ 3 \times 3, \ 5 \times 5\\) 的卷积核，分别提取特征

- 使用 \\(stride = 2\\) 的 \\(3 \times 3\\) pooling 提取特征，提高效率

- 将上述 \\(4\\) 组特征拼接后，作当前 Inception 节点的输出

#### 加速计算

- 特征图较多时，\\(3 \times 3, \ 5 \times 5\\) 卷积计算量巨大

- 因此先进行 \\(1 \times 1\\) 卷积，通过减少特征图通道数加速计算

### 主要改进

- 使用 Inception 结构后参数减少，缓解过拟合

- 网络变得更深、更宽，性能提高 \\(2-3\\) 倍

- 浅层使用普通卷积，深层使用 Inception 结构

- 训练时，为缓解网络加深导致的梯度消失，在中间层添加两个辅助 Softmax 传播梯度

- 测试时，去掉这两个 Softamx 层，只保留网络的最后输出

## Inception v2