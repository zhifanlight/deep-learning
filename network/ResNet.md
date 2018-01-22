<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# ResNet

## 基本思想

### 恒等映射

- 输入等于输出：\\(f(x)=x\\)

- 深层网络的性能不应该比浅层网络更差，恒等映射可以把原始信息传入更深的层，抑制信息退化

- 反向传播时，由于恒等映射这一支的导数为 \\(1\\)，可以将深层 loss 很好的保留到浅层，防止梯度消失

### 残差块

- 整个网络由多组残差块组成，同组残差块的特征图维度相同，而基本的残差块结构如下：

	![img](images/resnet_block.png)

- 实验表明，当恒等映射只跨越一层时效果较差，通常最少跨越两层

- 当 \\(x\\) 与 \\(F(x)\\) 维度相同时，直接按通道相加即可

- 当 \\(x\\) 与 \\(F(x)\\) 维度不同时，将 \\(W\_{s}x\\) 与 \\(F(x)\\) 按通道相加，其中 \\(W\_{s}\\) 是投影矩阵

### 去除池化层

- 在残差块之间，用步长为 \\(2\\) 的卷积代替池化实现下采样

- 除第一组残差块外，每组第一个残差块的第一层都进行上述下采样

### 加速计算

- 当网络较浅时，采用下图左侧的残差块

- 当网络较深时，由于计算量较大，采用下图右侧的残差块加速计算：

	- 先进行 \\(1 \times 1\\) 卷积减少特征图通道数

	- 再进行 \\(3 \times 3\\) 卷积

	- 最后进行 \\(1 \times 1\\) 卷积恢复特征图通道数

	![img](images/resnet_fast.png)

## 主要改进

- 使用残差块可以有效地训练 \\(1000\\) 层以上的网络

- 随着网络层数的加深，网络性能也变得更好

- 第一层使用普通卷积，第二层开始使用残差块

## 网络结构

![img](images/resnet.png)