# $\mathrm{ResNet}$

## 基本思想

### 网络退化

- 浅网络的解空间是深网络的子集：

  - 对于结构相似的两个网络，如果将浅网络的参数复制到深网络，将深网络的其他层限制为恒等映射，那么深网络的性能不会比浅网络差

  - 实验结果却表明发生了网络退化，即深网络的性能比浅网络更差

  - 深网络在训练集上的错误率更高，说明网络退化不是过拟合的原因，而是训练过程出了问题

### 残差块

- 整个网络由多个残差块组成，残差块内的特征图维度相同，基本的残差块结构如下：

  <center>
  <img src="images/resnet_block.png"/>
  </center>

- 实验表明，当恒等映射只跨越一层时效果较差，通常最少跨越两层

- 当 $x$ 与 $F \left( x \right)$ 维度相同时，直接按通道相加即可

- 当 $x$ 与 $F \left( x \right)$ 维度不同时，将 $W_{s} x$ 与 $F \left( x \right)$ 按通道相加，其中 $W_{s}$ 是投影矩阵

#### 去除池化层

- 除第一组残差块外，每组第一个残差块的第一层都进行下采样

  - 第一个残差块的 $\mathrm{shortcut}$ 分支使用 $\mathrm{stride} = 2$ 的 $1 \times 1$ 卷积实现 $\mathrm{pooling}$

  - 对于其他残差块，$\mathrm{shortcut}$ 分支和卷积分支直接相加

#### 加速计算

- 当网络较浅时，采用下图左侧的残差块

- 当网络较深时，由于计算量较大，采用下图右侧的 $\mathrm{bottleneck}$ 结构：

  - 先进行 $1 \times 1$ 卷积减少特征图通道数

  - 再进行 $3 \times 3$ 卷积

  - 最后进行 $1 \times 1$ 卷积恢复特征图通道数

  <center>
  <img src="images/resnet_fast.png"/>
  </center>

### 恒等映射

#### 数学推导

- 把网络中的连续几层看作一个整体，假设输入为 $x$，真实映射关系为 $G \left( x \right)$，网络学习到的映射关系为 $F \left( x \right)$

  - 对于普通网络，优化目标是 $F \left( x \right) \approx G \left( x \right)$

  - 对于 $\mathrm{ResNet}$，恒等映射不用学习，另一支的优化目标是 $F \left( x \right) \approx G \left( x \right) - x$

- 相对于浅网络，深网络的某些层应该逼近恒等映射，即 $G \left( x \right) \approx x$

  - 普通网络的优化目标是 $F \left( x \right) \approx x$

  - $\mathrm{ResNet}$ 的优化目标是 $F \left( x \right) \approx 0$

- 相比于普通网络，$\mathrm{ResNet}$ 更容易优化：

  - 恒等映射的卷积核为 $\left[ \begin{matrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0\end{matrix} \right]$，零映射的卷积核为 $\left[ \begin{matrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0\end{matrix} \right]$

  - 神经网络初始化时，多为均值为 $0$ 的某种分布，学习零映射的卷积核比恒等映射的卷积核更容易

- 如果不考虑 $\mathrm{ReLU}$ 的作用，$\mathrm{ResNet}$ 在正向计算时可以将信息直接从浅层传到深层，反向传播时可以将深层 $\mathrm{loss}$ 直接从深层传到浅层，缓解梯度问题

  - 残差网络的一般形式如下：

    $$
    y_{l} = H \left( x_{l} \right) + F \left( x_{l}, \ w_{l} \right)
    $$

    $$
    x_{l + 1} = \delta \left( y_{l} \right)
    $$

    - 其中 $x_{l}$ 是第 $l$ 层输入，$H \left( x_{l} \right)$ 表示 $\mathrm{shortcut}$ 分支，$F \left( x_{l}, \ w_{l} \right)$ 表示卷积分支，$\delta$ 表示 $\mathrm{ReLU}$ 函数

  - 当 $\mathrm{shortcut}$ 是恒等映射，且不考虑 $\mathrm{ReLU}$ 的作用时：

    $$
    x_{l + 1} = x_{l} + F \left( x_{l}, \ w_{l} \right)
    $$

    - 当跨越多层时，递归计算如下：

      $$
      x_{L} = x_{l} + \sum_{i = 1}^{L - 1} F \left( x_{i}, \ w_{i} \right)
      $$

      - 相比于普通网络 $x_{L} = \sum_{i = 0}^{L - 1} w_{i} x_{0}$，正向计算时，浅层信息可以不经变换直接传递到深层

    - 在反向传播时，误差计算如下：

      $$
      \frac{\partial{J}}{\partial{x_{l}}} = \frac{\partial{J}}{\partial{x_{L}}} \cdot \frac{\partial{x_{L}}}{\partial{x_{l}}} = \frac{\partial{J}}{\partial{x_{L}}} \cdot \left( 1 + \frac{\partial}{\partial{x_{l}}} \sum_{i = 1}^{L - 1} F \left( x_{i}, \ w_{i} \right) \right)
      $$

      - 由于最后一项不可能总为 $-1$，深层梯度可以很好的传递到浅层

  - 如果$\mathrm{ shortcut}$ 不是恒等映射，即 $H \left( x \right) = \lambda x$ 时，由于 $\lambda_{i}$ 的叠乘，正向计算和反向传播的过程中容易产生梯度问题

#### $\mathrm{Pre-Activation}$

- 经典残差块结构为 $\mathrm{Conv \rightarrow BN \rightarrow ReLU}$，当使用 $\mathrm{BN \rightarrow ReLU \rightarrow Conv}$ 结构时，可进一步降低误差

  - 网络的两支相加后，不再经过激活函数，$\mathrm{shortcut}$ 分支成为真正的恒等映射

  - $\mathrm{Batch \ Normalization}$ 层前置，正则化效果更加明显

    - 原始 $\mathrm{ResNet}$ 中，卷积分支经过 $\mathrm{Batch \ Normalization}$ 后再与 $\mathrm{shortcut}$ 分支相加，会削弱 $\mathrm{Batch \ Normalization}$ 的作用

## 主要改进

- 使用残差块可以有效地训练 $1000$ 层以上的网络

- 随着网络层数的加深，网络性能也变得更好

- 第一层使用普通卷积，第二层开始使用残差块

## 网络结构

- 第一层是步长为 $2$ 的 $7 \times 7$ 卷积

- 第二层是步长为 $2$ 的 $3 \times 3 \ \mathrm{max-pooling}$

- 最后一层通过 $\mathrm{global \ average \ pooling}$ 后进行 $\mathrm{softmax}$ 分类

- 其余层分为 $4$ 组残差块，每组包含若干个残差块

</center>
<img src="images/resnet.png"/>
</center>