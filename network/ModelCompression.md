# 模型压缩

## 背景介绍

- 主流的 $\mathrm{CNN}$ 存在以下问题：

  - 模型参数量巨大，模型本身占空间较大

  - 卷积层和全连接层需要大量的矩阵乘法，计算开销也大

- 这样的 $\mathrm{CNN}$ 在终端部署、低延迟需求场景下难以应用

- 一般来说，$\mathrm{CNN}$ 的模型参数主要来自全连接层，计算开销主要来自卷积层

- 根据压缩角度，可以分为两类：

  - 从网络权重角度

  - 从网络结构角度

- 从计算速度方面，也可以分为两类：

  - 仅压缩尺寸

  - 压缩尺寸的同时，提升速度

## 方法比较

方法 | 压缩角度 | 计算速度
:---: | :---: | :---:
$\mathrm{Deep \ Compression}$ | 权重 | $\mathrm{No}$
$\mathrm{SqueezeNet}$ | 结构 | $\mathrm{No}$
$\mathrm{MobileNet}$ | 结构 | $\mathrm{Yes}$
$\mathrm{ShuffleNet}$ | 结构 | $\mathrm{Yes}$

## $\mathrm{Deep \ Compression}$

- 不影响精度的前提下，把 $\mathrm{AlexNet}$ 模型压缩 $35$ 倍，把 $\mathrm{VGG}$ 模型压缩 $49$ 倍

- 计算速度提升 $3 - 4$ 倍，能耗降低 $3 - 7$ 倍

### 网络剪枝

- 对 $\mathrm{CNN}$ 进行正常预训练

- 删除权重小于一定阈值的所有连接

- 重新训练得到的稀疏网络

- 对于得到的稀疏权值矩阵，只存储非零值、与前一个非零值的索引距离：

  - 对于卷积层，每个距离占 $d = 8 \ \mathrm{bit}$

  - 对于全连接层，每个距离占 $d = 5 \ \mathrm{bit}$

  - 如果索引 $a, \ b$ 的距离超过 $2^{d}$，需要存储 $a + 2^{d}$ 处的零值，以防距离溢出

  <center>
  <img src="images/compression_pruning.png"/>
  </center>

### 权值共享与量化

- 将同一层的所有权值划分为 $k$ 个簇，计算每个权值对应聚类中心：

  - 对于每个权值，用相应的簇中心进行替换

  - 假设原始权值有 $n$ 个，每个权值占 $b$ 位，压缩前需要的空间为 $n b$

  - 压缩后，存储簇中心需要的空间为 $k b$

  - 对于每个权值，只需记录与其对应的簇索引，需要的总空间为 $n \log_{2} k$

    - 对于卷积层，每个索引为 $8 \ \mathrm{bit}$，即 $256$ 种共享权值

    - 对于全连接层，每个索引为 $5 \ \mathrm{bit}$，即 $32$ 种共享权值

  - 压缩率计算如下：

    $$
    r = \frac{nb}{n \log_{2} k + k b}
    $$

- 反向传播时，将簇内梯度和作为簇的总梯度，并更新簇中心

- 使用 $\mathrm{k-means}$ 进行聚类：

  - 计算当前层权值的边界 $\mathrm{min}, \ \mathrm{max}$，将 $\left[ \mathrm{min}, \ \mathrm{max} \right]$ 切为等长区间，将切分点作为初始聚类中心

  - 大的权值对 $\mathrm{CNN}$ 影响较大，但是占的比例较小

  - 无论是基于密度，还是随机选择聚类中心，都会降低大权值被选中的概率

### $\mathrm{Huffman}$ 编码

- 训练结束后，通过 $\mathrm{Huffman}$ 编码处理权值和索引，可以进一步压缩存储空间

## $\mathrm{SqueezeNet}$

- 不影响精度的前提下，把 $\mathrm{AlexNet}$ 的参数量压缩 $50$ 倍

- 结合 $\mathrm{Deep \ Compression}$ 的思想，可以将 $\mathrm{AlexNet}$ 模型压缩 $510$ 倍

### 设计思想

- 用 $1 \times 1$ 的卷积核代替 $3 \times 3$ 的卷积核

  - $1 \times 1$ 的卷积核可以将参数量压缩 $9$ 倍

  - 为了不影响精度，只替换部分 $3 \times 3$ 卷积核

- 减少输入 $3 \times 3$ 卷积的特征图通道数

  - 将原来的卷积层拆解为两层，并封装为一个 $\mathrm{Fire \ Module}$

- 减少或延迟 $\mathrm{pooling}$ 操作

  - 特征图分辨率越大，分类精度越高，但相应的计算量也越大

  - 减少或延迟 $\mathrm{pooling}$，可以在深层得到更大的特征图

- 用 $\mathrm{global \ average \ pooling}$ 层代替最后的全连接层，最大程度减少参数量

### $\mathrm{Fire \ Module}$

<center>
<img src="images/squeezenet.png"/>
</center>

- 首先通过 $\mathrm{squeeze}$ 层对特征图通道进行压缩

  - 使用 $1 \times 1$ 卷积核

  - 减少特征图通道数，不改变特征图维度

- 在 $\mathrm{expand}$ 层使用等量的 $1 \times 1$ 和 $3 \times 3$ 卷积核，之后对特征图进行拼接

- $\mathrm{squeeze}$ 层和 $\mathrm{expand}$ 层都使用 $\mathrm{ReLU}$ 激活函数

- 经过最后一个 $\mathrm{Fire \ Module}$ 后，按 $0.5$ 的概率进行 $\mathrm{Dropout}$

## $\mathrm{MobileNet}$

- 关于 $\mathrm{MobileNet}$，参考 [$\mathrm{MobileNet.md}$](MobileNet.md)

## $\mathrm{ShuffleNet}$

- 关于 $\mathrm{ShuffleNet}$，参考 [$\mathrm{ShuffleNet.md}$](ShuffleNet.md)